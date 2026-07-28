//! CLI-transport LLM provider for the backend Agent.
//!
//! Why this exists: `LlmConfig::is_usable_for_backend_http` deliberately
//! returns `false` for `claude-code` because that provider is a local
//! subprocess, not an HTTP endpoint. Until now nothing filled the gap on
//! the backend side, so a project whose active provider was `claude-code`
//! reached `AgentRuntime` with *no usable generator at all* and silently
//! degraded to the ranked-listing template — the chat route answered in
//! ~70ms having never invoked a model. (QA finding F-8.)
//!
//! The desktop UI never hit this because it owns a parallel transport in
//! `src/lib/claude-cli-transport.ts`, which drives the same binary through
//! the `claude_cli_spawn` Tauri command and webview events. That transport
//! is unreachable from the HTTP API and MCP surfaces: there is no webview
//! to emit events into. This module is the backend-side equivalent —
//! request/response instead of event streaming, but the same binary, the
//! same argv, and the same stream-json wire format.
//!
//! Scope: text completion only. `claude` runs with `-p` (non-interactive
//! print mode); its agentic tools, MCP servers, and session resumption are
//! not used. This is a *local subprocess*, not a network call: it makes no
//! request that the configured provider would not already make.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use super::provider::{AgentLlmProvider, LlmConfig};
use super::types::AgentImage;
use crate::commands::cli_resolver::{child_path_env, find_cli_command};

/// Hard ceiling on a single CLI turn. The HTTP API caller is blocked for
/// the duration, and `tiny_http` has no request timeout of its own, so an
/// authenticated-but-wedged `claude` would otherwise hold the connection
/// (and an in-flight slot) open indefinitely.
const CLI_TURN_TIMEOUT: Duration = Duration::from_secs(300);

/// Cap on captured-but-unparsed stdout retained for diagnostics. Mirrors
/// the frontend transport's buffer: enough to surface an auth error the
/// parser did not classify, small enough not to echo a whole transcript
/// into an error string.
const UNPARSED_BUFFER_CAP: usize = 4096;

/// Providers whose transport is a local CLI subprocess rather than HTTP.
/// `codex-cli` is deliberately absent: its argv and wire format differ,
/// and shipping an untested second transport would be a guess.
pub fn is_cli_transport_provider(provider: &str) -> bool {
    provider == "claude-code"
}

/// CLI-transport counterpart to `LlmConfig::is_usable_for_backend_http`.
/// A CLI provider needs no API key — authentication lives in the user's
/// `~/.claude` OAuth state — so a model name is the only requirement.
pub fn is_usable_for_cli_transport(config: &LlmConfig) -> bool {
    is_cli_transport_provider(config.provider.as_str()) && !config.model.trim().is_empty()
}

pub struct ClaudeCliProvider {
    model: String,
    working_directory: PathBuf,
    isolate_local_config: bool,
    /// Explicit binary path, bypassing PATH resolution. Tests point this at
    /// a stub that speaks stream-json so the subprocess contract — argv,
    /// stdin framing, stdout parsing, exit handling — is exercised for real
    /// without depending on an installed and authenticated `claude`.
    command_override: Option<PathBuf>,
}

impl ClaudeCliProvider {
    pub fn new(config: &LlmConfig, working_directory: impl Into<PathBuf>) -> Self {
        Self {
            model: config.model.trim().to_string(),
            working_directory: working_directory.into(),
            isolate_local_config: config.local_cli_isolation.unwrap_or(false),
            command_override: None,
        }
    }

    #[cfg(test)]
    fn with_command(mut self, command: impl Into<PathBuf>) -> Self {
        self.command_override = Some(command.into());
        self
    }

    async fn run_turn(
        &self,
        system: &str,
        user: &str,
        images: &[AgentImage],
        mut on_delta: Option<Box<dyn FnMut(&str) + Send + '_>>,
    ) -> Result<String, String> {
        let claude = match self.command_override.clone() {
            Some(path) => path,
            None => find_cli_command("claude", &["claude.cmd", "claude.exe"])
                .await
                .map_err(|err| {
                    format!(
                        "Claude Code CLI not found ({err}). Install `claude` or pick an HTTP \
                         provider in Settings → LLM Provider."
                    )
                })?,
        };

        let mut content = vec![json!({ "type": "text", "text": merge_system_preamble(system, user) })];
        for image in images {
            content.push(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data_base64,
                },
            }));
        }

        let mut cmd = Command::new(&claude);
        suppress_windows_console(&mut cmd);
        // Desktop apps do not inherit the login-shell PATH, and an
        // npm-installed `claude` is a Node shim that needs `node` on PATH
        // to run at all. Same treatment as the frontend transport.
        if let Some(path_env) = child_path_env().await {
            cmd.env("PATH", path_env);
        }
        cmd.args(build_cli_args(&self.model, self.isolate_local_config));
        cmd.current_dir(&self.working_directory);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn claude: {e}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Missing stdin handle".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Missing stdout handle".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "Missing stderr handle".to_string())?;

        // `content` must be a block array, never a bare string: the CLI
        // probes each block for `tool_use_id` and crashes on a raw string.
        let line = format!(
            "{}\n",
            json!({
                "type": "user",
                "message": { "role": "user", "content": content },
            })
        );
        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| format!("Failed to write to claude stdin: {e}"))?;
        stdin
            .flush()
            .await
            .map_err(|e| format!("Failed to flush claude stdin: {e}"))?;
        drop(stdin);

        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            let mut collected = String::new();
            while let Ok(Some(line)) = lines.next_line().await {
                collected.push_str(&line);
                collected.push('\n');
            }
            collected
        });

        let mut parser = ClaudeStreamParser::default();
        let mut answer = String::new();
        let mut unparsed = String::new();
        let mut reader = BufReader::new(stdout).lines();

        let drain = async {
            while let Some(line) = reader
                .next_line()
                .await
                .map_err(|e| format!("Failed to read claude stdout: {e}"))?
            {
                match parser.parse_line(&line) {
                    Some(text) => {
                        if let Some(sink) = on_delta.as_mut() {
                            sink(&text);
                        }
                        answer.push_str(&text);
                    }
                    None => {
                        if unparsed.len() < UNPARSED_BUFFER_CAP && !line.trim().is_empty() {
                            unparsed.push_str(&line);
                            unparsed.push('\n');
                        }
                    }
                }
            }
            Ok::<(), String>(())
        };

        match tokio::time::timeout(CLI_TURN_TIMEOUT, drain).await {
            Ok(result) => result?,
            Err(_) => {
                let _ = child.kill().await;
                return Err(format!(
                    "claude CLI timed out after {}s",
                    CLI_TURN_TIMEOUT.as_secs()
                ));
            }
        }

        let status = child
            .wait()
            .await
            .map_err(|e| format!("Failed to wait for claude: {e}"))?;
        let stderr_text = stderr_task.await.unwrap_or_default();

        if !status.success() {
            return Err(build_exit_error(status.code(), &stderr_text, &unparsed));
        }

        // Deltas exhausted without text: fall back to the terminal
        // `result` event, which carries the full answer on CLI builds
        // that emit no assistant/stream_event pair.
        let answer = if answer.trim().is_empty() {
            parser.result_text.clone().unwrap_or(answer)
        } else {
            answer
        };

        if answer.trim().is_empty() {
            let details = if !stderr_text.trim().is_empty() {
                stderr_text.trim().to_string()
            } else {
                unparsed.trim().to_string()
            };
            return Err(if details.is_empty() {
                "Claude Code CLI exited successfully but returned no content.".to_string()
            } else {
                format!("Claude Code CLI exited successfully but returned no content:\n{details}")
            });
        }

        Ok(answer)
    }
}

impl AgentLlmProvider for ClaudeCliProvider {
    fn provider_name(&self) -> &str {
        "claude-code"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn generate_text<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
        images: &'a [AgentImage],
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
        Box::pin(async move { self.run_turn(system, user, images, None).await })
    }

    fn generate_text_stream<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
        images: &'a [AgentImage],
        on_delta: Box<dyn FnMut(&str) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
        Box::pin(async move { self.run_turn(system, user, images, Some(on_delta)).await })
    }
}

/// The CLI has no portable system-prompt flag across supported versions,
/// so the system text is folded into the user turn — same approach as the
/// frontend transport, which keeps behaviour identical across surfaces.
fn merge_system_preamble(system: &str, user: &str) -> String {
    let system = system.trim();
    if system.is_empty() {
        return user.to_string();
    }
    format!("{system}\n\n{user}")
}

fn build_cli_args(model: &str, isolate_local_config: bool) -> Vec<String> {
    let mut args = vec![
        "-p".to_string(),
        "--output-format".to_string(),
        "stream-json".to_string(),
        "--input-format".to_string(),
        "stream-json".to_string(),
        "--verbose".to_string(),
    ];
    if isolate_local_config {
        args.extend([
            "--setting-sources".to_string(),
            "project".to_string(),
            "--strict-mcp-config".to_string(),
            "--mcp-config".to_string(),
            "{\"mcpServers\":{}}".to_string(),
            "--disable-slash-commands".to_string(),
            "--tools".to_string(),
            String::new(),
            "--no-session-persistence".to_string(),
            "--prompt-suggestions".to_string(),
            "false".to_string(),
        ]);
    }
    args.extend(["--model".to_string(), model.to_string()]);
    args
}

fn suppress_windows_console(_cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        _cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

pub fn build_exit_error(code: Option<i32>, stderr: &str, unparsed_stdout: &str) -> String {
    let code = code
        .map(|c| c.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let lower = stderr.to_ascii_lowercase();
    if lower.contains("unauthenticated")
        || lower.contains("authentication failed")
        || lower.contains("please log in")
    {
        return format!(
            "Claude Code CLI is not authenticated. Run `claude` in a terminal to complete the \
             OAuth login, then retry. LLM Wiki only spawns the binary — it cannot run the login \
             flow on your behalf. (exit {code})"
        );
    }
    if !stderr.trim().is_empty() {
        return format!("claude CLI exited with code {code}: {}", stderr.trim());
    }
    if !unparsed_stdout.trim().is_empty() {
        return format!(
            "claude CLI exited with code {code} (no stderr). Unparsed stdout:\n{}",
            unparsed_stdout.trim()
        );
    }
    format!("claude CLI exited silently with code {code}.")
}

/// Incremental stream-json reader. Port of `createClaudeCodeStreamParser`
/// in `src/lib/claude-cli-transport.ts` — kept behaviourally identical so
/// the API and the desktop UI cannot disagree about what the CLI said.
///
/// `assistant` events carry the whole in-progress message on every
/// emission (NOT a delta), while `stream_event` passthrough carries real
/// token deltas. Emitting both would double the answer, so once a delta
/// is seen the fat `assistant` events are ignored.
#[derive(Default)]
pub struct ClaudeStreamParser {
    saw_delta: bool,
    emitted_from_assistant: String,
    pub result_text: Option<String>,
}

impl ClaudeStreamParser {
    pub fn parse_line(&mut self, raw_line: &str) -> Option<String> {
        let line = raw_line.trim();
        if line.is_empty() {
            return None;
        }
        let event: Value = serde_json::from_str(line).ok()?;
        match event.get("type").and_then(Value::as_str)? {
            "stream_event" => {
                let inner = event.get("event")?;
                if inner.get("type").and_then(Value::as_str)? != "content_block_delta" {
                    return None;
                }
                let delta = inner.get("delta")?;
                if delta.get("type").and_then(Value::as_str)? != "text_delta" {
                    return None;
                }
                let text = delta.get("text").and_then(Value::as_str)?;
                self.saw_delta = true;
                Some(text.to_string())
            }
            "assistant" => {
                let content = event.get("message")?.get("content")?.as_array()?;
                let text: String = content
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect();
                if text.is_empty() || self.saw_delta {
                    return None;
                }
                if let Some(novel) = text.strip_prefix(&self.emitted_from_assistant) {
                    let novel = novel.to_string();
                    self.emitted_from_assistant = text;
                    return if novel.is_empty() { None } else { Some(novel) };
                }
                self.emitted_from_assistant = text.clone();
                Some(text)
            }
            "result" => {
                // Terminal summary. Captured, never emitted inline: on a
                // normal turn its text duplicates the assistant message.
                if let Some(text) = event.get("result").and_then(Value::as_str) {
                    if !text.trim().is_empty() {
                        self.result_text = Some(text.to_string());
                    }
                }
                None
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(provider: &str, model: &str) -> LlmConfig {
        serde_json::from_value(json!({
            "provider": provider,
            "model": model,
            "apiKey": "",
        }))
        .unwrap()
    }

    #[test]
    fn claude_code_is_usable_over_cli_but_not_over_http() {
        let cfg = config("claude-code", "sonnet");
        // The HTTP predicate must stay false: this is the invariant that
        // kept the agent loop from trying to POST to a subprocess.
        assert!(!cfg.is_usable_for_backend_http());
        assert!(is_usable_for_cli_transport(&cfg));
    }

    #[test]
    fn cli_transport_requires_a_model_and_a_cli_provider() {
        assert!(!is_usable_for_cli_transport(&config("claude-code", "   ")));
        assert!(!is_usable_for_cli_transport(&config("openai", "gpt-4o")));
        // codex-cli has a different argv and wire format; claiming it here
        // would spawn `claude` for a provider the user did not choose.
        assert!(!is_usable_for_cli_transport(&config("codex-cli", "gpt-5")));
    }

    #[test]
    fn parser_prefers_deltas_and_ignores_fat_assistant_events() {
        let mut parser = ClaudeStreamParser::default();
        assert_eq!(
            parser.parse_line(
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}}"#
            ),
            Some("Hello".to_string())
        );
        assert_eq!(
            parser.parse_line(
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}}"#
            ),
            Some(" world".to_string())
        );
        // Same text arriving as a full assistant message must not double.
        assert_eq!(
            parser.parse_line(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]}}"#
            ),
            None
        );
    }

    #[test]
    fn parser_diffs_successive_assistant_messages_when_no_deltas_arrive() {
        let mut parser = ClaudeStreamParser::default();
        assert_eq!(
            parser.parse_line(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Ver"}]}}"#
            ),
            Some("Ver".to_string())
        );
        assert_eq!(
            parser.parse_line(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Verónica"}]}}"#
            ),
            Some("ónica".to_string())
        );
    }

    #[test]
    fn parser_ignores_session_init_tool_use_and_malformed_lines() {
        let mut parser = ClaudeStreamParser::default();
        assert_eq!(parser.parse_line(r#"{"type":"system","subtype":"init"}"#), None);
        assert_eq!(
            parser.parse_line(r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"x"}]}}"#),
            None
        );
        assert_eq!(parser.parse_line("not json at all"), None);
        assert_eq!(parser.parse_line("   "), None);
    }

    #[test]
    fn parser_captures_result_text_without_emitting_it() {
        let mut parser = ClaudeStreamParser::default();
        assert_eq!(
            parser.parse_line(r#"{"type":"result","subtype":"success","result":"final answer"}"#),
            None
        );
        assert_eq!(parser.result_text.as_deref(), Some("final answer"));
    }

    #[test]
    fn system_preamble_is_folded_into_the_user_turn() {
        assert_eq!(merge_system_preamble("SYS", "USER"), "SYS\n\nUSER");
        assert_eq!(merge_system_preamble("   ", "USER"), "USER");
    }

    #[test]
    fn cli_args_carry_print_mode_stream_json_and_the_model() {
        let args = build_cli_args("sonnet", false);
        assert!(args.contains(&"-p".to_string()));
        assert_eq!(
            args.windows(2)
                .find(|w| w[0] == "--model")
                .map(|w| w[1].clone()),
            Some("sonnet".to_string())
        );
        assert!(!args.contains(&"--strict-mcp-config".to_string()));
        let isolated = build_cli_args("sonnet", true);
        assert!(isolated.contains(&"--strict-mcp-config".to_string()));
        assert!(isolated.contains(&"--no-session-persistence".to_string()));
    }

    /// Write an executable stub that behaves like `claude -p
    /// --output-format stream-json`: it drains stdin, then emits the given
    /// stdout lines and exits with `exit_code`. Unix-only because the
    /// stub is a shell script; the code under test is platform-neutral.
    #[cfg(unix)]
    fn stream_json_stub(name: &str, stdout_lines: &[&str], stderr: &str, exit_code: i32) -> PathBuf {
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt as _;

        let path = std::env::temp_dir().join(format!("llm-wiki-claude-stub-{name}"));
        let mut script = String::from("#!/bin/sh\ncat > /dev/null\n");
        for line in stdout_lines {
            // Single-quote the payload; stream-json lines contain no
            // single quotes, and printf keeps backslashes literal.
            script.push_str(&format!("printf '%s\\n' '{line}'\n"));
        }
        if !stderr.is_empty() {
            script.push_str(&format!("printf '%s\\n' '{stderr}' >&2\n"));
        }
        script.push_str(&format!("exit {exit_code}\n"));

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        drop(file);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    fn stub_provider(stub: PathBuf) -> ClaudeCliProvider {
        ClaudeCliProvider::new(&config("claude-code", "sonnet"), std::env::temp_dir())
            .with_command(stub)
    }

    /// The whole point of the order: a `claude-code` config must produce
    /// model text through a spawned subprocess. This exercises the real
    /// spawn, the real stdin framing, and the real stdout parse.
    #[cfg(unix)]
    #[tokio::test]
    async fn cli_transport_round_trips_stream_json_into_prose() {
        let stub = stream_json_stub(
            "round-trip",
            &[
                r#"{"type":"system","subtype":"init","session_id":"abc"}"#,
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Verónica Pierce is "}}}"#,
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"cédula 9-0110-0855."}}}"#,
                r#"{"type":"result","subtype":"success","result":"ignored summary"}"#,
            ],
            "",
            0,
        );
        let answer = stub_provider(stub)
            .generate_text("SYS", "Who is Verónica Pierce?", &[])
            .await
            .expect("stub must round trip");
        assert_eq!(answer, "Verónica Pierce is cédula 9-0110-0855.");
        // The terminal `result` event duplicates the answer; emitting it
        // too would double every reply.
        assert!(!answer.contains("ignored summary"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cli_transport_streams_deltas_to_the_callback_in_order() {
        let stub = stream_json_stub(
            "streaming",
            &[
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"one "}}}"#,
                r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"two"}}}"#,
            ],
            "",
            0,
        );
        let collected = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink = std::sync::Arc::clone(&collected);
        let answer = stub_provider(stub)
            .generate_text_stream(
                "SYS",
                "q",
                &[],
                Box::new(move |delta: &str| sink.lock().unwrap().push(delta.to_string())),
            )
            .await
            .unwrap();
        assert_eq!(answer, "one two");
        assert_eq!(*collected.lock().unwrap(), vec!["one ", "two"]);
    }

    /// A non-zero exit must surface as an error, never as an empty answer:
    /// an empty answer would be composed into the chat window as though
    /// the model had nothing to say.
    #[cfg(unix)]
    #[tokio::test]
    async fn cli_transport_reports_a_failed_exit_with_its_stderr() {
        let stub = stream_json_stub("failure", &[], "Unauthenticated: session expired", 1);
        let error = stub_provider(stub)
            .generate_text("SYS", "q", &[])
            .await
            .expect_err("non-zero exit must be an error");
        assert!(error.contains("not authenticated"), "{error}");
    }

    /// Exit 0 with no assistant text is the silent-failure case. It must
    /// also be an error so the caller falls back visibly instead of
    /// rendering a blank answer.
    #[cfg(unix)]
    #[tokio::test]
    async fn cli_transport_treats_a_silent_success_as_an_error() {
        let stub = stream_json_stub(
            "silent",
            &[r#"{"type":"system","subtype":"init"}"#],
            "",
            0,
        );
        let error = stub_provider(stub)
            .generate_text("SYS", "q", &[])
            .await
            .expect_err("empty success must be an error");
        assert!(error.contains("returned no content"), "{error}");
    }

    /// Live transport proof. Ignored by default because it spawns the real
    /// `claude` binary and consumes the developer's Claude Code session;
    /// run deliberately with:
    ///     cargo test -p llm-wiki claude_cli_round_trip -- --ignored --nocapture
    ///
    /// This is the assertion that the whole order rests on: that a
    /// `claude-code` provider can, in fact, be invoked from the backend
    /// Agent. A green unit-test suite proves the parser; only this proves
    /// the subprocess.
    #[tokio::test]
    #[ignore = "spawns the real claude CLI and uses the developer's session"]
    async fn claude_cli_round_trip_returns_model_prose() {
        let provider = ClaudeCliProvider::new(&config("claude-code", "haiku"), std::env::temp_dir());
        let answer = provider
            .generate_text(
                "You are a terse test fixture. Reply with exactly one short sentence.",
                "In one sentence, what colour is a clear midday sky?",
                &[],
            )
            .await
            .expect("claude CLI must be invocable from the backend agent");
        println!("--- claude CLI round trip ---\n{answer}\n---");
        assert!(!answer.trim().is_empty());
        assert!(answer.to_lowercase().contains("blue"));
    }

    #[test]
    fn exit_error_names_the_auth_failure_explicitly() {
        let message = build_exit_error(Some(1), "Unauthenticated: token expired", "");
        assert!(message.contains("not authenticated"));
        // A bare exit code is unactionable; stderr must survive otherwise.
        assert!(build_exit_error(Some(2), "boom", "").contains("boom"));
        assert!(build_exit_error(Some(2), "", "{\"type\":\"error\"}").contains("Unparsed stdout"));
        assert!(build_exit_error(None, "", "").contains("unknown"));
    }
}
