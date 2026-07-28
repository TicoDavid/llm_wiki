use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_HISTORY_CONTENT_BYTES: usize = 512 * 1024;
const MAX_ENTRIES_PER_FILE: usize = 30;
static HISTORY_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileHistoryEntry {
    pub id: String,
    pub path: String,
    pub timestamp: i64,
    pub author: String,
    pub tool: String,
    pub content: String,
}

/// Return provenance only, never historical content, for Agent retrieval
/// briefings. Reading the same bounded store as the timeline keeps attribution
/// consistent without expanding prompt size or exposing rollback snapshots.
pub fn latest_file_version(path: &Path) -> Option<(i64, String, String)> {
    let root = project_root_for(path)?;
    let _guard = HISTORY_LOCK.lock().ok()?;
    let raw = fs::read_to_string(history_path(&root, path)).ok()?;
    let entries: Vec<FileHistoryEntry> = serde_json::from_str(&raw).ok()?;
    entries
        .last()
        .map(|entry| (entry.timestamp, entry.author.clone(), entry.tool.clone()))
}

fn project_root_for(path: &Path) -> Option<PathBuf> {
    let mut cursor = path.parent();
    while let Some(dir) = cursor {
        if dir.join(".llm-wiki").is_dir() {
            return Some(dir.to_path_buf());
        }
        cursor = dir.parent();
    }
    None
}

/// Reduce a path to the platform-neutral string the history key is derived
/// from: drop any Windows verbatim (`\\?\`) prefix and spell every separator as
/// `/`, so a path that came back from `canonicalize()` and the raw forward-slash
/// path the UI hands us reduce to the same text.
fn key_text(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    match text.strip_prefix("//?/") {
        // `\\?\UNC\server\share` is the verbatim spelling of `\\server\share`.
        Some(rest) => match rest.strip_prefix("UNC/") {
            Some(share) => format!("//{share}"),
            None => rest.to_string(),
        },
        None => text,
    }
}

/// Address the history store by a key both sides of the feature agree on.
///
/// The writer is handed the caller's path as given while the readers go through
/// `checked_file()`, which canonicalises. Hashing whatever string each side
/// happened to hold made them disagree in two ways: `canonicalize()` rewrites
/// every separator to `\` and returns a `\\?\` form on Windows, and it resolves
/// symlinks on every platform. Either way the reader opened a store the writer
/// never wrote. Resolving and normalising here — identically, whichever side
/// calls — is what keeps the two in sync; callers may pass any spelling.
fn history_path(root: &Path, path: &Path) -> PathBuf {
    // All-or-nothing: mixing a resolved path with an unresolved root would
    // reintroduce exactly the mismatch this function exists to prevent.
    let (root_key, path_key) = match (root.canonicalize(), path.canonicalize()) {
        (Ok(resolved_root), Ok(resolved_path)) => {
            (key_text(&resolved_root), key_text(&resolved_path))
        }
        _ => (key_text(root), key_text(path)),
    };
    let relative = match path_key.strip_prefix(root_key.trim_end_matches('/')) {
        // Require a separator so `/proj-old/page.md` is not read as living
        // under `/proj`.
        Some(rest) if rest.starts_with('/') => rest.trim_start_matches('/'),
        _ => path_key.as_str(),
    };
    // Fixed FNV-1a keeps history addresses stable across Rust/toolchain upgrades.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in relative.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let key = format!("{hash:016x}");
    root.join(".llm-wiki/history").join(format!("{key}.json"))
}

pub fn record_file_version(path: &Path, author: &str, tool: &str) {
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    if !metadata.is_file() || metadata.len() as usize > MAX_HISTORY_CONTENT_BYTES {
        return;
    }
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let Some(root) = project_root_for(path) else {
        return;
    };
    if path.starts_with(root.join(".llm-wiki")) {
        return;
    }
    let Ok(_guard) = HISTORY_LOCK.lock() else {
        return;
    };
    let store_path = history_path(&root, path);
    let mut entries: Vec<FileHistoryEntry> = fs::read_to_string(&store_path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default();
    if entries.last().is_some_and(|entry| entry.content == content) {
        return;
    }
    entries.push(FileHistoryEntry {
        id: Uuid::new_v4().to_string(),
        path: path.to_string_lossy().replace('\\', "/"),
        timestamp: Utc::now().timestamp_millis(),
        author: author.to_string(),
        tool: tool.to_string(),
        content,
    });
    if entries.len() > MAX_ENTRIES_PER_FILE {
        entries.drain(..entries.len() - MAX_ENTRIES_PER_FILE);
    }
    if let Some(parent) = store_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(raw) = serde_json::to_string(&entries) {
        let _ = fs::write(store_path, raw);
    }
}

fn checked_file(project_path: &str, file_path: &str) -> Result<(PathBuf, PathBuf), String> {
    let root = Path::new(project_path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let file = Path::new(file_path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !file.starts_with(&root) || file.starts_with(root.join(".llm-wiki")) {
        return Err("History path must stay inside the project".to_string());
    }
    Ok((root, file))
}

#[tauri::command]
pub async fn list_file_history(
    project_path: String,
    file_path: String,
) -> Result<Vec<FileHistoryEntry>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (root, file) = checked_file(&project_path, &file_path)?;
        let raw =
            fs::read_to_string(history_path(&root, &file)).unwrap_or_else(|_| "[]".to_string());
        let mut entries: Vec<FileHistoryEntry> = serde_json::from_str(&raw).unwrap_or_default();
        entries.reverse();
        Ok(entries)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn restore_file_history(
    project_path: String,
    file_path: String,
    entry_id: String,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let (root, file) = checked_file(&project_path, &file_path)?;
        let raw = fs::read_to_string(history_path(&root, &file)).map_err(|e| e.to_string())?;
        let entries: Vec<FileHistoryEntry> =
            serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let entry = entries
            .into_iter()
            .find(|entry| entry.id == entry_id)
            .ok_or_else(|| "History entry not found".to_string())?;
        fs::write(&file, &entry.content).map_err(|e| e.to_string())?;
        record_file_version(&file, "human", "history.restore");
        Ok(entry.content)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn records_and_restores_append_only_versions() {
        let root = std::env::temp_dir().join(format!("llm-wiki-history-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join(".llm-wiki")).unwrap();
        fs::create_dir_all(root.join("wiki")).unwrap();
        let file = root.join("wiki/page.md");
        fs::write(&file, "before").unwrap();
        record_file_version(&file, "baseline", "before.test");
        fs::write(&file, "after").unwrap();
        record_file_version(&file, "agent", "test.write");

        let entries = list_file_history(
            root.to_string_lossy().into_owned(),
            file.to_string_lossy().into_owned(),
        )
        .await
        .unwrap();
        assert_eq!(entries.len(), 2);
        let old = entries
            .iter()
            .find(|entry| entry.content == "before")
            .unwrap();
        restore_file_history(
            root.to_string_lossy().into_owned(),
            file.to_string_lossy().into_owned(),
            old.id.clone(),
        )
        .await
        .unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "before");
        let restored = list_file_history(
            root.to_string_lossy().into_owned(),
            file.to_string_lossy().into_owned(),
        )
        .await
        .unwrap();
        assert_eq!(restored.first().unwrap().tool, "history.restore");
        let _ = fs::remove_dir_all(root);
    }

    /// The writer keys the store off the path as given; the readers key it off
    /// `canonicalize()`. Every spelling of one logical file must therefore land
    /// on one key, or history is recorded at an address nothing ever reads.
    /// These paths do not exist, so this exercises the string normalisation on
    /// every lane rather than whatever the local filesystem happens to resolve.
    #[test]
    fn history_key_is_identical_across_path_spellings() {
        let key_of = |root: &str, file: &str| {
            history_path(Path::new(root), Path::new(file))
                .file_name()
                .expect("history path has a file name")
                .to_string_lossy()
                .into_owned()
        };

        // Pinned, not merely self-consistent: this is FNV-1a of `wiki/page.md`,
        // the key the writer has always produced. Existing stores stay readable.
        let expected = "915f2613ee9ad2ac.json";
        assert_eq!(key_of("/llm-wiki-absent", "/llm-wiki-absent/wiki/page.md"), expected);

        for (root_form, file_form) in [
            // forward slash, as `build_tree` emits and the UI passes through
            ("C:/llm-wiki-absent", "C:/llm-wiki-absent/wiki/page.md"),
            // plain backslash
            (r"C:\llm-wiki-absent", r"C:\llm-wiki-absent\wiki\page.md"),
            // the mixed spelling `root.join("wiki/page.md")` really produces
            (r"C:\llm-wiki-absent", r"C:\llm-wiki-absent\wiki/page.md"),
            // the verbatim long-name form `canonicalize()` returns on Windows
            (r"\\?\C:\llm-wiki-absent", r"\\?\C:\llm-wiki-absent\wiki\page.md"),
            // writer spelling vs reader spelling — the actual divergence
            ("C:/llm-wiki-absent", r"\\?\C:\llm-wiki-absent\wiki\page.md"),
            // verbatim UNC, which spells `\\server\share`
            (r"\\?\UNC\server\share", r"\\server\share\wiki\page.md"),
        ] {
            assert_eq!(
                key_of(root_form, file_form),
                expected,
                "`{file_form}` under `{root_form}` must key the same store"
            );
        }

        // A shared textual prefix is not containment: the separator is required.
        assert_ne!(
            key_of("/llm-wiki-absent", "/llm-wiki-absent-other/wiki/page.md"),
            expected,
        );
    }

    /// The PR 4 probe, kept as a permanent guard. `canonicalize()` resolving a
    /// symlink rewrites the *relative* portion of the path, which is the same
    /// writer/reader divergence Windows hits unconditionally through separator
    /// rewriting. Keeping it here holds the bug class covered on the unix lanes.
    #[cfg(unix)]
    #[tokio::test]
    async fn records_and_restores_through_a_symlinked_directory() {
        let root = std::env::temp_dir().join(format!("llm-wiki-history-link-{}", Uuid::new_v4()));
        fs::create_dir_all(root.join(".llm-wiki")).unwrap();
        fs::create_dir_all(root.join("real")).unwrap();
        std::os::unix::fs::symlink("real", root.join("wiki")).unwrap();

        // The writer is handed `wiki/page.md`; the readers canonicalise it to
        // `real/page.md`. Before the fix those hashed to different stores.
        let file = root.join("wiki/page.md");
        fs::write(&file, "before").unwrap();
        record_file_version(&file, "baseline", "before.test");
        fs::write(&file, "after").unwrap();
        record_file_version(&file, "agent", "test.write");

        let entries = list_file_history(
            root.to_string_lossy().into_owned(),
            file.to_string_lossy().into_owned(),
        )
        .await
        .unwrap();
        assert_eq!(entries.len(), 2, "reader must see what the writer recorded");
        let old = entries
            .iter()
            .find(|entry| entry.content == "before")
            .unwrap();
        restore_file_history(
            root.to_string_lossy().into_owned(),
            file.to_string_lossy().into_owned(),
            old.id.clone(),
        )
        .await
        .unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap(), "before");
        let _ = fs::remove_dir_all(root);
    }
}
