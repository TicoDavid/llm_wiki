# SECURITY_TAURI.md — Tauri v2 Security Posture Scorecard

**Agent:** Dr. Insane · **Order:** DRINSANE-TAURI-AUDIT
**Repo:** TicoDavid/llm_wiki · **Branch:** `drinsane/tauri-audit` · **Base commit:** `9262b07` (origin/main)
**App version:** 0.6.6 (`src-tauri/tauri.conf.json:4`) · **Date:** 27 Jul 2026
**Doctrine:** bella-casefile `00_SOP/App_Setup_Doctrine.md` §2.1 (signed deployment) and §2.2 (security model)

> No secret value, private key, token, or certificate content was read or is reproduced in this document.
> This is a read-only audit: no configuration was changed, no control weakened, no workflow file authored or edited.

---

## Headline verdict

**The signed-release update path is NOT viable today.**

There is no updater plugin, no public key, no `createUpdaterArtifacts`, and no Tauri signing secret in CI.
Every deploy must therefore be a **manual install of an unsigned artifact** downloaded from the GitHub
release page. On Windows those artifacts also carry **no Authenticode signature**, so SmartScreen will
warn on every install. This is a deliberate, documented product decision — not an oversight — recorded in
`src/lib/update-check.ts:10-14`.

Doctrine §2.1's "blocker to check first" resolves to: **signing keys are NOT configured.**

---

## Scorecard

| # | Control | State | Evidence |
|---|---------|-------|----------|
| **NETWORK BINDING** ||||
| 1 | Default bind is loopback only | **PASS** | `src-tauri/src/server_bind.rs:6` — `DEFAULT_BIND_HOST = "127.0.0.1"`; used as the `unwrap_or_else` fallback at `:13` |
| 2 | `allowLanAccess` defaults to false | **PASS** | `src-tauri/src/server_bind.rs:56-62` — `.unwrap_or(false)`; test at `:104` asserts absent key ⇒ false |
| 3 | `tauri.conf.json` exposes no `0.0.0.0` / LAN path | **PASS** | `src-tauri/tauri.conf.json` — only network value is `devUrl: "http://localhost:1420"` at `:8` (dev-only); no `0.0.0.0` anywhere in the file |
| 4 | Both HTTP listeners share one bind policy | **PASS** | API server port 19828 `src-tauri/src/api_server.rs:20` binds via `configured_bind_host` at `:112`; clip server port 19827 `src-tauri/src/clip_server.rs:17` binds via the same helper at `:63` |
| 5 | LAN exposure is explicit opt-in via app state | **PASS (opt-in)** | `src-tauri/src/server_bind.rs:22-31` — `0.0.0.0` reached only when `apiConfig.allowLanAccess == true` in `app-state.json` |
| 6 | Env override cannot silently widen the bind | **FAIL** | `src-tauri/src/server_bind.rs:11-13` — `LLM_WIKI_BIND_HOST` is consulted **before** the store and wins; `sanitize_bind_host` (`:41-54`) validates charset only and explicitly accepts `0.0.0.0` (test `:72`). See Finding F-1 |
| **AUTHENTICATION** ||||
| 7 | Unauthenticated API request returns 401 | **PASS** | `src-tauri/src/api_server.rs:255-256` — `if !is_authorized(...) { return err(401, "Unauthorized") }` |
| 8 | Missing token fails closed (not open) | **PASS** | `src-tauri/src/api_server.rs:439-441` — `let Some(token) = api_token(app) else { return false }` |
| 9 | `allowUnauthenticated` defaults to false | **PASS** | `src-tauri/src/api_server.rs:506-515` — `.unwrap_or(false)` |
| 10 | Agent-chat endpoint is token-gated regardless of `allowUnauthenticated` | **PASS** | `src-tauri/src/api_server.rs:252-253` — checks `is_token_authorized` directly, bypassing the `allowUnauthenticated` escape |
| 11 | Token comparison is constant-time | **PASS** | `src-tauri/src/api_server.rs:557-566` — length-folded XOR accumulate, no early return |
| 12 | Clip server (19827) authenticates loopback callers | **FAIL (accepted risk)** | `src-tauri/src/clip_server.rs:124` — `if !request_is_loopback(&request) && !request_is_authorized(...)`; loopback short-circuits the token check. Rationale documented at `:119-123`. See Finding F-3 |
| 13 | Browser-origin CORS allowlist is exact-match, not prefix | **PASS** | `src-tauri/src/cors.rs:11-22`; negative tests reject `http://localhost.evil.com` and `http://127.0.0.1.evil.com` at `:69-70` |
| **CSP** ||||
| 14 | CSP is present | **PASS** | `src-tauri/tauri.conf.json:25` |
| 15 | `default-src` is restrictive | **PASS** | `src-tauri/tauri.conf.json:25` — `default-src 'self'` |
| 16 | Script execution is restricted (no `unsafe-inline` / `unsafe-eval` for scripts) | **PASS** | `src-tauri/tauri.conf.json:25` — no `script-src` directive, so scripts fall back to `default-src 'self'`. The sharpest sink is closed |
| 17 | `connect-src` is restrictive | **FAIL (permissive)** | `src-tauri/tauri.conf.json:25` — `connect-src 'self' https: http:` permits any host on any origin, including cleartext HTTP. See Finding F-4 |
| 18 | `style-src` avoids `unsafe-inline` | **FAIL** | `src-tauri/tauri.conf.json:25` — `style-src 'self' 'unsafe-inline'`. See Finding F-5 |
| 19 | `img-src` / `media-src` scoped | **PARTIAL** | `src-tauri/tauri.conf.json:25` — scoped to `'self'`, `asset:`, `asset.localhost`, plus `blob:` and `data:` on `img-src` |
| **CAPABILITIES & IPC SCOPING** ||||
| 20 | Exactly one capability file, scoped to a named window | **PASS** | `src-tauri/capabilities/default.json:5` — `"windows": ["main"]` |
| 21 | No `remote` origins granted IPC access | **PASS** | `src-tauri/capabilities/default.json` — no `remote` key; capability applies to local windows only |
| 22 | No filesystem plugin permission granted | **PASS** | `src-tauri/capabilities/default.json:6-29` — no `fs:` permission in the set |
| 23 | No shell / process-spawn plugin permission granted | **PASS** | `src-tauri/capabilities/default.json:6-29` — no `shell:` permission in the set |
| 24 | HTTP plugin scope is least-privilege | **FAIL** | `src-tauri/capabilities/default.json:14-28` — `http:default` allowed against `http://**` and `https://**` (10 wildcard patterns). See Finding F-2 |
| 25 | HTTP plugin does not enable forbidden-header override | **FAIL** | `src-tauri/Cargo.toml:31` — `tauri-plugin-http = { version = "2", features = ["unsafe-headers"] }`. See Finding F-2 |
| 26 | Asset protocol scope is least-privilege | **FAIL** | `src-tauri/tauri.conf.json:26-28` — `assetProtocol.enable: true` with `"scope": ["**"]`. See Finding F-6 |
| 27 | No `withGlobalTauri` / `dangerous*` / `freezePrototype` escape hatches | **PASS** | Repo-wide grep across `src-tauri/**/*.{json,rs}` returns zero hits for `withGlobalTauri`, `dangerous`, `freezePrototype` |
| **UPDATER & SIGNING** ||||
| 28 | Updater plugin block in config | **ABSENT** | `src-tauri/tauri.conf.json` is 44 lines; top-level keys are `$schema`, `productName`, `version`, `identifier`, `build`, `app`, `bundle` — there is **no `plugins` block** |
| 29 | Updater public key (`pubkey`) in config | **ABSENT** | Repo-wide grep for `pubkey` (excluding `node_modules`, `.git`, `package-lock.json`) returns **zero hits** |
| 30 | Updater `endpoints` configured | **ABSENT** | No `plugins.updater` block exists to hold them (`src-tauri/tauri.conf.json:32-43` is the entire `bundle` block: `active`, `targets`, `icon`) |
| 31 | `createUpdaterArtifacts` enabled | **ABSENT** | Repo-wide grep for `createUpdaterArtifacts` returns **zero hits**; `bundle` block `src-tauri/tauri.conf.json:32-43` does not contain it |
| 32 | `tauri-plugin-updater` dependency present | **ABSENT** | `src-tauri/Cargo.toml:21-68` — not among the `[dependencies]` entries |
| 33 | Updater permission granted in capabilities | **ABSENT** | `src-tauri/capabilities/default.json:6-29` — no `updater:` permission (consistent with #32) |
| 34 | Signing private key material committed to the repo | **ABSENT (correct)** | `git ls-files` matching `\.(key\|pem\|pub\|p12\|pfx)$\|signing\|secret` returns **zero tracked files**. `.env` / `.env.local` are gitignored at `.gitignore:20-22` |
| 35 | `TAURI_SIGNING_PRIVATE_KEY` wired into CI | **ABSENT** | `.github/workflows/build.yml:113-120` — the `tauri-action` `env:` block contains only `GITHUB_TOKEN` and six `APPLE_*` secrets. No Tauri signing variable |
| **CI ARTIFACT SIGNING** ||||
| 36 | A Windows build lane exists | **PASS** | `.github/workflows/build.yml:29-31` — `platform: windows-latest` in the matrix |
| 37 | Windows artifact is Authenticode-signed | **FAIL** | `.github/workflows/build.yml:111-132` — no `WINDOWS_CERTIFICATE` / thumbprint / `signCommand` env; `src-tauri/tauri.conf.json:32-43` and `src-tauri/tauri.windows.conf.json` carry no `bundle.windows` signing config |
| 38 | Windows artifact carries a Tauri updater signature (`.sig`) | **ABSENT** | Follows from #31 — `createUpdaterArtifacts` is off, so `tauri-action` emits no `.sig` and no `latest.json` |
| 39 | Windows artifacts are published to the release | **PASS** | `.github/workflows/build.yml:141-147` — portable zip uploaded via `gh release upload`; msi / nsis exe published by `tauri-action` (`:121-132`, glob at `:163-164`) |
| 40 | macOS signing/notarization secrets are wired | **PASS (names only)** | `.github/workflows/build.yml:115-120` — `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`. Whether these repo secrets are **populated** is not determinable from the tree and was not probed |
| 41 | PDFium binaries are checksum-verified in CI | **PASS (partial)** | `.github/workflows/build.yml:80-82` — `shasum -a 256 -c src-tauri/pdfium/SHA256SUMS`, but the step is skipped on `windows-latest` (`if: matrix.platform != 'windows-latest'`) |

**Tally:** 22 PASS · 8 FAIL · 2 PARTIAL · 8 ABSENT · 1 PASS(names only)

---

## Findings

### F-1 — `LLM_WIKI_BIND_HOST` outranks `allowLanAccess`, and `/health` then misreports the posture
**Severity: Medium** · `src-tauri/src/server_bind.rs:11-13`, `:41-54` · `src-tauri/src/api_server.rs:234`

`configured_bind_host` evaluates the environment variable first and returns on the first `Some`, so the
store-backed `allowLanAccess` flag is never consulted when `LLM_WIKI_BIND_HOST` is set. `sanitize_bind_host`
only rejects empty strings and non-`[A-Za-z0-9.\-_:\[\]]` characters — `0.0.0.0` passes (asserted by the test
at `:72`). The consequence is a reporting split: the server can be bound to `0.0.0.0` while
`/health` returns `"allowLanAccess": false` (`api_server.rs:234` reads the store, not the live bind address).
An operator reading `/health` would conclude the app is loopback-only when it is not.

*Note:* auth still applies on the API port, so this is exposure widening rather than an auth bypass — but it
combines badly with F-3.

**Recommendation (not applied):** have `/health` report the **resolved** bind host rather than the stored flag,
and require `allowLanAccess` to also be true before honouring a non-loopback `LLM_WIKI_BIND_HOST`.

### F-2 — `http:default` is granted global wildcard scope, with `unsafe-headers` enabled
**Severity: Medium** · `src-tauri/capabilities/default.json:14-28` · `src-tauri/Cargo.toml:31`

This is the **broadest capability in the set**. The WebView can issue HTTP requests to any host on any port
over either scheme, and the `unsafe-headers` Cargo feature lifts the forbidden-header restriction, so
frontend JS can set headers a browser would normally reserve (`Origin`, `Cookie`, `Host`). Together these
mean the Rust-side HTTP plugin imposes no meaningful egress boundary — the CSP `connect-src` (F-4) is the
only remaining check, and it is equally wide.

The product justification is real: the app is explicitly designed for user-supplied LLM/embedding endpoints
(`src/lib/embedding.ts`, `src/components/settings/llm-presets.ts`) and `Origin` override is used deliberately
for LAN embedding servers (`src-tauri/src/commands/search.rs:1457`, `:1589`; test `src/lib/embedding.test.ts:446`).
Flagged as **broader than needed** rather than wrong: a runtime-scoped allowlist derived from the user's
configured providers would achieve the same feature set with a far smaller surface.

### F-3 — Clip server trusts loopback without a token
**Severity: Medium** · `src-tauri/src/clip_server.rs:124` (rationale `:119-123`)

On port 19827 the token check is skipped entirely for loopback callers. Any local process running as the
user — including a malicious npm postinstall script or a browser extension talking to `127.0.0.1:19827` —
can reach the clip/project endpoints without credentials, which the code's own comment notes would
"leak project paths and permit writes". The LAN path is correctly gated. Called out as a deliberate,
documented compatibility choice, but it is the widest unauthenticated surface in the app.

### F-4 — `connect-src 'self' https: http:` is permissive
**Severity: Low** · `src-tauri/tauri.conf.json:25`

Permits exfiltration to any host, including cleartext `http:`. Same product justification as F-2 (arbitrary
user-configured endpoints), and the same mitigation would apply. Recording it as *permissive*, per the order's
ask, rather than as a defect.

### F-5 — `style-src 'unsafe-inline'`
**Severity: Low** · `src-tauri/tauri.conf.json:25`

Standard for a Tailwind/shadcn frontend (`components.json` present). Enables CSS-based exfiltration and
UI-redress if an injection sink is ever found; it does **not** enable script execution — `script-src` correctly
falls back to `default-src 'self'` (see control #16).

### F-6 — `assetProtocol.scope: ["**"]`
**Severity: Medium** · `src-tauri/tauri.conf.json:26-28`

Grants the `asset:` protocol read access to the entire filesystem reachable by the user, not just the wiki
vault. Any content-injection bug in the WebView becomes a whole-disk read primitive, since `img-src` and
`media-src` both admit `asset:` (`:25`). Notably, the app grants **no** `fs:` plugin permission (control #22) —
so `assetProtocol` is the one path by which arbitrary local files reach the WebView, and it is unbounded.
Scoping to the vault root and app-data directory would preserve every current feature.

### F-7 — PDFium checksum verification is skipped on the Windows lane
**Severity: Low** · `.github/workflows/build.yml:80-82`

`pdfium.dll` is the only bundled native binary whose SHA256 is not checked before it is packaged into the
shipped installer. Combined with the absence of Authenticode signing (control #37), the Windows artifact has
the weakest supply-chain evidence of the three platforms.

---

## Deploy-path verdict

### Is the updater path viable today? **No.**

| Requirement (doctrine §2.1) | State |
|---|---|
| `tauri-plugin-updater` dependency | Absent — `src-tauri/Cargo.toml` |
| `plugins.updater` block with `pubkey` | Absent — `src-tauri/tauri.conf.json` has no `plugins` block |
| `plugins.updater.endpoints` | Absent |
| `bundle.createUpdaterArtifacts: true` | Absent — `src-tauri/tauri.conf.json:32-43` |
| `updater:` capability permission | Absent — `src-tauri/capabilities/default.json:6-29` |
| `TAURI_SIGNING_PRIVATE_KEY` in CI | Absent — `.github/workflows/build.yml:113-120` |
| Windows Authenticode certificate | Absent — no signing config in any conf file or workflow |

**Therefore: every deploy is a manual install of an unsigned artifact.** The user downloads the `.msi`,
NSIS `.exe`, or portable `.zip` from the GitHub release page and installs it by hand, accepting a Windows
SmartScreen warning. The in-app "update" feature is a **notifier, not an updater**: `src/lib/update-check.ts`
polls the GitHub Releases API, compares against the build-time version, and opens the release page in the
user's browser — it "intentionally [doesn't] download or install" (`src/lib/update-check.ts:4-8`).

This is a documented product decision, not drift. `src/lib/update-check.ts:10-14` states the reasoning
verbatim: *"a real auto-install flow needs Tauri-signed release manifests plus a paid Windows code-signing
cert to avoid SmartScreen warnings. Worth doing later, but for a free OSS distribution a polite 'here's the
new version, click to download' covers 95% of the value."*

Doctrine §2.1's escalation applies: **the repo has no signing key configured, so the updater path does not
work and every deploy is a manual install** — with the added qualifier that the artifact is not merely
un-updater-signed but also un-Authenticode-signed on Windows.

### Where the private key would come from

The Tauri updater key is a **minisign** keypair generated off-repo with `npm run tauri signer generate`.
The private half and its passphrase belong in GitHub Actions repository secrets as
`TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`, injected into the `tauri-action` step's
`env:` block. Only the **public** half is committed, as `plugins.updater.pubkey` in `tauri.conf.json`.
Neither half exists in this repository today, and no key material was read, generated, or printed during
this audit.

**Two distinct signatures — do not conflate them:**

1. **Tauri updater signature (minisign)** — what the updater verifies before installing. Free. Needs only
   the generated keypair. Fixes controls #28–#33, #35, #38.
2. **Windows Authenticode signature** — what suppresses SmartScreen. Requires a **purchased** code-signing
   certificate (OV or EV) from a CA. Fixes control #37.

Enabling the updater without (2) yields working auto-updates that still trigger a SmartScreen warning on
each install.

### Path to viability (reported, not applied)

1. Generate the minisign keypair off-repo; store the private key and passphrase as repository secrets.
2. Add `tauri-plugin-updater` to `src-tauri/Cargo.toml` and register it in the builder.
3. Add `plugins.updater` (`pubkey` + `endpoints`) and `bundle.createUpdaterArtifacts: true` to `src-tauri/tauri.conf.json`.
4. Add `updater:default` to `src-tauri/capabilities/default.json`.
5. Add the two signing env vars to the `tauri-action` step in `.github/workflows/build.yml`.
6. Separately, budget for an Authenticode certificate to clear SmartScreen.
7. Extend the PDFium `SHA256SUMS` check to the Windows lane (F-7).

Steps 1 and 5 touch a workflow file and CI secrets and are **outside this agent's remit** — reported for
ADAM and David, not performed. Nothing in this audit modified any configuration.

---

## Scope, method and limitations

- **Method:** static read of the checked-out tree at `9262b07`. Nothing was built, run, installed, or probed
  over the network. No running instance of the app was contacted.
- **Doctrine location:** `00_SOP/App_Setup_Doctrine.md` is **not on `main`** in bella-casefile. It exists only
  on branch `jordan/obsidian-bases` (commit `d0d6147`), which is where §2.1 and §2.2 were read from. Worth
  landing on `main` so the doctrine an audit cites is not itself unmerged.
- **Not verifiable from the tree:** whether the six `APPLE_*` repository secrets are actually populated
  (control #40). Confirming that requires repository-settings access, and its result would not change the
  Windows verdict.
- **Runtime-state controls** (#2, #5, #9, #12) were assessed from their code defaults and unit tests. A
  machine whose `app-state.json` has already been hand-edited, or whose environment sets
  `LLM_WIKI_BIND_HOST` / `LLM_WIKI_API_TOKEN`, can differ from the shipped defaults recorded here.
- **File placement:** the scorecard is at `security/SECURITY_TAURI.md` rather than `docs/` because `docs/` is
  gitignored as "Internal docs (not shipped)" (`.gitignore:25`) and a report written there could not be
  committed.

---

*Dr. Insane · DRINSANE-TAURI-AUDIT · read-only audit, no configuration changed, no secret read.*
*Agents do not merge — Zane gates, David merges.*
