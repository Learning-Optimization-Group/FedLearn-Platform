# 14 — LOW-LEVEL DESIGN (LLD): Desktop Orchestrator (Tauri v2)

**Document type:** Production build specification — Low-Level Design (LLD) for one deployable unit.
**Unit:** the desktop FL (Federated Learning) orchestrator, rebuilt as a **Tauri v2** application that reuses the existing React renderer and replaces the Electron main process with a small Rust command layer.
**Audience:** a mid-sized local LLM (Large Language Model, ~30 billion parameters, e.g. Qwen / Llama 32B on an Apple M4 Max) that implements the function bodies. Everything here is **pre-decided**: exact pinned versions, exact file paths, full Rust + TypeScript (TS) signatures, exact environment-variable names, exact commands. Do not choose alternatives, do not invent APIs, do not add unrequested features.
**Status:** authoritative for the v2 desktop build. Supersedes the Electron unit at `fedlearn-desktop/`.
**Date authored:** 2026-05-29.

> **Conformance:** this document conforms to the three foundation docs and must not contradict them:
> - `docs/v2/build/02-TECH-STACK.md` (§1.4 Rust, §16.1 Tauri, §16.2 bollard, §24.7 pin table) — the locked, version-pinned stack.
> - `docs/v2/build/03-DATA-MODEL.md` — the control-plane schema. The desktop **owns no rows** in that schema; it is a pure client of the REST/STOMP API and a local-only client of the OS (Operating System) keychain (see §7).
> - `docs/v2/build/04-API-CONTRACTS.md` — the REST, STOMP, and gRPC contracts the desktop consumes. Referenced by exact endpoint / topic / DTO (Data Transfer Object) names throughout.
> Audit findings are cited inline as `A5-Cn`/`A5-Hn` (`docs/audit/2026-05-29/A5-desktop.md`), `B5 §n` (`docs/audit/2026-05-29/B5-desktop-strategy.md`), and `C5 §n` (`docs/audit/2026-05-29/C5-design-ux.md`). Existing-code claims cite `file:line` against the current Electron source under `fedlearn-desktop/`.

---

## 0. Abbreviations (first-use expansions, repeated here for self-containment)

The first occurrence of each acronym in the prose below is expanded in parentheses; this table is the master key.

| Short form | Full form |
|---|---|
| LLD | Low-Level Design |
| LLM | Large Language Model |
| OS | Operating System |
| TS | TypeScript |
| FL | Federated Learning |
| DeComFL | Dimension-Free Communication Federated Learning (zeroth-order optimization; the v1 wiki's "Decomposed" expansion is wrong per the paper) |
| FedAvg | Federated Averaging |
| IPC | Inter-Process Communication |
| RCE | Remote Code Execution |
| JWT | JSON Web Token (JSON = JavaScript Object Notation) |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| STOMP | Simple Text Oriented Messaging Protocol |
| WS | WebSocket |
| HTTP | HyperText Transfer Protocol |
| HTTPS | HTTP Secure |
| URL | Uniform Resource Locator |
| UUID | Universally Unique Identifier |
| gRPC | Google Remote Procedure Call |
| CSP | Content-Security-Policy |
| UI | User Interface |
| UX | User Experience |
| GPU | Graphics Processing Unit |
| MPS | Metal Performance Shaders (Apple Silicon GPU backend) |
| CUDA | Compute Unified Device Architecture (NVIDIA GPU backend) |
| CPU | Central Processing Unit |
| ARM64 | 64-bit Advanced RISC Machines architecture (aarch64) |
| L4T | Linux for Tegra (NVIDIA Jetson base) |
| PID | Process Identifier |
| WebView | OS-provided embedded web rendering engine (WebView2 / WKWebView / WebKitGTK) |
| WKWebView | WebKit WebView (macOS/iOS) |
| WebKitGTK | WebKit port for the GTK toolkit (Linux) |
| CLI | Command-Line Interface |
| CI | Continuous Integration |
| TOML | Tom's Obvious Minimal Language (config file format) |
| JSON | JavaScript Object Notation |
| SHA / sha256 | Secure Hash Algorithm (256-bit) |
| Ed25519 | Edwards-curve Digital Signature Algorithm using Curve25519 |
| TLS | Transport Layer Security |
| mTLS | mutual TLS |
| HMAC | Hash-based Message Authentication Code |
| DTO | Data Transfer Object |
| CN | Common Name (of a TLS certificate) |
| W3C | World Wide Web Consortium |
| OTel | OpenTelemetry |
| CVE | Common Vulnerabilities and Exposures |

---

## 1. Purpose & single responsibility

**Single responsibility:** the desktop unit is a **thin, privileged orchestrator** that lets an end user (a federation participant) authenticate against the control plane, pick local hardware and a dataset directory, **launch / monitor / stop the FL client as an out-of-process subprocess** (a native PyInstaller bundle on most hardware; a Docker container on Jetson), and receive code-signed auto-updates that cannot be tampered with. It renders the **same React UI** used by the web frontend (`frontend/`) and adds exactly the privileged operations a browser cannot do: spawn a local process / Docker container, mount a local dataset directory, and hold the JWT (JSON Web Token) in the OS keychain.

It is **not**:

- a training engine — training runs in a separate OS process (the PyInstaller-bundled Python client, or the Jetson Docker container). The desktop never links libtorch in-process. This process-isolation boundary is the unit's primary robustness property (`B5 §2.3`) and is preserved verbatim from the salvaged v1 `DockerService` dispatcher (`B5 §6`, verdict "salvage").
- a database owner — it owns **zero** rows of the `03-DATA-MODEL.md` schema. All durable state lives in the control plane; the only local persistent state is the keychain entry and a tiny settings file (§7).
- a second FL implementation — the in-process C++ mobile core is **killed for v2** on desktop (`B5 §2`, §9). RNG (Random Number Generator) parity for DeComFL (Dimension-Free Communication Federated Learning) stays free because the desktop ships the **same** Python/PyTorch client as the server.

**Why this unit exists at all (vs just using the web app):** a browser cannot spawn a local training process, cannot bind-mount a user's dataset directory into a container, and cannot reach the Jetson Docker socket. The desktop is the minimal privileged shell around those three capabilities plus secure token storage.

---

## 2. Position in the system — depends-on / depended-by

```
                 ┌─────────────────────────────────────────────┐
                 │           DESKTOP ORCHESTRATOR (this unit)    │
                 │  ┌───────────────┐      ┌──────────────────┐  │
   user ───────► │  │ React renderer│◄────►│ Rust command     │  │
                 │  │ (system       │ Tauri│ layer (privileged)│  │
                 │  │  WebView)     │ invoke│  - process spawn  │  │
                 │  └──────┬────────┘ /event│  - bollard Docker │  │
                 │         │ HTTPS/WS        │  - keyring (JWT)  │  │
                 └─────────┼─────────────────┴─────────┬─────────┘
                           │                            │ spawn / mount
        REST + STOMM (cookie│JWT)                        ▼
                           ▼                   ┌───────────────────┐
                 ┌───────────────────┐         │  FL CLIENT         │
                 │ Spring Boot        │         │  (separate OS proc)│
                 │ control plane :8081│         │  - PyInstaller bin │
                 │ (04-API-CONTRACTS) │         │  - or L4T Docker   │
                 └─────────┬──────────┘         │    (Jetson)        │
                           │ spawns FL server   └─────────┬─────────┘
                           ▼                              │ gRPC fedlearn.v2
                 ┌───────────────────┐ ◄──────────────────┘ (TLS+mTLS;
                 │ Python FL server   │                      plaintext only dev)
                 │ (k8s Job/ECS/local)│
                 └────────────────────┘
```

### 2.1 Interfaces CONSUMED (by exact name from `04-API-CONTRACTS.md`)

| Interface consumed | From `04-API-CONTRACTS.md` | Used by | Purpose |
|---|---|---|---|
| `POST /api/auth/login` → `MeResponse` + `Set-Cookie: jwtToken` | §2 | Rust command `login` | Authenticate; obtain the HttpOnly cookie. |
| `GET /api/auth/me` → `MeResponse` | §2 | Rust command `check_auth` | Silent session probe (the 401 is swallowed only here). |
| `POST /api/auth/logout` (204, `Set-Cookie` Max-Age=0) | §2 | Rust command `logout` | Clear session + keychain. |
| `GET /api/projects` → `ProjectResponseDto[]` | §3 | renderer (via proxied fetch) | Project list for the orchestrator UI. |
| `GET /api/projects/{projectId}/runs`, `GET /api/runs/{runId}/status` → `RunStatusDto` | §4 | renderer polling | Run status for the live view. |
| STOMP topics `/topic/logs/{projectId}`, `/topic/results/{projectId}`, `/topic/status/{projectId}`, `/topic/runs/{projectId}` | §11 | renderer STOMP client | Live logs / metrics / lifecycle (the renderer subscribes directly, exactly as the web frontend does). |
| gRPC `fedlearn.v2.FederatedLearningService` (`RegisterClient`, `Heartbeat`, `GetGlobalModelStream`, `SubmitGradientScalars`, …) | §10 | **the spawned FL client subprocess** (NOT the desktop shell) | The desktop does not speak gRPC itself; it spawns the client that does. Listed because the desktop passes the run's `grpc_endpoint`, `run_id`, and enrollment token into that subprocess's environment/arguments. |

**Auth-transport conformance (locked):** the renderer's REST/WS calls set `withCredentials: true` and rely on the HttpOnly `jwtToken` cookie (`04-API-CONTRACTS.md §1`). The desktop does **not** read the cookie value into JS-readable storage. See §6.3 for how this fixes `A5-C2`.

### 2.2 Interfaces EXPOSED

The desktop exposes exactly **one** internal interface: the **Tauri command + event bridge** (the IPC boundary between the privileged Rust core and the sandboxed React renderer). It is fully enumerated in §5. It exposes **no** network service and **no** public API.

### 2.3 Depended-by

Nothing depends on the desktop. It is a leaf client. The FL client subprocess it spawns depends on the control plane (for the run token) and the FL server (over gRPC), but those are independent contracts the desktop merely forwards into the child environment.

---

## 3. Tech stack for this unit (pinned, from `02-TECH-STACK.md`)

| Component | Pinned version | Source in `02-TECH-STACK.md` | One-line reasoning |
|---|---|---|---|
| Rust toolchain | **`1.87.0`** stable (pin in `rust-toolchain.toml` as `channel = "1.87.0"`) | §1.4, §24.6 | Tauri v2's privileged process is Rust; pin the exact toolchain so CI (Continuous Integration) is reproducible. |
| Tauri | **`2.11.2`** (crates `tauri`, `tauri-build`) | §16.1, §24.7 | Its updater **requires** minisign-signed artifacts, making the v1 unsigned auto-install RCE (Remote Code Execution) class structurally impossible (`B5 §4`, `A5-C1`). |
| tauri-bundler | **`2.9.2`** | §16.1, §24.7 | Pinned to the same release set as Tauri 2.11.2 for a coherent build. |
| wry | **`0.55.1`** | §16.1, §24.7 | The WebView (OS-provided embedded web rendering engine) layer Tauri builds on; pinned with the release set. |
| tao | **`0.35.3`** | §16.1, §24.7 | The windowing layer; pinned with the release set. |
| `tauri-plugin-updater` | **`2.x`** (pin exact to the `2.11.2` plugin release set; `verify-before-use`) | §16.1 (updater is locked) | The minisign-verifying updater plugin; the structural fix for `A5-C1`. |
| `tauri-plugin-shell` | **`2.x`** (pin exact, `verify-before-use`) | §16.1 / §16.2 (subprocess model) | Sidecar/external-binary spawn + stdout streaming for the PyInstaller client. |
| `bollard` (Rust Docker client) | **pin exact in `Cargo.toml`** (`verify-before-use` via `cargo search bollard`) | §16.2, §24.7 | Rust equivalent of `dockerode`; covers `create_container`/`start`/`logs(follow)`/`stop`/`remove` and demuxes the multiplexed stream the v1 code hand-rolled buggily (`A5-H3`). |
| `keyring` (Rust OS-keychain crate) | **pin exact in `Cargo.toml`** (`verify-before-use` via `cargo search keyring`) | §16.2 ("OS keychain holds the JWT via Tauri's keychain command; the renderer never sees the token") | Cross-platform OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service); holds the JWT opaquely so the renderer never sees it. |
| `tokio` (async runtime) | **pin exact** (`verify-before-use`; required transitively by `bollard`) | implied by `bollard` async | Async runtime for the Docker log stream and process I/O. |
| `serde` / `serde_json` | **pin exact** (`verify-before-use`) | implied (Tauri command payloads) | Serialize/deserialize command args + event payloads across the IPC boundary. |
| Node.js + TypeScript | TS **`5.x`** (one version across `frontend` + desktop renderer + mobile) | §1.3, §14 (TS pin) | Builds the React renderer bundle Tauri loads; unify TS to remove drift. |
| React renderer | reused from `frontend/` (React 19 + Tailwind v4 + shadcn/ui + `@fedlearn/tokens`) | §14, §15, C5 §7 | The renderer ports almost as-is from web; one design-token source kills the v1 three-palette problem (`C5 §1`). |

**Explicitly NOT in this unit's stack (rejected, with reasoning):**

- **Electron** — rejected: Electron requires *adding* signature verification, whereas Tauri makes it the default contract (`B5 §4`). Migrating removes Node from the privileged process entirely.
- **Per-OS native (Swift/WinUI/GTK)** — killed: 3× UI surface for an orchestrator UI, no payoff for a small team (`B5 §3`).
- **In-process C++ mobile core** — killed for v2: no bundle win (libtorch dominates either way), adds a fragile cross-language RNG-parity invariant, collapses process isolation (`B5 §2`).
- **`dockerode`** (Node) — replaced by `bollard` (`B5 §6`).
- **`electron-updater`** — replaced by `tauri-plugin-updater` (the entire `A5-C1` fix).

---

## 4. Module / file structure (exact directory tree)

The unit lives at `fedlearn-desktop/` (the v2 rebuild replaces the Electron `src/main/*` with `src-tauri/`; the renderer moves under `src/`). One-line responsibility per file.

```
fedlearn-desktop/
├── package.json                      # renderer build scripts (vite), tauri CLI devDependency
├── tsconfig.json                     # TS 5.x config for the renderer
├── vite.config.ts                    # Vite build for the renderer (outputs to dist/)
├── index.html                        # renderer entry; ships the PRODUCTION CSP (no 'unsafe-eval') (A5-H1)
├── rust-toolchain.toml               # channel = "1.87.0" (pins Rust)
├── src/                              # === REACT RENDERER (reused from frontend/) ===
│   ├── main.tsx                      # React root; calls ensureBridge() before render (fail-closed, A5-C3)
│   ├── App.tsx                       # app shell + auth gate; NO fake preview-user fallback (A5-C3)
│   ├── bridge/
│   │   ├── tauri.ts                  # typed TS wrappers over `invoke(...)` + event listeners (the binding half of §5)
│   │   └── types.ts                  # request/response TS types shared with Rust (mirror serde structs)
│   ├── views/                        # Login, HardwareSelector, ProjectList, RunView, Settings, UpdateBanner
│   └── components/                   # shadcn/ui components on @fedlearn/tokens (C5 §7)
└── src-tauri/                        # === RUST PRIVILEGED COMMAND LAYER ===
    ├── Cargo.toml                    # pins tauri 2.11.2, bollard, keyring, tokio, serde (exact)
    ├── Cargo.lock                    # committed lockfile with hashes (02-TECH-STACK reproducibility)
    ├── tauri.conf.json               # app config; updater pubkey + endpoints; CSP; capabilities (§8)
    ├── build.rs                      # tauri-build codegen
    ├── capabilities/
    │   └── default.json              # capability allowlist: which commands the renderer may invoke (deny-by-default)
    └── src/
        ├── main.rs                   # Tauri builder; registers commands, plugins, single-instance, lifecycle hooks
        ├── error.rs                  # `DesktopError` enum + serde serialization to the renderer (§9)
        ├── auth.rs                   # `login`/`logout`/`check_auth` commands; cookie-jar reqwest client (§6.3)
        ├── keychain.rs              # keyring wrapper: store/get/delete the JWT opaquely (A5-C2 fix)
        ├── hardware.rs               # `detect_hardware` command; nvidia-smi/arch probe → HardwareProfile
        ├── launcher.rs               # `start_training`/`stop_training` dispatch: native vs docker (B5 §6)
        ├── native_runner.rs          # spawn the PyInstaller sidecar; stream stdout → events; SIGTERM→SIGKILL
        ├── docker_runner.rs          # bollard: create/start/logs/stop/remove; JETSON_DEVICE_MOUNTS (§6.5)
        ├── dataset_path.rs           # validate + sanitize a user-chosen dataset dir (traversal guard, salvage)
        ├── updater.rs                # check/download/install via tauri-plugin-updater; emit progress events
        └── state.rs                  # `AppState` (Arc<Mutex<...>>): child handle map, http client, run context
```

**Why this split:** each Rust file maps to exactly one command-group or one cross-cutting concern, so the ~30B local model implements one file at a time with a clear contract. The renderer/Rust boundary is the only trust boundary; `bridge/tauri.ts` + `src-tauri/src/*.rs` are the two halves of every command in §5.

---

## 5. Key interfaces & type signatures (FULL)

This is the **complete** Tauri command + event surface. Each command is given as the **Rust `#[tauri::command]` fn** (the implementer writes the body) **and** its **TypeScript binding** in `src/bridge/tauri.ts`. The shapes are pre-decided; do not add fields.

### 5.1 Shared types (Rust serde structs ↔ TS interfaces)

Rust (`src-tauri/src/state.rs` and per-module), all `#[derive(Serialize, Deserialize, Clone)]` with `#[serde(rename_all = "camelCase")]`:

```rust
// --- auth ---
pub struct LoginRequest  { pub username: String, pub password: String }
pub struct MeResponse {                       // mirrors 04-API-CONTRACTS.md §2.1 MeResponse
    pub user_id: i64,                         // users.id is BIGINT -> i64 (JSON number)
    pub username: String,
    pub email: String,
    pub platform_role: String,                // "USER" | "PLATFORM_ADMIN"
    pub orgs: Vec<OrgMembership>,
    pub email_verified: bool,
}
pub struct OrgMembership { pub org_id: String, pub org_name: String, pub org_role: String }
pub struct AuthResult { pub success: bool, pub authenticated: bool, pub me: Option<MeResponse> }

// --- hardware ---
#[serde(rename_all = "UPPERCASE")]
pub enum HardwareProfile { Mps, Cuda, Cpu, Jetson, Discrete }  // serialized "MPS"/"CUDA"/...
pub struct HardwareInfo {
    pub profile: HardwareProfile,
    pub has_nvidia_smi: bool,
    pub gpu_name: Option<String>,
    pub arch: String,                         // process arch, e.g. "aarch64","x86_64"
    pub os: String,                           // "macos"|"windows"|"linux"
}

// --- training launch ---
pub struct TrainingConfig {
    pub run_id: String,                       // fl_runs.id (UUID); passed to the client subprocess
    pub project_id: String,                   // UUID (display/log + STOMP topic key)
    pub server_address: String,               // host:port of the FL server gRPC endpoint
    pub client_id: String,                    // client-chosen handle (display only; NOT authz, see §10 proto)
    pub enrollment_token: String,             // backend-minted; binds client identity (anti-Sybil)
    pub dataset_path: String,                 // absolute local dir; validated before container bind
    pub profile: HardwareProfile,             // selects native vs docker dispatch
    pub docker_image: Option<String>,         // override; default per profile (L4T for Jetson)
}
pub struct TrainingHandle { pub run_id: String, pub kind: RunnerKind, pub started: bool }
#[serde(rename_all = "lowercase")]
pub enum RunnerKind { Native, Docker }

// --- updater (mirror tauri-plugin-updater shapes; A5-M1: do NOT use `any`) ---
pub struct UpdateStatus {
    pub available: bool,
    pub version: Option<String>,
    pub notes: Option<String>,
    pub date: Option<String>,
}
pub struct UpdateProgress { pub downloaded: u64, pub content_length: Option<u64> }
```

TypeScript mirrors in `src/bridge/types.ts` (exact field names, `camelCase`):

```typescript
export interface LoginRequest { username: string; password: string }
export interface OrgMembership { orgId: string; orgName: string; orgRole: string }
export interface MeResponse {
  userId: number; username: string; email: string;
  platformRole: "USER" | "PLATFORM_ADMIN";
  orgs: OrgMembership[]; emailVerified: boolean;
}
export interface AuthResult { success: boolean; authenticated: boolean; me: MeResponse | null }
export type HardwareProfile = "MPS" | "CUDA" | "CPU" | "JETSON" | "DISCRETE";
export interface HardwareInfo {
  profile: HardwareProfile; hasNvidiaSmi: boolean;
  gpuName: string | null; arch: string; os: string;
}
export interface TrainingConfig {
  runId: string; projectId: string; serverAddress: string; clientId: string;
  enrollmentToken: string; datasetPath: string; profile: HardwareProfile;
  dockerImage?: string | null;
}
export interface TrainingHandle { runId: string; kind: "native" | "docker"; started: boolean }
export interface UpdateStatus { available: boolean; version: string | null; notes: string | null; date: string | null }
export interface UpdateProgress { downloaded: number; contentLength: number | null }
```

### 5.2 Commands (Rust fn + TS binding) — the full IPC surface

Every command returns `Result<T, DesktopError>` (§9); the renderer always receives either a typed value or a typed error — there is **no silent success** (the structural fix for `A5-C3`).

| # | Command | Rust signature (in module) | TS binding | Purpose |
|---|---|---|---|---|
| 1 | `check_auth` | `async fn check_auth(state: State<'_, AppState>) -> Result<AuthResult, DesktopError>` (`auth.rs`) | `checkAuth(): Promise<AuthResult>` | Probe `GET /api/auth/me`; swallow 401 → `{authenticated:false}`. |
| 2 | `login` | `async fn login(req: LoginRequest, state: State<'_, AppState>) -> Result<AuthResult, DesktopError>` (`auth.rs`) | `login(req: LoginRequest): Promise<AuthResult>` | `POST /api/auth/login`; persist cookie in Rust cookie jar; store opaque token in keychain. |
| 3 | `logout` | `async fn logout(state: State<'_, AppState>) -> Result<(), DesktopError>` (`auth.rs`) | `logout(): Promise<void>` | `POST /api/auth/logout`; clear cookie jar + keychain. |
| 4 | `detect_hardware` | `async fn detect_hardware() -> Result<HardwareInfo, DesktopError>` (`hardware.rs`) | `detectHardware(): Promise<HardwareInfo>` | Probe `nvidia-smi` + arch/os → profile. |
| 5 | `pick_dataset_dir` | `async fn pick_dataset_dir(app: AppHandle) -> Result<Option<String>, DesktopError>` (`dataset_path.rs`) | `pickDatasetDir(): Promise<string \| null>` | OS folder picker; returns absolute path or null (cancelled). |
| 6 | `start_training` | `async fn start_training(config: TrainingConfig, state: State<'_, AppState>, app: AppHandle) -> Result<TrainingHandle, DesktopError>` (`launcher.rs`) | `startTraining(config: TrainingConfig): Promise<TrainingHandle>` | Validate dataset path; dispatch native vs docker; stream stdout to events. |
| 7 | `stop_training` | `async fn stop_training(run_id: String, state: State<'_, AppState>) -> Result<(), DesktopError>` (`launcher.rs`) | `stopTraining(runId: string): Promise<void>` | SIGTERM→SIGKILL the native child, or `bollard.stop_container` + `remove`. |
| 8 | `get_training_status` | `fn get_training_status(run_id: String, state: State<'_, AppState>) -> Result<TrainingHandle, DesktopError>` (`launcher.rs`) | `getTrainingStatus(runId: string): Promise<TrainingHandle>` | Local liveness of the child (not the run's server-side status — that comes from REST/STOMP). |
| 9 | `check_for_update` | `async fn check_for_update(app: AppHandle) -> Result<UpdateStatus, DesktopError>` (`updater.rs`) | `checkForUpdate(): Promise<UpdateStatus>` | Query the updater endpoint; verify signature metadata; report only. |
| 10 | `download_and_install_update` | `async fn download_and_install_update(app: AppHandle) -> Result<(), DesktopError>` (`updater.rs`) | `downloadAndInstallUpdate(): Promise<void>` | **User-initiated** download+install with minisign verification; emits `update://progress`. |

### 5.3 Events (Rust → renderer, via `app.emit`/`Window::emit`)

| Event name | Payload (TS) | Emitted by | Renderer handler |
|---|---|---|---|
| `training://log` | `{ runId: string; line: string }` | `native_runner.rs` / `docker_runner.rs` | append to the log buffer (render as **text node**, never `dangerouslySetInnerHTML` — `A5-M2`) |
| `training://exit` | `{ runId: string; code: number \| null; signal: string \| null }` | runner modules | mark run-local process finished |
| `update://progress` | `UpdateProgress` | `updater.rs` | drive the `UpdateBanner` progress bar |
| `update://error` | `{ message: string }` | `updater.rs` | surface a non-fatal updater error |

### 5.4 The fail-closed bridge initializer (renderer side)

`src/main.tsx` must call this **before** rendering the app shell. It replaces the v1 `ensureFedLearnBridge()` that installed a fake authenticated `preview-user` fallback (`App.tsx:100-105` in the Electron code), the `A5-C3` Critical.

```typescript
// src/bridge/tauri.ts — fail CLOSED. No fake preview-user, ever (A5-C3).
import { invoke } from "@tauri-apps/api/core";

export function bridgeAvailable(): boolean {
  // In a Tauri window, the IPC global is injected by the runtime. In a plain
  // browser (vite dev preview), it is absent.
  return typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== "undefined";
}

// Compile-time dev flag, injected by Vite `define`. CANNOT be true in a packaged build.
declare const __FEDLEARN_PREVIEW__: boolean;

export function ensureBridge(): { ok: true } | { ok: false; reason: string } {
  if (bridgeAvailable()) return { ok: true };
  // Only a dev preview build may proceed without the bridge — and even then it
  // does NOT fabricate auth; checkAuth() below returns authenticated:false.
  if (typeof __FEDLEARN_PREVIEW__ !== "undefined" && __FEDLEARN_PREVIEW__) {
    return { ok: true };
  }
  // Packaged build with no bridge => HARD ERROR screen, never an authed shell.
  return { ok: false, reason: "Desktop bridge failed to initialize — reinstall the application." };
}

export const checkAuth = (): Promise<import("./types").AuthResult> =>
  bridgeAvailable()
    ? invoke("check_auth")
    : Promise.resolve({ success: true, authenticated: false, me: null }); // preview => logged OUT, not in
```

---

## 6. Core algorithms & flows

### 6.1 App startup + fail-closed gate (fixes `A5-C3`)

```
main.tsx
  └─ ensureBridge()
       ├─ ok  → render <App/>;  App calls checkAuth()
       │         ├─ authenticated:true  → render authenticated shell
       │         └─ authenticated:false → render <LoginView/>
       └─ !ok → render <FatalBridgeError/>  (NEVER the authenticated shell)
```

**Why:** in the Electron code the gate was "is `window.fedLearnAPI` present", which is *also* false in a broken packaged build, so a packaging regression renders the full authenticated app to an unauthenticated user (`A5-C3`). Here the only non-bridge path is gated on the compile-time `__FEDLEARN_PREVIEW__` flag (injected by Vite `define`, impossible to be true in a release build), and even that path returns `authenticated:false`. Fail-open auth is **non-representable**.

### 6.2 Login + opaque keychain storage (fixes `A5-C2`)

```
renderer: login({username,password})
      │ invoke("login")
      ▼
Rust auth.rs::login
  1. POST /api/auth/login via the shared reqwest client that owns a COOKIE JAR.
  2. On 200: the Set-Cookie `jwtToken` is captured INTO THE COOKIE JAR by reqwest.
     The Rust layer does NOT regex-extract the token value into a struct field.
  3. Persist the cookie jar (serialized opaque blob) into the OS keychain via keyring:
        keychain::store("fedlearn", "session", <serialized cookie jar bytes>)
  4. Return AuthResult{ success:true, authenticated:true, me: <MeResponse body> }.
```

```rust
// keychain.rs — keyring wrapper. The renderer NEVER receives the token (A5-C2).
use keyring::Entry;
const SERVICE: &str = "com.fedlearn.desktop";
const ACCOUNT: &str = "session";

pub fn store(blob: &[u8]) -> Result<(), DesktopError> {
    let entry = Entry::new(SERVICE, ACCOUNT)?;
    entry.set_secret(blob)?;          // OS keychain: macOS Keychain / Win Cred Mgr / Secret Service
    Ok(())
}
pub fn load() -> Result<Option<Vec<u8>>, DesktopError> {
    match Entry::new(SERVICE, ACCOUNT)?.get_secret() {
        Ok(b) => Ok(Some(b)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.into()),
    }
}
pub fn delete() -> Result<(), DesktopError> {
    match Entry::new(SERVICE, ACCOUNT)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.into()),
    }
}
```

**Why opaque cookie-jar, not a Bearer string:** the v1 Electron client read the `Set-Cookie` header, regex-extracted the `jwtToken` value, stored it, and replayed it as `Authorization: Bearer <jwt>` (`auth.service.ts:128-140, 259`), making the desktop the one client that extracts the HttpOnly token into application-readable storage — the exact thing `HttpOnly` exists to prevent (`A5-C2`). Option (ii) from the audit (`A5-C2` recommendation) is implemented: keep the cookie in a Rust cookie jar, persist the jar opaquely in the keychain, and **never** expose the value. This restores the platform's cookie-only invariant (`04-API-CONTRACTS.md §1`, auth transport) and removes the dead `accessToken` branch entirely (`A5-C2`, the latent footgun). On keychain-unavailable (headless Linux without Secret Service), fall back to in-memory only (never write reversible bytes to disk — the salvaged v1 posture, `A5` "what's good").

### 6.3 Hardware detection → profile (salvage, with the documented nit)

```rust
// hardware.rs::detect_hardware — port of the salvaged hardware.probe.ts (A5 "low/hygiene")
pub async fn detect_hardware() -> Result<HardwareInfo, DesktopError> {
    let os = std::env::consts::OS;       // "macos"|"windows"|"linux"
    let arch = std::env::consts::ARCH;   // "aarch64"|"x86_64"
    // execFile-equivalent: no shell, args arrayified, 2s timeout (safe; A5 low note).
    let smi = run_with_timeout("nvidia-smi", &["--query-gpu=name","--format=csv,noheader"], 2000).await;
    let has_smi = smi.is_ok();
    let profile = match (os, arch, has_smi) {
        ("macos", "aarch64", _)        => HardwareProfile::Mps,
        (_, _, true) if os != "linux"  => HardwareProfile::Cuda,
        ("linux", "aarch64", _)        => HardwareProfile::Jetson,   // DOCUMENTED NIT (A5 low): Asahi/ARM
        ("linux", _, true)             => HardwareProfile::Discrete, //   server would misclassify; document it
        _                              => HardwareProfile::Cpu,
    };
    Ok(HardwareInfo { profile, has_nvidia_smi: has_smi, gpu_name: smi.ok(), arch: arch.into(), os: os.into() })
}
```

> **Documented coupling (carry the `A5` low note):** deriving `Jetson` purely from `linux + aarch64` misclassifies an Apple-Silicon Asahi-Linux box or an ARM server as Jetson. This is an accepted edge case; document it in the Settings view's hardware override so a user can force the profile.

### 6.4 Training launch dispatch (salvage the v1 dispatcher; `B5 §6`)

```
launcher.rs::start_training(config)
  1. dataset_path::validate(&config.dataset_path)   // NUL reject, resolve, residual ".." reject,
     │                                              // exists + is_dir (salvage from ipc.handlers.ts:47-82)
  2. match config.profile {
       Jetson                       => docker_runner::start(config, app, state).await,  // ONLY Docker path
       Mps | Cuda | Cpu | Discrete  => native_runner::start(config, app, state).await,  // dominant path
     }
```

**Why this dispatch and not "always Docker":** `B5 §1` corrects the framing — the dominant path is a **bundled native PyInstaller subprocess**; Docker is the **exception** used only on Jetson. The v1 dispatcher (`docker.service.ts:90-96`) already had this right; v2 keeps the single source of truth.

### 6.5 Jetson Docker path (`bollard`) — the device-mount rule (locked)

The Jetson invariant is **data, not Electron-specific code** (`B5 §6`), and ports verbatim to a `bollard` `HostConfig`. The exact device set is the one verified in the v1 code (`docker.service.ts:41-47`).

```rust
// docker_runner.rs — bollard. The /dev/nvhost-* device set + NO --runtime nvidia (locked).
use bollard::Docker;
use bollard::container::{Config, CreateContainerOptions, LogsOptions};
use bollard::models::{HostConfig, DeviceMapping};

// EXACT Jetson device set (matches docker.service.ts:41-47, verified).
fn jetson_device_mounts() -> Vec<DeviceMapping> {
    ["/dev/nvhost-ctrl", "/dev/nvhost-ctrl-gpu", "/dev/nvhost-dbg-gpu",
     "/dev/nvhost-prof-gpu", "/dev/nvmap", "/dev/nvhost-gpu"]
        .iter().map(|p| DeviceMapping {
            path_on_host: Some(p.to_string()),
            path_in_container: Some(p.to_string()),
            cgroup_permissions: Some("rwm".to_string()),
        }).collect()
}

pub async fn start(cfg: TrainingConfig, app: AppHandle, state: State<'_, AppState>)
    -> Result<TrainingHandle, DesktopError>
{
    let docker = Docker::connect_with_local_defaults()?;     // unix socket; NO socket mount into container
    let image = cfg.docker_image.clone()
        .unwrap_or_else(|| "fedlearn-client:jetson".to_string()); // L4T base (the project conventions Jetson rule)

    let host_config = HostConfig {
        // RULE (LOCKED): on Jetson, set Devices to the /dev/nvhost-* set.
        // DO NOT set Runtime = Some("nvidia") — it HANGS on Jetson searching the
        // device tree for PCIe discrete-GPU metadata (docker.service.ts:300 comment; the project conventions).
        devices: Some(jetson_device_mounts()),
        // dataset bind mount (validated path):
        binds: Some(vec![format!("{}:/data", cfg.dataset_path)]),
        auto_remove: Some(false),        // keep for post-mortem (salvage: docker.service.ts:336-341)
        ..Default::default()
    };

    let create = docker.create_container(
        Some(CreateContainerOptions { name: format!("fl-client-{}", cfg.run_id), platform: None }),
        Config {
            image: Some(image),
            cmd: Some(vec![
                "--server-address".into(), cfg.server_address.clone(),
                "--client-id".into(),      cfg.client_id.clone(),
                "--run-id".into(),         cfg.run_id.clone(),
            ]),
            env: Some(client_env(&cfg)),     // FEDLEARN_RUN_ID / _RUN_TOKEN / TRACEPARENT, see §8.2
            host_config: Some(host_config),
            ..Default::default()
        }).await?;

    docker.start_container::<String>(&create.id, None).await?;
    // bollard demuxes the multiplexed stream FOR US — fixes the hand-rolled
    // demuxDockerStream partial-frame bug (A5-H3).
    spawn_log_stream(docker.clone(), create.id.clone(), cfg.run_id.clone(), app);
    state.register_docker(&cfg.run_id, &create.id);
    Ok(TrainingHandle { run_id: cfg.run_id, kind: RunnerKind::Docker, started: true })
}
```

> **Discrete-GPU path (non-Jetson Linux/Windows with nvidia-smi):** use bollard's device-request equivalent of `--gpus all` (the v1 `DeviceRequests: [{ Count: -1, Capabilities: [['gpu']] }]`, `docker.service.ts:307`), **not** the Jetson device list. The two paths must stay distinct.

### 6.6 Native PyInstaller subprocess path (dominant)

```rust
// native_runner.rs — spawn the PyInstaller-bundled Python client as a sidecar/external bin.
pub async fn start(cfg: TrainingConfig, app: AppHandle, state: State<'_, AppState>)
    -> Result<TrainingHandle, DesktopError>
{
    // Resolve the bundled binary path (extraResources equivalent: tauri externalBin/resource dir).
    // Windows: "fedlearn-client.exe"; otherwise "fedlearn-client" (docker.service.ts:174).
    let bin = resolve_client_binary(&app)?;       // returns DesktopError::ClientBinaryMissing if absent
    let mut child = tokio::process::Command::new(bin)
        .args(["--server-address", &cfg.server_address,
               "--client-id",      &cfg.client_id,
               "--run-id",         &cfg.run_id])
        .envs(client_env(&cfg))                    // §8.2
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())      // merged into the log stream
        .spawn()?;
    stream_child_stdout_stderr(&mut child, cfg.run_id.clone(), app);   // -> training://log events
    state.register_native(&cfg.run_id, child);     // store the Child handle for stop_training
    Ok(TrainingHandle { run_id: cfg.run_id, kind: RunnerKind::Native, started: true })
}
```

**Stop escalation (salvage the v1 logic, `docker.service.ts:102-126`):** `stop_training` issues SIGTERM, waits a bounded grace period, then SIGKILL, guarded on the actual `exitCode`/`signalCode` (not a misleading `killed` flag). For Docker, `bollard.stop_container` (which does the SIGTERM→SIGKILL escalation) then `remove_container`.

### 6.7 Auto-updater flow (the structural `A5-C1` fix)

```
ASCII: user-initiated, signature-verified update (NO auto-install-on-quit)

 startup ──► check_for_update()  ──► UpdateStatus{available:true, version, notes}
                                          │  (advisory only; render UpdateBanner)
 user clicks "Update now"  ──► download_and_install_update()
        │
        ▼
 tauri-plugin-updater:
   1. fetch latest.json from tauri.conf updater.endpoints  (HTTPS)
   2. download the artifact
   3. VERIFY the artifact's minisign signature against updater.pubkey baked
      into tauri.conf.json — MANDATORY; a bad/missing signature ABORTS the install
      (the framework refuses it — this is what makes A5-C1 structurally impossible)
   4. emit update://progress events during download
   5. install + relaunch on user confirmation
```

**Why this kills `A5-C1` by construction:** the v1 Electron updater set `autoDownload=true` + `autoInstallOnAppQuit=true` over **unsigned** GitHub releases (`updater.ts:13-14`, verified above), so anyone who could publish a release got silent RCE on every machine on next quit. Tauri's updater **requires** a minisign-signed artifact and rejects unsigned ones in the framework itself (`B5 §4`, `02-TECH-STACK.md §16.1`). v2 additionally makes updates **user-initiated** (no auto-download, no auto-install-on-quit) — the `A5-C1` Phase-0 posture is the *permanent* posture here.

---

## 7. Data it owns

**Control-plane schema (`03-DATA-MODEL.md`):** the desktop owns **no tables and no columns**. It is a client of the REST/STOMP API and references server-owned entities only by id on the wire:

- `fl_runs.id` (UUID) — carried in `TrainingConfig.run_id`, passed to the subprocess as `--run-id` / `FEDLEARN_RUN_ID`. Never written by the desktop.
- `projects.id` (UUID) — `TrainingConfig.project_id`, used as the STOMP topic key (`/topic/logs/{projectId}`, §11 of `04-API-CONTRACTS.md`).
- `users.id` (BIGINT → `i64`) — surfaced read-only inside `MeResponse.user_id`.

**Local persistent state (desktop-owned, outside the database):**

| Store | Location | Contents | Lifetime |
|---|---|---|---|
| OS keychain entry | macOS Keychain / Windows Credential Manager / Linux Secret Service, service `com.fedlearn.desktop`, account `session` | the serialized **opaque cookie jar** (the `jwtToken` cookie + attributes), never the bare token | until `logout` or keychain deletion |
| settings file | Tauri app-config dir (`tauri::path::app_config_dir`), `settings.json` | non-secret prefs: backend base URL override, last hardware profile, last dataset dir | until user clears |

**In-memory structures (`src-tauri/src/state.rs`):**

```rust
pub struct AppState {
    pub http: reqwest::Client,                              // owns the cookie jar (cookie_store(true))
    pub backend_base_url: String,                           // from FEDLEARN_BACKEND_URL / settings (§8)
    pub native_children: Mutex<HashMap<String, tokio::process::Child>>, // run_id -> child (for stop)
    pub docker_containers: Mutex<HashMap<String, String>>,  // run_id -> container_id (for stop/remove)
}
```

This replaces the v1 in-memory `ConcurrentHashMap`-style child tracking; it tracks only **local** children for `stop_training`. The durable run record lives in `fl_runs` on the server (`03-DATA-MODEL.md §4.1`); the desktop never duplicates it.

---

## 8. Configuration & environment variables

### 8.1 Desktop-process configuration

| Name | Type | Default | Source / mode | Purpose |
|---|---|---|---|---|
| `FEDLEARN_BACKEND_URL` | string (URL) | `https://fedlearn.duckdns.org` (release); `http://localhost:8081` (dev) | env or `settings.json` | Base URL for REST/WS. In a packaged release, HTTPS only. |
| `__FEDLEARN_PREVIEW__` | compile-time bool | `false` | Vite `define`, dev-only | The only switch that lets the renderer run without the Tauri bridge (§5.4). Never true in a release build. |
| `NODE_ENV` | string | `production` (release) | renderer build | Renderer build mode; does **not** gate the updater (the updater is gated on `tauri::is_dev()` / packaged state, not `NODE_ENV` — kills the `forceDevUpdateConfig` footgun `A5-C1`). |

### 8.2 Environment passed INTO the spawned FL client subprocess (locked names from `04-API-CONTRACTS.md §13` and §14)

`client_env(&cfg)` (used by both `native_runner.rs` and `docker_runner.rs`) sets exactly:

| Env var | Value | Source |
|---|---|---|
| `FEDLEARN_RUN_ID` | `cfg.run_id` (the `fl_runs.id` UUID) | `04-API-CONTRACTS.md §13` |
| `FEDLEARN_RUN_TOKEN` | the per-run scoped token `flrun_<...>` (HMAC-signed; the subprocess sets `Authorization: Bearer ${FEDLEARN_RUN_TOKEN}` on its `/api/internal/runs/{runId}/**` callbacks) | `04-API-CONTRACTS.md §13` |
| `FEDLEARN_BACKEND_URL` | base URL for `/api/internal/...` (HTTPS outside dev) | `04-API-CONTRACTS.md §13` |
| `FEDLEARN_PROJECT_ID` | `cfg.project_id` (display/log convenience) | `04-API-CONTRACTS.md §13` |
| `TRACEPARENT` | the W3C (World Wide Web Consortium) `traceparent` of the launch span, e.g. `00-<32hex>-<16hex>-01` | `04-API-CONTRACTS.md §14` |

> **Where these come from:** the desktop receives `run_id`, `enrollment_token`, the per-run token, and `grpc_endpoint` when it starts a run via the control plane (`POST /api/projects/{projectId}/runs` → `RunDto`, `04-API-CONTRACTS.md §4`). The desktop forwards them into the child; it never mints them.

### 8.3 `tauri.conf.json` — the updater + CSP + capabilities (load-bearing)

```jsonc
{
  "productName": "FedLearn",                 // ONE brand; retire "FedLearn Desktop" (C5 §8)
  "version": "2.0.0",
  "app": {
    "security": {
      // PRODUCTION CSP — NO 'unsafe-eval' (fixes A5-H1); NO third-party CDN (fixes A5-H2 Perplexity entry).
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self' https://fedlearn.duckdns.org wss://fedlearn.duckdns.org"
    }
  },
  "plugins": {
    "updater": {
      "endpoints": [
        "https://releases.fedlearn.example/desktop/{{target}}/{{arch}}/{{current_version}}"
      ],
      // minisign PUBLIC key (base64). The PRIVATE key signs artifacts in CI and is NEVER in the repo.
      "pubkey": "<MINISIGN_PUBLIC_KEY_BASE64>",
      "windows": { "installMode": "passive" }
    }
  }
}
```

`src-tauri/capabilities/default.json` (deny-by-default capability allowlist — the renderer may invoke only the §5 commands):

```jsonc
{
  "identifier": "default",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "updater:default",
    "shell:allow-execute",        // only the bundled client binary, scoped by the externalBin/resource entry
    { "identifier": "core:event:allow-listen", "allow": ["training://log","training://exit","update://progress","update://error"] }
  ]
}
```

**Why move release hosting off the public mono-repo:** `A5-C1` Phase-1+ requires that "can push code" ≠ "can push an update to every user". The `endpoints` point at a dedicated release host, and the minisign **private** key lives only in CI signing secrets, not the repo.

---

## 9. Error handling & edge cases

`src-tauri/src/error.rs` defines one error enum serialized to the renderer as `{ code, message }` (matching the spirit of the platform error envelope, `04-API-CONTRACTS.md §12`). The renderer switches on `code`, shows `message`.

```rust
#[derive(thiserror::Error, Debug)]
pub enum DesktopError {
    #[error("network error: {0}")]              Network(String),       // code "NETWORK"
    #[error("not authenticated")]               NotAuthenticated,      // code "NOT_AUTHENTICATED"
    #[error("bad credentials")]                 BadCredentials,        // code "BAD_CREDENTIALS"
    #[error("keychain error: {0}")]             Keychain(String),      // code "KEYCHAIN"
    #[error("dataset path invalid: {0}")]       DatasetPath(String),   // code "DATASET_PATH_INVALID"
    #[error("client binary missing")]           ClientBinaryMissing,   // code "CLIENT_BINARY_MISSING"
    #[error("docker error: {0}")]               Docker(String),        // code "DOCKER"
    #[error("process spawn failed: {0}")]       Spawn(String),         // code "SPAWN_FAILED"
    #[error("run not found locally: {0}")]      RunNotFound(String),   // code "RUN_NOT_FOUND"
    #[error("update error: {0}")]               Update(String),        // code "UPDATE"
    #[error("update signature invalid")]        UpdateSignatureInvalid,// code "UPDATE_SIGNATURE_INVALID"
}
impl serde::Serialize for DesktopError { /* -> { "code": <stable>, "message": <self.to_string()> } */ }
```

Enumerated failure modes and exact handling:

| # | Failure mode | Detection | Handling |
|---|---|---|---|
| 1 | Preload/bridge missing in a packaged build | `ensureBridge()` returns `{ok:false}` (§5.4) | Render `<FatalBridgeError/>`; **never** an authenticated shell (fixes `A5-C3`). |
| 2 | Keychain unavailable (headless Linux, no Secret Service) | `keyring::Error` on `store`/`load` | Fall back to in-memory cookie jar only; never write reversible bytes to disk (salvaged v1 posture). Warn once. |
| 3 | Login 401 | `POST /api/auth/login` → 401 | Return `DesktopError::BadCredentials`; renderer shows the login error inline. |
| 4 | `/api/auth/me` 401 (silent probe) | `check_auth` sees 401 | Swallow → `AuthResult{authenticated:false}` (the *only* endpoint where 401 is non-fatal, `04-API-CONTRACTS.md §2`). |
| 5 | Dataset path traversal / NUL / non-existent | `dataset_path::validate` | Reject with `DatasetPath`; never interpolate into the `:/data` bind (salvage `ipc.handlers.ts:47-82`). |
| 6 | Client binary missing (broken bundle) | `resolve_client_binary` returns `Err` | `ClientBinaryMissing`; renderer shows "reinstall" guidance (relates to unsigned-bundle `A5-M7`, fixed by signing). |
| 7 | Docker daemon down / Jetson socket unreachable | `Docker::connect_*` / `create_container` error | `Docker(...)`; surface to renderer; do not crash the shell. |
| 8 | `--runtime nvidia` accidentally set on Jetson | code review + the locked rule (§6.5) | Never set `host_config.runtime` for Jetson; the device-mount path is the only path (the project conventions / `docker.service.ts:300` comment). |
| 9 | Multiplexed Docker stream split across chunks | bollard handles demux | Use bollard's demuxed stream; do **not** hand-roll the 8-byte-header parser (the v1 `demuxDockerStream` partial-frame corruption, `A5-H3`). |
| 10 | Child crash / OOM during training | `training://exit` with non-zero code | Process isolation means the UI survives; mark run-local finished; the server-side run status comes from REST/STOMP, unaffected (`B5 §2.3`). |
| 11 | Update artifact signature invalid/missing | `tauri-plugin-updater` verification fails | Abort install; emit `update://error`; return `UpdateSignatureInvalid`. The framework refuses unsigned artifacts (the `A5-C1` structural guarantee). |
| 12 | Update check while offline | endpoint fetch fails | `UpdateStatus{available:false}`; non-fatal; retried on next manual check. |
| 13 | Quit while a child is running | Tauri `on_window_event` / exit hook | Drain native children (SIGTERM→SIGKILL) and `stop`+`remove` Docker containers (salvage the v1 `before-quit` lifecycle, `main.ts:144-159`). |
| 14 | Renderer tries to render a malicious log line | log lines arrive as `string` via `training://log` | Render as a **text node** only (never `dangerouslySetInnerHTML`); with `unsafe-eval` removed (A5-H1) a regression is not exploitable (`A5-M2`). |

---

## 10. Testing strategy

| Layer | Framework | What to test | Named test → assertion |
|---|---|---|---|
| Rust unit | `cargo test` (built-in) | `dataset_path::validate` | `rejects_nul_byte` → input with `\0` returns `DatasetPath`. `rejects_parent_traversal` → `../../etc` after resolve returns `DatasetPath`. `accepts_existing_dir` → a real temp dir returns `Ok`. |
| Rust unit | `cargo test` | Jetson device set | `jetson_mounts_exact` → the returned `Vec<DeviceMapping>` equals the 6 `/dev/nvhost-*`+`/dev/nvmap` entries, all `rwm`. `jetson_never_sets_runtime` → the constructed `HostConfig.runtime` is `None` (the locked rule, `A5` Jetson verdict + the project conventions). |
| Rust unit | `cargo test` | keychain wrapper | `store_load_roundtrip` (mock keyring) → bytes in == bytes out. `load_missing_is_none` → `NoEntry` maps to `Ok(None)`. `renderer_never_gets_token` → `AuthResult` has no token field (compile-time / shape assertion). |
| Rust unit | `cargo test` | updater config | `updater_pubkey_present` → `tauri.conf.json` parses and `plugins.updater.pubkey` is non-empty. `no_auto_install` → there is no `autoInstallOnAppQuit`-style auto path in `updater.rs` (the `A5-C1` regression gate; mirrors `A5-H4`(d)). |
| Renderer unit | **Vitest** (`02-TECH-STACK.md §15` aligns desktop renderer to Vitest) | fail-closed bridge | `ensureBridge_fails_closed_in_packaged` → with `__FEDLEARN_PREVIEW__` false and no bridge, returns `{ok:false}` (the `A5-C3` gate, mirrors `A5-H4`(c)). `checkAuth_preview_is_logged_out` → preview path returns `authenticated:false`, never `true`. |
| Renderer unit | Vitest | log rendering | `log_line_renders_as_text` → a `training://log` payload containing `<img onerror=...>` renders escaped (no HTML execution), guarding `A5-M2`. |
| Integration | `cargo test` + a fake reqwest server (`wiremock`) | login/keychain | `login_persists_opaque_jar` → after `login`, the keychain holds bytes and no struct field exposes the token (the `A5-C2` gate, mirrors `A5-H4`(b)). |
| End-to-end (manual gate) | `cargo tauri build` + a clean VM | signed update | install N, publish N+1 signed, confirm update applies; publish an **unsigned** artifact, confirm the framework **refuses** it (the `A5-C1` proof). |

**Coverage target rationale:** the v1 unit had **zero tests** on the three highest-risk modules (updater, auth, docker stream — `A5-H4`). The named tests above make each `A5` Critical/High a CI gate, exactly as `A5-H4` recommends.

---

## 11. Build & run (this unit in isolation)

```bash
# --- one-time toolchain (pinned) ---
rustup toolchain install 1.87.0          # matches rust-toolchain.toml (channel = "1.87.0")
cargo install tauri-cli --version 2.11.2 # or use the project-local devDependency @tauri-apps/cli

# --- install renderer deps ---
cd fedlearn-desktop
npm install                              # React renderer + @tauri-apps/api + @tauri-apps/cli

# --- run in dev (renderer hot-reload + Rust core) ---
npm run tauri dev                        # = `tauri dev`: vite dev server + cargo run; updater gated OFF in dev

# --- lint / typecheck the renderer ---
npm run lint                             # ESLint (frontend config)
npx tsc --noEmit                         # TS 5.x typecheck

# --- Rust checks ---
cd src-tauri
cargo fmt --check
cargo clippy -- -D warnings
cargo test                               # all §10 Rust + integration tests

# --- production build (signed) ---
cd ..
export TAURI_SIGNING_PRIVATE_KEY="$(cat ~/.fedlearn/minisign.key)"   # CI secret; NEVER in repo
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="..."                       # CI secret
npm run tauri build                      # produces signed installers per OS + latest.json

# --- verify the unit in isolation ---
# 1. dev run reaches the login screen and authenticates against a local backend (FEDLEARN_BACKEND_URL=http://localhost:8081)
# 2. `cargo test` is green (the A5-C1/C2/C3 + Jetson gates pass)
# 3. the production bundle runs WITHOUT 'unsafe-eval' in the CSP (A5-H1 verification)
# 4. an unsigned update artifact is REFUSED by the updater (A5-C1 proof)
```

**macOS/Windows signing (per `02-TECH-STACK.md §16.2` budget):** Apple Developer ID + notarization ($99/yr, notarization free); Windows Azure Trusted Signing (~$120/yr, US/Canada + legal entity) or OV/EV Authenticode (~$200–580/yr). This is **OS-code-signing** and is **separate from** and **in addition to** the minisign update signature (defense in depth, `B5 §5`).

---

## 12. Reasoning & alternatives

| Decision | Chosen | Rejected | Why (audit) |
|---|---|---|---|
| Shell | **Tauri v2** | Electron | Tauri's updater **requires** minisign-signed artifacts → the `A5-C1` unsigned-auto-install RCE class is structurally impossible, not merely patched (`B5 §4`, `A5-C1`). Also removes Node from the privileged process (`B5 §4`). |
| Shell | **Tauri v2** | Per-OS native (Swift/WinUI/GTK) | 3× UI surface for an orchestrator UI, identical signing cost, no payoff for a small team (`B5 §3`). |
| Training engine | **out-of-process PyInstaller subprocess** (Docker only on Jetson) | in-process C++ mobile core | No bundle win (libtorch dominates either way ~267 MB–1.9 GB), adds a fragile cross-language RNG-parity invariant, collapses the process-isolation boundary (`B5 §2.1–2.3`). Same-language Python keeps DeComFL RNG parity free. |
| Updater posture | **check-only + user-initiated install**, minisign-verified | auto-download + auto-install-on-quit | The v1 `autoDownload=true`+`autoInstallOnAppQuit=true` over unsigned public-repo releases = silent RCE on every machine (`A5-C1`, verified `updater.ts:13-14`). |
| Token storage | **opaque cookie jar in OS keychain**, renderer never sees it | extract the cookie value → replay as `Bearer` | The v1 desktop laundered the HttpOnly token into app-readable storage, the one client that breaks the cookie-only invariant (`A5-C2`). Option (ii) of the `A5-C2` recommendation restores it. |
| Auth gate | **fail-closed**; preview path returns `authenticated:false` | fail-open to a fake `preview-user` | The v1 bridge rendered the authenticated shell to an unauthenticated user on any packaging regression (`A5-C3`, Critical). |
| Docker client | **bollard** | dockerode / hand-rolled demux | bollard demuxes the multiplexed stream, removing the v1 `demuxDockerStream` partial-frame corruption (`A5-H3`); the Jetson device set ports verbatim (`B5 §6`). |
| Jetson GPU access | **`/dev/nvhost-*` device mounts**, never `--runtime nvidia` | `--runtime nvidia` | `--runtime nvidia` hangs on Jetson searching for PCIe discrete-GPU metadata (the project conventions / `docker.service.ts:300`). |
| Renderer CSP | **no `'unsafe-eval'`, no third-party CDN** | v1 packaged CSP with `'unsafe-eval'` + Perplexity `font-src` | Widens RCE blast radius (`A5-H1`); the Perplexity entry is copy-paste leakage with live third-party egress in a healthcare-vertical app (`A5-H2`). |
| Updater payload types | **typed `UpdateStatus`/`UpdateProgress`** | `any` across the bridge | `any` on data crossing the trust boundary is exactly where to be strict (`A5-M1`). |
| Design system | **shared `@fedlearn/tokens` + shadcn/ui in the renderer** | desktop's two conflicting `:root` token blocks | The v1 desktop shipped two non-matching accents, neither matching web (`C5 §1`); one OKLCH token source unifies all surfaces (`C5 §7`). |

**Open risks to re-verify at build time (flagged, not blockers, from `02-TECH-STACK.md §16.1` and `B5 §10`):**
1. WebKitGTK rendering parity for framer-motion / recharts on Linux — smoke-test before committing.
2. Tauri sidecar code-signing issues #11778 / #9981 — re-check against the pinned `2.11.2` release; if unresolved, sign the FL client as a **non-sidecar external binary** the build signs itself.
3. Azure Trusted Signing eligibility (US/Canada + legal entity) — confirm before budgeting the cheaper Windows path.

---

## 13. Build task checklist for the ~30B local model (ordered, dependency-first)

Each task is one file/feature with a clear done-condition. Do them in order.

1. **Scaffold the Tauri project.** Create `src-tauri/Cargo.toml` pinning `tauri 2.11.2`, `tauri-build 2.11.2`, `tauri-plugin-updater 2.x`, `tauri-plugin-shell 2.x`, `bollard` (exact), `keyring` (exact), `tokio`, `serde`, `serde_json`, `thiserror`, `reqwest` (with `cookies` feature); add `rust-toolchain.toml` (`channel = "1.87.0"`). **Done:** `cargo check` compiles an empty app.
2. **`error.rs`.** Implement `DesktopError` + the `Serialize` impl emitting `{code,message}` (§9). **Done:** `cargo test` builds; a unit test asserts each variant serializes its stable `code`.
3. **`state.rs`.** Implement `AppState` with a `reqwest::Client` (`cookie_store(true)`), `backend_base_url`, and the two `Mutex<HashMap>` child maps (§7). **Done:** `AppState::new()` returns and is `manage`-able.
4. **`keychain.rs`.** Implement `store`/`load`/`delete` over `keyring` (§6.2). **Done:** `store_load_roundtrip` and `load_missing_is_none` tests pass against a mock keyring.
5. **`auth.rs`.** Implement `login`/`logout`/`check_auth` against `04-API-CONTRACTS.md §2`; persist the cookie jar opaquely via `keychain` (§6.2); swallow 401 only on `check_auth`. **Done:** `login_persists_opaque_jar` integration test (wiremock) passes; `AuthResult` exposes no token field.
6. **`dataset_path.rs`.** Implement `validate` (NUL reject, `resolve`, residual `..` reject, exists + is_dir) (§6.4). **Done:** the three `dataset_path` unit tests pass.
7. **`hardware.rs`.** Implement `detect_hardware` with the 2s-timeout `nvidia-smi` probe and the profile match (§6.3). **Done:** returns a `HardwareInfo`; document the Asahi/ARM-server nit.
8. **`docker_runner.rs`.** Implement `jetson_device_mounts()` (exact 6-entry set), `start` (bollard create/start with `devices` set, **no** `runtime`), bollard-demuxed log stream → `training://log`, and container registration. **Done:** `jetson_mounts_exact` + `jetson_never_sets_runtime` tests pass.
9. **`native_runner.rs`.** Implement `resolve_client_binary` (platform-aware name), `start` (tokio spawn with `client_env`), stdout/stderr stream → `training://log`, child registration; SIGTERM→SIGKILL stop helper. **Done:** spawning a stub binary emits `training://log` and `training://exit`.
10. **`launcher.rs`.** Implement `start_training` (validate → dispatch native vs docker by profile), `stop_training` (native escalation or bollard stop+remove), `get_training_status` (§6.4). **Done:** Jetson config routes to docker, all others to native.
11. **`updater.rs`.** Implement `check_for_update` and `download_and_install_update` over `tauri-plugin-updater`, emitting `update://progress`/`update://error`; gate any updater code behind packaged/`!is_dev()` (§6.7). **Done:** `updater_pubkey_present` + `no_auto_install` tests pass.
12. **`client_env` helper.** Implement the exact env map (`FEDLEARN_RUN_ID`, `FEDLEARN_RUN_TOKEN`, `FEDLEARN_BACKEND_URL`, `FEDLEARN_PROJECT_ID`, `TRACEPARENT`) (§8.2). **Done:** a unit test asserts all five keys are set.
13. **`main.rs`.** Build the Tauri app: register all §5 commands, the updater + shell plugins, `AppState`, single-instance, and the quit/window-close hook that drains children (§9 #13). **Done:** `cargo tauri dev` launches and the renderer loads.
14. **`tauri.conf.json` + `capabilities/default.json`.** Set the production CSP (no `'unsafe-eval'`, no third-party CDN), `updater.endpoints` + `pubkey`, and the deny-by-default capability allowlist (§8.3). **Done:** the build embeds the CSP and pubkey; a test asserts the pubkey is non-empty.
15. **`src/bridge/types.ts` + `src/bridge/tauri.ts`.** Mirror every serde struct as a TS interface; implement the typed `invoke` wrappers + event listeners; implement `ensureBridge()` fail-closed (§5). **Done:** `tsc --noEmit` clean; `ensureBridge_fails_closed_in_packaged` Vitest passes.
16. **`src/main.tsx` + `App.tsx`.** Call `ensureBridge()` before render; render `<FatalBridgeError/>` on `{ok:false}`; auth gate reads `AuthResult.authenticated` (no fake preview-user) (§5.4, §6.1). **Done:** with no bridge in a non-preview build the app shows the fatal screen, never the shell.
17. **Renderer views.** Port Login, HardwareSelector (with override), ProjectList, RunView (subscribes STOMP `/topic/{logs,results,status,runs}/{projectId}` directly), Settings, UpdateBanner — all on shadcn/ui + `@fedlearn/tokens`; log lines render as text nodes (§9 #14). **Done:** `npm run lint` + `tsc` clean; UI reaches a live run view.
18. **Tests + CI gates.** Wire `cargo test`, `cargo clippy -D warnings`, Vitest, `npm audit --omit=dev`, and an `electronegativity`-equivalent Tauri capability lint into the unit CI (`desktop.yml`) per `A5-H4`/`A5-H5`. **Done:** CI is green and the `A5-C1/C2/C3` regression tests are required checks.
