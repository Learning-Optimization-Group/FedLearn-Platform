# FedLearn Desktop — Overview & Architecture

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [What is FedLearn Desktop?](#what-is-fedlearn-desktop)
2. [Technology Stack](#technology-stack)
3. [High-Level Architecture](#high-level-architecture)
4. [Three-Process Model](#three-process-model)
5. [Execution Paths](#execution-paths)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Directory Structure](#directory-structure)
8. [Key Design Decisions](#key-design-decisions)

---

## What is FedLearn Desktop?

FedLearn Desktop is a cross-platform Electron application that serves as the **local training orchestrator** for the FedLearn federated learning platform. It provides participants (clients) in a federated learning network with a graphical interface to:

- **Authenticate** against the FedLearn backend server
- **Pick a model to train** from the list of projects the signed-in user is allowed to join — the gRPC address, the server-assigned partition id, the model type, the aggregation strategy, the training arm, and the FL connection token all come back from the backend, not from a form
- **Choose** a local dataset directory (or explicitly skip it and use the recipe's built-in data) and, if needed, override the detected hardware profile
- **Launch** a containerized (Docker) or native (PyInstaller binary) Python training client
- **Stream real-time logs** from the running training process
- **Monitor** training status via live status polling
- **Run inference** against an already-trained model — image, numeric vector, text classification, or streamed text generation — through the Model Playground

The desktop app is the bridge between the **end user's machine** (with its local dataset) and the **central FedLearn Spring Boot server** that coordinates federated learning rounds. Critically, **raw data never leaves the client machine** — only model updates (weights/pseudo-gradients) are transmitted.

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Application Shell | Electron | `^42.4.0` | Cross-platform desktop container |
| UI Framework | React | `18.3.1` | Renderer UI components |
| Icons | lucide-react | `^0.487.0` | Icon set (replaced the emoji-based UI in `74cda60`) |
| Language | TypeScript | `5.7.3` | Type-safe development across all processes |
| Build Tool | Webpack | `^5.106.2` | Bundles main, preload, and renderer separately |
| Packaging | electron-builder | `^26.8.1` | Produces DMG, NSIS, AppImage, deb distributables |
| Auto-update | electron-updater | `^6.8.3` | Background download + restart-to-install |
| Docker Control | dockerode | `^4.0.10` | Node.js Docker Engine API client (Jetson path only) |
| Secure Storage | electron-store + safeStorage | `8.2.0` | OS-encrypted JWT and saved-credential persistence |
| HTTP Client | axios | `^1.15.0` | Backend REST calls from Main, through one shared instance (`src/main/http.ts`) |
| Streaming | `@stomp/stompjs` + `ws` | `^7.3.0` / `^8.21.0` | Main-process STOMP bridge that relays generation tokens to the renderer |
| Fonts | `@fontsource/hanken-grotesk`, `@fontsource/jetbrains-mono` | `^5.2.8` | Self-hosted Ledger typefaces — no remote font host in the CSP |
| Logging | electron-log | `5.3.0` | Structured log to file + console |

Version pins are read from `fedlearn-desktop/package.json`. There is one `overrides` entry — `protobufjs >= 7.5.5` — forcing a patched transitive copy.

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   FedLearn Desktop (Electron)                  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  RENDERER PROCESS  (sandboxed, context-isolated)         │  │
│  │                                                          │  │
│  │   React App (App.tsx) — icon rail + section outlet        │  │
│  │   ├── AuthModal        ← login + server URL (auth gate)   │  │
│  │   ├── TrainSection     ← guided setup / running run       │  │
│  │   │     ├── HardwareProfilePicker (Advanced disclosure)   │  │
│  │   │     └── LogPanel   ← filtered, severity-coloured logs │  │
│  │   ├── ModelPlayground  ← "Use a model" inference + chat   │  │
│  │   ├── SettingsSection  ← server URL, updates, about       │  │
│  │   ├── UpdateBanner     ← auto-update layer (shell level)  │  │
│  │   └── StatusBar        ← host · hardware · StatusIndicator│  │
│  │                                                          │  │
│  │   window.fedLearnAPI (contextBridge — frozen surface)     │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │ IPC (validated)                 │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  PRELOAD SCRIPT  (trusted context bridge)                │  │
│  │   preload.ts                                             │  │
│  │   • Exposes window.fedLearnAPI via contextBridge         │  │
│  │   • Validates ALL inputs against allowlists              │  │
│  │   • Blocks invalid calls before Main ever sees them      │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │ ipcMain.handle                  │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  MAIN PROCESS  (full Node.js privileges)                 │  │
│  │                                                          │  │
│  │   main.ts            ← BrowserWindow, menu, CSP, quit    │  │
│  │   ipc.handlers.ts    ← every ipcMain registration        │  │
│  │   ├── validators.ts          ← pure input validation     │  │
│  │   ├── dataset-consent.ts     ← user-chosen path allowlist│  │
│  │   ├── docker.service.ts      ← container + native client │  │
│  │   ├── auth.service.ts        ← JWT, credentials, expiry  │  │
│  │   ├── http.ts                ← shared axios + 401 hook   │  │
│  │   ├── client-projects.service.ts ← "models I can train"  │  │
│  │   ├── inference.service.ts   ← one-shot inference        │  │
│  │   ├── inference-stream.service.ts ← STOMP token bridge   │  │
│  │   ├── deviceCapabilities.collector.ts ← RAM/disk/OS      │  │
│  │   ├── hardware.probe.ts      ← GPU/platform detection    │  │
│  │   └── updater.ts             ← electron-updater wiring   │  │
│  └───────────┬────────────────────────────────┬────────────┘  │
│              │                                │               │
└──────────────┼────────────────────────────────┼───────────────┘
               │                                │
       Docker Daemon                     FedLearn Backend
   (Jetson profile only)          (Spring Boot REST + STOMP)
```

`src/shared/` holds the modules imported from **both** processes (and from jest): `urlSecurity.ts` (loopback / plaintext-HTTP policy), `evaluateEligibility.ts` + `deviceCapabilities.types.ts` (the advisory device self-gate), and `bundleVariants.ts` (the shippable platform/arch/GPU variant manifest).

---

## Three-Process Model

Electron uses a **multi-process architecture** for security and stability. FedLearn Desktop leverages all three processes:

### 1. Main Process (`src/main/`)

The **privileged** process. It:
- Creates and manages `BrowserWindow`, and sets a standard-roles-only application menu (no custom items, so no menu item needs an IPC channel)
- Has access to Node.js APIs (file system, child processes, Docker socket)
- Manages the JWT in memory/encrypted storage — **never exposed to the renderer**
- Owns every backend HTTP call, through one shared axios instance that carries the `X-FedLearn-Client` marker and the single 401 session-expiry interceptor
- Runs `DockerService`, `AuthService`, `ClientProjectService`, `InferenceService`, `InferenceStreamService`, the hardware probe and the device-capability collector
- Registers all `ipcMain.handle` listeners
- Drains any running training on `before-quit` (with a 15 s hard cap) so quitting mid-run cannot orphan a Jetson container

```typescript
// main.ts — The entry point
app.whenReady().then(() => {
  createWindow(); // Creates BrowserWindow with strict security settings
});
```

### 2. Preload Script (`src/preload/preload.ts`)

The **bridge** between renderer and main. It:
- Runs in a special context that has access to **both** the DOM (like the renderer) and `ipcRenderer` (to talk to main)
- Exposes a **strictly typed, allowlist-validated API** via `contextBridge.exposeInMainWorld`
- Never exposes raw `ipcRenderer` — the renderer can only call the surface defined in `window.fedLearnAPI`

```typescript
// preload.ts — The security boundary
contextBridge.exposeInMainWorld('fedLearnAPI', {
  startTraining: async (config) => {
    // Validate EVERY field before forwarding
    if (!isValidHardwareProfile(config.hardwareProfile)) {
      return { success: false, error: 'Invalid hardware profile' };
    }
    // ... more validation ...
    return ipcRenderer.invoke('docker:start-training', config);
  },
  // ...
});
```

### 3. Renderer Process (`src/renderer/`)

The **sandboxed UI** process. It:
- Runs like a regular web page inside Chromium
- Has **no access to Node.js** (`nodeIntegration: false`)
- Can only call the functions exposed on `window.fedLearnAPI`
- Renders the React component tree
- Listens for pushed events via the `onTrainingLog`, `onInferenceToken`, `onSessionExpired` and `onUpdate*` callbacks

---

## Execution Paths

A key design feature is that training supports **two execution paths** depending on hardware:

### Path A — Native Binary (macOS MPS, Windows CUDA/CPU)

```
Renderer → IPC → Main → spawn(fedlearn-client binary)
                         ↓
               PyInstaller bundle at <resourcesPath>/fedlearn-client/
                         ↓
               Python training client (no Docker required)
                         ↓
               stdout/stderr → IPC push → Renderer LogPanel
```

**When used:** `mps`, `discrete` (Windows), `cpu` profiles.  
**Why:** PyInstaller bundles the entire Python runtime — no system Python, no `pip install`, no Docker required on the end user's machine.

In **dev mode** (`app.isPackaged === false`) the same path spawns `python3 -u fl-runtime/client.py` from the repo checkout with `framework/src` prepended to `PYTHONPATH`, so a Python edit does not require rebuilding the bundle.

### Path B — Docker Container (Jetson SoC)

```
Renderer → IPC → Main → DockerService → Docker daemon
                                         ↓
                               docker.createContainer()
                               docker.start()
                                         ↓
                            attach log stream (multiplexed)
                                         ↓
                           demux → IPC push → Renderer LogPanel
```

**When used:** `jetson` profile only.  
**Why:** NVIDIA Jetson uses a very specific JetPack/L4T PyTorch wheel that cannot be PyInstaller-bundled. A prebuilt container image with the matching L4T stack is the practical path. *How* the container is granted GPU access is the part that has changed: see [07 — NVIDIA Jetson SoC](./07-hardware-profiles.md#nvidia-jetson-soc-docker), which documents a measured correction to the shipped device-mount approach on JetPack 6.

---

## Data Flow Diagram

```
User picks a model + dataset     → TrainSection (Renderer)
         ↓ getProjectConnection(projectId)
backend resolves the run          → client:get-connection → ClientProjectService
   (gRPC address, partitionId, modelType, strategy, trainingArm, token)
         ↓ window.fedLearnAPI.startTraining(config)
validate in preload allowlists    → preload.ts
         ↓ ipcRenderer.invoke('docker:start-training', config)
re-validate + canonicalize        → ipc.handlers.ts → validators.ts
   sanitizeDatasetPath() + isDatasetPathConsented()
         ↓ dockerService.startTraining(validConfig)
route by hardwareProfile          → docker.service.ts
         ↓
  [jetson] → Docker container     OR   [mps/cpu/discrete] → Native process
         ↓                                     ↓
  demuxed container stdout               child stdout/stderr
         ↓                                     ↓
  sendLog(text) → mainWindow.webContents.send('docker:training-log', text)
         ↓
  ipcRenderer.on('docker:training-log') in preload
         ↓
  onTrainingLog callback in App.tsx
         ↓
  RAF-batched setLogs() → TrainSection → LogPanel re-render
```

The renderer never composes a project id, gRPC address or partition id by hand; those come from `GET /api/client/projects/{id}/connection` (after an idempotent `POST .../join`) and are only re-validated on the way through.

---

## Directory Structure

```
fedlearn-desktop/
├── src/
│   ├── main/                          # Main Process (Node.js context)
│   │   ├── main.ts                    # Entry point, BrowserWindow, menu, CSP, quit drain
│   │   ├── ipc.handlers.ts            # All ipcMain.handle registrations
│   │   ├── validators.ts              # Pure input validation (electron-free, unit-tested)
│   │   ├── dataset-consent.ts         # In-memory allowlist of user-picked dataset dirs
│   │   ├── docker.service.ts          # Docker + native process orchestration
│   │   ├── auth.service.ts            # JWT + saved credentials, session expiry
│   │   ├── http.ts                    # Shared axios instance, SE-9 marker, 401 hook
│   │   ├── client-projects.service.ts # GET /client/projects + /connection
│   │   ├── inference.service.ts       # POST /inference/{id}, GET /inference/models
│   │   ├── inference-stream.service.ts# STOMP bridge for generation tokens
│   │   ├── deviceCapabilities.collector.ts # RAM / free storage / OS probe
│   │   ├── hardware.probe.ts          # GPU / platform detection
│   │   └── updater.ts                 # electron-updater registration (once per process)
│   │
│   ├── preload/
│   │   └── preload.ts                 # contextBridge API surface (security boundary)
│   │
│   ├── shared/                        # Imported by BOTH processes and by jest
│   │   ├── urlSecurity.ts             # Loopback detection + plaintext-HTTP policy text
│   │   ├── evaluateEligibility.ts     # Advisory device self-gate rule
│   │   ├── deviceCapabilities.types.ts
│   │   └── bundleVariants.ts          # Shippable platform/arch/GPU variant manifest
│   │
│   └── renderer/                      # Renderer Process (Chromium context)
│       ├── index.tsx                  # React root mount (StrictMode)
│       ├── index.html                 # HTML shell; CSP meta baked in at build time
│       ├── tokens.css                 # GENERATED from design/tokens.json — do not edit
│       ├── fonts.css                  # Self-hosted @fontsource imports
│       ├── styles.css                 # Global Ledger styles (imports tokens + fonts)
│       ├── client.types.ts            # ClientProject / ProjectConnection shapes
│       ├── inference.types.ts         # InferableModel / InferenceResult shapes
│       ├── App.tsx                    # Shell: rail, section outlet, StatusBar, IPC wiring
│       └── components/
│           ├── AuthModal.tsx          # Login, server URL, save-password, show-password
│           ├── TrainSection.tsx       # Guided setup ⇄ running-run layout
│           ├── trainFlow.ts           # Pure phase/readiness/format logic for TrainSection
│           ├── HardwareSelector.tsx   # HardwareProfilePicker (controlled card grid)
│           ├── LogPanel.tsx           # Filtered, severity-coloured, follow-tail log view
│           ├── logView.ts             # Incremental log-line cache + filter (pure)
│           ├── runNotifications.ts    # Run completed/failed desktop notifications (pure)
│           ├── ModelPlayground.tsx    # "Use a model" inference + chat
│           ├── SettingsSection.tsx    # Server URL, updates, about
│           ├── StatusBar.tsx          # Persistent bottom strip
│           ├── StatusIndicator.tsx    # Status badge (rendered inside StatusBar)
│           ├── UpdateBanner.tsx       # Auto-update layer (mounted at shell level)
│           └── sections.css           # Section-scoped styles
│
├── src/__tests__/                     # 22 jest suites (node env, ts-jest)
├── src/__mocks__/                     # electron / electron-log / electron-store / CSS stubs
├── scripts/
│   ├── check-native-bundle.js         # Packaging preflight (bundle must exist)
│   └── generate-checksums.js          # afterAllArtifactBuild → release/SHA256SUMS.txt
│
├── build/                             # Static build assets (icons, entitlements)
├── dist/                              # Compiled output (webpack)
├── release/                           # electron-builder output (DMG/EXE/AppImage/deb)
│
├── webpack.main.config.js             # Dev webpack config for Main Process
├── webpack.preload.config.js          # Dev webpack config for Preload Script
├── webpack.renderer.config.js         # Dev webpack config for Renderer (dev server + HMR)
├── webpack.prod.config.js             # Production build — exports all 3 configs as an array
├── webpack.csp.js                     # Single source of truth for the renderer <meta> CSP
├── electron-builder.yml               # Packaging config (macOS/Win/Linux)
├── eslint.config.mjs                  # ESLint 9 flat config (CI-gated)
├── jest.config.js                     # ts-jest, node env, coverage thresholds
├── tsconfig.json / tsconfig.test.json # TypeScript configs
└── package.json                       # Scripts + dependencies
```

---

## Key Design Decisions

### 1. JWT Never Reaches the Renderer

The authentication token is stored exclusively in the Main Process (via `electron-store` + `safeStorage`). The renderer only ever receives `{ success: boolean }`. This means even if the renderer is compromised (e.g., via XSS from a malicious log line), it cannot steal the JWT.

### 2. Double Validation (Preload + Main)

Every input is validated **twice**: once in `preload.ts` (renderer-side, for fast UX feedback) and once in Main (`ipc.handlers.ts` delegating to `validators.ts`, as defense-in-depth). This ensures that even a crafted `ipcRenderer.invoke` call from a compromised renderer cannot inject malicious data into Docker bind mounts or child process arguments.

The main-side validators live in their own `electron`-free module precisely so they can be unit-tested against the shipped code. `ipc.handlers.ts` used to carry a diverged inline copy — and the copy got the empty-dataset-path case wrong.

### 2a. Dataset Paths Need Consent, Not Just Validation

Proving a path is an existing absolute directory does **not** prove the *user* chose it. `dataset-consent.ts` keeps an in-memory set of the directories returned by the native `dialog:open-directory` picker; `docker:start-training` refuses to bind-mount any non-empty path that is not in that set. A compromised renderer therefore cannot get `~/.ssh` mounted into the training container even with a perfectly well-formed path.

### 3. MPS is Native-Only

Apple Silicon MPS (Metal Performance Shaders) cannot run inside Docker containers on macOS — Docker Desktop on Mac runs containers inside a Linux VM that has no access to the Apple GPU. Therefore, MPS training always uses the native PyInstaller binary path.

### 4. Jetson is Docker-Only

NVIDIA Jetson is the inverse case: its L4T PyTorch wheels are pinned to specific JetPack firmware, making PyInstaller bundling impractical. A prebuilt L4T container image is the reliable approach. The shipped code grants that container GPU access through direct `/dev/nvhost-*` device mounts and explicitly refuses `--runtime nvidia`; that refusal was **measured to be wrong on JetPack 6** and is documented in full, with the honest scope of what was and was not re-tested, in [07](./07-hardware-profiles.md#nvidia-jetson-soc-docker).

### 5. Docker Is Never Probed Eagerly

`DockerService`'s constructor builds a `dockerode` client but does **not** ping the daemon — `dockerode` opens the socket lazily, so constructing it costs nothing. The daemon is probed on demand inside `startDockerTraining()`, which is only reachable from the Jetson profile. An earlier eager ping produced a spurious "Docker is not running" banner for the overwhelming majority of users (macOS/Windows on MPS/CUDA/CPU) who never touch Docker at all; that banner and its `docker:daemon-unavailable` push channel were removed.

### 6. Log Batching via requestAnimationFrame

Container output can arrive in rapid bursts (hundreds of lines per second). Instead of calling `setState` on every IPC event (which would thrash React's reconciler), the app buffers incoming lines in a `useRef` and flushes the entire buffer in a single `setState` call on the next animation frame. This keeps the UI smooth even during heavy training output.

### 7. Sections Stay Mounted

The authenticated shell mounts Train, Models and Settings once and toggles visibility with CSS rather than unmounting. That is deliberate: the Model Playground's chat thread and streaming state, and the training log buffer, survive a section switch. `UpdateBanner` is mounted at shell level *forever* because its preload listeners have no removal API.

---

*Next: [02 — Security Model](./02-security-model.md)*
