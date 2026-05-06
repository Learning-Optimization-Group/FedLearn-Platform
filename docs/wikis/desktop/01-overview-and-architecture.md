# FedLearn Desktop — Overview & Architecture

> **Part of:** [FedLearn Platform Docs](../../README.md) → [Desktop Wiki](./README.md)

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
- **Configure** a training job: hardware profile, project, partition, model architecture, and local dataset
- **Launch** a containerized (Docker) or native (PyInstaller binary) Python training client
- **Stream real-time logs** from the running training process
- **Monitor** training status via live status polling

The desktop app is the bridge between the **end user's machine** (with its local dataset) and the **central FedLearn Spring Boot server** that coordinates federated learning rounds. Critically, **raw data never leaves the client machine** — only model updates (gradients/weights) are transmitted.

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Application Shell | Electron | `^34.5.8` | Cross-platform desktop container |
| UI Framework | React | `18.3.1` | Renderer UI components |
| Language | TypeScript | `5.7.3` | Type-safe development across all processes |
| Build Tool | Webpack | `^5.106.2` | Bundles main, preload, and renderer separately |
| Packaging | electron-builder | `^26.8.1` | Produces DMG, NSIS, AppImage distributable |
| Docker Control | dockerode | `^4.0.10` | Node.js Docker Engine API client |
| Secure Storage | electron-store + safeStorage | `8.2.0` | OS-encrypted JWT persistence |
| HTTP Client | axios | `^1.15.0` | Backend API calls (auth only, from Main) |
| Logging | electron-log | `5.3.0` | Structured log to file + console |

---

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   FedLearn Desktop (Electron)             │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  RENDERER PROCESS  (sandboxed, context-isolated)    │  │
│  │                                                     │  │
│  │   React App (App.tsx)                               │  │
│  │   ├── AuthModal       ← login form                  │  │
│  │   ├── HardwareSelector ← training config + start    │  │
│  │   ├── LogPanel        ← real-time log stream        │  │
│  │   ├── StatusIndicator ← container state badge       │  │
│  │   └── SettingsModal   ← server URL settings         │  │
│  │                                                     │  │
│  │   window.fedLearnAPI (contextBridge — read only)    │  │
│  └────────────────────────┬────────────────────────────┘  │
│                           │ IPC (validated)               │
│  ┌─────────────────────────▼────────────────────────────┐  │
│  │  PRELOAD SCRIPT  (trusted context bridge)            │  │
│  │   preload.ts                                         │  │
│  │   • Exposes window.fedLearnAPI via contextBridge     │  │
│  │   • Validates ALL inputs against allowlists          │  │
│  │   • Blocks invalid calls before Main ever sees them  │  │
│  └────────────────────────┬────────────────────────────┘  │
│                           │ ipcMain.handle               │
│  ┌─────────────────────────▼────────────────────────────┐  │
│  │  MAIN PROCESS  (full Node.js privileges)             │  │
│  │                                                     │  │
│  │   main.ts          ← BrowserWindow + CSP + security │  │
│  │   ipc.handlers.ts  ← ipcMain registrations          │  │
│  │   ├── docker.service.ts  ← container lifecycle      │  │
│  │   ├── auth.service.ts    ← JWT + backend API calls  │  │
│  │   └── hardware.probe.ts  ← GPU/platform detection   │  │
│  └───────────┬───────────────────────────┬─────────────┘  │
│              │                           │                 │
└──────────────┼───────────────────────────┼─────────────────┘
               │                           │
       Docker Daemon                FedLearn Backend
   (container lifecycle)            (Spring Boot API)
```

---

## Three-Process Model

Electron uses a **multi-process architecture** for security and stability. FedLearn Desktop leverages all three processes:

### 1. Main Process (`src/main/`)

The **privileged** process. It:
- Creates and manages `BrowserWindow`
- Has access to Node.js APIs (file system, child processes, Docker socket)
- Manages the JWT in memory/encrypted storage — **never exposed to the renderer**
- Runs `DockerService`, `AuthService`, and `HardwareProbe`
- Registers all `ipcMain.handle` listeners

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
- Listens for pushed log events via `onTrainingLog` callback

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
**Why:** NVIDIA Jetson uses a very specific JetPack/L4T PyTorch wheel that cannot be PyInstaller-bundled. Docker with direct `/dev/nvhost-*` device mounts is the supported path.

---

## Data Flow Diagram

```
User fills form                  → HardwareSelector (Renderer)
         ↓ onClick
validate client-side             → HardwareSelector (Renderer)
         ↓ window.fedLearnAPI.startTraining(config)
validate in preload allowlists   → preload.ts
         ↓ ipcRenderer.invoke('docker:start-training', config)
validate again in IPC handler    → ipc.handlers.ts / sanitizeDatasetPath()
         ↓ dockerService.startTraining(config)
route by hardwareProfile         → docker.service.ts
         ↓
  [jetson] → Docker container    OR   [mps/cpu/discrete] → Native process
         ↓                                    ↓
  stdout of container                   stdout of child process
         ↓                                    ↓
  sendLog(text) → ipcMain.send('docker:training-log', text)
         ↓
  ipcRenderer.on('docker:training-log') in preload
         ↓
  onTrainingLog callback in App.tsx
         ↓
  RAF-batched setLogs() → LogPanel re-render
```

---

## Directory Structure

```
fedlearn-desktop/
├── src/
│   ├── main/                    # Main Process (Node.js context)
│   │   ├── main.ts              # Entry point, BrowserWindow, CSP, app lifecycle
│   │   ├── ipc.handlers.ts      # All ipcMain.handle registrations
│   │   ├── docker.service.ts    # Docker + native process orchestration
│   │   ├── auth.service.ts      # JWT storage, login, logout
│   │   └── hardware.probe.ts    # GPU / platform detection
│   │
│   ├── preload/
│   │   └── preload.ts           # contextBridge API surface (security boundary)
│   │
│   └── renderer/                # Renderer Process (Chromium context)
│       ├── index.tsx            # React root mount
│       ├── index.html           # HTML shell (CSP meta tag for packaged builds)
│       ├── styles.css           # Global CSS (dark theme, components)
│       ├── App.tsx              # Root React component, state management
│       └── components/
│           ├── AuthModal.tsx    # Login form + server URL config
│           ├── HardwareSelector.tsx  # Profile cards + training config form
│           ├── LogPanel.tsx     # Real-time log stream viewer
│           ├── StatusIndicator.tsx   # Container status badge
│           └── SettingsModal.tsx     # Settings overlay (server URL)
│
├── build/                       # Static build assets (icons)
├── dist/                        # Compiled output (webpack)
├── release/                     # electron-builder output (DMG/EXE/AppImage)
│
├── webpack.main.config.js       # Webpack config for Main Process
├── webpack.preload.config.js    # Webpack config for Preload Script
├── webpack.renderer.config.js   # Webpack config for Renderer (with HMR)
├── webpack.prod.config.js       # Combined production build (all 3 targets)
├── electron-builder.yml         # Packaging config (macOS/Win/Linux)
├── tsconfig.json                # TypeScript config
└── package.json                 # Scripts + dependencies
```

---

## Key Design Decisions

### 1. JWT Never Reaches the Renderer

The authentication token is stored exclusively in the Main Process (via `electron-store` + `safeStorage`). The renderer only ever receives `{ success: boolean }`. This means even if the renderer is compromised (e.g., via XSS from a malicious log line), it cannot steal the JWT.

### 2. Double Validation (Preload + Main)

Every input is validated **twice**: once in `preload.ts` (renderer-side, for fast UX feedback) and once in `ipc.handlers.ts` (main-side, as defense-in-depth). This ensures that even a crafted `ipcRenderer.invoke` call from a compromised renderer cannot inject malicious data into Docker bind mounts or child process arguments.

### 3. MPS is Native-Only

Apple Silicon MPS (Metal Performance Shaders) cannot run inside Docker containers on macOS — Docker Desktop on Mac runs containers inside a Linux VM that has no access to the Apple GPU. Therefore, MPS training always uses the native PyInstaller binary path.

### 4. Jetson is Docker-Only

NVIDIA Jetson is the inverse case: its L4T PyTorch wheels are pinned to specific JetPack firmware, making PyInstaller bundling impractical. Docker with direct device mounts (`/dev/nvhost-*`) is the only reliable approach.

### 5. Log Batching via requestAnimationFrame

Container output can arrive in rapid bursts (hundreds of lines per second). Instead of calling `setState` on every IPC event (which would thrash React's reconciler), the app buffers incoming lines in a `useRef` and flushes the entire buffer in a single `setState` call on the next animation frame. This keeps the UI smooth even during heavy training output.

---

*Next: [02 — Security Model](./02-security-model.md)*
