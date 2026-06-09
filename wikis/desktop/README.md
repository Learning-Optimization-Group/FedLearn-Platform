# FedLearn Desktop — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)

The `fedlearn-desktop` module is the cross-platform Electron application that serves as the local training orchestrator for FedLearn participants. It provides a secure graphical interface to configure, launch, and monitor federated learning training sessions — either through a PyInstaller native binary (macOS MPS, Windows CUDA/CPU) or a Docker container (NVIDIA Jetson SoC).

---

## Document Index

| # | Document | Description |
|---|---|---|
| 01 | [Overview & Architecture](./01-overview-and-architecture.md) | Three-process model, execution paths, data flow, directory structure, key design decisions |
| 02 | [Security Model](./02-security-model.md) | BrowserWindow hardening, CSP, contextBridge, double-validation, JWT confinement, safeStorage, XSS prevention |
| 03 | [Main Process Deep Dive](./03-main-process.md) | `main.ts`, `ipc.handlers.ts`, `docker.service.ts`, `auth.service.ts`, `hardware.probe.ts`, IPC channel reference |
| 04 | [Preload & IPC Bridge](./04-preload-ipc-bridge.md) | contextBridge API, validation strategy, complete `window.fedLearnAPI` reference, push channels, TypeScript integration |
| 05 | [Renderer & React Components](./05-renderer-components.md) | `App.tsx` state machine, `AuthModal`, `HardwareSelector`, `LogPanel`, `StatusIndicator`, `SettingsModal`, styles, performance patterns |
| 06 | [Build, Packaging & Distribution](./06-build-and-packaging.md) | Webpack configs, dev workflow, production build, electron-builder, PyInstaller bundle, platform details, code signing, release artifacts |
| 07 | [Hardware Profiles & Training Execution](./07-hardware-profiles.md) | MPS, CUDA, CPU, Jetson profiles; execution flows; Docker device mounts; model architecture; lifecycle state machine; troubleshooting |
| 08 | [Developer Guide & Contributing](./08-developer-guide.md) | Prerequisites, setup, script reference, conventions, adding IPC channels/components/profiles, debugging, testing checklist, gotchas |

---

## Quick Reference

### Technology Stack

```
Electron 42.x  ←→  React 18 + TypeScript 5.7
     │
     ├── Main Process (Node.js)
     │   ├── dockerode  — Docker Engine API
     │   ├── electron-store + safeStorage — encrypted JWT
     │   ├── axios — backend HTTP auth
     │   └── electron-log — structured logging
     │
     ├── Preload (contextBridge)
     │   └── Input validation + ipcRenderer
     │
     └── Renderer (Chromium sandbox)
         └── React SPA (no Node.js access)
```

### Execution Paths

| Profile | Backend | GPU Runtime |
|---|---|---|
| `mps` | PyInstaller binary | Apple Metal Performance Shaders |
| `discrete` | PyInstaller binary | NVIDIA CUDA |
| `cpu` | PyInstaller binary | CPU only |
| `jetson` | Docker container | NVIDIA Tegra (direct `/dev/nvhost-*` mounts) |

### Development Quick Start

```bash
cd fedlearn-desktop
npm install
npm run dev       # Starts all 4 processes (webpack + Electron)
```

### Build & Package

```bash
npm run build                # Production webpack compile
npm run package:mac          # macOS DMG (arm64 + x64)
npm run package:win:cuda     # Windows NSIS (CUDA variant)
npm run package:linux        # Linux AppImage + deb
```

### Key Security Invariants

1. **JWT stays in Main.** Renderer only receives `{ success: boolean }`.
2. **All inputs validated twice.** Preload + Main process IPC handlers.
3. **Docker socket never mounted into containers.** Only `DockerService` accesses it.
4. **Log output rendered as plain text.** No `innerHTML`, no `dangerouslySetInnerHTML`.
5. **No new window creation from renderer.** `setWindowOpenHandler` blocks all opens.

---

## Related Documentation

- [Backend Wiki](../backend/README.md) — Spring Boot API that coordinates federated learning rounds
- [Frontend Wiki](../frontend/README.md) — Web dashboard for project management
- [Framework Wiki](../framework/README.md) — Python federated learning framework (Flower-based)
