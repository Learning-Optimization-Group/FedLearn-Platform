# FedLearn Desktop — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)

The `fedlearn-desktop` module is the cross-platform Electron application that serves as the local training orchestrator for FedLearn participants. It provides a secure graphical interface to configure, launch, and monitor federated learning training sessions — either through a PyInstaller native binary (macOS MPS, Windows CUDA/CPU) or a Docker container (NVIDIA Jetson SoC) — and to run inference against already-trained models.

> **Design-system history.** The renderer moved to the **Ember** system in `3.1.0-beta` (`0da835b`), which had itself superseded **Instrument** (`74cda60`). Both are now history: `2c50672` replaced Ember with **Ledger** — navy structural ink on quiet paper surfaces, light-first — and `3d54484` rolled Ledger across web, desktop and mobile. `src/renderer/tokens.css` is generated from `design/tokens.json` by `design/build-tokens.mjs` and carries a `DO NOT EDIT` header; a CI job (`scripts/check_design_tokens.sh`) fails the build if it drifts from the source of truth. Bricolage Grotesque was retired with Ember — the font stack is Hanken Grotesk (sans **and** display) plus JetBrains Mono, both self-hosted via `@fontsource`.
>
> The three-process security model, the contextBridge IPC contract, and the two execution paths below are unaffected by any of those cycles.

**Current version: `3.2.0-beta`** (`fedlearn-desktop/package.json`, mirrored in [`wikis/VERSIONS.md`](../VERSIONS.md)).

---

## Document Index

| # | Document | Description |
|---|---|---|
| 01 | [Overview & Architecture](./01-overview-and-architecture.md) | Three-process model, execution paths, data flow, directory structure, key design decisions |
| 02 | [Security Model](./02-security-model.md) | BrowserWindow hardening, CSP, contextBridge, double-validation, dataset-path consent, JWT confinement, safeStorage, XSS prevention, dependency posture |
| 03 | [Main Process Deep Dive](./03-main-process.md) | `main.ts`, `ipc.handlers.ts`, `validators.ts`, `docker.service.ts`, `auth.service.ts`, `http.ts`, `hardware.probe.ts`, the inference/client-project services, IPC channel reference |
| 04 | [Preload & IPC Bridge](./04-preload-ipc-bridge.md) | contextBridge API, validation strategy, complete `window.fedLearnAPI` reference, push channels, TypeScript integration |
| 05 | [Renderer & React Components](./05-renderer-components.md) | Shell layout, `App.tsx` state, `TrainSection`, `ModelPlayground`, `SettingsSection`, `AuthModal`, `LogPanel`, `StatusBar`, `UpdateBanner`, styles, performance patterns |
| 06 | [Build, Packaging & Distribution](./06-build-and-packaging.md) | Webpack configs, dev workflow, production build, electron-builder, PyInstaller bundle, platform details, code signing, release artifacts, CI gates |
| 07 | [Hardware Profiles & Training Execution](./07-hardware-profiles.md) | MPS, CUDA, CPU, Jetson profiles; execution flows; Jetson GPU access (the `--runtime nvidia` correction); launch arguments; lifecycle state machine; troubleshooting |
| 08 | [Developer Guide & Contributing](./08-developer-guide.md) | Prerequisites, setup, script reference, conventions, adding IPC channels/components/profiles, debugging, the jest suite, gotchas |

---

## Quick Reference

### Technology Stack

```
Electron 42.x  ←→  React 18.3 + TypeScript 5.7
     │
     ├── Main Process (Node.js)
     │   ├── dockerode          — Docker Engine API (Jetson path only)
     │   ├── electron-store + safeStorage — encrypted JWT + saved credentials
     │   ├── axios (shared `http` instance) — backend REST
     │   ├── @stomp/stompjs + ws — STOMP bridge for streaming inference tokens
     │   ├── electron-updater   — background auto-update
     │   └── electron-log       — structured logging
     │
     ├── Preload (contextBridge)
     │   └── Input validation + ipcRenderer
     │
     └── Renderer (Chromium sandbox)
         └── React SPA + lucide-react icons (no Node.js access)
```

### Execution Paths

| Profile | Backend | GPU Runtime |
|---|---|---|
| `mps` | PyInstaller binary | Apple Metal Performance Shaders |
| `discrete` | PyInstaller binary | NVIDIA CUDA |
| `cpu` | PyInstaller binary | CPU only |
| `jetson` | Docker container | NVIDIA Tegra — see [07](./07-hardware-profiles.md#nvidia-jetson-soc-docker) before trusting the device-mount recipe on JetPack 6+ |

`DockerService.startTraining()` dispatches on `hardwareProfile` alone: `jetson` → Docker, everything else → the bundled native client.

### Development Quick Start

```bash
cd fedlearn-desktop
npm install
npm run dev       # Starts 4 processes (webpack main + preload + renderer dev server + Electron)
```

### Build & Package

```bash
npm run build                # Production webpack compile (all three targets)
npm run package:mac          # macOS DMG + zip (arm64 only)
npm run package:win:cpu      # Windows NSIS (CPU client bundle)
npm run package:win:cuda     # Windows NSIS (CUDA client bundle)
npm run package:linux        # Linux AppImage + deb (x64 + arm64)
```

Every `package:*` script runs `scripts/check-native-bundle.js` **first** — packaging aborts loudly if the PyInstaller client bundle is missing rather than shipping an installer that fails on "Start training".

### Key Security Invariants

1. **JWT stays in Main.** The renderer only receives `{ success: boolean }`; the token is encrypted with `safeStorage` (OS keychain) and never crosses the bridge.
2. **All inputs validated twice.** Preload (`preload.ts`) and again in Main (`validators.ts`, called from `ipc.handlers.ts`).
3. **Only a user-chosen dataset directory can be bind-mounted.** `dataset-consent.ts` records paths returned by the native directory dialog; `docker:start-training` refuses any other path.
4. **Docker socket never mounted into containers.** Only `DockerService` in Main touches it — and it is not even probed until the Jetson path needs it.
5. **Log output rendered as plain text.** No `innerHTML`, no `dangerouslySetInnerHTML`.
6. **No new window creation from the renderer.** `setWindowOpenHandler` denies every open; `will-navigate` blocks off-app navigation.
7. **Remote plaintext `http://` server URLs are refused** unless the user explicitly acknowledges the risk (DE-13).

---

## Related Documentation

- [Backend Wiki](../backend/README.md) — Spring Boot API that coordinates federated learning rounds
- [Frontend Wiki](../frontend/README.md) — Web dashboard for project management
- [Framework Wiki](../framework/README.md) — Python federated learning framework. It is **entirely custom** — its own server, client, strategies and protobuf, with no Flower FL semantics. `flwr`/`flwr-datasets` are no longer dependencies of any unit either (dropped in `65048b6`); the desktop app never depended on them.
- [Client Docker Wiki](../client-docker/README.md) — the container image the Jetson profile runs, and the PyInstaller packaging scripts the desktop installers embed
- [Mobile Wiki](../mobile/README.md) — the on-device mobile FL client
