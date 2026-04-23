# FedLearn Desktop

> Electron client for federated-learning training. Ships with a native
> PyInstaller-bundled Python client on Mac and Windows — no Docker, no
> Python, no repo checkout needed on end-user machines. Jetson clients still
> go through Docker because NVIDIA's L4T torch wheel is firmware-pinned.

---

## For end users (install-and-run)

1. Download the installer for your platform:
   - **macOS (Apple Silicon)** — `FedLearn Desktop-X.Y.Z-arm64.dmg`
   - **Windows x64 with NVIDIA GPU** — `FedLearn-Desktop-Setup-X.Y.Z-cuda.exe`
   - **Windows x64 without GPU** — `FedLearn-Desktop-Setup-X.Y.Z-cpu.exe`
   - **Jetson AGX Orin** — install via `.deb` or AppImage, requires Docker + NVIDIA Container Toolkit (pre-installed with JetPack)
2. Install and launch. The app auto-detects your hardware (Apple Silicon, NVIDIA GPU, or CPU) and pre-selects the right profile.
3. Enter the server address + project ID and click **Start Training**.

---

## For developers

### Prerequisites

| Requirement | Notes |
|---|---|
| **Node.js** ≥ 18.x | For building and running the Electron app |
| **Python 3.11+** | Only needed for dev mode (the packaged installer bundles its own) |
| **Docker Engine** | Only needed for the Jetson profile |
| **FedLearn Backend** | Spring Boot API running at `http://localhost:8081` |

### Dev mode (system Python + repo checkout)

The `DockerService` falls back to spawning `python3 client-docker/scripts/client.py`
when `app.isPackaged === false`, so you don't need to rebuild the PyInstaller
bundle after every Python edit. Make sure you have the framework installed:

```bash
pip install -e framework
pip install -r client-docker/packaging/requirements-client.txt
```

### Building a distributable installer

Distributables bundle the native client as a PyInstaller-produced binary
inside `resources/`. Build the client first, then package Electron:

```bash
# 1. Build the native client (pick the one matching your installer target)
cd client-docker/packaging
./build-mac.sh             # Mac arm64 (MPS)
# or on Windows:
.\build-win-cpu.ps1        # Windows CPU
.\build-win-cuda.ps1       # Windows CUDA 12.4

# 2. Package the Electron installer
cd ../../fedlearn-desktop
npm run package:mac        # Mac
npm run package:win:cpu    # Windows CPU
npm run package:win:cuda   # Windows CUDA
```

See `client-docker/packaging/README.md` for bundle size, troubleshooting,
and the rationale for the PyInstaller approach.

### Jetson (Docker path — unchanged)

Jetson deployment still goes through `fedlearn-client:latest`:

```bash
docker build -t fedlearn-client:latest \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -f client-docker/Dockerfile .
```

The desktop app picks the Docker path automatically when the user selects
the **Jetson SoC** hardware profile; other profiles use the native bundle.

---

## Development Setup

```bash
# 1. Install dependencies
cd fedlearn-desktop
npm install

# 2. Start the dev environment (renderer + main + preload, concurrent)
npm run dev

# 3. In a separate terminal, launch Electron
npm run dev:electron
```

The renderer dev server runs on `http://localhost:9000` with HMR. The Electron main process loads from this URL in development mode.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FEDLEARN_API_URL` | `http://localhost:8081/api` | Backend API base URL |
| `NODE_ENV` | — | Set to `production` for packaged builds |

---

## Production Build

```bash
# Build all three webpack targets (main + preload + renderer)
npm run build

# Package for current platform
npm run package

# Package for specific platforms
npm run package:mac    # macOS (.dmg)
npm run package:linux  # Linux (.deb + .rpm)
npm run package:win    # Windows (.exe via NSIS)
```

### What the Production Build Does

1. **Webpack** compiles Main, Preload, and Renderer bundles in production mode
2. **TerserPlugin** with `drop_console: true` strips all `console.*` calls from Renderer + Preload bundles (prevents JWT/path leakage to DevTools)
3. **electron-builder** packages the app with `asar` compression

---

## Hardware Profile Guide

The Electron app supports three hardware profiles that control how Docker containers access GPU resources:

### Discrete GPU (`discrete`)

For standard Linux/Windows workstations with NVIDIA GPUs (e.g., RTX 3090, A100).

- **Docker flag**: `--gpus all` (via `DeviceRequests`)
- **Requires**: NVIDIA Container Toolkit installed on the host
- **Typical use**: Lab workstations, cloud VMs with GPU passthrough

### Jetson SoC (`jetson`)

For NVIDIA Jetson edge devices (Orin, Xavier, Nano) with integrated Tegra GPUs.

- **Docker flag**: Direct device mounts (`/dev/nvhost-ctrl`, `/dev/nvhost-ctrl-gpu`, etc.)
- **Critical**: The standard `--runtime nvidia` flag is **PROHIBITED** on Jetson — it searches for PCIe discrete GPU metadata in the kernel device tree and **hangs indefinitely**
- **Requires**: L4T-based Docker image built with `BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`
- **Typical use**: Edge inference nodes, IoT deployments

### CPU Only (`cpu`)

For machines without GPU acceleration.

- **Docker flag**: None (no GPU configuration)
- **Typical use**: Development testing, non-GPU nodes, ARM devices

---

## Security Architecture

### Non-Negotiable Security Settings

| Setting | Value | Enforced In |
|---|---|---|
| `nodeIntegration` | `false` | `main.ts` |
| `contextIsolation` | `true` | `main.ts` |
| `sandbox` | `true` | `main.ts` |
| `remote` module | **Not used** | Project-wide |
| CSP Headers | Set on every window | `main.ts` (session-level) |

### IPC Security Model

1. **Renderer** calls `window.fedLearnAPI.*` (exposed via `contextBridge`)
2. **Preload** validates all inputs against explicit allowlists before forwarding
3. **Main Process** re-validates inputs (defense-in-depth) before executing
4. **Rejections** are logged via `electron-log`

### JWT Containment

- JWT is extracted from the backend's `Set-Cookie` header in Main Process
- Encrypted using `safeStorage` (OS keychain) and stored via `electron-store`
- Renderer **never** receives the token — only `{ success: boolean }` responses
- `AuthService.getAuthHeader()` is available only to Main Process services

### Console Stripping

- TerserPlugin with `drop_console: true` is applied to Renderer + Preload bundles
- Prevents accidental JWT, path, or config leakage to DevTools in production
- Main Process retains `electron-log` transport for operational logging

---

## Project Structure

```
fedlearn-desktop/
├── src/
│   ├── main/
│   │   ├── main.ts               # BrowserWindow + CSP + app lifecycle
│   │   ├── ipc.handlers.ts       # All ipcMain.handle registrations
│   │   ├── docker.service.ts     # DockerService using dockerode
│   │   └── auth.service.ts       # JWT storage + backend API calls
│   ├── preload/
│   │   └── preload.ts            # contextBridge API + input validation
│   └── renderer/
│       ├── App.tsx               # Main application layout
│       ├── index.tsx             # React 18 entry point
│       ├── index.html            # HTML template
│       ├── styles.css            # Design system + global styles
│       └── components/
│           ├── HardwareSelector.tsx   # Hardware profile cards + config
│           ├── LogPanel.tsx           # Plain-text log viewer
│           ├── StatusIndicator.tsx    # Container status badge
│           └── AuthModal.tsx          # Login modal
├── webpack.main.config.js        # Webpack: electron-main
├── webpack.renderer.config.js    # Webpack: renderer (web target)
├── webpack.preload.config.js     # Webpack: electron-preload
├── webpack.prod.config.js        # Webpack: production multi-config
├── electron-builder.yml          # Cross-platform packaging
├── tsconfig.json
└── package.json
```

---

## License

See the root [LICENSE](../LICENSE) file.
