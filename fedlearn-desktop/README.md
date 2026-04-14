# FedLearn Desktop

> Docker-based Federated Learning Orchestrator for edge devices and GPU workstations.

FedLearn Desktop is an Electron application that acts as a **system orchestrator** — it does not bundle PyTorch, CUDA, or the Python runtime. Instead, it controls the host's pre-installed Docker daemon to spin up `fedlearn-client` training containers with hardware-aware device mappings.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Node.js** ≥ 18.x | For building and running the Electron app |
| **Docker Engine** | Must be installed and running on the host |
| **NVIDIA Container Toolkit** | Required for `discrete` GPU profile only |
| **FedLearn Client Docker Image** | `fedlearn-client:latest` must be built/pulled locally |
| **FedLearn Backend** | Spring Boot API running at `http://localhost:8081` (default) |

### Building the Client Docker Image

From the repository root:

```bash
docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

For NVIDIA Jetson, use the L4T base image:

```bash
docker build -t fedlearn-client:latest \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -f client-docker/Dockerfile .
```

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
