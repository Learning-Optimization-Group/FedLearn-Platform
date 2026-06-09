# FedLearn Desktop — Build, Packaging & Distribution

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Build System Overview](#build-system-overview)
2. [Webpack Configuration](#webpack-configuration)
3. [Development Workflow](#development-workflow)
4. [Production Build](#production-build)
5. [electron-builder Packaging](#electron-builder-packaging)
6. [Native Client Bundle (PyInstaller)](#native-client-bundle-pyinstaller)
7. [Platform-Specific Details](#platform-specific-details)
8. [Code Signing](#code-signing)
9. [Release Artifacts](#release-artifacts)
10. [Environment Variables Reference](#environment-variables-reference)

---

## Build System Overview

FedLearn Desktop uses a **Webpack-based multi-target build system**. Each Electron process gets its own webpack configuration because they run in different runtime environments and have different module requirements:

```
Source (TypeScript + React)
    │
    ├── webpack.main.config.js      → dist/main/main.js
    │   (Target: electron-main)       + dist/main/ipc.handlers.js
    │                                 + dist/main/docker.service.js
    │                                 + dist/main/auth.service.js
    │                                 + dist/main/hardware.probe.js
    │
    ├── webpack.preload.config.js   → dist/preload/preload.js
    │   (Target: electron-preload)
    │
    └── webpack.renderer.config.js  → dist/renderer/
        (Target: web)                 + index.html
                                      + bundle.js
                                      + styles.css
```

In production, all three targets are built sequentially by `webpack.prod.config.js`.

---

## Webpack Configuration

### Main Process Config (`webpack.main.config.js`)

```javascript
// webpack.main.config.js
module.exports = {
  target: 'electron-main',  // Node.js context, Electron APIs available
  entry: './src/main/main.ts',
  output: {
    path: path.resolve(__dirname, 'dist/main'),
    filename: 'main.js',
  },
  module: {
    rules: [{ test: /\.tsx?$/, use: 'ts-loader' }],
  },
  externals: {
    // Don't bundle these — they must be required at runtime from node_modules
    'electron-log': 'commonjs electron-log',
    'electron-store': 'commonjs electron-store',
    'dockerode': 'commonjs dockerode',
  },
};
```

**Why are native modules external?** `electron-log`, `electron-store`, and `dockerode` contain native Node.js addons (`.node` files) or rely on dynamic `require()` patterns. Webpack cannot bundle these correctly — they must be resolved at runtime from the `node_modules` directory.

### Preload Config (`webpack.preload.config.js`)

```javascript
module.exports = {
  target: 'electron-preload', // Has ipcRenderer, but sandboxed
  entry: './src/preload/preload.ts',
  output: {
    path: path.resolve(__dirname, 'dist/preload'),
    filename: 'preload.js',
  },
};
```

The `electron-preload` target is important — it tells webpack which globals are available (e.g., `ipcRenderer` from Electron) and which aren't (e.g., full Node.js APIs in sandbox mode).

### Renderer Config (`webpack.renderer.config.js`)

```javascript
module.exports = {
  target: 'web',  // Standard browser environment
  entry: './src/renderer/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist/renderer'),
    filename: 'bundle.js',
  },
  plugins: [
    new HtmlWebpackPlugin({ template: './src/renderer/index.html' }),
  ],
  devServer: {
    port: 9000,
    hot: true,  // Hot Module Replacement
  },
};
```

### Production Config (`webpack.prod.config.js`)

The production config builds all three targets with:
- `mode: 'production'` (minification, tree-shaking)
- Source maps disabled or set to `'source-map'` for crash debugging
- `TerserPlugin` for JS minification

---

## Development Workflow

### Starting the Dev Environment

```bash
cd fedlearn-desktop
npm run dev
```

This runs four processes concurrently via `concurrently`:

```json
"dev": "concurrently --kill-others
  \"npm run dev:preload\"    ← webpack watch (preload)
  \"npm run dev:main\"       ← webpack watch (main)
  \"npm run dev:renderer\"   ← webpack-dev-server (renderer, port 9000, HMR)
  \"sleep 3 && npm run dev:electron\" ← Electron (after 3s build delay)"
```

The `sleep 3` ensures webpack has finished its initial compilation before Electron tries to load `dist/main/main.js`. Without it, Electron starts while the bundle is still being written and fails to launch.

### Process Breakdown

| Process | Script | Port | Purpose |
|---|---|---|---|
| Preload watcher | `dev:preload` | — | Rebuilds `preload.js` on changes |
| Main watcher | `dev:main` | — | Rebuilds `main.js` on changes |
| Renderer dev server | `dev:renderer` | 9000 | Serves renderer with HMR |
| Electron | `dev:electron` | — | Loads `dist/main/main.js` |

### Hot Module Replacement

The renderer dev server (webpack-dev-server) supports HMR for React components. When you modify a `.tsx` or `.css` file, the browser tab updates in-place without a full reload.

**Main and Preload do NOT support HMR.** When you modify `main.ts`, `docker.service.ts`, or `preload.ts`, you must **quit and restart Electron** to pick up the changes. The watcher will rebuild the file, but Electron has already loaded the old version.

### Connecting to the Backend in Development

The app defaults to `http://localhost:8081`. To use a different backend:

1. Launch the app
2. Click "Sign In" → expand the server config section
3. Change the URL → Save
4. Log in

Or set `FEDLEARN_API_URL` before launching:
```bash
FEDLEARN_API_URL=http://192.168.1.50:8081/api npm run dev:electron
```

---

## Production Build

```bash
npm run build
```

This runs `webpack.prod.config.js` which builds all three targets in production mode. The output is written to `dist/`:

```
dist/
├── main/
│   └── main.js           ← Minified Main Process bundle
├── preload/
│   └── preload.js        ← Minified Preload bundle
└── renderer/
    ├── index.html         ← HTML with injected bundle tags
    ├── bundle.js          ← Minified React app
    └── styles.css         ← (if extracted)
```

**Do not run `npm run build` in development** — it produces minified output which makes debugging very difficult. Use `npm run dev` for development.

---

## electron-builder Packaging

After the webpack build, `electron-builder` packages the app into a distributable installer.

### Quick Commands

```bash
# macOS (arm64 + x64 DMG)
npm run package:mac

# Windows (x64 NSIS installer)
npm run package:win:cpu    # CPU/MPS variant (no CUDA bundle)
npm run package:win:cuda   # CUDA variant (with CUDA native bundle)

# Linux (AppImage + deb)
npm run package:linux

# Current platform (auto-detected)
npm run package
```

All commands run `npm run build` first to ensure fresh compiled output.

### What Gets Included

From `electron-builder.yml`:

```yaml
files:
  - dist/**/*                  # Compiled webpack output
  - "!dist/**/*.map"           # Exclude source maps
  - node_modules/dockerode/**/* # Native modules (not bundled by webpack)
  - node_modules/electron-store/**/*
  - node_modules/electron-log/**/*
  - node_modules/axios/**/*

asar: true                     # Package files into asar archive
asarUnpack:
  - node_modules/dockerode/**/* # Unpack dockerode (native .node file)
```

`asar: true` packages most files into an archive for faster load times and to obscure the source. `asarUnpack` for `dockerode` is necessary because it contains a native `.node` addon that must be on the filesystem (not inside the archive) to be loaded by Node.js.

### Dependency Audit Posture

The shipped dependency tree is gated by `npm audit`. Electron is pinned to `^42.4.0` (bumped from `^34.5.8`) to clear all high/critical Chromium/Electron CVEs that `npm audit` flagged against the older `34.x` line — `tsc` and the Jest suite pass against Electron 42 with no application-code changes. The tree is now **clean at `--audit-level=high`**.

Four **moderate** advisories remain (the `uuid` buffer-bounds-check issue, reached transitively via `dockerode` 4.0.x and `webpack-dev-server`/`sockjs`). Their only fix is a breaking `dockerode@5.0.0` major, so they are deferred and tracked for a future `dockerode` upgrade rather than force-fixed. See [Security Model → Dependency Vulnerability Posture](./02-security-model.md#dependency-vulnerability-posture) for detail.

---

## Native Client Bundle (PyInstaller)

For macOS and Windows, the app ships a **PyInstaller bundle** of the Python training client. This eliminates the need for Python, pip, or any ML dependencies on the end user's machine.

### Build Location

```
client-docker/packaging/dist/fedlearn-client/
├── fedlearn-client        (macOS/Linux binary)
├── fedlearn-client.exe    (Windows binary)
└── _internal/             (PyInstaller support files)
```

### Packaging Scripts

```bash
# macOS (arm64 — must run on an Apple Silicon Mac)
client-docker/packaging/build-mac.sh

# Windows CPU (run on Windows)
client-docker/packaging/build-win-cpu.ps1

# Windows CUDA (run on Windows with CUDA toolkit)
client-docker/packaging/build-win-cuda.ps1
```

These scripts run PyInstaller and output to `client-docker/packaging/dist/fedlearn-client/`.

### How It's Included in the Electron App

In `electron-builder.yml`:
```yaml
extraResources:
  - from: ../client-docker/packaging/dist/fedlearn-client
    to: fedlearn-client
    filter:
      - "**/*"
```

This copies the entire bundle into the app's `resources/` directory. At runtime, `DockerService.resolveNativeInvocation()` finds it at:
```
process.resourcesPath + '/fedlearn-client/fedlearn-client'
```

### The Build Order (Critical)

```
1. Build the PyInstaller bundle (on the target platform)
2. npm run build      ← webpack compile
3. electron-builder  ← package into DMG/NSIS/AppImage
```

If you skip step 1, `extraResources` will fail because the source directory doesn't exist. `electron-builder` will either error out or produce a broken package.

---

## Platform-Specific Details

### macOS

```yaml
mac:
  category: public.app-category.developer-tools
  target:
    - target: dmg
      arch: [arm64, x64]
    - target: zip
      arch: [arm64, x64]
  titleBarStyle: hiddenInset     # Inset traffic lights
  darkModeSupport: true
  identity: null                 # Ad-hoc signed (no Apple Developer ID)
  hardenedRuntime: false
  gatekeeperAssess: false
```

**arm64 vs x64:** Both architectures are built. arm64 is for Apple Silicon (M1/M2/M3/M4); x64 is for Intel Macs. They are separate DMG files.

**Ad-hoc signing (`identity: null`):** Without an Apple Developer ID certificate, the app is ad-hoc signed. macOS Gatekeeper will block the first launch with "App is from an unidentified developer" or "damaged" warning. Users must right-click → Open, or run:
```bash
xattr -cr /Applications/FedLearn\ Desktop.app
```

**Why not use Hardened Runtime?** Hardened Runtime (required for notarization) prevents the app from executing arbitrary code, which conflicts with spawning the Python subprocess. Implementing it would require runtime exception entitlements and Apple Developer Program enrollment.

### Windows

```yaml
win:
  target:
    - target: nsis
      arch: [x64]

nsis:
  oneClick: false                          # Show installer wizard
  allowToChangeInstallationDirectory: true # Custom install path
  perMachine: false                        # Per-user install (no admin required)
```

**NSIS installer:** Produces a standard Windows `.exe` installer with a wizard UI.

**`perMachine: false`:** Per-user installation doesn't require administrator privileges. Users can install without IT approval.

**Two Windows variants:** The `package:win:cpu` and `package:win:cuda` scripts exist because the PyInstaller bundle includes PyTorch. The CPU variant includes `torch` without CUDA; the CUDA variant includes `torch` with CUDA support. Shipping both in one installer would double the size.

### Linux

```yaml
linux:
  target:
    - target: AppImage
      arch: [x64, arm64]
    - target: deb
      arch: [x64, arm64]
```

**AppImage:** A portable executable that runs on any Linux distribution without installation. Users `chmod +x FedLearn.AppImage && ./FedLearn.AppImage`.

**deb:** Debian/Ubuntu package. For systems that prefer traditional package management.

**arm64 on Linux:** Targets NVIDIA Jetson devices running Ubuntu 20.04/22.04 (L4T). However, Jetson training uses the **Docker path** (not native), so the Linux native bundle may not include CUDA support by default.

---

## Code Signing

### macOS

For production releases with a valid Apple Developer ID:

```bash
export CSC_LINK=/path/to/certificate.p12
export CSC_KEY_PASSWORD=your-p12-password
export APPLE_ID=your@apple.id
export APPLE_APP_SPECIFIC_PASSWORD=xxxx-xxxx-xxxx-xxxx
export APPLE_TEAM_ID=XXXXXXXXXX

# Uncomment notarize script in electron-builder.yml first
npm run package:mac
```

### Windows

```bash
export CSC_LINK=/path/to/certificate.pfx
export CSC_KEY_PASSWORD=your-pfx-password

npm run package:win:cuda
```

Without signing, Windows SmartScreen will show an "Unknown Publisher" warning on first run.

---

## Release Artifacts

After packaging, output is written to `release/`:

```
release/
├── FedLearn Desktop-2.2.4-beta-arm64.dmg     ← macOS Apple Silicon
├── FedLearn Desktop-2.2.4-beta-x64.dmg       ← macOS Intel
├── FedLearn Desktop-2.2.4-beta-arm64-mac.zip ← macOS zip (update mechanism)
├── FedLearn Desktop-Setup-2.2.4-beta.exe     ← Windows NSIS installer
├── FedLearn Desktop-2.2.4-beta.AppImage      ← Linux portable
└── fedlearn-desktop_2.2.4-beta_amd64.deb     ← Linux Debian package
```

Version is taken from `package.json`:
```json
{
  "version": "2.2.4-beta"
}
```

---

## Environment Variables Reference

| Variable | Used By | Default | Description |
|---|---|---|---|
| `NODE_ENV` | `main.ts` | `development` | Set to `production` for production builds |
| `FEDLEARN_API_URL` | `auth.service.ts` | `http://localhost:8081/api` | Backend API base URL (used if no saved URL) |
| `FEDLEARN_API_ORIGINS` | `main.ts` | *(empty)* | Comma-separated additional CSP `connect-src` origins |
| `FEDLEARN_CLIENT_IMAGE` | `docker.service.ts` | `fedlearn-client:latest` | Override Docker image name for training container |
| `PYTHONUNBUFFERED` | `docker.service.ts` | Set to `1` by app | Forces Python stdout to be unbuffered (real-time logs) |
| `PYTHONPATH` | `docker.service.ts` | Set by app | Adds `framework/src` to Python module search path (dev mode) |
| `CSC_LINK` | electron-builder | — | Path to code signing certificate |
| `CSC_KEY_PASSWORD` | electron-builder | — | Code signing certificate password |
| `APPLE_ID` | electron-builder | — | Apple ID for notarization |
| `APPLE_APP_SPECIFIC_PASSWORD` | electron-builder | — | App-specific password for notarization |
| `APPLE_TEAM_ID` | electron-builder | — | Apple Team ID for notarization |

---

*Next: [07 — Hardware Profiles & Training Execution](./07-hardware-profiles.md)*  
*Previous: [05 — Renderer & Components](./05-renderer-components.md)*
