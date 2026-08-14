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
10. [CI & Release Automation](#ci--release-automation)
11. [Environment Variables Reference](#environment-variables-reference)

---

## Build System Overview

FedLearn Desktop uses a **Webpack-based multi-target build system**. Each Electron process gets its own webpack configuration because they run in different runtime environments and have different module requirements:

```
Source (TypeScript + React)
    │
    ├── webpack.main.config.js      → dist/main/main.js
    │   (Target: electron-main)       (single bundle — every src/main/*.ts is
    │                                  bundled into main.js, not emitted per file)
    │
    ├── webpack.preload.config.js   → dist/preload/preload.js
    │   (Target: electron-preload)
    │
    └── webpack.renderer.config.js  → dist/renderer/
        (Target: web)                 + index.html
                                      + renderer.js  (dev)
```

Those three are the **development** configs. `webpack.prod.config.js` is a separate file that exports an **array** of three production configs — `module.exports = [mainConfig, preloadConfig, rendererConfig]` — so `npm run build` builds all three in one webpack invocation.

`webpack.csp.js` is shared by both renderer configs and is the single source of truth for the `<meta>` CSP baked into `index.html`.

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
    rules: [{ test: /\.ts$/, use: 'ts-loader', exclude: /node_modules/ }],
  },
  externals: {
    // Don't bundle these — they must be required at runtime from node_modules
    dockerode: 'commonjs dockerode',
    'electron-store': 'commonjs electron-store',
    'electron-log': 'commonjs electron-log',
    electron: 'commonjs electron',
  },
  node: { __dirname: false, __filename: false },  // keep real paths for resourcesPath resolution
};
```

**Why are these external?** `electron-log`, `electron-store`, and `dockerode` pull in optional native addons or rely on dynamic `require()` patterns webpack cannot follow; `electron` itself is provided by the runtime. All four are resolved from `node_modules` at runtime and are listed in `electron-builder.yml`'s `files` so they ship.

`node: { __dirname: false }` matters: the main bundle computes the preload path and the dev-mode repo root from `__dirname`, so webpack must not replace it with `"/"`.

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
  mode: 'development',
  target: 'web',  // Standard browser environment
  entry: './src/renderer/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist/renderer'),
    filename: 'renderer.js',
    publicPath: '/',
  },
  module: { rules: [
    { test: /\.tsx?$/, use: 'ts-loader', exclude: /node_modules/ },
    { test: /\.css$/, use: ['style-loader', 'css-loader'] },       // HMR needs runtime <style>
    { test: /\.(png|svg|jpg|jpeg|gif|ico)$/i, type: 'asset/resource' },
    { test: /\.(woff|woff2|eot|ttf|otf)$/i,  type: 'asset/resource' },  // self-hosted fonts
  ]},
  plugins: [
    new webpack.DefinePlugin({ __APP_VERSION__: JSON.stringify(require('./package.json').version) }),
    new HtmlWebpackPlugin({
      template: './src/renderer/index.html',
      templateParameters: { csp: buildRendererCsp({ allowEval: true, allowInlineStyle: true }) },
    }),
  ],
  devServer: { port: 9000, hot: true, /* + its own CSP header for direct browser hits */ },
};
```

Both renderer configs start with a shim that defines a stub `global.localStorage` — a workaround for a Node 22+ `SecurityError` thrown by `html-webpack-plugin`.

**`__APP_VERSION__`** is injected from `package.json` by `DefinePlugin` so the About card and StatusBar never ship a hardcoded version string (`2b02173`; pinned by `src/__tests__/webpack-app-version.test.ts`).

### Production Config (`webpack.prod.config.js`)

Exports `[mainConfig, preloadConfig, rendererConfig]`, all at `mode: 'production'` with `devtool: false`. Differences from the dev configs that matter:

| Aspect | Production behaviour |
|---|---|
| Minification | Two `TerserPlugin` instances. The **renderer and preload** bundles use `drop_console: true` + `pure_funcs` for `console.log/info/debug/warn`, stripping every `console.*` call so nothing leaks to Chromium DevTools. The **main** bundle deliberately keeps `console`, because `electron-log` may use it as a transport fallback. |
| CSS | `MiniCssExtractPlugin` extracts to `styles.[contenthash].css`, loaded via `<link rel="stylesheet">` — which is what lets the CSP drop `'unsafe-inline'` from `style-src`. |
| Filenames | `renderer.[contenthash].js`, `publicPath: './'` (required under a `file://` origin), `clean: true`. |
| Chunking | `splitChunks` with a `vendor` cache group for `node_modules`. |
| CSP | `buildRendererCsp({ allowEval: false, allowInlineStyle: false })`. |
| HTML | `HtmlWebpackPlugin` minifies (collapse whitespace, remove comments/redundant attributes). |

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
│   └── main.js                      ← Minified Main Process bundle
├── preload/
│   └── preload.js                   ← Minified Preload bundle (console stripped)
└── renderer/
    ├── index.html                   ← HTML with injected tags + the <meta> CSP
    ├── renderer.<contenthash>.js    ← Minified React app (console stripped)
    ├── vendor.<contenthash>.js      ← node_modules split chunk
    ├── styles.<contenthash>.css     ← Extracted CSS
    └── *.woff / *.woff2             ← Self-hosted font assets
```

**Do not run `npm run build` in development** — it produces minified output which makes debugging very difficult. Use `npm run dev` for development.

---

## electron-builder Packaging

After the webpack build, `electron-builder` packages the app into a distributable installer.

### Quick Commands

```bash
# macOS (arm64 DMG + zip — no Intel x64 target)
npm run package:mac

# Windows (x64 NSIS installer)
npm run package:win:cpu    # CPU variant (no CUDA in the client bundle)
npm run package:win:cuda   # CUDA variant

# Linux (AppImage + deb, x64 + arm64)
npm run package:linux

# Current platform (auto-detected)
npm run package
```

Each script is a three-step chain, and the **first** step is the one to know about:

```json
"package:mac": "node scripts/check-native-bundle.js mac && npm run build && electron-builder --mac --config electron-builder.yml"
```

`scripts/check-native-bundle.js` is a packaging preflight. Without it, running `npm run package:*` before building the PyInstaller bundle produces an installer with **no native client** — the app launches fine and then fails the instant the user clicks Start ("Native training bundle not found at `<resources>/fedlearn-client`"). The guard turns that silent shipping bug into a loud, actionable failure that names the exact build script to run.

It reads the platform→binary and platform→build-command maps out of `src/shared/bundleVariants.ts`, transpiling that TypeScript file in memory with the TypeScript compiler API (the script runs *before* `npm run build`, so no compiled output exists yet). `bundleVariants.ts` is deliberately import-free and self-contained so that load stays trivial.

The two Windows scripts additionally override the artifact name so both variants can coexist:

```
-c.nsis.artifactName=FedLearn-Desktop-Setup-${version}-cpu.exe    (or -cuda.exe)
```

### What Gets Included

From `electron-builder.yml`:

```yaml
files:
  - dist/**/*                   # Compiled webpack output
  - "!dist/**/*.map"            # Exclude source maps
  - node_modules/dockerode/**/* # webpack externals — must exist at runtime
  - node_modules/electron-store/**/*
  - node_modules/electron-log/**/*
  - node_modules/axios/**/*

asar: true                      # Package files into asar archive
asarUnpack:
  - node_modules/dockerode/**/* # Unpack dockerode (optional native addons)

npmRebuild: false               # See below
afterAllArtifactBuild: scripts/generate-checksums.js
```

`asar: true` packages most files into an archive for faster load times. `asarUnpack` for `dockerode` keeps it on the real filesystem.

**`npmRebuild: false`** (`8568ec2`) skips `@electron/rebuild` entirely. The app ships **no first-party native module**; the only native addons in the tree are `cpu-features` and ssh2's `sshcrypto`, both *optional transitive* deps of `dockerode → docker-modem → ssh2`. `DockerService` only ever uses `new Docker({ socketPath })` — never `ssh://` — so they are never exercised at runtime. Rebuilding `cpu-features` against Electron's V8 fails on macOS and Linux (nan/V8 incompatibility under clang/gcc; MSVC happens to compile it), which broke the mac-arm64, linux-x64 and linux-arm64 release jobs at the packaging step. The config carries the standing note: if a real first-party native module is ever added, remove this and rebuild selectively instead.

**`afterAllArtifactBuild`** runs `scripts/generate-checksums.js` (DE-12, `53e4c9b`), which writes `release/SHA256SUMS.txt` in the canonical `<hex>  <name>` format covering every installer artifact — plus **one rolled-up deterministic digest of the embedded native client bundle**. The bundle is a one-dir tree of hundreds of files, so it is hashed as a single digest over `(relpath + file-hash)` in sorted order rather than hundreds of lines; any added, removed or changed file flips it. `*.yml` update metadata and any pre-existing `SHA256SUMS.txt` are excluded; blockmaps are kept.

### Dependency Audit Posture

Measured on **2026-08-13**, `npm audit` reports **31 vulnerabilities (2 low, 8 moderate, 19 high, 2 critical)** for this unit — so the older "clean at `--audit-level=high`" claim no longer holds, and **CI does not gate on it**. Most entries are build-time only and have non-breaking fixes available; the `dockerode` → `uuid` chain is the one whose only fix is a breaking major. The full breakdown, with the honest caveat that any `npm audit` number drifts as the advisory DB moves, is in [Security Model → Dependency Vulnerability Posture](./02-security-model.md#dependency-vulnerability-posture).

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

# Linux x64 / arm64 (run on the matching Linux host)
client-docker/packaging/build-linux.sh
```

These scripts run PyInstaller against `client-docker/packaging/fedlearn-client.spec` and output to `client-docker/packaging/dist/fedlearn-client/`. PyInstaller does not cross-compile, so each OS/arch target needs its own native host.

Only **one** variant can exist in `dist/fedlearn-client/` at a time — the `extraResources` entry points at a single fixed directory, so building the Windows CUDA bundle overwrites the CPU one. Pick the variant you are shipping before invoking `electron-builder`.

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
2. node scripts/check-native-bundle.js  ← preflight (run automatically by package:*)
3. npm run build      ← webpack compile
4. electron-builder   ← package into DMG/NSIS/AppImage/deb
```

If you skip step 1, the preflight in step 2 fails loudly and names the build script for your platform. (Invoking `electron-builder` directly, bypassing the npm scripts, skips the preflight and will happily produce a broken package.)

---

## Platform-Specific Details

### macOS

```yaml
mac:
  category: public.app-category.developer-tools
  target:
    - target: dmg
      arch: [arm64]
    - target: zip
      arch: [arm64]
  icon: build/icon.icns
  darkModeSupport: true
  identity: null                 # Ad-hoc signed (no Apple Developer ID)
  hardenedRuntime: false
  gatekeeperAssess: false

dmg:
  artifactName: "${productName}-${version}-${arch}.dmg"
```

**arm64 only — the Intel x64 target was deliberately dropped** (`6ba588f`). The embedded PyInstaller client bundle is produced per platform+arch (`build-mac.sh` targets the host arch), and the whole platform is oriented at arm64 (Apple Silicon dev machines plus Jetson arm64 edge clients). An x64 mac artifact would be built and shipped without ever being exercised on real Intel hardware, so it is not published. The release workflow adds a second reason: the `macos-13` runner queue is unreliable as GitHub phases it out, and arm64 PyInstaller cannot cross-compile to x64. Intel Mac users can run the arm64 build under Rosetta. Re-adding `x64` means editing **both** `electron-builder.yml` and `src/shared/bundleVariants.ts` — and validating it end to end first.

**Ad-hoc signing (`identity: null`):** Without an Apple Developer ID certificate the app is ad-hoc signed. macOS Gatekeeper blocks the first launch with an "unidentified developer" or "damaged" warning. Users must right-click → Open, or run:
```bash
xattr -cr /Applications/FedLearn\ Desktop.app
```

**Why `hardenedRuntime: false`?** Hardened Runtime has to be paired with a Developer ID *and* notarization; enabling it without them makes macOS reject the launch outright ("damaged / check with developer") even after a successful install. The config says exactly this.

Note `titleBarStyle: 'hiddenInset'` is set in `BrowserWindow`'s options (`main.ts`), not in `electron-builder.yml`.

### Windows

```yaml
win:
  target:
    - target: nsis
      arch: [x64]
  icon: build/icon.ico

nsis:
  oneClick: false                          # Show installer wizard
  allowToChangeInstallationDirectory: true # Custom install path
  perMachine: false                        # Per-user install (no admin required)
  artifactName: "${productName}-Setup-${version}.exe"
```

The `package:win:cpu` / `package:win:cuda` scripts override `nsis.artifactName` on the command line so the two variants produce distinct filenames.

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

**arm64 on Linux:** Targets NVIDIA Jetson devices running an L4T Ubuntu. Jetson training itself uses the **Docker path**, not the native bundle, so the Linux arm64 native client is only relevant if the user selects a CPU profile on that device.

The `linux` block also sets `category: Development`, `icon: build/icons` (the PNG set), and a `.desktop` entry with `StartupNotify=true` / `Terminal=false`.

### Publish target

```yaml
publish:
  provider: github
  owner: anurag2796
  repo: FedLearn-Platform
```

This is what `electron-updater` resolves releases from at runtime. The release workflow itself packages with `--publish never` and uploads separately.

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

After packaging, output is written to `release/`. The shippable set is enumerated once, in `src/shared/bundleVariants.ts`, and mirrored by `electron-builder.yml` + the packaging scripts:

| Variant | `nativeBundleId` | Artifact |
|---|---|---|
| macOS Apple Silicon | `mac-arm64` | `FedLearn Desktop-${version}-arm64.dmg` (+ `-mac.zip` for the update channel) |
| Linux x64 | `linux-x64` | `FedLearn Desktop-${version}-x64.AppImage` (+ `.deb`) |
| Linux arm64 | `linux-arm64` | `FedLearn Desktop-${version}-arm64.AppImage` (+ `.deb`) |
| Windows x64 CPU | `win-x64-cpu` | `FedLearn-Desktop-Setup-${version}-cpu.exe` |
| Windows x64 CUDA | `win-x64-cuda` | `FedLearn-Desktop-Setup-${version}-cuda.exe` |

Plus `SHA256SUMS.txt`, `latest-mac.yml` / `latest*.yml` update metadata, and `.blockmap` files.

Version comes from `package.json` — currently:
```json
{ "version": "3.2.0-beta" }
```

Adding or removing a build target means editing `BUNDLE_VARIANTS` in `src/shared/bundleVariants.ts` **and** the corresponding electron-builder target; `src/__tests__/bundleVariants.test.ts` guards the manifest.

---

## CI & Release Automation

Two workflows touch this unit.

### `.github/workflows/ci.yml` → the `desktop` job

Path-filtered on `fedlearn-desktop/**`, runs on **Node 24**:

```yaml
- name: No skipped/focused jest tests
  run: bash scripts/check_no_skipped_tests.sh fedlearn-desktop   # TE-10
- name: Install + lint + test
  working-directory: fedlearn-desktop
  run: |
    npm ci
    npm run lint
    npm run test:coverage    # TE-11: coverageThreshold enforced from jest.config.js
```

> **Asymmetry worth knowing: there is no standalone `tsc --noEmit` gate here.** `frontend/` and `mobile_client/` both run one; the desktop job does not. Types are only checked incidentally, by `ts-loader` at build time and by `ts-jest` on test-reachable sources — and CI never runs `npm run build` for this unit. Run `npx tsc --noEmit` locally before pushing a type-level change.

`scripts/check_design_tokens.sh` runs in a separate, **unfiltered** job — a hand-edit of the generated `src/renderer/tokens.css`, or a `design/tokens.json` change without regenerating, fails the build regardless of which unit changed.

Nothing in CI runs `npm audit`.

### `.github/workflows/release-desktop.yml`

Tag-triggered on **`desktop-v*`** (a per-unit prefix, so frontend/mobile tags don't fire it), plus `workflow_dispatch`. One matrix row per shippable variant, each on a native runner because PyInstaller cannot cross-compile:

| Row | Runner | Native step | electron-builder |
|---|---|---|---|
| `mac-arm64` | `macos-latest` | `./build-mac.sh` | `--mac --arm64 --publish never` |
| `win-x64-cpu` | `windows-latest` | `.\build-win-cpu.ps1` | `--win --x64` + the `-cpu.exe` artifact name |
| `win-x64-cuda` | `windows-latest` | `.\build-win-cuda.ps1` | `--win --x64` + the `-cuda.exe` artifact name |
| `linux-x64` | `ubuntu-latest` | `./build-linux.sh` | `--linux AppImage:x64 deb:x64` |
| `linux-arm64` | `ubuntu-24.04-arm` | `./build-linux.sh` | `--linux AppImage:arm64 deb:arm64` |

Two details from the config comments: the `target:arch` pairs on Linux *replace* the config's target list, which is what stops `@electron/rebuild` from also running for the other arch on the wrong host (aarch64 gcc has no `-m64`); and `mac-x64` was dropped for the runner/cross-compile reasons noted above.

> **Version pins differ between the two workflows.** `ci.yml`'s desktop job uses Node 24 (matching `.nvmrc` and `.tool-versions`), while `release-desktop.yml` sets `NODE_VERSION: '22'` and `PYTHON_VERSION: '3.11'`. So the app is *tested* on Node 24 and *released* from Node 22, and the release Python (3.11) is not the repo's pinned 3.12.9. That is a real inconsistency in the workflow files, not a documented policy.

Jobs are capped at `timeout-minutes: 45` so a queued or hung runner cannot stall a release.

---

## Environment Variables Reference

| Variable | Used By | Default | Description |
|---|---|---|---|
| `NODE_ENV` | `main.ts`, `updater.ts` | `development` | `production` disables dev paths; `development` also sets `autoUpdater.forceDevUpdateConfig` |
| `FEDLEARN_API_URL` | `auth.service.ts` | `http://localhost:8081/api` | Backend API base URL, used only when no URL is saved in the store |
| `FEDLEARN_API_ORIGINS` | `main.ts` | *(empty)* | Comma-separated extra CSP `connect-src` origins. **Dev mode only** — the packaged build registers no response-header handler, and its `<meta>` CSP is fixed. See [02 → Content Security Policy](./02-security-model.md#content-security-policy). |
| `FEDLEARN_CLIENT_IMAGE` | `docker.service.ts` | `fedlearn-client:latest` | Overrides the training container image |
| `FEDLEARN_CONNECTION_TOKEN` | set *by* the app on the spawned client | — | The FL connection token; injected into the child env / container env, never a CLI flag |
| `PYTHONUNBUFFERED` | `docker.service.ts` | Set to `1` by the app | Forces unbuffered Python stdout (real-time logs) |
| `PYTHONPATH` | `docker.service.ts` | Prepended by the app | Adds `framework/src` ahead of any existing value (dev mode) |
| `CSC_LINK` | electron-builder | — | Path to the code-signing certificate |
| `CSC_KEY_PASSWORD` | electron-builder | — | Code-signing certificate password |
| `APPLE_ID` | electron-builder | — | Apple ID for notarization |
| `APPLE_APP_SPECIFIC_PASSWORD` | electron-builder | — | App-specific password for notarization |
| `APPLE_TEAM_ID` | electron-builder | — | Apple Team ID for notarization |

---

*Next: [07 — Hardware Profiles & Training Execution](./07-hardware-profiles.md)*  
*Previous: [05 — Renderer & Components](./05-renderer-components.md)*
