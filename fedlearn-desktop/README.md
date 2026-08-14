# FedLearn Desktop

> Electron client for federated-learning training. Ships with a native
> PyInstaller-bundled Python client on Mac, Windows and Linux — no Docker, no
> Python, no repo checkout needed on end-user machines. Jetson clients still
> go through Docker because NVIDIA's L4T torch wheel is firmware-pinned.

---

## For end users (install-and-run)

1. Download the installer for your platform:
   - **macOS (Apple Silicon)** — `FedLearn Desktop-X.Y.Z-arm64.dmg`
   - **Windows x64 with NVIDIA GPU** — `FedLearn-Desktop-Setup-X.Y.Z-cuda.exe`
   - **Windows x64 without GPU** — `FedLearn-Desktop-Setup-X.Y.Z-cpu.exe`
   - **Linux / Jetson AGX Orin** — `.AppImage` or `.deb` (x64 and arm64). The Jetson
     path additionally needs Docker + the NVIDIA Container Toolkit (pre-installed with JetPack)
2. Install and launch. The app probes your hardware (Apple Silicon → `mps`, visible
   NVIDIA GPU → `discrete`, otherwise `cpu`) and pre-selects a profile; you can
   always override it on the hardware cards.
3. Sign in. The app lists the models you're allowed to train (`GET /api/client/projects`)
   — pick one and click **Start Training**. It joins the project and fetches the gRPC
   address, partition id, aggregation strategy, training arm, and connection token for
   you; there's no manual server address / project ID / partition entry.

There is no Intel (x64) macOS build. It was dropped deliberately rather than shipped
untested — Apple Silicon users are the target, and Intel Macs can run the arm64 build
under Rosetta if needed. `src/shared/bundleVariants.ts` is the authoritative list of
shippable variants.

---

## For developers

### Prerequisites

| Requirement | Notes |
|---|---|
| **Node.js 24** | For building and running the Electron app (repo `.nvmrc` = `24`; the PR gate in `ci.yml` runs on 24) |
| **Python 3.10+** | Only needed for dev mode (the packaged installer bundles its own). Repo pins 3.12.9 |
| **Docker Engine** | Only needed for the Jetson profile |
| **FedLearn Backend** | Spring Boot API running at `http://localhost:8081` |

### Dev mode (system Python + repo checkout)

`DockerService.resolveNativeInvocation()` falls back to spawning
`python3 -u fl-runtime/client.py` (with `framework/src` on `PYTHONPATH`) when
`app.isPackaged === false`, so you don't need to rebuild the PyInstaller bundle
after every Python edit. Make sure you have the framework installed (from the repo root):

```bash
pip install -e framework
pip install -r client-docker/packaging/requirements-client.txt
```

### Development setup

```bash
# 1. Install dependencies
cd fedlearn-desktop
npm install

# 2. Start the dev environment (preload + main + renderer, then Electron — all concurrent)
npm run dev
```

`npm run dev` already launches Electron for you (it waits ~3s for the bundles, then
runs `dev:electron`). Use `npm run dev:electron` on its own only when you want to
re-attach Electron to bundles that are already built.

The renderer dev server runs on `http://localhost:9000` with HMR. The Electron main
process loads from this URL in development mode.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `FEDLEARN_API_URL` | `http://localhost:8081/api` | Backend API base URL (a URL saved in-app takes precedence) |
| `FEDLEARN_API_ORIGINS` | — | Comma-separated extra origins appended to the `connect-src` of the **dev** session-header CSP. `main.ts` reads it inside the `isDev` branch only, so it has **no effect on a packaged build** — that carries the static `<meta>` CSP baked in at build time from `webpack.csp.js`, which no runtime env var can extend |
| `FEDLEARN_CLIENT_IMAGE` | `fedlearn-client:latest` | Overrides the client image used by the Docker (Jetson) path |
| `NODE_ENV` | — | Set to `production` for packaged builds |

### Building a distributable installer

Distributables bundle the native client as a PyInstaller-produced binary
inside `resources/`. Build the client first, then package Electron:

```bash
# 1. Build the native client (pick the one matching your installer target)
cd client-docker/packaging
./build-mac.sh             # Mac arm64 (MPS)
./build-linux.sh           # Linux
# or on Windows:
.\build-win-cpu.ps1        # Windows CPU
.\build-win-cuda.ps1       # Windows CUDA

# 2. Package the Electron installer
cd ../../fedlearn-desktop
npm run package:mac        # Mac
npm run package:linux      # Linux
npm run package:win:cpu    # Windows CPU
npm run package:win:cuda   # Windows CUDA
```

Only one Windows native-client variant can exist on disk at a time — build the one
you're shipping. `scripts/check-native-bundle.js` runs before every `package:*` and
fails loudly if the bundle for the target is missing, so you can't ship an installer
whose "Start" button dead-ends.

See [`client-docker/packaging/README.md`](../client-docker/packaging/README.md) for bundle size,
troubleshooting, and the rationale for the PyInstaller approach.

### Packaging targets that actually exist

```bash
npm run build             # webpack: main + preload + renderer (production)
npm run package           # current platform
npm run package:mac       # macOS (.dmg + .zip, arm64 only)
npm run package:linux     # Linux (AppImage + .deb, x64 + arm64)
npm run package:win:cpu   # Windows CPU (.exe via NSIS, x64)
npm run package:win:cuda  # Windows CUDA (.exe via NSIS, x64)
```

There is no bare `package:win` script — Windows ships as two variants (`cpu` /
`cuda`) that differ only in the bundled native client, distinguished by an NSIS
`artifactName` override on the npm script.

Tagged releases are cut by `.github/workflows/release-desktop.yml` on a `desktop-v*`
tag; it builds the PyInstaller bundle on a native runner per target (PyInstaller does
not cross-compile) before running electron-builder. Note that the release workflow
currently pins Node 22 while the PR gate uses 24.

### What the production build does

1. **`check-native-bundle.js`** verifies the PyInstaller client bundle is present for the target before anything else runs
2. **Webpack** compiles Main, Preload, and Renderer bundles in production mode
3. **TerserPlugin** with `drop_console: true` strips all `console.*` calls from Renderer + Preload bundles (prevents JWT/path leakage to DevTools)
4. **electron-builder** packages the app with `asar` (dockerode is `asarUnpack`ed)
5. **`generate-checksums.js`** (`afterAllArtifactBuild`) emits `release/SHA256SUMS.txt` covering every installer plus a digest of the embedded native client

---

## Process model

Three processes, one bridge:

| Process | Trust | Responsibility |
|---|---|---|
| **Main** (`src/main/`) | Full Node + OS access | Owns the JWT, the Docker socket, the native client child process, HTTP to the backend, and every `ipcMain.handle` |
| **Preload** (`src/preload/preload.ts`) | Sandboxed, no Node globals leaked | The **only** bridge. Exposes `window.fedLearnAPI` via `contextBridge` and validates every input before it reaches `ipcRenderer.invoke` |
| **Renderer** (`src/renderer/`) | Sandboxed, untrusted | React 18 UI. Talks only to `window.fedLearnAPI`; never sees a token, a socket, or a filesystem path it didn't get back from a dialog |

`src/shared/` holds the dependency-free modules all three (and jest, and the plain-node
preflight script) can import — the shippable-variant manifest, device-capability types,
eligibility evaluation, and URL transport-security helpers.

---

## Security architecture

### Non-negotiable settings

| Setting | Value | Enforced in |
|---|---|---|
| `nodeIntegration` | `false` | `main.ts` |
| `contextIsolation` | `true` | `main.ts` |
| `sandbox` | `true` | `main.ts` |
| `remote` module | **Not used** | Project-wide |
| CSP | Session headers in dev; baked `<meta>` tag in packaged builds | `main.ts` + `webpack.csp.js` |

The packaged build carries its CSP as a `<meta>` tag rather than a response header,
because Chromium's reading of `'self'` under a `file://` origin is inconsistent. The dev
header allows `'unsafe-eval'` (webpack's `eval` devtool needs it); the packaged meta CSP
does not. Fonts are bundled locally (`src/renderer/fonts.css`), so no remote font host
appears in either policy.

### IPC contract

1. **Renderer** calls `window.fedLearnAPI.*` — the only surface exposed by `contextBridge`.
2. **Preload** validates against explicit allowlists and patterns before forwarding:
   hardware profile ∈ `{discrete, jetson, cpu, mps}`, project id `/^[a-zA-Z0-9_-]{1,128}$/`,
   partition id `/^[0-9]{1,10}$/`, server address `/^[a-zA-Z0-9._:/-]{1,256}$/`,
   bounded string/vector/image sizes. Rejections are logged and never reach Main.
3. **Main** re-validates the same inputs (defense-in-depth) in `ipc.handlers.ts` using the
   shared, unit-tested predicates in `main/validators.ts` — a compromised renderer that
   bypasses preload still hits a closed door.
4. **Dataset paths carry consent, not just validity.** A bind-mounted host directory must
   be one the user actually picked through the native `dialog:open-directory` — `main/dataset-consent.ts`
   records those, and `docker:start-training` refuses any other path. Proving a path exists
   is not the same as proving the user chose it.

### JWT containment

- The JWT is extracted in **Main** from the backend login response (`Set-Cookie: jwtToken=...`).
- It is encrypted with Electron's `safeStorage` (OS keychain) and persisted through `electron-store`.
- If `safeStorage.isEncryptionAvailable()` is false (typically headless Linux with no keyring),
  the token is held in main-process memory for the session only. It is **never** written as
  reversible base64 — that would be obfuscation posing as encryption.
- The **Renderer never receives the token** — IPC replies are shapes like `{ success: boolean }`.
  `AuthService.getAuthHeader()` is a Main-process-only call.
- Saved login credentials ("Save password") follow the same rule: `safeStorage`-encrypted or
  refused outright.

### Console stripping

- TerserPlugin `drop_console: true` on the Renderer + Preload bundles.
- Prevents accidental JWT, path, or config leakage to DevTools in production
  (DevTools itself is only enabled in dev builds).
- Main keeps its `electron-log` transport for operational logging.

---

## Hardware profiles

There are **four** profiles (`src/main/docker.service.ts` → `HardwareProfile`), and the
profile is the sole dispatcher in `startTraining()`:

| Profile | Path | Notes |
|---|---|---|
| `mps` | Native bundle | Apple Silicon. Auto-selected on `darwin`/`arm64`. Explicitly rejected by the Docker path. |
| `discrete` | Native bundle | Workstation NVIDIA GPU (auto-selected when `nvidia-smi` answers, off Linux). Runs whichever native client the installer shipped — on Windows that is the `cuda` or `cpu` variant chosen at package time. |
| `cpu` | Native bundle | No accelerator. Also the fallback on Linux, where a CUDA bundle isn't shipped yet. |
| `jetson` | **Docker** | The only profile that builds a container. |

**`discrete` does not go through Docker and passes no `--gpus` flag.** Every non-`jetson`
profile is routed to the bundled native client, and `startDockerTraining()` throws if it is
ever reached with one — a routing regression fails loudly instead of quietly building a
container for a native profile.

### Jetson and `--runtime nvidia` — corrected

This README used to say `--runtime nvidia` is *prohibited* on Jetson because it hangs while
searching the device tree for PCIe GPU metadata. **That guidance does not hold on JetPack 6.**
Measured on an AGX Orin (L4T R36.5.0 / JetPack 6.2, `nvidia-container-toolkit 1.19.0-1`, with
`nvidia` listed among `docker info`'s runtimes):

| Approach | Result on that Orin |
|---|---|
| `docker run --runtime nvidia` | **Works.** `torch.cuda.is_available()` → `True`, device reported as `Orin`. No hang. |
| Hand-rolled `/dev/nvhost-*` device mounts, **no** `--runtime nvidia` | **Fails.** `cuInit → 801` (`CUDA_ERROR_NOT_SUPPORTED`), then a segfault — the in-container `libcuda.so.1` is a stub. |

Two related details:

- **`/dev/nvhost-ctrl` does not exist on L4T R36.5.** Passing it makes Docker hard-error with
  `no such file or directory`. That device-node set is JetPack-5-era.
- The **`nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`** base tag below is likewise JetPack-5-era,
  two major L4T generations behind an R36.5 box.

**Honest scope of the correction:** the original hang was plausibly real on the older
JetPack 5 / `nvidia-container-runtime` it was written against — that is an inference, not a
re-test, because no JetPack 5 hardware was available. What was actually measured is that the
ban does not hold on JetPack 6.

**Recommendation:** on JetPack 6+, try `--runtime nvidia` first and keep device mounts only as
a fallback; on older L4T, keep device mounts. Either way, **re-verify GPU access on the target
device's actual L4T release** rather than trusting either rule. Note that
`DockerService`'s Jetson branch still uses the device-mount list (including `/dev/nvhost-ctrl`)
and does not pass `--runtime nvidia` — so on an R36.5 device the desktop's Jetson flow needs
that list revisited before it will start a container.

### Building the Jetson client image

The Dockerfile `COPY`s both `framework/` and `fl-runtime/`, so **the build context is the
repo root** — run this from the repository root, not from `client-docker/`:

```bash
docker build -f client-docker/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:latest .
```

The container is configured by **environment variables, not CLI flags**:
`entrypoint.sh` exits 1 if `PROJECT_ID` / `SERVER_ADDRESS` / `PARTITION_ID` are unset, then
builds the `--project-id` / `--server-address` / `--partition-id` flags itself (plus
`--model-type` / `--strategy` / `--training-arm` from `MODEL_TYPE` / `STRATEGY` /
`TRAINING_ARM`) and forwards anything else you pass. The desktop's `buildContainerEnv()`
sets all of these, including `FEDLEARN_CONNECTION_TOKEN`, so the app never has to hand-build
a command line.

---

## Design system — Ledger

The renderer is styled from **Ledger**: navy structural ink on quiet paper surfaces,
light-first (canvas `#F6F3EE`, ink `#191A1C`, accent `#1C314D`), Hanken Grotesk +
JetBrains Mono, both self-hosted via `@fontsource` so the packaged CSP needs no remote
font host.

`design/tokens.json` at the repo root is the single source of truth across frontend,
desktop and mobile; `design/build-tokens.mjs` generates `src/renderer/tokens.css`. **That
file is generated — do not hand-edit it**, and don't hardcode a colour in a component.
CI runs `scripts/check_design_tokens.sh` on every PR and fails on any drift.

One literal has to be duplicated by hand: `BrowserWindow`'s `backgroundColor` in `main.ts`,
because the main process cannot read CSS variables. Keep it in sync on a palette change.

*History: Ledger (2026-07-17) replaced **Ember** (burnt orange on warm paper, Bricolage
Grotesque display), which replaced Instrument. If you find Ember values or Bricolage in a
comment here, it is two cycles stale.*

---

## Testing & CI gates

```bash
npm test                                          # jest
npm run test:coverage                             # jest --coverage (thresholds from jest.config.js)
npm run lint                                      # ESLint 9 flat config
bash ../scripts/check_no_skipped_tests.sh fedlearn-desktop
```

The `desktop` job in `.github/workflows/ci.yml` runs only when `fedlearn-desktop/**` changes,
and gates on exactly three things:

1. **`check_no_skipped_tests.sh`** — jest has no forbid-skip switch, so `.skip` / `.only` /
   `xit` / `fit` are statically rejected. A skipped test can't ride a green run.
2. **`npm run lint`** — ESLint 9, hard-gated.
3. **`npm run test:coverage`** — jest with `coverageThreshold` enforced from `jest.config.js`
   (a regression floor set just under the measured baseline, not an aspirational target).

**There is no standalone `tsc --noEmit` gate here** — unlike `frontend/` and `mobile_client/`,
which both run one. Types are only checked incidentally, by ts-jest compiling the sources the
`node`-environment suite actually reaches; the renderer `.tsx` components have no jsdom/RTL
harness in this unit and are excluded from coverage collection for the same reason. A plain
`npx tsc --noEmit` passes today, but nothing keeps it that way — run it yourself before
opening a PR.

---

## Project structure

```
fedlearn-desktop/
├── src/
│   ├── main/
│   │   ├── main.ts                     # BrowserWindow + CSP + app lifecycle + menu
│   │   ├── ipc.handlers.ts             # All ipcMain.handle registrations (re-validates every input)
│   │   ├── docker.service.ts           # Native-client spawn + dockerode (Jetson) orchestration
│   │   ├── auth.service.ts             # safeStorage-encrypted JWT + backend auth calls
│   │   ├── client-projects.service.ts  # "models I can train" + /connection flow
│   │   ├── inference.service.ts        # Local inference on a trained model
│   │   ├── inference-stream.service.ts # Streaming (token-by-token) inference output
│   │   ├── hardware.probe.ts           # Hardware auto-detection (nvidia-smi, platform/arch)
│   │   ├── deviceCapabilities.collector.ts
│   │   ├── dataset-consent.ts          # Only user-dialog-chosen paths may be bind-mounted
│   │   ├── http.ts                     # HTTP client + auth interceptor
│   │   ├── updater.ts                  # electron-updater wiring
│   │   └── validators.ts               # Shared, unit-tested input validation
│   ├── preload/
│   │   └── preload.ts                  # contextBridge `fedLearnAPI` + input validation
│   ├── renderer/
│   │   ├── App.tsx                     # Main application layout
│   │   ├── index.tsx                   # React 18 entry point
│   │   ├── tokens.css                  # GENERATED from design/tokens.json — do not edit
│   │   ├── fonts.css                   # Self-hosted @fontsource faces
│   │   ├── styles.css
│   │   └── components/
│   │       ├── TrainSection.tsx        # The guided Train flow: model picker, dataset, start/stop
│   │       ├── HardwareSelector.tsx    # Hardware profile card grid, embedded under TrainSection's Advanced
│   │       ├── SettingsSection.tsx     # Server URL + preferences (supersedes the old SettingsModal)
│   │       ├── AuthModal.tsx           # Login modal
│   │       ├── LogPanel.tsx            # Plain-text log viewer
│   │       ├── StatusBar.tsx           # Persistent bottom strip (backend/run status), outside the section outlet
│   │       ├── StatusIndicator.tsx     # Run status badge
│   │       ├── ModelPlayground.tsx     # Inference playground
│   │       ├── UpdateBanner.tsx        # Auto-update prompt
│   │       └── logView.ts / trainFlow.ts / runNotifications.ts   # Pure, unit-tested view logic
│   ├── shared/                         # Import-free modules usable from every process + jest
│   │   ├── bundleVariants.ts           # Authoritative shippable-variant manifest
│   │   ├── evaluateEligibility.ts      # Device-requirement matching
│   │   ├── deviceCapabilities.types.ts
│   │   └── urlSecurity.ts              # Plaintext-HTTP policy for user-supplied server URLs
│   ├── __tests__/                      # jest suites (node environment)
│   └── __mocks__/                      # electron / electron-log / electron-store / CSS stubs
├── scripts/
│   ├── check-native-bundle.js          # Pre-package guard: native client present?
│   └── generate-checksums.js           # SHA256SUMS.txt over the released artifacts
├── webpack.{main,renderer,preload,prod}.config.js
├── webpack.csp.js                      # Shared CSP definition (baked into packaged index.html)
├── electron-builder.yml                # Cross-platform packaging
├── eslint.config.mjs                   # ESLint 9 flat config (CI-gated)
├── jest.config.js
├── tsconfig.json / tsconfig.test.json
└── package.json
```

---

## Adjacent docs

- Wiki / deeper architecture: [`wikis/desktop/`](../wikis/desktop/)
- Native client packaging: [`client-docker/packaging/README.md`](../client-docker/packaging/README.md)

## License

See the root [LICENSE](../LICENSE) file.
