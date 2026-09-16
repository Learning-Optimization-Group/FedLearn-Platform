# FedLearn Desktop — Main Process Deep Dive

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Entry Point: `main.ts`](#entry-point-maints)
2. [IPC Handler Registry: `ipc.handlers.ts`](#ipc-handler-registry-ipchandlersts)
3. [Validators: `validators.ts`](#validators-validatorsts)
4. [Training Orchestration: `docker.service.ts`](#training-orchestration-dockerservicets)
5. [Authentication Service: `auth.service.ts`](#authentication-service-authservicets)
6. [Shared HTTP Client: `http.ts`](#shared-http-client-httpts)
7. [Backend-Facing Services](#backend-facing-services)
8. [Hardware Probe: `hardware.probe.ts`](#hardware-probe-hardwareprobets)
9. [Device Capabilities: `deviceCapabilities.collector.ts`](#device-capabilities-devicecapabilitiescollectorts)
10. [Auto-Updater: `updater.ts`](#auto-updater-updaterts)
11. [IPC Channel Reference](#ipc-channel-reference)

---

## Entry Point: `main.ts`

`main.ts` is the first file executed by Electron's Main Process. It sets up the entire application shell: window creation, Content Security Policy, IPC handler registration, and app lifecycle management.

### Window Creation

```typescript
// src/main/main.ts
function createWindow(): void {
  mainWindow = new BrowserWindow({
    // Shell layout budget: 64px rail + ~380px setup column + usable log pane
    // needs >= 1024 wide; drag strip + checklist + logs + status bar needs
    // >= 700 tall.
    width: 1360,
    height: 860,
    minWidth: 1024,
    minHeight: 700,
    title: 'FedLearn Desktop',
    backgroundColor: '#F6F3EE',   // Ledger light canvas — see note below
    titleBarStyle: 'hiddenInset', // macOS native traffic lights inset into app
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, '..', 'preload', 'preload.js'),
      devTools: isDev,
      webSecurity: true,
      allowRunningInsecureContent: false,
      experimentalFeatures: false,
    },
  });
}
```

**Notable choices:**
- `backgroundColor: '#F6F3EE'` — the Ledger light canvas, painted *before* the renderer loads so there is no flash while HTML/CSS is parsed. This literal mirrors `--canvas` in `design/tokens.json`; the main process cannot read CSS variables, so **it must be updated by hand on any palette swap** (the code says so in a comment). It was `#0a0a0f` under the older dark-first systems.
- `titleBarStyle: 'hiddenInset'` — macOS only. Hides the title bar chrome but keeps the traffic-light buttons in the window, positioned at the specified offsets. `App.tsx` renders its own drag strip and keys the inset off `navigator.userAgent`.
- `minWidth: 1024, minHeight: 700` — sized from the actual shell layout budget (icon rail + setup column + log pane), not a round number.

### Application Menu

`setApplicationMenu()` installs a template of **standard roles only** — `appMenu` (macOS), `editMenu`, a View submenu of zoom/fullscreen roles, and `windowMenu`. That restores the system copy/paste, zoom and window shortcuts without introducing any custom item.

The omission is deliberate: the section-switching shortcuts (`Cmd/Ctrl+1..3`) live in a renderer `keydown` listener instead, because a menu item that reached the renderer would require a new IPC channel — a new hole in the bridge for a keyboard shortcut.

### Dev vs. Production Loading

```typescript
if (isDev) {
  // Dev: Load from webpack-dev-server (supports HMR)
  mainWindow.loadURL('http://localhost:9000');
} else {
  // Production: Load from packaged asar bundle
  mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}
```

`isDev` is computed as:
```typescript
const isDev = process.env.NODE_ENV !== 'production' && !app.isPackaged;
```

The double check (`NODE_ENV` AND `!app.isPackaged`) prevents development-only code paths from running if the app is accidentally built without setting `NODE_ENV=production`.

### Logging Setup

`electron-log` is the sole logging mechanism. It replaces the built-in `console` in Main so all `console.log`, `console.error`, etc. calls go through the structured logger:

```typescript
log.transports.file.level = 'info';    // File log level
log.transports.console.level = 'debug'; // Console log level (more verbose in dev)
log.initialize();
Object.assign(console, log.functions); // Replace console with electron-log
```

Log files are written to the OS app data directory:
- **macOS:** `~/Library/Logs/FedLearn Desktop/main.log`
- **Windows:** `%USERPROFILE%\AppData\Roaming\FedLearn Desktop\logs\main.log`
- **Linux:** `~/.config/FedLearn Desktop/logs/main.log`

### Crash Reporter

```typescript
crashReporter.start({ uploadToServer: false });
```

Crash dumps are written to `app.getPath('crashDumps')` but **not uploaded** anywhere. This prevents telemetry from leaking sensitive environment information (environment variables, memory contents, etc.) to a remote server.

### App Lifecycle

```typescript
// Re-create window on macOS dock click (standard macOS behavior)
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Quit on all windows closed (except macOS — dock app stays running)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Disable GPU hardware acceleration
// GPU compute happens inside Docker/native process — Electron's UI doesn't need it
app.disableHardwareAcceleration();
```

### Draining Training on Quit

`before-quit` intercepts the **first** quit and best-effort drains any running training before exiting:

```typescript
let isDraining = false;
app.on('before-quit', (event) => {
  if (isDraining) return;              // drain in progress; let app.exit(0) proceed
  const docker = getDockerService();   // undefined only if IPC registration threw
  if (!docker) return;
  isDraining = true;
  event.preventDefault();

  let exited = false;
  const exit = () => { if (!exited) { exited = true; app.exit(0); } };
  const hardTimeout = setTimeout(() => { log.warn(/* … */); exit(); }, 15000);

  Promise.resolve(docker.stopTraining())
    .catch((err) => log.error('[Main] before-quit: stopTraining failed', err))
    .finally(() => { clearTimeout(hardTimeout); exit(); });
});
```

Why it exists: containers are created with `AutoRemove: false` and the native client is spawned non-detached, so quitting mid-run would otherwise orphan a Jetson container until the next run lazily cleaned it up.

Why **15 s** specifically: `stopDockerContainer()` does `container.stop({ t: 10 })` — up to ~10 s for a SIGTERM-ignoring container — and *then* `container.remove()`. An 8 s cap would force-exit between the two and orphan exactly the container it was meant to clean up. 15 s covers the slow-but-responsive path; only a genuinely wedged daemon (where cleanup is impossible anyway) hits the backstop, and the cap guarantees a hung daemon can never make the app unquittable.

On macOS, closing the window does **not** quit the app, so training simply keeps running under the still-live app — nothing is orphaned. On other platforms `window-all-closed` → `app.quit()` fires `before-quit`, so the drain covers that path too.

`getDockerService()` is exported from `ipc.handlers.ts` purely as the accessor for this handler.

---

## IPC Handler Registry: `ipc.handlers.ts`

All `ipcMain.handle` registrations are centralized in `registerIpcHandlers()`. This is called once from `main.ts` after the window is created.

```typescript
// src/main/ipc.handlers.ts
export function registerIpcHandlers(mainWindow: BrowserWindow): void {
  dockerService = new DockerService(mainWindow);
  authService = new AuthService(mainWindow);            // window needed to push auth:session-expired
  inferenceService = new InferenceService(authService);
  clientProjectService = new ClientProjectService(authService);
  const inferenceStreamService = new InferenceStreamService(authService, mainWindow);

  // Register each IPC channel...
}
```

Every backend-facing service is constructed with the `AuthService` instance rather than reaching for the token itself — that is what keeps `getAuthHeader()` a single, Main-only call site per request.

The function is wrapped in a try/catch in `main.ts` so that a registration failure (e.g., a crash in `DockerService` constructor) **does not prevent the renderer from loading**:

```typescript
// main.ts
try {
  registerIpcHandlers(mainWindow);
} catch (err) {
  log.error('[Main] registerIpcHandlers failed; renderer will still load', err);
}
```

This is important because if the window stays black (no renderer), the user has no way to recover — they can't even log out or change the server URL.

### Channel: `dialog:open-directory`

Opens a native OS directory picker dialog — and records the result as consented:

```typescript
ipcMain.handle('dialog:open-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    title: 'Select Dataset Directory'
  });
  if (result.canceled || result.filePaths.length === 0) {
    return { success: false, error: 'User canceled.' };
  }
  // Only a path the user physically picked here may later be bind-mounted.
  recordConsentedDatasetPath(result.filePaths[0]);
  return { success: true, path: result.filePaths[0] };
});
```

Using the **native dialog** for path selection (rather than a text input) is a security choice — paths chosen through the OS dialog are real, existing paths the user intentionally selected. The raw path is still sanitized by `sanitizeDatasetPath` in the `docker:start-training` handler, *and* checked against the consent allowlist, before use. See [02 → Dataset-Path Consent](./02-security-model.md#dataset-path-consent-dataset-consentts).

---

## Validators: `validators.ts`

The main-side validation predicates live in their own module, not inline in `ipc.handlers.ts`. The module's own header states the contract: *"Pure validation utilities extracted for testability. Do not import `electron` or `electron-log` here."* Its only non-type import from `docker.service` is `type { HardwareProfile }`, which is erased at compile time — so the module stays free of the Electron runtime and `src/__tests__/validators.test.ts` exercises the **shipped** predicates rather than a copy.

```typescript
export const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> =
  new Set(['discrete', 'jetson', 'cpu', 'mps']);

export const PROJECT_ID_PATTERN     = /^[a-zA-Z0-9_-]{1,128}$/;
export const PARTITION_ID_PATTERN   = /^[0-9]{1,10}$/;
export const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
export const MAX_DATASET_PATH_LEN   = 2048;
export const MAX_SERVER_URL_LEN     = 512;
```

| Export | Purpose |
|---|---|
| `sanitizeDatasetPath(raw)` | `'' \| absolutePath \| null`. Empty/whitespace normalizes to `''` ("use the built-in dataset"); otherwise resolve → no `..` → absolute → `statSync` is a directory. |
| `validateHardwareProfile(p)` | Type guard narrowing to the `HardwareProfile` union. |
| `validateProjectId(id)` / `validatePartitionId(id)` / `validateServerAddress(a)` | Anchored-regex type guards. |
| `validateStringInput(v, maxLength)` | Non-empty bounded string. |
| `evaluateServerUrl(raw, allowInsecureHttp)` | The whole `auth:set-server-url` decision — shape, protocol, DE-13 plaintext policy, and `/api` normalization. Returns `{ ok, url?, warning?, error?, code? }`. |

These mirror the constants in `preload.ts`. Keeping the two in sync matters; Main is the second line of defense, and a profile added to one but not the other is silently rejected at whichever layer was missed.

---

## Training Orchestration: `docker.service.ts`

`DockerService` is the core of the desktop application. It manages the full lifecycle of a training session, routing between two execution backends based on the hardware profile.

### Constructor

```typescript
export class DockerService {
  private docker: Docker;
  private mainWindow: BrowserWindow;
  private activeContainerId: string | null = null;
  private logStream: NodeJS.ReadableStream | null = null;
  private nativeProcess: ChildProcess | null = null;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;

    // Platform-appropriate Docker socket
    const socketPath = process.platform === 'win32'
      ? '//./pipe/docker_engine'
      : '/var/run/docker.sock';

    // dockerode opens the socket lazily — constructing the client does NOT connect.
    // The daemon is deliberately NOT probed here; see below.
    this.docker = new Docker({ socketPath });
    log.info(`[DockerService] Initialized with socket: ${socketPath} (daemon probed lazily — Jetson path only)`);
  }
}
```

**There is no startup ping.** An earlier build fired a non-blocking `docker.ping()` in the constructor and pushed a `docker:daemon-unavailable` event to the renderer on failure. That produced a spurious *"Docker is not running: connect ENOENT \\\\.\\pipe\\docker_engine"* banner for the overwhelming majority of users — Windows and macOS on CPU/CUDA/MPS, who run the bundled native client and never touch Docker at all. `4d7d3a4` removed the eager probe, the banner, and the push channel.

Docker is now probed exactly where it is needed: at the top of `startDockerTraining()`, which is only reachable from the Jetson profile, and which surfaces an actionable, platform-specific error at the moment it matters:

```typescript
try {
  await this.docker.ping();
} catch (err) {
  const hint = process.platform === 'win32'
    ? 'Start Docker Desktop and wait for the whale icon to say "Engine running."'
    : 'Start the Docker daemon.';
  const full = `Docker daemon unreachable: ${message}. ${hint}`;
  this.sendLog(`[System] ${full}\n`);
  throw new Error(full);
}
```

### Execution Routing

```typescript
async startTraining(config: TrainingConfig): Promise<void> {
  if (config.hardwareProfile === 'jetson') {
    await this.startDockerTraining(config);
    return;
  }
  await this.startNativeProcess(config);
}
```

Simple: `jetson` → Docker, everything else → native binary. `hardwareProfile` is the **sole** dispatcher.

### Native Process Resolution

The native client can be in two locations depending on runtime context:

```typescript
private resolveNativeInvocation() {
  const binaryName = process.platform === 'win32' ? 'fedlearn-client.exe' : 'fedlearn-client';

  if (app.isPackaged) {
    // PRODUCTION: Use the PyInstaller bundle shipped in extraResources
    const bundleDir = path.join(process.resourcesPath, NATIVE_BUNDLE_DIR);  // 'fedlearn-client'
    const binary = path.join(bundleDir, binaryName);

    if (!fs.existsSync(binary)) {
      log.error(`[Native] Packaged bundle missing at ${binary}`);
      return null;  // Caller surfaces a clear error — never a silent dev fallback
    }

    return {
      command: binary,
      baseArgs: [],
      cwd: bundleDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    };
  }

  // DEVELOPMENT: system python3 + the repo's fl-runtime client
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const clientScript = path.join(repoRoot, 'fl-runtime', 'client.py');
  const frameworkSrc = path.join(repoRoot, 'framework', 'src');

  if (!fs.existsSync(clientScript)) {
    log.error(`[Native] Dev-mode script missing at ${clientScript}`);
    return null;
  }

  const pythonPathSep = process.platform === 'win32' ? ';' : ':';
  const existingPythonPath = process.env.PYTHONPATH || '';

  return {
    command: process.platform === 'win32' ? 'python' : 'python3',
    baseArgs: ['-u', clientScript],  // -u: unbuffered output (critical for real-time logs)
    cwd: path.dirname(clientScript),
    env: {
      ...process.env,
      // PREPEND rather than replace — never clobber a developer's own PYTHONPATH
      PYTHONPATH: existingPythonPath ? `${frameworkSrc}${pythonPathSep}${existingPythonPath}` : frameworkSrc,
      PYTHONUNBUFFERED: '1',
    },
  };
}
```

> The dev-mode script is **`fl-runtime/client.py`**. `client-docker/scripts/client.py` no longer exists — DA-5 (`d2cc757`) consolidated onto `fl-runtime/` as the single client source of truth, consumed by the Docker image, the PyInstaller bundle, and this dev fallback alike.

### Starting a Native Process

```typescript
private async startNativeProcess(config: TrainingConfig): Promise<void> {
  if (this.nativeProcess) {
    // Fully DRAIN the previous client before respawning — see stopTraining below.
    await this.stopTraining();
  }

  const invocation = this.resolveNativeInvocation();
  if (!invocation) {
    const msg = app.isPackaged
      ? `Native training bundle not found at <resources>/${NATIVE_BUNDLE_DIR}. Reinstall the app or rebuild with the client bundle.`
      : 'Dev-mode client.py not found. Run the app from the repo root.';
    this.sendLog(`[System] ${msg}\n`);
    throw new Error(msg);
  }

  const args = [
    ...invocation.baseArgs,
    '--project-id', config.projectId,
    '--server-address', config.serverAddress,
    '--partition-id', config.partitionId,
  ];

  if (config.modelType)   args.push('--model-type', config.modelType);
  if (config.trainingArm) args.push('--training-arm', config.trainingArm);
  if (config.strategy)    args.push('--strategy', config.strategy);

  const datasetPath = config.datasetPath?.trim();
  if (datasetPath)        args.push('--dataset-path', datasetPath);

  const child = spawn(invocation.command, args, {
    env: withConnectionTokenEnv(invocation.env, config),  // FEDLEARN_CONNECTION_TOKEN
    cwd: invocation.cwd,
  });

  this.nativeProcess = child;

  child.stdout?.on('data', (data: Buffer) => this.sendLog(data.toString('utf-8')));
  child.stderr?.on('data', (data: Buffer) => this.sendLog(`[stderr] ${data.toString('utf-8')}`));
  child.on('error', (err) => { /* log + sendLog, clear nativeProcess */ });
  child.on('exit',  (code, signal) => { /* log + sendLog, clear nativeProcess */ });
}
```

Each optional flag is **omitted entirely** when absent rather than passed empty, so the client's own defaults apply (`--strategy` absent ⇒ FedAvg; `--training-arm` absent ⇒ `FULL`).

> **`--dataset-path` currently has no receiver.** `fl-runtime/client.py` — the dev-mode target *and* the entry point of the PyInstaller bundle (`client-docker/packaging/fedlearn-client.spec` sets `CLIENT_ENTRY` to it) — declares no `--dataset-path` argument and parses with strict `parse_args()`, so the flag is an *unrecognised argument* and the client exits 2 before training starts. DE-2 (`d8f1b60`) added the argument to `client-docker/scripts/client.py`; DA-5 (`d2cc757`) deleted that fork and consolidated onto `fl-runtime/client.py`, which never carried it. Choosing a dataset folder on a native profile therefore fails today; leaving it blank (or ticking "skip") omits the flag and works. The Docker path is unaffected — the directory travels as the `${datasetPath}:/data` bind and `entrypoint.sh` never turns it into a flag.

`--use-llm` is no longer passed by the desktop (it survives in `fl-runtime/client.py` only as a deprecated alias). The client derives `USE_LLM` from `--model-type TRANSFORMER` itself, and `modelType` is now a recipe key coming from the backend's connection payload rather than a hardcoded dropdown value.

The FL connection token travels as an **environment variable**, not an argument, on both paths — the framework reads `FEDLEARN_CONNECTION_TOKEN` straight from `os.environ` (`fedlearn/security/client_interceptor.maybe_wrap_channel`). `withConnectionTokenEnv()` returns the base env unchanged when no token is set, so a gate-off server still accepts the legacy no-token flow.

**Why `PYTHONUNBUFFERED=1`?** Python buffers its output when stdout is not a TTY (which it isn't when spawned by Node). Without unbuffered mode, log lines only appear once the buffer fills (typically 8KB), making real-time logging useless.

### Stopping Training — Await the Drain

```typescript
async stopTraining(): Promise<void> {
  if (this.nativeProcess) {
    const proc = this.nativeProcess;
    // exitCode is set on normal exit; signalCode when killed by a signal.
    // proc.killed is NOT an "exited" signal — it is true the instant .kill()
    // DELIVERS a signal — so it must not be used to detect liveness.
    if (proc.exitCode !== null || proc.signalCode !== null) {
      this.nativeProcess = null;
      return;                      // nothing to wait for; avoids a spurious 5s grace on quit
    }
    proc.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        // Gate escalation on EXIT status, not proc.killed (already true from the
        // SIGTERM above) — otherwise SIGKILL would never reach a SIGTERM-ignoring child.
        if (proc.exitCode === null && proc.signalCode === null) {
          try { proc.kill('SIGKILL'); } catch { /* already gone */ }
        }
        resolve();
      }, 5000);
      proc.once('exit', () => { clearTimeout(timer); resolve(); });
    });
    this.nativeProcess = null;     // cleared only AFTER the drain completes
    return;
  }

  if (this.activeContainerId) { await this.stopDockerContainer(); return; }
  log.warn('[DockerService] No active training process to stop');
}
```

Three details here are load-bearing, and all three were bugs at some point (`9a74e17`):

1. **It awaits.** A fire-and-forget kill let a respawn connect to the FL server on the same partition while the old client was still alive — a double-client race on one partition. `startNativeProcess` awaiting `stopTraining` guarantees at most one live native process per partition across a respawn.
2. **`this.nativeProcess` stays set during the drain.** Clearing it early would let a concurrent `docker:get-status` poll observe `idle`, re-enable the Start button, and allow that second spawn.
3. **Liveness is `exitCode`/`signalCode`, never `proc.killed`.** `killed` is true the moment a signal is delivered, so gating on it would both skip the SIGKILL escalation and impose a pointless 5 s wait on an already-dead child at quit time.

### Docker Container Lifecycle

#### Creating and Starting

```typescript
private async startDockerTraining(config: TrainingConfig): Promise<void> {
  // 1. Verify Docker is reachable
  await this.docker.ping();

  // 2. Clean up any leftover container from a previous session
  await this.cleanupExistingContainer();

  // 3. Build HostConfig based on hardware profile
  const hostConfig: Docker.HostConfig = {
    AutoRemove: false,
    Binds: [`${config.datasetPath}:/data`],  // Dataset bind mount
  };

  switch (config.hardwareProfile) {
    case 'jetson':
      hostConfig.Devices = JETSON_DEVICE_MOUNTS;
      break;
    case 'mps':
      throw new Error('MPS profile cannot run under Docker');
    default:
      // Jetson is the ONLY profile that uses the Docker path — startTraining()
      // routes discrete/cpu/mps to the bundled native client. Reaching here with
      // a non-jetson profile means a routing regression, so fail loudly.
      throw new Error(
        `Profile '${config.hardwareProfile}' does not use the Docker path — only 'jetson' runs under Docker`,
      );
  }

  // 4. Pass config as environment variables (buildContainerEnv)
  const env = buildContainerEnv(config);

  // 5. Create + start container
  const container = await this.docker.createContainer({
    Image: DOCKER_IMAGE,           // FEDLEARN_CLIENT_IMAGE ?? 'fedlearn-client:latest'
    name: CONTAINER_NAME,          // 'fedlearn-training-client'
    Env: env,
    HostConfig: hostConfig,
    Tty: false,
    AttachStdout: true,
    AttachStderr: true,
  });

  this.activeContainerId = container.id;
  await container.start();
  this.attachLogStream(container);
}
```

> **The `discrete` case is gone from this switch.** `DeviceRequests: [{ Count: -1, Capabilities: [['gpu']] }]` (the `--gpus all` equivalent) is no longer here, because `discrete` never reaches the Docker path — it runs the bundled native client with CUDA torch. `2b02173` made the profile routing truthful in both the switch and the profile-card copy; a `discrete` value arriving here now throws rather than quietly building a container for a native profile.

**Why `AutoRemove: false`?** If set to `true`, Docker removes the container immediately on exit — before the log stream has fully drained. With `false`, the container stays until explicitly removed in `stopDockerContainer()`, giving the log stream time to deliver the final bytes.

#### Container Environment (`buildContainerEnv`)

Exported separately from the class so it can be unit-tested (`src/__tests__/trainingArmPropagation.test.ts`, `clientAuthEnv.test.ts`):

```typescript
export function buildContainerEnv(config: TrainingConfig): string[] {
  const env = [
    `PROJECT_ID=${config.projectId}`,
    `SERVER_ADDRESS=${config.serverAddress}`,
    `PARTITION_ID=${config.partitionId}`,
    `MODEL_TYPE=${config.modelType}`,
    `DATASET_PATH=/data`,
  ];
  if (config.strategy)        env.push(`STRATEGY=${config.strategy}`);
  if (config.trainingArm)     env.push(`TRAINING_ARM=${config.trainingArm}`);
  if (config.connectionToken) env.push(`FEDLEARN_CONNECTION_TOKEN=${config.connectionToken}`);
  return env;
}
```

`client-docker/entrypoint.sh` turns `MODEL_TYPE`, `STRATEGY` and `TRAINING_ARM` back into `--model-type` / `--strategy` / `--training-arm` flags, and omits each one entirely when its variable is unset — giving the Docker path argv parity with the native path. `FEDLEARN_CONNECTION_TOKEN` is read directly from the environment by the framework and needs no flag.

#### Jetson-Specific Device Mounts

```typescript
const JETSON_DEVICE_MOUNTS: Docker.DeviceMapping[] = [
  { PathOnHost: '/dev/nvhost-ctrl',     PathInContainer: '/dev/nvhost-ctrl',     CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-ctrl-gpu', PathInContainer: '/dev/nvhost-ctrl-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-dbg-gpu',  PathInContainer: '/dev/nvhost-dbg-gpu',  CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-prof-gpu', PathInContainer: '/dev/nvhost-prof-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvmap',           PathInContainer: '/dev/nvmap',           CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-gpu',      PathInContainer: '/dev/nvhost-gpu',      CgroupPermissions: 'rwm' },
];
```

The source comment beside the `case 'jetson'` branch states that `--runtime nvidia` is *"PROHIBITED on Jetson — it searches for PCIe discrete-GPU metadata in the kernel device tree and hangs indefinitely"*.

**That prohibition was measured to be wrong on JetPack 6, and this device list does not work there either.** On an AGX Orin running L4T R36.5.0 / JetPack 6.2 with `nvidia-container-toolkit 1.19.0-1`, `docker run --runtime nvidia` succeeded (7.9 s, `torch.cuda.is_available()` True, device reported as "Orin") while the device-mount path *without* `--runtime nvidia` failed with `cuInit → 801` (`CUDA_ERROR_NOT_SUPPORTED`) and then segfaulted — the in-container `libcuda.so.1` is a stub. Separately, `/dev/nvhost-ctrl` — the first entry above — **does not exist on L4T R36.5**, and Docker hard-errors (`no such file or directory`) if you pass it. That node set is JetPack-5-era.

The full measurement, including the honest scope of what was and was not re-tested, is in [07 → NVIDIA Jetson SoC](./07-hardware-profiles.md#nvidia-jetson-soc-docker). Read it before trusting either recipe on a specific device.

### Docker Multiplexed Log Stream Demultiplexing

Docker's attach/log stream uses a **multiplexed binary protocol** where each chunk is prefixed with an 8-byte header:

```
[stream_type: 1 byte][padding: 3 bytes][payload_size: 4 bytes][payload: payload_size bytes]
```

Where `stream_type` is `1` for stdout and `2` for stderr. Raw chunks must be demultiplexed before they can be displayed as text:

```typescript
private demuxDockerStream(chunk: Buffer, state: { partial: string }): string {
  let output = state.partial;
  let offset = 0;

  while (offset < chunk.length) {
    // Not enough bytes for a full header
    if (offset + 8 > chunk.length) {
      output += chunk.slice(offset).toString('utf-8');
      break;
    }

    // Read payload size from bytes 4-7 (big-endian uint32)
    const payloadSize = chunk.readUInt32BE(offset + 4);

    if (payloadSize === 0) {
      offset += 8;
      continue;
    }

    // Read payload
    if (offset + 8 + payloadSize > chunk.length) {
      output += chunk.slice(offset + 8).toString('utf-8');
      break;
    }

    output += chunk.slice(offset + 8, offset + 8 + payloadSize).toString('utf-8');
    offset += 8 + payloadSize;
  }

  state.partial = '';
  return output;
}
```

### Status Polling

```typescript
async getStatus(): Promise<string> {
  // Native process takes priority
  if (this.nativeProcess) {
    if (this.nativeProcess.exitCode === null) return 'running';
    return this.nativeProcess.exitCode === 0 ? 'completed' : 'error';
  }

  if (!this.activeContainerId) return 'idle';

  try {
    const container = this.docker.getContainer(this.activeContainerId);
    const info = await container.inspect();

    if (info.State.Running)    return 'running';
    if (info.State.Restarting) return 'restarting';
    if (info.State.Paused)     return 'paused';
    if (info.State.Dead)       return 'error';

    return info.State.ExitCode === 0 ? 'completed' : 'error';
  } catch (err) {
    // A container removed out from under us is not an error — it is 'idle'.
    if (message.includes('No such container') || message.includes('404')) {
      this.activeContainerId = null;
      return 'idle';
    }
    return 'error';
  }
}
```

The renderer polls this every 3 seconds via the `getDockerStatus()` IPC call. Note that a container removed externally clears `activeContainerId` and reports `idle` rather than leaving the UI wedged in `error`.

---

## Authentication Service: `auth.service.ts`

### State Machine

`AuthService` maintains authentication state through two storage levels:

```
Login attempt
    │
    ▼
http.post(`${apiBaseUrl}/auth/login`, { username, password })
    │   (the shared instance — carries X-FedLearn-Client; validateStatus < 500
    │    so a 401 is a returned response, not a throw)
    │
    ├── status !== 200 → return false
    │
    └── status 200 → extract JWT
          │
          ├── From response body (accessToken field) [preferred]
          └── From Set-Cookie header (jwtToken cookie) [fallback]
                │
                ▼
         safeStorage.isEncryptionAvailable()?
                │
           ┌────┴────┐
          YES        NO
           │          │
    encrypt → disk   memory only
    (OS keychain)    (no disk persist)
```

### JWT Extraction Strategy

The backend sends the token in two places (for compatibility):

```typescript
// 1. Check response body first (always present)
if (response.data && typeof response.data.accessToken === 'string') {
  jwt = response.data.accessToken;
}

// 2. Fallback: parse the Set-Cookie header
if (!jwt) {
  const setCookieHeaders = response.headers['set-cookie'];
  if (setCookieHeaders) {
    for (const cookie of setCookieHeaders) {
      const match = cookie.match(/jwtToken=([^;]+)/);
      if (match) {
        jwt = match[1];
        break;
      }
    }
  }
}
```

### Server URL Persistence

The backend URL is persisted to `electron-store` across restarts:

```typescript
const DEFAULT_API_BASE_URL = 'http://localhost:8081/api';

constructor(mainWindow: BrowserWindow | null = null) {
  this.mainWindow = mainWindow;                 // used to push auth:session-expired

  this.store = new Store<AuthStoreSchema>({
    name: 'fedlearn-auth',
    clearInvalidConfig: true,  // Recover from corrupt store (e.g., schema changes)
  });

  // Priority: saved URL > env var > localhost default.
  // Read DIRECTLY, not via setApiUrl — see the session note below.
  const savedUrl = this.store.get(SERVER_URL_KEY) as string | undefined;
  this.apiBaseUrl = savedUrl || process.env.FEDLEARN_API_URL || DEFAULT_API_BASE_URL;

  installUnauthorizedHandler(() => this.handleSessionExpired());
}
```

`clearInvalidConfig: true` is important — without it, a `SyntaxError` reading a corrupt store file would crash the IPC handler registration chain, leaving a black window.

The `mainWindow` parameter defaults to `null` so `AuthService` stays constructible in isolation (unit tests, any future headless use) without a real `BrowserWindow`.

The store schema holds three keys: `serverUrl`, `auth` (`{ encryptedJwt, expiresAt, username }`), and `savedCredentials` (the "Save password" blob).

### Changing the URL Invalidates the Session

```typescript
setApiUrl(url: string): void {
  const changed = url !== this.apiBaseUrl;
  this.apiBaseUrl = url;
  this.store.set(SERVER_URL_KEY, url);
  if (changed) {
    this.handleSessionExpired();   // clear JWT + credentials, signal the renderer
  }
}
```

A JWT is minted by one backend and must never be sent to another. This covers both a legitimate server switch and a compromised renderer calling `setServerUrl('https://attacker…')`. The constructor reads `apiBaseUrl` directly rather than calling `setApiUrl`, so a persisted session survives a normal relaunch (`77cb95e`).

### URL Validation in the IPC Handler

The handler no longer contains the policy — it delegates to `evaluateServerUrl` in `validators.ts` and forwards the machine-readable outcome:

```typescript
// ipc.handlers.ts
ipcMain.handle('auth:set-server-url', async (_event, url: unknown, opts?: unknown) => {
  const allowInsecureHttp =
    !!opts && typeof opts === 'object' && (opts as Record<string, unknown>).allowInsecureHttp === true;

  const evaluation = evaluateServerUrl(url, allowInsecureHttp);
  if (!evaluation.ok) {
    // DE-13: 'INSECURE_HTTP' is the code the renderer keys its override UI off
    return { success: false, error: evaluation.error, code: evaluation.code };
  }

  authService.setApiUrl(evaluation.url as string);
  return evaluation.warning
    ? { success: true, url: evaluation.url, warning: evaluation.warning }
    : { success: true, url: evaluation.url };
});
```

Normalization is unchanged in effect: `http://192.168.1.100:8081` becomes `http://192.168.1.100:8081/api`. What is new is that a **remote plaintext `http://`** URL is now *refused* unless the caller passes `{ allowInsecureHttp: true }`, and is accepted with a persistent `warning` even then. See [02 → Transport Policy](./02-security-model.md#transport-policy--refusing-remote-plaintext-http).

### Saved Credentials

`saveCredentials` / `getSavedCredentials` / `clearSavedCredentials` back the "Save password" checkbox. They mirror the JWT posture exactly — `safeStorage`-encrypted or not persisted at all — and a blob that can no longer be decrypted is scrubbed rather than surfaced as an error.

---

## Shared HTTP Client: `http.ts`

Every main-process backend call goes through **one** axios instance:

```typescript
export const NATIVE_CLIENT_HEADER = 'X-FedLearn-Client';
export const NATIVE_CLIENT_VALUE = 'fedlearn-desktop';

export const http: AxiosInstance = axios.create();
http.defaults.headers.common[NATIVE_CLIENT_HEADER] = NATIVE_CLIENT_VALUE;
```

Two reasons it is centralized:

1. **SE-9 marker.** The backend honours `Authorization: Bearer` only when the request also carries the `X-FedLearn-Client` marker; browsers stay cookie-only. Setting it as an instance default means it rides on every request rather than being remembered per call site. (`src/__tests__/nativeClientHeader.test.ts` pins this.)
2. **One 401 handler.** `installUnauthorizedHandler(cb)` installs the single active response interceptor, ejecting any previous one so handlers can never stack. It fires on a 401 from any request *except* the auth handshake (`isAuthHandshakeRequest` matches `/auth/(login|me)`), and works whether the 401 arrives as a resolved response or a rejected promise.

---

## Backend-Facing Services

| Service | Endpoints | Notes |
|---|---|---|
| `ClientProjectService` | `GET /client/projects`, `POST /client/projects/{id}/join`, `GET /client/projects/{id}/connection` | Drives "models I can train". `getConnection` **joins first** (idempotent) — the `/connection` endpoint enrolls only owner-or-CLIENT members, so a merely-*discovered* PUBLIC project would otherwise 403 "Access denied" (`43f4d7e`). The connection payload carries `serverAddress`, `partitionId`, `modelType`, `status`, `connectionToken`, `strategy` and `trainingArm`. |
| `InferenceService` | `GET /inference/models`, `POST /inference/{projectId}` | One-shot inference. Uses a permissive `validateStatus` so the backend's own 4xx message can be surfaced verbatim instead of thrown away. |
| `InferenceStreamService` | STOMP `/ws-logs` → `/topic/inference/{projectId}`, `POST /inference/{id}/generate`, `POST /inference/{id}/generate/stop` | Main-process STOMP bridge. Subscribes **before** firing the generate POST so no tokens are missed, forwards each token to the renderer as an `inference:token` push, and resolves with the full result. The WS upgrade carries both the Bearer header and the SE-9 marker. Connection wait is capped at 8 s and `onStompError`/`onWebSocketError` both resolve, so a transport stall degrades to "HTTP result only" rather than hanging. |

The broker URL is derived from the API URL, not configured separately: `getApiUrl()` → `http(s)://host:port/api` becomes `ws(s)://host:port/ws-logs`.

---

## Hardware Probe: `hardware.probe.ts`

The hardware probe runs once on startup to detect the local GPU environment and pre-select the appropriate training profile.

### Detection Logic

```typescript
export async function detectHardware(): Promise<HardwareDetection> {
  const platform = process.platform;
  const arch = process.arch;
  const bundleAvailable = nativeBundleExists();
  const { available: cudaAvailable, info: cudaInfo } = await probeNvidiaSmi();

  let recommendedProfile: HardwareProfile;

  if (platform === 'darwin' && arch === 'arm64') {
    recommendedProfile = 'mps';          // Apple Silicon → MPS
  } else if (cudaAvailable && platform !== 'linux') {
    recommendedProfile = 'discrete';     // Windows CUDA → Discrete GPU
    // Note: Linux CUDA falls through to 'cpu' — no shipped Linux CUDA bundle yet
  } else {
    recommendedProfile = 'cpu';          // Default
  }

  return { platform, arch, recommendedProfile, nativeBundleAvailable: bundleAvailable, cudaAvailable, cudaInfo };
}
```

### nvidia-smi Probe

```typescript
function probeNvidiaSmi(): Promise<{ available: boolean; info?: string }> {
  return new Promise((resolve) => {
    // 2-second timeout — nvidia-smi can hang on degraded driver installs
    const timeout = setTimeout(() => resolve({ available: false }), 2000);

    execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], (err, stdout) => {
      clearTimeout(timeout);
      if (err) { resolve({ available: false }); return; }
      const info = stdout.trim().split('\n')[0] || 'NVIDIA GPU';
      resolve({ available: true, info });
    });
  });
}
```

The 2-second timeout is critical. On machines with a problematic NVIDIA driver (e.g., driver installed but display adapter disabled), `nvidia-smi` can block for 30+ seconds. The timeout ensures the UI loads promptly even on degraded configurations.

### Native Bundle Check

```typescript
function nativeBundleExists(): boolean {
  if (!app.isPackaged) return true; // Dev mode: always true (uses python3 fallback)
  
  const binaryName = process.platform === 'win32' ? 'fedlearn-client.exe' : 'fedlearn-client';
  const binary = path.join(process.resourcesPath, 'fedlearn-client', binaryName);
  return fs.existsSync(binary);
}
```

If `nativeBundleAvailable` is `false` in a packaged build, `TrainSection` raises a **warning** readiness row ("Native client bundle missing — reinstall to enable training"). There is no Docker fallback for the non-Jetson profiles: they run the bundled client or nothing.

---

## Device Capabilities: `deviceCapabilities.collector.ts`

A second, separate probe backing the advisory eligibility self-gate. Node stdlib only — no Electron import — so it stays unit-testable:

```typescript
export function collectDeviceCapabilities(): DeviceCapabilities {
  return {
    ramGb: os.totalmem() / 1024 ** 3,
    freeStorageGb: freeStorageGb(),   // fs.statfsSync on the home volume (drive root on Windows)
    osName: osName(),                 // 'macos' | 'windows' | 'linux'
    osVersion: os.release(),
    npuTops: undefined,               // not probed on desktop
    batteryPct: undefined,            // not probed on desktop
    onWifi: undefined,                // not probed on desktop
  };
}
```

The three `undefined` fields are treated as "unknown" by `evaluateEligibility` (`src/shared/`), which makes them soft warnings and never a hard failure. The rule's hard failures are RAM, free storage, and the phone-only checks; the result is advisory and never blocks Start.

---

## Auto-Updater: `updater.ts`

```typescript
let updaterInitialized = false;

export function initializeUpdater(mainWindow: BrowserWindow) {
  if (updaterInitialized) return;    // autoUpdater is a PROCESS-WIDE singleton
  updaterInitialized = true;
  ...
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.forceDevUpdateConfig = process.env.NODE_ENV === 'development';
  autoUpdater.checkForUpdatesAndNotify();
}
```

The `updaterInitialized` guard is load-bearing: `createWindow()` can run more than once per process (macOS `activate` re-creates the window after all windows are closed), and `autoUpdater` is a singleton — without the guard every listener would stack and each IPC message would fire N times (`2b02173`).

Updates download in the background, so there is no separate download prompt; `UpdateBanner` surfaces "downloading → progress → restart to install", and `autoInstallOnAppQuit` applies an already-downloaded update on the next quit. The `updater:check` IPC handler additionally attaches one-shot `update-not-available` / `error` relays so a *manual* check gets feedback, which the passive event set does not provide.

---

## IPC Channel Reference

Every registered channel, as of `3.2.0-beta`:

| Channel | Direction | Handler | Description |
|---|---|---|---|
| `dialog:open-directory` | invoke → handle | `ipc.handlers.ts` | Native directory picker; records the result as a consented dataset path; returns `{ success, path }` |
| `docker:start-training` | invoke → handle | `DockerService.startTraining()` | Validates + canonicalizes, then starts the container or native process |
| `docker:stop-training` | invoke → handle | `DockerService.stopTraining()` | Stops (and drains) the active container or native process |
| `docker:get-status` | invoke → handle | `DockerService.getStatus()` | Returns the current training state |
| `docker:training-log` | **push** | `DockerService.sendLog()` | Streams training output to the renderer |
| `hardware:detect` | invoke → handle | `detectHardware()` | Platform, arch, CUDA, bundle presence, recommended profile |
| `device:capabilities` | invoke → handle | `collectDeviceCapabilities()` | RAM / free storage / OS for the eligibility self-gate |
| `auth:login` | invoke → handle | `AuthService.login()` | Authenticates and stores the JWT; returns `{ success }` only |
| `auth:logout` | invoke → handle | `AuthService.logout()` | Clears the JWT from store and memory |
| `auth:check` | invoke → handle | `AuthService.isAuthenticated()` | Returns `{ authenticated: boolean }` |
| `auth:set-server-url` | invoke → handle | `evaluateServerUrl()` → `AuthService.setApiUrl()` | Validates + normalizes; may return `code: 'INSECURE_HTTP'` or a `warning` |
| `auth:get-server-url` | invoke → handle | `AuthService.getApiUrl()` | Returns the current backend URL |
| `auth:save-credentials` | invoke → handle | `AuthService.saveCredentials()` | "Save password" opt-in; `{ success: false }` when `safeStorage` is unavailable |
| `auth:get-credentials` | invoke → handle | `AuthService.getSavedCredentials()` | Pre-fills the login form; `{ success: false }` when none stored |
| `auth:clear-credentials` | invoke → handle | `AuthService.clearSavedCredentials()` | Forgets saved credentials |
| `auth:session-expired` | **push** | `AuthService.emitSessionExpired()` | Session went valid → invalid (401 or proactive expiry) |
| `client:list-projects` | invoke → handle | `ClientProjectService.listProjects()` | Projects the user may train |
| `client:get-connection` | invoke → handle | `ClientProjectService.getConnection()` | Joins, then resolves the live gRPC connection payload |
| `inference:list-models` | invoke → handle | `InferenceService.listModels()` | The user's runnable trained models |
| `inference:run` | invoke → handle | `InferenceService.runInference()` | One-shot inference (image / vector / text) |
| `inference:run-generation` | invoke → handle | `InferenceStreamService.runGeneration()` | Streaming text generation; resolves with the full result |
| `inference:stop-generation` | invoke → handle | `InferenceStreamService.stopGeneration()` | Cancels an in-flight generation (best-effort) |
| `inference:token` | **push** | `InferenceStreamService` | One streamed generation token |
| `updater:install` | invoke → handle | `autoUpdater.quitAndInstall()` | Restart to install a downloaded update |
| `updater:check` | invoke → handle | `autoUpdater.checkForUpdates()` | Manual check; relays not-available/error back |
| `updater:update-available` / `updater:download-progress` / `updater:update-downloaded` / `updater:not-available` / `updater:error` | **push** | `updater.ts` + the `updater:check` handler | Auto-update lifecycle events |

> **Push channels** are unidirectional — Main sends via `mainWindow.webContents.send()`. The renderer subscribes through the preload bridge (`onTrainingLog`, `onInferenceToken`, `onSessionExpired`, `onUpdate*`).
>
> `docker:daemon-unavailable` **no longer exists.** It was removed with the eager startup ping in `4d7d3a4`; the preload exposes no `onDockerUnavailable`, and Docker errors now surface in the log panel at the moment the Jetson path needs the daemon.
>
> The `[IPC] All handlers registered: …` log line at the bottom of `registerIpcHandlers` lists only a subset of the channels above — it has not been kept current and is not a reliable inventory.

---

*Next: [04 — Preload & IPC Bridge](./04-preload-ipc-bridge.md)*  
*Previous: [02 — Security Model](./02-security-model.md)*
