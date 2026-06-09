# FedLearn Desktop — Main Process Deep Dive

> **Part of:** [FedLearn Platform Docs](../../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Entry Point: `main.ts`](#entry-point-maints)
2. [IPC Handler Registry: `ipc.handlers.ts`](#ipc-handler-registry-ipchandlersts)
3. [Training Orchestration: `docker.service.ts`](#training-orchestration-dockerservicets)
4. [Authentication Service: `auth.service.ts`](#authentication-service-authservicets)
5. [Hardware Probe: `hardware.probe.ts`](#hardware-probe-hardwareprobets)
6. [IPC Channel Reference](#ipc-channel-reference)

---

## Entry Point: `main.ts`

`main.ts` is the first file executed by Electron's Main Process. It sets up the entire application shell: window creation, Content Security Policy, IPC handler registration, and app lifecycle management.

### Window Creation

```typescript
// src/main/main.ts
function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    title: 'FedLearn Desktop',
    backgroundColor: '#0a0a0f',   // Dark background, prevents flash on load
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
- `backgroundColor: '#0a0a0f'` — Sets the window background to the app's dark theme color *before* the renderer loads. This prevents the white flash that would otherwise appear while HTML/CSS is being parsed.
- `titleBarStyle: 'hiddenInset'` — macOS only. Hides the title bar chrome but keeps the traffic-light buttons (close/minimize/maximize) in the window, positioned at the specified offsets.
- `minWidth: 960, minHeight: 640` — Enforces a minimum usable window size, preventing layout breakage on very small displays.

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

---

## IPC Handler Registry: `ipc.handlers.ts`

All `ipcMain.handle` registrations are centralized in `registerIpcHandlers()`. This is called once from `main.ts` after the window is created.

```typescript
// src/main/ipc.handlers.ts
export function registerIpcHandlers(mainWindow: BrowserWindow): void {
  dockerService = new DockerService(mainWindow);
  authService = new AuthService();

  // Register each IPC channel...
}
```

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

### Validation Constants

```typescript
const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> = new Set(['discrete', 'jetson', 'cpu', 'mps']);
const PROJECT_ID_PATTERN    = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN  = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
const MAX_DATASET_PATH_LEN = 2048;
```

These mirror the constants in `preload.ts`. Keeping them in sync is important; they're the second line of defense.

### Channel: `dialog:open-directory`

Opens a native OS directory picker dialog:

```typescript
ipcMain.handle('dialog:open-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    title: 'Select Dataset Directory'
  });
  if (result.canceled || result.filePaths.length === 0) {
    return { success: false, error: 'User canceled.' };
  }
  return { success: true, path: result.filePaths[0] };
});
```

Using the **native dialog** for path selection (rather than a text input) is a security choice — paths chosen through the OS dialog are guaranteed to be valid, existing paths that the user intentionally selected. The raw path is still sanitized by `sanitizeDatasetPath` in the `docker:start-training` handler before use.

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

    this.docker = new Docker({ socketPath });

    // Non-blocking probe — logged but not surfaced to renderer unless needed
    this.docker.ping()
      .then(() => log.info('[DockerService] Docker daemon is reachable'))
      .catch((err) => log.warn(`[DockerService] Docker daemon unreachable: ${err.message}`));
  }
}
```

The daemon ping is **non-blocking** — it doesn't wait for a response before creating the window. The result is logged so developers can see connectivity status, but native-path users (MPS/CPU) should never see a Docker-related warning unless they click the Jetson profile.

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

Simple: `jetson` → Docker, everything else → native binary.

### Native Process Resolution

The native client can be in two locations depending on runtime context:

```typescript
private resolveNativeInvocation() {
  const binaryName = process.platform === 'win32' ? 'fedlearn-client.exe' : 'fedlearn-client';

  if (app.isPackaged) {
    // PRODUCTION: Use the PyInstaller bundle shipped in extraResources
    const bundleDir = path.join(process.resourcesPath, 'fedlearn-client');
    const binary = path.join(bundleDir, binaryName);

    if (!fs.existsSync(binary)) {
      log.error(`[Native] Packaged bundle missing at ${binary}`);
      return null;  // Caller surfaces a clear error
    }

    return {
      command: binary,
      baseArgs: [],
      cwd: bundleDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    };
  }

  // DEVELOPMENT: Use system python3 + source tree
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const clientScript = path.join(repoRoot, 'client-docker', 'scripts', 'client.py');
  const frameworkSrc = path.join(repoRoot, 'framework', 'src');

  return {
    command: process.platform === 'win32' ? 'python' : 'python3',
    baseArgs: ['-u', clientScript],  // -u: unbuffered output (critical for real-time logs)
    cwd: path.dirname(clientScript),
    env: {
      ...process.env,
      PYTHONPATH: frameworkSrc,
      PYTHONUNBUFFERED: '1',
    },
  };
}
```

### Starting a Native Process

```typescript
private async startNativeProcess(config: TrainingConfig): Promise<void> {
  // Kill any previous process
  if (this.nativeProcess) {
    this.nativeProcess.kill('SIGTERM');
    this.nativeProcess = null;
  }

  const invocation = this.resolveNativeInvocation();
  if (!invocation) {
    const msg = app.isPackaged
      ? `Native training bundle not found at <resources>/fedlearn-client.`
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

  // LLM flag for large model architectures
  if (config.modelType === 'OPT-125M' || config.modelType === 'Transformer') {
    args.push('--use-llm');
  }

  const child = spawn(invocation.command, args, {
    env: invocation.env,
    cwd: invocation.cwd,
  });

  this.nativeProcess = child;

  // Pipe stdout/stderr to renderer log panel
  child.stdout?.on('data', (data: Buffer) => this.sendLog(data.toString('utf-8')));
  child.stderr?.on('data', (data: Buffer) => this.sendLog(`[stderr] ${data.toString('utf-8')}`));

  child.on('error', (err) => {
    this.sendLog(`[System] Native process error: ${err.message}\n`);
    this.nativeProcess = null;
  });

  child.on('exit', (code, signal) => {
    this.sendLog(`[System] Native process exited (code=${code}, signal=${signal})\n`);
    this.nativeProcess = null;
  });
}
```

**Why `PYTHONUNBUFFERED=1`?** Python buffers its output by default when stdout is not a TTY (which it isn't when spawned by Node). Without unbuffered mode, log lines would only appear after the buffer fills up (typically 8KB), making real-time logging useless.

### Stopping Training (SIGTERM + SIGKILL Fallback)

```typescript
async stopTraining(): Promise<void> {
  if (this.nativeProcess) {
    this.nativeProcess.kill('SIGTERM');         // Graceful shutdown first
    const proc = this.nativeProcess;
    setTimeout(() => {
      if (proc && !proc.killed) proc.kill('SIGKILL'); // Force kill after 5s
    }, 5000);
    return;
  }
  if (this.activeContainerId) {
    await this.stopDockerContainer();
  }
}
```

A 5-second grace period allows the Python client to checkpoint its model state and perform clean shutdown before being forcefully terminated.

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
      // Direct device mounts — DO NOT use --runtime nvidia on Jetson
      hostConfig.Devices = JETSON_DEVICE_MOUNTS;
      break;
    case 'discrete':
      // PCIe GPU — standard DeviceRequests (equivalent to --gpus all)
      hostConfig.DeviceRequests = [{ Count: -1, Capabilities: [['gpu']] }];
      break;
    case 'cpu':
      break; // No GPU config
    case 'mps':
      throw new Error('MPS profile cannot run under Docker');
  }

  // 4. Pass config as environment variables
  const env = [
    `PROJECT_ID=${config.projectId}`,
    `SERVER_ADDRESS=${config.serverAddress}`,
    `PARTITION_ID=${config.partitionId}`,
    `MODEL_TYPE=${config.modelType}`,
    `DATASET_PATH=/data`,
  ];

  // 5. Create + start container
  const container = await this.docker.createContainer({
    Image: DOCKER_IMAGE,
    name: CONTAINER_NAME,
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

**Why `AutoRemove: false`?** If set to `true`, Docker removes the container immediately on exit — before the log stream has fully drained. With `false`, the container stays around until explicitly removed in `stopDockerContainer()`, giving the log stream time to deliver the final bytes.

#### Jetson-Specific Device Mounts

NVIDIA Jetson SoCs use a Tegra GPU that does not present as a standard PCIe device. The `--runtime nvidia` Docker flag (used for discrete GPUs) searches for PCIe device metadata in the kernel device tree and **hangs indefinitely** on Jetson. The correct approach is direct device node mapping:

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

All six device nodes must be mounted. Missing any one of them causes the container to hang or crash silently.

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

  const container = this.docker.getContainer(this.activeContainerId);
  const info = await container.inspect();

  if (info.State.Running)    return 'running';
  if (info.State.Restarting) return 'restarting';
  if (info.State.Paused)     return 'paused';
  if (info.State.Dead)       return 'error';

  return info.State.ExitCode === 0 ? 'completed' : 'error';
}
```

The renderer polls this every 3 seconds via `getDockerStatus()` IPC call.

---

## Authentication Service: `auth.service.ts`

### State Machine

`AuthService` maintains authentication state through two storage levels:

```
Login attempt
    │
    ▼
axios.post('/api/auth/login', { username, password })
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

constructor() {
  this.store = new Store({
    name: 'fedlearn-auth',
    clearInvalidConfig: true,  // Recover from corrupt store (e.g., schema changes)
  });
  
  // Priority: saved URL > env var > localhost default
  const savedUrl = this.store.get(SERVER_URL_KEY) as string | undefined;
  this.apiBaseUrl = savedUrl || process.env.FEDLEARN_API_URL || DEFAULT_API_BASE_URL;
}
```

`clearInvalidConfig: true` is important — without it, a `SyntaxError` reading a corrupt store file would crash the IPC handler registration chain, leaving a black window.

### URL Normalization in IPC Handler

```typescript
// ipc.handlers.ts
ipcMain.handle('auth:set-server-url', async (_event, url: unknown) => {
  // Require http(s):// protocol
  if (!/^https?:\/\//i.test(url.trim())) {
    return { success: false, error: 'URL must start with http:// or https://' };
  }
  
  // Normalize: strip trailing slashes, ensure /api suffix
  let normalized = url.trim().replace(/\/+$/, '');
  if (!normalized.endsWith('/api')) {
    normalized += '/api';
  }
  
  authService.setApiUrl(normalized);
  return { success: true, url: normalized };
});
```

This means a user entering `http://192.168.1.100:8081` automatically becomes `http://192.168.1.100:8081/api`.

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

If `nativeBundleAvailable` is `false` in a packaged build, the `HardwareSelector` shows a warning label and the Docker path may be required.

---

## IPC Channel Reference

A complete reference of all registered IPC channels:

| Channel | Direction | Handler | Description |
|---|---|---|---|
| `dialog:open-directory` | invoke → handle | `ipc.handlers.ts` | Opens native directory picker; returns `{ success, path }` |
| `docker:start-training` | invoke → handle | `DockerService.startTraining()` | Starts Docker container or native process |
| `docker:stop-training` | invoke → handle | `DockerService.stopTraining()` | Stops active container or native process |
| `docker:get-status` | invoke → handle | `DockerService.getStatus()` | Returns current training state |
| `docker:training-log` | **push** (Main → Renderer) | `DockerService.sendLog()` | Streams training output to renderer |
| `docker:daemon-unavailable` | **push** (Main → Renderer) | *(planned)* | Notifies renderer if Docker is not running |
| `hardware:detect` | invoke → handle | `detectHardware()` | Returns platform, GPU, recommended profile |
| `auth:login` | invoke → handle | `AuthService.login()` | Authenticates and stores JWT |
| `auth:logout` | invoke → handle | `AuthService.logout()` | Clears JWT from store and memory |
| `auth:check` | invoke → handle | `AuthService.isAuthenticated()` | Returns `{ authenticated: boolean }` |
| `auth:set-server-url` | invoke → handle | `AuthService.setApiUrl()` | Saves backend URL with normalization |
| `auth:get-server-url` | invoke → handle | `AuthService.getApiUrl()` | Returns current backend URL |

> **Push channels** (`docker:training-log`, `docker:daemon-unavailable`) are unidirectional — Main pushes to Renderer via `mainWindow.webContents.send()`. The renderer registers listeners via the preload bridge's `onTrainingLog()` and `onDockerUnavailable()` callbacks.

---

*Next: [04 — Preload & IPC Bridge](./04-preload-ipc-bridge.md)*  
*Previous: [02 — Security Model](./02-security-model.md)*
