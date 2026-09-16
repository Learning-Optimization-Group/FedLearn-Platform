# FedLearn Desktop — Preload Script & IPC Bridge

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [The Role of the Preload Script](#the-role-of-the-preload-script)
2. [contextBridge API Surface](#contextbridge-api-surface)
3. [Validation Strategy in Depth](#validation-strategy-in-depth)
4. [Complete `window.fedLearnAPI` Reference](#complete-windowfedlearnapi-reference)
5. [Event-Based IPC (Push Channels)](#event-based-ipc-push-channels)
6. [TypeScript Integration in the Renderer](#typescript-integration-in-the-renderer)
7. [Common Pitfalls](#common-pitfalls)

---

## The Role of the Preload Script

The preload script (`src/preload/preload.ts`) executes in a **special sandboxed context** — it has access to `ipcRenderer` (to talk to the Main Process) but runs in an isolated JavaScript world separate from the renderer's page context. This isolation is enforced by `contextIsolation: true` in `BrowserWindow`.

```
                  ┌─────────────────────────────────────────┐
                  │  Renderer World (window.js context)     │
                  │  React App, Components, etc.            │
                  │                                         │
                  │  window.fedLearnAPI ← (read-only view)  │
                  └──────────────────┬──────────────────────┘
                                     │ structured clone
                  ┌──────────────────▼──────────────────────┐
                  │  Preload World (isolated context)        │
                  │  preload.ts                              │
                  │  ├── contextBridge.exposeInMainWorld()  │
                  │  ├── Validation functions               │
                  │  └── ipcRenderer.invoke() calls         │
                  └──────────────────┬──────────────────────┘
                                     │ IPC
                  ┌──────────────────▼──────────────────────┐
                  │  Main Process (Node.js)                  │
                  │  ipcMain.handle() registrations          │
                  └─────────────────────────────────────────┘
```

**Key invariant:** The renderer cannot call `ipcRenderer.invoke` directly. It can only call the functions exposed via `contextBridge`, which means every IPC message goes through the validation layer in the preload first.

---

## contextBridge API Surface

The entire exposed API is declared in a single `contextBridge.exposeInMainWorld('fedLearnAPI', { ... })` call. This registers a named object on `window` that the renderer can access as `window.fedLearnAPI`.

```typescript
// src/preload/preload.ts
contextBridge.exposeInMainWorld('fedLearnAPI', {
  // ── Training ──────────────────────────────────────────────
  startTraining:             async (config)             => { /* ... */ },
  stopTraining:              async ()                   => { /* ... */ },
  getDockerStatus:           async ()                   => { /* ... */ },
  onTrainingLog:             (callback)                 => { /* ... */ },
  removeTrainingLogListener: ()                         => { /* ... */ },
  selectDatasetPath:         async ()                   => { /* ... */ },

  // ── Auth / session ────────────────────────────────────────
  login:                     async (username, password) => { /* ... */ },
  logout:                    async ()                   => { /* ... */ },
  checkAuth:                 async ()                   => { /* ... */ },
  onSessionExpired:          (callback)                 => { /* ... */ },
  removeSessionExpiredListener: ()                      => { /* ... */ },
  setServerUrl:              async (url, opts)          => { /* ... */ },
  getServerUrl:              async ()                   => { /* ... */ },
  saveCredentials:           async (username, password) => { /* ... */ },
  getSavedCredentials:       async ()                   => { /* ... */ },
  clearSavedCredentials:     async ()                   => { /* ... */ },

  // ── "Models I can train" ──────────────────────────────────
  listTrainableProjects:     async ()                   => { /* ... */ },
  getProjectConnection:      async (projectId)          => { /* ... */ },

  // ── "Use a model" (inference) ─────────────────────────────
  listModels:                async ()                   => { /* ... */ },
  runInference:              async (projectId, payload) => { /* ... */ },
  runGeneration:             async (projectId, payload) => { /* ... */ },
  stopGeneration:            async (projectId)          => { /* ... */ },
  onInferenceToken:          (callback)                 => { /* ... */ },
  removeInferenceTokenListener: ()                      => { /* ... */ },

  // ── Device / hardware ─────────────────────────────────────
  detectHardware:            async ()                   => { /* ... */ },
  getDeviceCapabilities:     ()                         => { /* ... */ },

  // ── Auto-updater ──────────────────────────────────────────
  onUpdateAvailable:         (callback)                 => { /* ... */ },
  onUpdateProgress:          (callback)                 => { /* ... */ },
  onUpdateDownloaded:        (callback)                 => { /* ... */ },
  onUpdateNotAvailable:      (callback)                 => { /* ... */ },
  onUpdateError:             (callback)                 => { /* ... */ },
  installUpdate:             async ()                   => { /* ... */ },
  checkForUpdates:           async ()                   => { /* ... */ },
});
```

There is **no `onDockerUnavailable`**. It was removed together with the eager Docker daemon ping and the `docker:daemon-unavailable` channel (`4d7d3a4`); Docker failures now surface in the training log at the moment the Jetson path needs the daemon.

All five `onUpdate*` listeners have **no removal counterpart** — nothing removes them, by design. That is why `App.tsx` mounts `UpdateBanner` once at shell level and never unmounts it: re-mounting would stack listeners with no way to clear them. The other three families — `onTrainingLog`, `onSessionExpired` and `onInferenceToken` — *do* have removers and must be cleaned up.

`electron-updater` appears in the preload as a **type-only** import (`import type { UpdateInfo, ProgressInfo }`), erased at compile time — the preload bundle never pulls in its runtime code, only the shapes Main forwards over IPC.

**Important:** `contextBridge` serializes all values using the **structured clone algorithm**. This means:
- Functions passed TO the renderer are callable but not inspectable
- Objects passed FROM the renderer are deep-cloned before reaching the preload/main
- Prototype chains are NOT preserved — class instances become plain objects
- This prevents prototype pollution attacks from the renderer

---

## Validation Strategy in Depth

### Allowlist Constants

```typescript
// Explicit enum of valid profiles — not derived from any runtime data
const ALLOWED_HARDWARE_PROFILES = ['discrete', 'jetson', 'cpu', 'mps'] as const;

// Regex patterns with explicit length bounds
const PROJECT_ID_PATTERN     = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN   = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
const MAX_STRING_LENGTH      = 256;

// Payload bounds for the inference channels
const MAX_IMAGE_BASE64_LEN = 14 * 1024 * 1024;  // ~10 MB decoded
const MAX_VECTOR_LEN       = 100_000;
const MAX_TEXT_LEN         = 10_000;            // matches the backend's @Size(max = 10_000)
```

All patterns use **anchored regex** (both `^` and `$`) to prevent partial matches. Every pattern includes an explicit **maximum length** to prevent resource exhaustion from arbitrarily long strings.

Two validators are worth calling out separately:

```typescript
// Optional: absent means the legacy no-auth flow. When present it must be a
// bounded token-charset string (an HMAC-JWT: three base64url segments, dots).
function isValidConnectionToken(val: unknown): boolean {
  if (val === undefined || val === null) return true;
  if (typeof val !== 'string' || val.length === 0 || val.length > 8192) return false;
  return /^[A-Za-z0-9._-]+$/.test(val);
}

// Exactly one of imageBase64 / values / text, each within bounds.
function isValidInferencePayload(payload: unknown): payload is InferencePayloadInput { /* ... */ }
```

`isValidInferencePayload` checks in order — `imageBase64` (non-empty, ≤ `MAX_IMAGE_BASE64_LEN`), then `values` (non-empty, ≤ `MAX_VECTOR_LEN`, every element a finite number), then `text` (non-blank after trim, ≤ `MAX_TEXT_LEN`) — and rejects a payload carrying none of them. `ipc.handlers.ts` re-implements the same rules in `sanitizeInferencePayload`, which additionally returns a *clean* payload object rather than a boolean.

### Validation Function Pattern

Each validator follows the same pattern: check type first (to avoid exceptions), then check constraints:

```typescript
function isValidHardwareProfile(profile: unknown): boolean {
  // 1. Type guard — protect against undefined, number, object, etc.
  if (typeof profile !== 'string') {
    console.error(`[Preload:Validation] Hardware profile is not a string: ${typeof profile}`);
    return false;
  }
  // 2. Allowlist check — only explicit known values
  const valid = (ALLOWED_HARDWARE_PROFILES as readonly string[]).includes(profile);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected hardware profile: "${profile}"`);
  }
  return valid;
}

function isValidProjectId(id: unknown): boolean {
  if (typeof id !== 'string') { /* ... */ return false; }
  const valid = PROJECT_ID_PATTERN.test(id);
  if (!valid) { /* ... */ }
  return valid;
}

function isValidDatasetPath(val: unknown): boolean {
  if (typeof val !== 'string') { /* ... */ return false; }
  // Length-only check here — deep path validation happens in Main
  // (Main has access to fs.statSync; preload doesn't in sandbox mode).
  // '' is deliberately ALLOWED — it means "use the built-in dataset".
  if (val.length > 2048) { /* ... */ return false; }
  return true;
}
```

**Why does `isValidDatasetPath` only check length, not the actual path?** Because `preload.ts` runs in the Electron sandboxed renderer context — it has access to `ipcRenderer` but NOT to `fs`. The full path sanitization (resolve, stat, directory check) happens in Main where `fs` is available. The preload's job is only to block obviously invalid inputs.

### The `startTraining` Validation Flow

```typescript
startTraining: async (config: TrainingConfigInput) => {
  // Validate every field before ANY ipcRenderer call
  if (!isValidHardwareProfile(config.hardwareProfile))
    return { success: false, error: 'Invalid hardware profile' };

  if (!isValidProjectId(config.projectId))
    return { success: false, error: 'Invalid project ID' };

  if (!isValidServerAddress(config.serverAddress))
    return { success: false, error: 'Invalid server address' };

  if (!isValidPartitionId(config.partitionId))
    return { success: false, error: 'Invalid partition ID' };

  if (!isValidModelType(config.modelType))
    return { success: false, error: 'Invalid model type' };

  if (!isValidDatasetPath(config.datasetPath))
    return { success: false, error: 'Invalid dataset path' };

  if (!isValidConnectionToken(config.connectionToken))
    return { success: false, error: 'Invalid connection token' };

  // strategy is optional and comes from the backend connection payload; reject a
  // malformed one rather than forwarding garbage. Main re-validates identically.
  if (config.strategy !== undefined && !/^[a-zA-Z0-9_\-.]{1,64}$/.test(config.strategy))
    return { success: false, error: 'Invalid strategy' };

  // All checks passed — forward to Main
  return ipcRenderer.invoke('docker:start-training', {
    // Explicitly reconstruct the object — DO NOT spread config directly.
    // Explicit fields prevent additional unexpected properties from being forwarded.
    hardwareProfile: config.hardwareProfile,
    projectId:       config.projectId,
    serverAddress:   config.serverAddress,
    partitionId:     config.partitionId,
    modelType:       config.modelType,
    datasetPath:     config.datasetPath,
    connectionToken: config.connectionToken,
    strategy:        config.strategy,
  });
},
```

**Why reconstruct the object instead of spreading?** Spreading `{ ...config }` would forward any extra properties on the `config` object to Main. While Main validates the expected fields, it is cleaner — and safer — to be explicit about what gets sent.

> **The cost of that pattern, live in this file.** `TrainingConfigInput` declares `trainingArm?: string`, `TrainSection` populates it from the backend connection payload, and `App.tsx`'s `Window.fedLearnAPI` declaration types it — but the reconstructed object above does **not** include `trainingArm`. The field is therefore dropped at the bridge, and Main's `docker:start-training` handler always observes `cfg.trainingArm === undefined`.
>
> Downstream, `if (config.trainingArm)` in `DockerService` is never true, so neither `--training-arm` (native path) nor `TRAINING_ARM` (container path) is ever emitted from the desktop app. `buildContainerEnv` and the argv construction are both correct and unit-tested (`src/__tests__/trainingArmPropagation.test.ts` calls them directly), which is exactly why the gap survives a green suite: nothing tests the preload's forwarded payload shape.
>
> The commit that added the field (`2eefee1`) touched the interface and every layer below the bridge, but not the invoke object. Adding a field to `TrainingConfigInput` **and** to the invoke payload is one change, not two.

---

## Complete `window.fedLearnAPI` Reference

### `startTraining(config)`

Starts a federated learning training session.

```typescript
startTraining(config: {
  hardwareProfile: 'discrete' | 'jetson' | 'cpu' | 'mps';
  projectId: string;        // Pattern: /^[a-zA-Z0-9_-]{1,128}$/
  serverAddress: string;    // Pattern: /^[a-zA-Z0-9._:/-]{1,256}$/
  partitionId: string;      // Pattern: /^[0-9]{1,10}$/
  modelType: string;        // Pattern: /^[a-zA-Z0-9_\-\.]{1,128}$/  (a recipe key)
  datasetPath: string;      // Existing directory, or '' for "use the built-in dataset"
  connectionToken?: string; // FL connection token from the backend (SE-14)
  strategy?: string;        // Aggregation strategy from the connection payload
  trainingArm?: string;     // DECLARED but NOT FORWARDED — see the note above
}): Promise<{ success: boolean; error?: string }>
```

Everything from `projectId` through `trainingArm` originates in `GET /api/client/projects/{id}/connection`, not in a form the user filled in. `datasetPath` is the only field the user supplies directly, and only through the native dialog.

**Returns:**
- `{ success: true }` — training started successfully
- `{ success: false, error: string }` — validation failed at either layer, or Docker/spawn error

---

### `stopTraining()`

Stops the currently running training container or native process.

```typescript
stopTraining(): Promise<{ success: boolean; error?: string }>
```

---

### `getDockerStatus()`

Returns the current status of the training process (polled every 3 seconds by `App.tsx`).

```typescript
getDockerStatus(): Promise<{
  success: boolean;
  status?: 'idle' | 'running' | 'completed' | 'error' | 'restarting' | 'paused';
}>
```

---

### `login(username, password)`

Authenticates with the FedLearn backend. The JWT is stored in Main and **never returned to the renderer**.

```typescript
login(username: string, password: string): Promise<{ success: boolean }>
```

- `username` and `password` must each be non-empty strings up to 256 characters
- Returns `{ success: true }` on success, `{ success: false }` on failure (no error details to prevent credential enumeration)

---

### `logout()`

Clears the stored JWT from the OS keychain and in-memory fallback.

```typescript
logout(): Promise<{ success: boolean }>
```

---

### `checkAuth()`

Checks whether a valid (non-expired) JWT is currently stored.

```typescript
checkAuth(): Promise<{ success: boolean; authenticated?: boolean }>
```

Called on app startup in `App.tsx` to determine whether to show the login screen or the dashboard.

---

### `onTrainingLog(callback)`

Registers a callback to receive real-time log output from the training process.

```typescript
onTrainingLog(callback: (logLine: string) => void): void
```

This is a one-way push channel — the Main Process sends log lines via `mainWindow.webContents.send('docker:training-log', text)` and the preload forwards them to the registered callback.

```typescript
// Preload implementation
onTrainingLog: (callback: (logLine: string) => void): void => {
  ipcRenderer.on('docker:training-log', (_event, value: string) => {
    if (typeof value === 'string') {  // Type guard on the incoming value
      callback(value);
    }
  });
},
```

**Note:** Multiple calls to `onTrainingLog` register multiple listeners. Call `removeTrainingLogListener()` before re-registering to avoid duplicates.

---

### `removeTrainingLogListener()`

Removes all listeners registered for the `docker:training-log` event. Should be called in `useEffect` cleanup:

```typescript
removeTrainingLogListener(): void
```

```typescript
// Usage in App.tsx
useEffect(() => {
  window.fedLearnAPI.onTrainingLog((logLine) => { /* ... */ });
  
  return () => {
    window.fedLearnAPI.removeTrainingLogListener(); // Cleanup on unmount
  };
}, [isAuthenticated]);
```

---

### `onSessionExpired(callback)` / `removeSessionExpiredListener()`

Registers a callback fired when Main invalidates the current session mid-use — a 401 from the backend, a locally-detected expired token, or a server-URL change. The renderer reacts by clearing its own auth state and showing the login screen again.

```typescript
onSessionExpired(callback: () => void): void
removeSessionExpiredListener(): void
```

`App.tsx` registers this **unconditionally on mount**, not gated on `isAuthenticated` — it is precisely what detects the authenticated → expired transition, so gating it on the state it is meant to change would make it useless.

---

### `setServerUrl(url, opts?)`

Sets the backend server URL. The URL is validated (must start with `http://` or `https://`) and `/api` is appended automatically if not present.

```typescript
setServerUrl(
  url: string,
  opts?: { allowInsecureHttp?: boolean },
): Promise<{
  success: boolean;
  url?: string;      // The normalized URL that was saved
  error?: string;
  code?: string;     // 'INSECURE_HTTP' when refused for plaintext transport
  warning?: string;  // Present when accepted via the insecure override
}>
```

Plaintext `http://` to a **non-loopback** host is refused with `code: 'INSECURE_HTTP'` unless `opts.allowInsecureHttp` is set — and even then the response carries a `warning` the caller must surface, because credentials and the session token would traverse the network unencrypted (DE-13). The preload forwards only the single known flag, never the caller's whole `opts` object.

Setting a **different** URL clears the current session (see [03 → Changing the URL Invalidates the Session](./03-main-process.md#changing-the-url-invalidates-the-session)).

**Example:**
```typescript
await window.fedLearnAPI.setServerUrl('http://192.168.1.50:8081');
// → { success: false, code: 'INSECURE_HTTP', error: '…would cross the network unencrypted…' }

await window.fedLearnAPI.setServerUrl('http://192.168.1.50:8081', { allowInsecureHttp: true });
// → { success: true, url: 'http://192.168.1.50:8081/api', warning: 'Insecure server URL: …' }

await window.fedLearnAPI.setServerUrl('http://localhost:8081');
// → { success: true, url: 'http://localhost:8081/api' }   — loopback, warning-free
```

---

### `getServerUrl()`

Returns the currently configured backend URL (includes `/api` suffix).

```typescript
getServerUrl(): Promise<{ success: boolean; url?: string }>
```

---

### `selectDatasetPath()`

Triggers the native OS directory picker dialog.

```typescript
selectDatasetPath(): Promise<{
  success: boolean;
  path?: string;   // Absolute path to selected directory
  error?: string;
}>
```

Returns `{ success: false, error: 'User canceled.' }` if the user closes the dialog.

---

### `detectHardware()`

Runs a one-shot hardware detection probe.

```typescript
detectHardware(): Promise<{
  success: boolean;
  detection?: {
    platform: string;                // 'darwin' | 'win32' | 'linux'
    arch: string;                    // 'arm64' | 'x64'
    recommendedProfile: string;      // 'mps' | 'discrete' | 'cpu'
    nativeBundleAvailable: boolean;  // Is the PyInstaller bundle present?
    cudaAvailable: boolean;          // Was nvidia-smi successful?
    cudaInfo?: string;               // GPU name from nvidia-smi, e.g., 'NVIDIA RTX 4090'
  };
  error?: string;
}>
```

---

### Saved credentials

```typescript
saveCredentials(username: string, password: string): Promise<{ success: boolean }>
getSavedCredentials(): Promise<{ success: boolean; username?: string; password?: string }>
clearSavedCredentials(): Promise<{ success: boolean }>
```

Backs the "Save password" opt-in. `saveCredentials` returns `{ success: false }` when OS encryption is unavailable — Main refuses to write a reversible secret to disk rather than degrading silently. `getSavedCredentials` returns `{ success: false }` when nothing is stored or the blob can no longer be decrypted.

---

### `listTrainableProjects()` / `getProjectConnection(projectId)`

The pair that replaced manual project-id / server-address / partition entry.

```typescript
listTrainableProjects(): Promise<{ success: boolean; projects?: ClientProject[]; error?: string }>

getProjectConnection(projectId: string): Promise<{
  success: boolean;
  connection?: {
    projectId: string;
    name: string;
    modelType: string;
    serverAddress: string;
    partitionId: number;      // number here; TrainSection stringifies it for startTraining
    status: string;
    connectionToken?: string;
    strategy?: string;
    trainingArm?: string;
  };
  error?: string;
}>
```

`getProjectConnection` performs an idempotent `POST .../join` before the `GET .../connection` — without it, a PUBLIC project the user only *discovered* would 403 "Access denied" (`43f4d7e`). Backend 4xx messages are surfaced verbatim (e.g. "the FL server is not running").

`ClientProject` additionally carries an optional `requirements` block that feeds the advisory device self-gate.

---

### Inference

```typescript
listModels(): Promise<{ success: boolean; models?: InferableModel[]; error?: string }>

runInference(
  projectId: string,
  payload: { imageBase64?: string } | { values?: number[] } | { text?: string },
): Promise<{ success: boolean; result?: InferenceResult; error?: string }>

runGeneration(
  projectId: string,
  payload: {
    prompt: string;
    maxNewTokens: number;
    temperature: number;
    history?: { role: 'user' | 'assistant'; content: string }[];
  },
): Promise<{ success: boolean; result?: unknown; error?: string }>

stopGeneration(projectId: string): Promise<{ success: boolean; stopped?: boolean; error?: string }>

onInferenceToken(callback: (token: string) => void): void
removeInferenceTokenListener(): void
```

`InferableModel.inputKind` is `'image' | 'vector' | 'text' | 'generation' | null` and drives which input widget the Model Playground renders.

The preload validates `prompt` (non-blank, ≤ 10 000 chars) but leaves `maxNewTokens`/`temperature` to Main, which **clamps rather than rejects** them: `maxNewTokens` to `[1, 2048]` (default 256) and `temperature` to `[0, 2]` (default 0.7), and truncates `history` to 100 turns after filtering out malformed entries. `stopGeneration` is best-effort — the renderer keeps whatever partial text it already streamed regardless of the response.

---

### `getDeviceCapabilities()`

```typescript
getDeviceCapabilities(): Promise<{
  success: boolean;
  capabilities?: DeviceCapabilities;  // ramGb, freeStorageGb?, osName, osVersion, npuTops?, batteryPct?, onWifi?
  error?: string;
}>
```

Feeds `evaluateEligibility` (`src/shared/`) so the model picker can mark each project *recommended* / *limited* / *unsupported*. On desktop, `npuTops`, `batteryPct` and `onWifi` are always `undefined`, which the rule treats as "unknown" — a soft warning, never a hard failure. The result is advisory and never blocks Start.

---

### Auto-updater

```typescript
onUpdateAvailable(callback: (info: UpdateInfo) => void): void
onUpdateProgress(callback: (progress: ProgressInfo) => void): void
onUpdateDownloaded(callback: (info: UpdateInfo) => void): void
onUpdateNotAvailable(callback: () => void): void
onUpdateError(callback: (message: string) => void): void
installUpdate(): Promise<{ success: boolean; error?: string }>
checkForUpdates(): Promise<{ success: boolean; error?: string }>
```

None of the five listeners has a removal counterpart. `UpdateBanner` is therefore mounted once at shell level and never unmounted — see [05](./05-renderer-components.md#updatebanner--auto-update-layer).

`onUpdateNotAvailable` and `onUpdateError` only fire for a **manual** `checkForUpdates()`: the `updater:check` handler attaches those one-shot relays itself, whereas the passive listeners registered in `updater.ts` cover only `update-available`, `download-progress` and `update-downloaded`.

---

## Event-Based IPC (Push Channels)

Unlike `ipcRenderer.invoke` (request-response), some IPC communication is **unidirectional push** from Main to Renderer. Five families use it:

| Channel | Preload listener | Remover | Consumer |
|---|---|---|---|
| `docker:training-log` | `onTrainingLog` | `removeTrainingLogListener` | `App.tsx` → RAF batch → `LogPanel` |
| `inference:token` | `onInferenceToken` | `removeInferenceTokenListener` | `ModelPlayground` streaming bubble |
| `auth:session-expired` | `onSessionExpired` | `removeSessionExpiredListener` | `App.tsx` → back to the login screen |
| `updater:update-available` / `download-progress` / `update-downloaded` | `onUpdateAvailable` / `onUpdateProgress` / `onUpdateDownloaded` | *(none)* | `UpdateBanner` |
| `updater:not-available` / `updater:error` | `onUpdateNotAvailable` / `onUpdateError` | *(none)* | `UpdateBanner`, manual checks only |

The log-streaming case below is the one where efficiency matters most.

### How Push Works

```
Main Process                          Renderer Process
     │                                      │
     │  mainWindow.webContents              │
     │    .send('docker:training-log', txt) │
     │ ──────────────────────────────────► │
     │                                      │
     │                         ipcRenderer.on('docker:training-log',
     │                                     (_event, value) => callback(value))
     │                                      │
     │                                 callback(txt)  → setState → re-render
```

The preload wraps `ipcRenderer.on` in the `onTrainingLog` function to:
1. Validate the incoming value is a string
2. Present a clean callback-based API to the renderer instead of raw event emitter

### Why Push Instead of Polling?

For log streaming, push is significantly more efficient than polling:
- **Low latency:** Logs appear within milliseconds of being written
- **No wasted requests:** Polling at 100ms with empty results = 10 empty IPC calls per second
- **Batching:** The renderer's `requestAnimationFrame` batching works better with push since it batches everything received in a 16ms window

---

## TypeScript Integration in the Renderer

Since `window.fedLearnAPI` is injected at runtime by the preload, TypeScript doesn't know it exists by default. `App.tsx` declares it as a global interface augmentation:

```typescript
// src/renderer/App.tsx
declare global {
  interface Window {
    fedLearnAPI: {
      startTraining: (config: {
        hardwareProfile: string;
        projectId: string;
        serverAddress: string;
        partitionId: string;
        modelType: string;
        datasetPath: string;
        connectionToken?: string;
        strategy?: string;
        trainingArm?: string;
      }) => Promise<{ success: boolean; error?: string }>;

      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (username: string, password: string) => Promise<{ success: boolean }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean }>;
      onSessionExpired: (callback: () => void) => void;
      removeSessionExpiredListener: () => void;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
      listTrainableProjects: () => Promise<{ success: boolean; projects?: ClientProject[]; error?: string }>;
      getProjectConnection: (projectId: string)
        => Promise<{ success: boolean; connection?: ProjectConnection; error?: string }>;
      setServerUrl: (url: string, opts?: { allowInsecureHttp?: boolean })
        => Promise<{ success: boolean; url?: string; error?: string; code?: string; warning?: string }>;
      getServerUrl: () => Promise<{ success: boolean; url?: string }>;
      saveCredentials / getSavedCredentials / clearSavedCredentials: /* … */;
      selectDatasetPath: () => Promise<{ success: boolean; path?: string; error?: string }>;
      listModels / runInference / runGeneration / stopGeneration: /* … */;
      onInferenceToken: (callback: (token: string) => void) => void;
      removeInferenceTokenListener: () => void;
      detectHardware: () => Promise<{ success: boolean; detection?: { /* … */ }; error?: string }>;
      getDeviceCapabilities: () => Promise<{ success: boolean; capabilities?: DeviceCapabilities; error?: string }>;
      onUpdateAvailable / onUpdateProgress / onUpdateDownloaded
        / onUpdateNotAvailable / onUpdateError / installUpdate / checkForUpdates: /* … */;
    };
  }
}
```

(Abbreviated above; the file carries every signature in full.) The declaration references `ClientProject` / `ProjectConnection` from `src/renderer/client.types.ts` and `InferableModel` / `InferenceResult` from `src/renderer/inference.types.ts`, so the renderer's view of the backend payloads is typed rather than `unknown` — even though the **preload** deliberately types those same returns as `unknown[]`/`unknown` (it validates shape, not schema).

This declaration should ideally live in a separate `src/renderer/global.d.ts` file for cleaner separation, but is currently co-located in `App.tsx`.

> **Important:** Keep this declaration in sync with the actual `preload.ts` implementation. Drift between the two is only caught at runtime — and note it can drift in *either* direction. `trainingArm` is currently present in this declaration and in `TrainingConfigInput`, yet absent from the object the preload actually forwards, so the compiler is entirely happy while the value is silently discarded.

---

## Common Pitfalls

### 1. Adding a New IPC Channel — Three Files to Update

Adding any new capability requires updating **three files** in a specific order:

1. **`ipc.handlers.ts`** — Register `ipcMain.handle('new:channel', ...)`, with validation delegated to `validators.ts`
2. **`preload.ts`** — Add a typed wrapper to `contextBridge.exposeInMainWorld` with validation
3. **`App.tsx` (or `global.d.ts`)** — Add the TypeScript declaration to `Window.fedLearnAPI`

Missing any one of these causes either a silent failure (no channel registered), a TypeScript error (missing type), or a security gap (no preload validation).

### 1a. Adding a New *Field* to an Existing Channel — Four Edits, Not Three

A new field on an existing payload is easier to get wrong than a new channel, because every layer type-checks and none of them fails:

1. `TrainingConfigInput` (or the equivalent interface) in `preload.ts`
2. **the object literal passed to `ipcRenderer.invoke`** ← the one that gets forgotten
3. the Main-side validation + the `TrainingConfig` construction in `ipc.handlers.ts`
4. the `Window.fedLearnAPI` declaration in `App.tsx`

`trainingArm` currently has 1, 3 and 4 but not 2, so it type-checks end to end and is discarded at the bridge. If you add a field, grep the invoke object — not the interface — to confirm it actually crosses.

### 2. `electron-log` Cannot Be Used in Sandboxed Preload

The preload comment explicitly notes:
```typescript
// NOTE: electron-log cannot be used in sandboxed preload scripts.
// console.error is forwarded to the main process console automatically.
```

In sandbox mode (`sandbox: true`), the preload runs in a highly restricted environment that doesn't allow native module loading. `electron-log` is a native-adjacent module. Use `console.error` in the preload — Electron forwards it to the Main Process console.

### 3. Removing Listeners on Cleanup

`ipcRenderer.on()` registrations **stack** — calling `onTrainingLog` twice registers two listeners. The `useEffect` cleanup in `App.tsx` calls `removeTrainingLogListener()` (which calls `ipcRenderer.removeAllListeners('docker:training-log')`) to avoid duplicate log entries or memory leaks on re-mount.

### 4. Structured Clone Limitations

`contextBridge` uses structured clone, which means:
- **No functions** can be passed through the bridge as data (they ARE callable as exposed API methods, but cannot be transferred as properties of objects)
- **No class instances** — they become plain objects
- **No Symbols**, **no `undefined` in objects** (becomes missing key)
- **No Error objects with custom properties** — only standard `Error` properties survive

This is why all IPC responses use plain `{ success: boolean, ...data }` objects rather than typed class instances.

---

*Next: [05 — Renderer & Components](./05-renderer-components.md)*  
*Previous: [03 — Main Process Deep Dive](./03-main-process.md)*
