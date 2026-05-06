# FedLearn Desktop — Preload Script & IPC Bridge

> **Part of:** [FedLearn Platform Docs](../../README.md) → [Desktop Wiki](./README.md)

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
  startTraining:             async (config)           => { /* ... */ },
  stopTraining:              async ()                 => { /* ... */ },
  getDockerStatus:           async ()                 => { /* ... */ },
  login:                     async (username, password) => { /* ... */ },
  logout:                    async ()                 => { /* ... */ },
  checkAuth:                 async ()                 => { /* ... */ },
  onTrainingLog:             (callback)               => { /* ... */ },
  removeTrainingLogListener: ()                       => { /* ... */ },
  onDockerUnavailable:       (callback)               => { /* ... */ },
  setServerUrl:              async (url)              => { /* ... */ },
  getServerUrl:              async ()                 => { /* ... */ },
  selectDatasetPath:         async ()                 => { /* ... */ },
  detectHardware:            async ()                 => { /* ... */ },
});
```

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
const MAX_STRING_LENGTH = 256;
```

All patterns use **anchored regex** (both `^` and `$`) to prevent partial matches. Every pattern includes an explicit **maximum length** to prevent resource exhaustion from arbitrarily long strings.

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
  // (Main has access to fs.statSync; preload doesn't in sandbox mode)
  if (val.length === 0 || val.length > 2048) { /* ... */ return false; }
  return true;
}
```

**Why does `isValidDatasetPath` only check length, not the actual path?** Because `preload.ts` runs in the Electron sandboxed renderer context — it has access to `ipcRenderer` but NOT to `fs`. The full path sanitization (resolve, stat, directory check) happens in Main where `fs` is available. The preload's job is only to block obviously invalid inputs.

### The `startTraining` Validation Flow

```typescript
startTraining: async (config: TrainingConfigInput) => {
  // Validate all 6 fields before ANY ipcRenderer call
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
  });
},
```

**Why reconstruct the object instead of spreading?** Spreading `{ ...config }` would forward any extra properties on the `config` object to Main. While Main validates the expected fields, it's cleaner to be explicit about what gets sent.

---

## Complete `window.fedLearnAPI` Reference

### `startTraining(config)`

Starts a federated learning training session.

```typescript
startTraining(config: {
  hardwareProfile: 'discrete' | 'jetson' | 'cpu' | 'mps';
  projectId: string;       // Pattern: /^[a-zA-Z0-9_-]{1,128}$/
  serverAddress: string;   // Pattern: /^[a-zA-Z0-9._:/-]{1,256}$/
  partitionId: string;     // Pattern: /^[0-9]{1,10}$/
  modelType: string;       // Pattern: /^[a-zA-Z0-9_\-\.]{1,128}$/
  datasetPath: string;     // Existing directory on the local filesystem
}): Promise<{ success: boolean; error?: string }>
```

**Returns:**
- `{ success: true }` — training started successfully
- `{ success: false, error: string }` — validation failed or Docker error

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

### `onDockerUnavailable(callback)`

Registers a callback for when the Docker daemon is unreachable. Fired by Main if the initial ping fails.

```typescript
onDockerUnavailable(callback: (message: string) => void): void
```

---

### `setServerUrl(url)`

Sets the backend server URL. The URL is validated (must start with `http://` or `https://`) and `/api` is appended automatically if not present.

```typescript
setServerUrl(url: string): Promise<{
  success: boolean;
  url?: string;    // The normalized URL that was saved
  error?: string;
}>
```

**Example:**
```typescript
await window.fedLearnAPI.setServerUrl('http://192.168.1.50:8081');
// Stored as: 'http://192.168.1.50:8081/api'
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

## Event-Based IPC (Push Channels)

Unlike `ipcRenderer.invoke` (request-response), some IPC communication is **unidirectional push** from Main to Renderer. This is used for real-time log streaming where efficiency matters.

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
      }) => Promise<{ success: boolean; error?: string }>;
      
      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (username: string, password: string) => Promise<{ success: boolean }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean }>;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
      onDockerUnavailable: (callback: (message: string) => void) => void;
      setServerUrl: (url: string) => Promise<{ success: boolean; url?: string; error?: string }>;
      getServerUrl: () => Promise<{ success: boolean; url?: string }>;
      selectDatasetPath: () => Promise<{ success: boolean; path?: string; error?: string }>;
      detectHardware: () => Promise<{
        success: boolean;
        detection?: {
          platform: string;
          arch: string;
          recommendedProfile: string;
          nativeBundleAvailable: boolean;
          cudaAvailable: boolean;
          cudaInfo?: string;
        };
        error?: string;
      }>;
    };
  }
}
```

This declaration should ideally live in a separate `src/renderer/global.d.ts` file for cleaner separation, but is currently co-located in `App.tsx`.

> **Important:** Keep this declaration in sync with the actual `preload.ts` implementation. Drift between the two will only be caught at runtime.

---

## Common Pitfalls

### 1. Adding a New IPC Channel — Three Files to Update

Adding any new capability requires updating **three files** in a specific order:

1. **`ipc.handlers.ts`** — Register `ipcMain.handle('new:channel', ...)` with validation
2. **`preload.ts`** — Add a typed wrapper to `contextBridge.exposeInMainWorld` with validation
3. **`App.tsx` (or `global.d.ts`)** — Add the TypeScript declaration to `Window.fedLearnAPI`

Missing any one of these causes either a silent failure (no channel registered), a TypeScript error (missing type), or a security gap (no preload validation).

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
