# FedLearn Desktop — Developer Guide & Contributing

> **Part of:** [FedLearn Platform Docs](../../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [First-Time Setup](#first-time-setup)
3. [Development Scripts Reference](#development-scripts-reference)
4. [Project Conventions](#project-conventions)
5. [Adding a New IPC Channel](#adding-a-new-ipc-channel)
6. [Adding a New React Component](#adding-a-new-react-component)
7. [Adding a New Hardware Profile](#adding-a-new-hardware-profile)
8. [Debugging Techniques](#debugging-techniques)
9. [Testing Strategy](#testing-strategy)
10. [Common Mistakes & Gotchas](#common-mistakes--gotchas)

---

## Prerequisites

### Required Software

| Tool | Version | Purpose |
|---|---|---|
| Node.js | 20.x LTS | Runtime for Electron and webpack |
| npm | 10.x | Package management |
| TypeScript | 5.7.x (pinned in devDeps) | Type checking |
| Docker Desktop | Latest | Jetson profile testing |
| Python | 3.10+ | Dev-mode native client fallback |

### Platform-Specific Prerequisites

**macOS:**
```bash
# Xcode Command Line Tools (for native module compilation)
xcode-select --install
```

**Windows:**
```powershell
# Windows Build Tools (for node-gyp native module compilation)
npm install --global windows-build-tools
# Or via Visual Studio Build Tools with C++ workload
```

**Linux:**
```bash
sudo apt install build-essential libsecret-1-dev  # For safeStorage / libsecret
```

### Optional Tools

- **Electron DevTools Extensions:** React DevTools, Redux DevTools
- **Docker:** Required only for Jetson profile testing
- **nvidia-smi:** Required on CUDA machines for hardware detection

---

## First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/anurag2796/FedLearn-Platform.git
cd FedLearn-Platform/fedlearn-desktop

# 2. Install dependencies
npm install

# 3. Start the development environment
npm run dev
```

The `npm run dev` command starts four processes:
- Webpack watcher for Main Process
- Webpack watcher for Preload Script
- Webpack Dev Server for Renderer (port 9000, with HMR)
- Electron (after 3-second delay for initial compile)

### First Launch

On first launch the app will:
1. Show a loading spinner (auth check)
2. Show the login screen (no saved auth)
3. Default server URL: `http://localhost:8081`

Point the app at a running FedLearn backend server, or start the Spring Boot backend locally.

---

## Development Scripts Reference

```bash
# Start full dev environment (4 concurrent processes)
npm run dev

# Build individual targets (dev mode)
npm run dev:main        # Webpack watch — Main Process
npm run dev:preload     # Webpack watch — Preload Script
npm run dev:renderer    # Webpack Dev Server — Renderer (port 9000, HMR)
npm run dev:electron    # Start Electron (requires built main.js)

# Build for production
npm run build           # All targets via webpack.prod.config.js

# Build individual targets (production mode)
npm run build:main      # Main Process only
npm run build:preload   # Preload only
npm run build:renderer  # Renderer only

# Package distributable
npm run package         # Current platform, auto-detected
npm run package:mac     # macOS DMG (arm64 + x64)
npm run package:linux   # Linux AppImage + deb
npm run package:win:cpu   # Windows NSIS (CPU variant)
npm run package:win:cuda  # Windows NSIS (CUDA variant)
```

---

## Project Conventions

### File Organization

```
src/main/      → Main Process only. Node.js APIs allowed.
               → No React, no DOM.
               → All Docker, auth, and hardware logic lives here.

src/preload/   → Security boundary. One file only.
               → No Node.js APIs (sandbox mode).
               → Validation + contextBridge only.
               → No business logic.

src/renderer/  → Renderer only. No Node.js.
               → Only access backend via window.fedLearnAPI.
               → All UI components and React state here.
```

### Naming Conventions

| Item | Convention | Example |
|---|---|---|
| IPC channels | `domain:action` | `docker:start-training`, `auth:login` |
| Component files | PascalCase | `HardwareSelector.tsx` |
| Service files | camelCase + `.service.ts` | `docker.service.ts` |
| Utility files | camelCase + `.ts` | `hardware.probe.ts` |
| CSS classes | kebab-case | `.profile-card-active` |
| IDs on interactive elements | kebab-case | `#start-training-button` |

### TypeScript Rules

- Use `unknown` for IPC inputs, not `any`. Then narrow with type guards.
- Use `ReadonlySet<string>` for allowlists (prevents accidental mutation).
- All async functions must handle errors with try/catch and log them.
- Never use `as Type` without a prior type guard validation.

### Error Handling Pattern

```typescript
// Standard pattern for IPC handlers
ipcMain.handle('domain:action', async (_event, arg: unknown) => {
  try {
    // 1. Validate input
    if (!isValid(arg)) {
      log.error('[IPC:domain:action] Validation failed');
      return { success: false, error: 'Validation error message' };
    }
    
    // 2. Execute business logic
    const result = await service.doSomething(arg as ValidatedType);
    
    // 3. Return success
    return { success: true, data: result };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    log.error(`[IPC:domain:action] Failed: ${message}`);
    return { success: false, error: message };
  }
});
```

---

## Adding a New IPC Channel

Follow this checklist every time you add a new IPC capability:

### Step 1: Define the Main Process Handler (`ipc.handlers.ts`)

```typescript
// In registerIpcHandlers():
ipcMain.handle('myfeature:do-something', async (_event, input: unknown) => {
  try {
    // Validate input
    if (typeof input !== 'string' || input.length === 0) {
      log.error('[IPC:myfeature:do-something] Invalid input');
      return { success: false, error: 'Invalid input' };
    }
    
    // Execute
    const result = await someService.doSomething(input);
    return { success: true, result };
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    log.error(`[IPC:myfeature:do-something] Failed: ${message}`);
    return { success: false, error: message };
  }
});
```

### Step 2: Expose via Preload Bridge (`preload.ts`)

```typescript
// In contextBridge.exposeInMainWorld('fedLearnAPI', { ... }):
doSomething: async (input: string): Promise<{ success: boolean; result?: string; error?: string }> => {
  // Preload validation
  if (typeof input !== 'string' || input.length === 0 || input.length > 256) {
    console.error('[Preload:Validation] doSomething: invalid input');
    return { success: false, error: 'Invalid input' };
  }
  return ipcRenderer.invoke('myfeature:do-something', input);
},
```

### Step 3: Update TypeScript Declaration (`App.tsx` or `global.d.ts`)

```typescript
declare global {
  interface Window {
    fedLearnAPI: {
      // ... existing methods ...
      doSomething: (input: string) => Promise<{ success: boolean; result?: string; error?: string }>;
    };
  }
}
```

### Step 4: Update the IPC Channel Reference Table

Add the new channel to the table in `03-main-process.md`:
```markdown
| `myfeature:do-something` | invoke → handle | `myService.doSomething()` | Description of what it does |
```

### Step 5: Log Registrations

Add the new channel name to the registration log at the bottom of `registerIpcHandlers`:
```typescript
log.info('[IPC] All handlers registered: ..., myfeature:do-something');
```

---

## Adding a New React Component

### File Template

```typescript
// src/renderer/components/MyComponent.tsx
// =============================================================================
// FedLearn Desktop — MyComponent
// =============================================================================
// Brief description of what this component does.
// =============================================================================

import React, { useState, useCallback } from 'react';

interface MyComponentProps {
  onAction: (data: string) => void;
  isDisabled?: boolean;
}

const MyComponent: React.FC<MyComponentProps> = ({ onAction, isDisabled = false }) => {
  const [value, setValue] = useState('');

  const handleClick = useCallback(() => {
    if (!value.trim()) return;
    onAction(value);
  }, [value, onAction]);

  return (
    <div className="my-component">
      <input
        id="my-component-input"          // Unique ID for each interactive element
        className="form-input"
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={isDisabled}
      />
      <button
        id="my-component-button"
        className="btn btn-primary"
        onClick={handleClick}
        disabled={isDisabled}
        type="button"
      >
        Action
      </button>
    </div>
  );
};

export default MyComponent;
```

### CSS in `styles.css`

Add component styles under a clearly labeled section:
```css
/* ==================== MyComponent ==================== */
.my-component {
  display: flex;
  gap: var(--spacing-sm);
}
```

### Rules for New Components

1. **Never call `window.fedLearnAPI` directly in a leaf component** — pass callbacks down from `App.tsx`
2. **Give every interactive element a unique `id`** (for accessibility and testing)
3. **Wrap event handlers in `useCallback`** when they're passed as props
4. **No `dangerouslySetInnerHTML`** — never, for any reason
5. **No `innerHTML`** assignment — never, for any reason

---

## Adding a New Hardware Profile

To add a new hardware profile (e.g., `amd-rocm`):

### 1. Update the allowlist (3 places)

**`preload.ts`:**
```typescript
const ALLOWED_HARDWARE_PROFILES = ['discrete', 'jetson', 'cpu', 'mps', 'amd-rocm'] as const;
```

**`ipc.handlers.ts`:**
```typescript
const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> = new Set(['discrete', 'jetson', 'cpu', 'mps', 'amd-rocm']);
```

**`docker.service.ts`:**
```typescript
export type HardwareProfile = 'discrete' | 'jetson' | 'cpu' | 'mps' | 'amd-rocm';
```

### 2. Add routing logic in `DockerService.startTraining()`

```typescript
async startTraining(config: TrainingConfig): Promise<void> {
  if (config.hardwareProfile === 'jetson') {
    await this.startDockerTraining(config);
    return;
  }
  if (config.hardwareProfile === 'amd-rocm') {
    await this.startDockerTraining(config);  // or startNativeProcess
    return;
  }
  await this.startNativeProcess(config);
}
```

### 3. Add the HostConfig case in `startDockerTraining()`

```typescript
switch (config.hardwareProfile) {
  case 'amd-rocm':
    // ROCm requires /dev/kfd and /dev/dri
    hostConfig.Devices = [
      { PathOnHost: '/dev/kfd', PathInContainer: '/dev/kfd', CgroupPermissions: 'rwm' },
    ];
    hostConfig.Binds = [...(hostConfig.Binds || []), '/dev/dri:/dev/dri'];
    break;
  // ... existing cases
}
```

### 4. Add the profile card to `HardwareSelector.tsx`

```typescript
const HARDWARE_PROFILES: HardwareProfileOption[] = [
  // ... existing profiles
  {
    id: 'amd-rocm',
    label: 'AMD GPU (ROCm)',
    description: 'AMD Radeon GPU with ROCm compute stack.',
    icon: '🔴',
    dockerConfig: 'Devices: /dev/kfd, /dev/dri',
  },
];
```

### 5. Update hardware detection if applicable

Update `hardware.probe.ts` to detect AMD GPUs and set `recommendedProfile = 'amd-rocm'` when appropriate.

---

## Debugging Techniques

### Main Process Debugging

Add `--inspect` flag to enable Node.js inspector:
```bash
# In package.json dev:electron script, or directly:
electron --inspect=5858 dist/main/main.js
```

Then in Chrome/Edge: navigate to `chrome://inspect` and connect to `localhost:5858`.

### Renderer Debugging

DevTools are available in dev mode. Press `Cmd+Option+I` (macOS) or `Ctrl+Shift+I` (Windows/Linux) to open.

The renderer's `window.fedLearnAPI` is inspectable in the console:
```javascript
// In DevTools console:
window.fedLearnAPI.checkAuth().then(console.log)
// { success: true, authenticated: true }

window.fedLearnAPI.detectHardware().then(console.log)
// { success: true, detection: { platform: 'darwin', arch: 'arm64', ... } }
```

### IPC Logging

All IPC calls are logged in the Main Process via electron-log:
```
[IPC:docker:start-training] Starting training with profile=mps, project=my-project, model=CNN
[IPC:auth:login] Called for user: admin
[IPC:hardware:detect] Called
```

Log file location:
- **macOS:** `~/Library/Logs/FedLearn Desktop/main.log`
- **Windows:** `%APPDATA%\FedLearn Desktop\logs\main.log`
- **Linux:** `~/.config/FedLearn Desktop/logs/main.log`

### Preload Validation Errors

Preload validation errors go to `console.error`, which is forwarded to the Main Process log:
```
[Preload:Validation] Rejected hardware profile not in allowlist: "invalid-profile"
```

If a call to `window.fedLearnAPI.startTraining()` returns `{ success: false }` unexpectedly, check the Main Process log for `[Preload:Validation]` messages.

### Docker Container Debugging

```bash
# Check if the training container is running
docker ps | grep fedlearn-training-client

# View container logs directly (bypass the app)
docker logs fedlearn-training-client

# Inspect container state
docker inspect fedlearn-training-client | jq '.[0].State'

# Execute a shell in the running container
docker exec -it fedlearn-training-client /bin/bash
```

---

## Testing Strategy

FedLearn Desktop does not currently have an automated test suite. The following manual testing checklist should be verified before each release:

### Authentication Tests

```
[ ] Login with valid credentials → dashboard appears
[ ] Login with invalid credentials → error message shown
[ ] Login with unreachable server → connection failed error
[ ] Logout → auth modal reappears
[ ] App restart → auto-login via saved JWT (if keychain available)
[ ] JWT expiry (24h) → re-auth required on next launch
[ ] Settings: change server URL → new URL persisted after restart
```

### Training Tests (Native Path)

```
[ ] MPS profile (Apple Silicon Mac):
    [ ] Start training → logs appear in real-time
    [ ] Stop training → process terminated
    [ ] Status transitions: pulling → running → completed/error
    [ ] Log panel auto-scroll behavior

[ ] CPU profile (any machine):
    [ ] Start training → logs appear
    [ ] Stop → SIGTERM sent, then SIGKILL after 5s if not stopped
```

### Training Tests (Docker Path — Jetson)

```
[ ] Docker not running → warning banner appears
[ ] Jetson profile + Docker running + image exists → container starts
[ ] Jetson profile + image missing → helpful error in log panel
[ ] Stop training → container removed
[ ] Start training twice → old container cleaned up before new one starts
```

### Security Tests

```
[ ] Renderer cannot access Node.js: typeof require === 'undefined' in DevTools
[ ] window.fedLearnAPI is the only bridge: no other Node APIs on window
[ ] Dataset path with ../.. → rejected by ipc.handlers.ts
[ ] Hardware profile 'invalid' → rejected at preload + ipc.handlers
[ ] Log output containing <script> tags → rendered as plain text, not executed
```

### Packaging Tests

```
[ ] Build completes without errors: npm run build
[ ] Package completes: npm run package:mac (or platform equivalent)
[ ] Produced DMG/EXE can be installed and launched
[ ] App loads from packaged file:// origin (not localhost)
[ ] JWT auth works in packaged mode
[ ] Native bundle is found at <resources>/fedlearn-client/
```

---

## Common Mistakes & Gotchas

### 1. Calling `window.fedLearnAPI` Before It's Initialized

The preload script runs synchronously before the DOM is ready, so `window.fedLearnAPI` is available on page load. However, if you call it in the module body (outside of a component/effect), it may not be defined yet:

```typescript
// ❌ BAD — may run before contextBridge populates
const result = window.fedLearnAPI.checkAuth();

// ✅ GOOD — runs after React mounts
useEffect(() => {
  window.fedLearnAPI.checkAuth().then(/* ... */);
}, []);
```

### 2. Not Cleaning Up IPC Listeners

Calling `onTrainingLog` multiple times (e.g., on repeated login/logout cycles) stacks listeners:

```typescript
// ❌ BAD — listener accumulates on each login
useEffect(() => {
  window.fedLearnAPI.onTrainingLog(callback);
}, [isAuthenticated]);

// ✅ GOOD — cleanup removes listener on dependency change
useEffect(() => {
  if (!isAuthenticated) return;
  window.fedLearnAPI.onTrainingLog(callback);
  return () => window.fedLearnAPI.removeTrainingLogListener();
}, [isAuthenticated]);
```

### 3. Modifying `ALLOWED_HARDWARE_PROFILES` in Only One Place

The allowlist is defined separately in `preload.ts` and `ipc.handlers.ts`. If you add a new profile to one but not the other, calls will be rejected at the second validation layer with no obvious error message.

### 4. Forgetting `PYTHONUNBUFFERED=1`

Without this env var, Python buffers stdout when not connected to a TTY. Real-time logs will appear only after the buffer fills (8KB by default) or the process ends. The app sets this automatically, but if you spawn a Python process manually for testing, remember to add it.

### 5. Using `--runtime nvidia` on Jetson

This is explicitly documented in the code but still a common mistake when working with Docker + NVIDIA:
```typescript
// ❌ WRONG for Jetson — hangs indefinitely
hostConfig.Runtime = 'nvidia';

// ✅ CORRECT for Jetson
hostConfig.Devices = JETSON_DEVICE_MOUNTS;
```

### 6. Electron Reload Doesn't Reload Main Process

When you modify a Main Process file (`main.ts`, `docker.service.ts`, etc.) and webpack rebuilds it, **Electron does not automatically reload**. You must quit and restart Electron to pick up Main Process changes. Only renderer changes are hot-reloaded.

### 7. `electron-store` Schema Mismatch After Upgrades

If you change the schema of data stored in `electron-store` between versions, the old stored data will fail to parse. The `clearInvalidConfig: true` option handles this gracefully by resetting the store, but users will lose their saved server URL and will need to re-authenticate.

If you make breaking schema changes, consider implementing a migration in the `AuthService` constructor.

---

*Previous: [07 — Hardware Profiles & Training Execution](./07-hardware-profiles.md)*  
*Back to: [Desktop Wiki Index](./README.md)*
