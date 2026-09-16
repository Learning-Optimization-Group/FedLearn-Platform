# FedLearn Desktop — Developer Guide & Contributing

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

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
| Node.js | **24** (`.nvmrc` → `24`; `.tool-versions` → `nodejs 24.4.0`; CI runs 24) | Runtime for Electron and webpack |
| npm | ships with Node 24 | Package management |
| TypeScript | 5.7.3 (pinned in devDeps) | Type checking |
| Docker Engine / Desktop | Latest | **Jetson profile only** |
| Python | 3.10+ floor; the repo pins **3.12.9** | Dev-mode native client fallback |

> The repo-wide pins are `.tool-versions` + `.nvmrc`, and `ci.yml`'s desktop job matches them (Node 24). Note `release-desktop.yml` currently sets `NODE_VERSION: '22'` and `PYTHON_VERSION: '3.11'` — the release path does not match the test path. Any doc claiming "Node.js 18+" or "Node 20" for this unit is stale.

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

# Quality gates (what CI runs)
npm run lint            # ESLint 9 flat config — CI-gated
npm test                # jest
npm run test:coverage   # jest --coverage; coverageThreshold from jest.config.js — CI-gated
npx tsc --noEmit        # NOT gated in CI for this unit — run it yourself

# Package distributable (each runs the native-bundle preflight, then build, then electron-builder)
npm run check:bundle    # Preflight on its own
npm run package         # Current platform, auto-detected
npm run package:mac     # macOS DMG + zip (arm64 only)
npm run package:linux   # Linux AppImage + deb (x64 + arm64)
npm run package:win:cpu   # Windows NSIS (CPU client bundle)
npm run package:win:cuda  # Windows NSIS (CUDA client bundle)
```

---

## Project Conventions

### File Organization

```
src/main/      → Main Process only. Node.js APIs allowed.
               → No React, no DOM.
               → All Docker, auth, backend HTTP and hardware logic lives here.
               → validators.ts must stay electron-free (it is imported by jest).

src/preload/   → Security boundary. One file only.
               → No Node.js APIs (sandbox mode).
               → Validation + contextBridge only.
               → No business logic.

src/shared/    → Imported by BOTH processes AND by jest.
               → No electron, no node-only APIs that the renderer lacks.
               → urlSecurity, evaluateEligibility, deviceCapabilities.types,
                 bundleVariants (the last is also loaded by a plain-node script).

src/renderer/  → Renderer only. No Node.js.
               → Only reach the backend via window.fedLearnAPI.
               → All UI components and React state here.
               → tokens.css is GENERATED — never hand-edit it.
```

### Keep renderer logic pure where you can

The jest suite runs in a **`node` environment with no jsdom/RTL harness**, so `.tsx` components are effectively untestable here (`jest.config.js` says so explicitly, and excludes them from coverage). The established pattern is to extract the logic:

| Component | Pure module | What moved out |
|---|---|---|
| `TrainSection.tsx` | `trainFlow.ts` | phase derivation, readiness rules, formatters |
| `LogPanel.tsx` | `logView.ts` | incremental line cache, filtering, time formatting |
| `TrainSection.tsx` | `runNotifications.ts` | run-transition classification + notification |

New renderer logic that can be a pure function should be one.

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

**`validators.ts`** (this moved out of `ipc.handlers.ts`):
```typescript
export const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> =
  new Set(['discrete', 'jetson', 'cpu', 'mps', 'amd-rocm']);
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

Only if the profile actually uses Docker. The switch's `default` branch now **throws** for any non-`jetson` profile, so a Docker-path profile that isn't handled fails loudly instead of silently building a container.

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
export const HARDWARE_PROFILES: HardwareProfileOption[] = [
  // ... existing profiles
  {
    id: 'amd-rocm',
    label: 'AMD GPU (ROCm)',
    description: 'AMD Radeon GPU with ROCm compute stack. Runs in a Docker container.',
    icon: Cpu,                                    // a lucide component, NOT an emoji string
    dockerConfig: 'Docker container (/dev/kfd, /dev/dri)',
  },
];
```

`dockerConfig` must state how the profile **actually** executes — the `discrete` card claimed `--gpus all` for a long time while the code ran the native client, and `2b02173` had to correct both together. `icon` is a lucide component reference since `74cda60` replaced the emoji set.

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

FedLearn Desktop **does** have an automated test suite, and it is CI-gated. Any claim that it doesn't is stale.

### The jest suite

```bash
npm test               # jest
npm run test:coverage  # jest --coverage — what CI runs
```

`jest.config.js`: `ts-jest` preset, **`testEnvironment: 'node'`**, `testMatch: **/__tests__/**/*.test.ts`, using `tsconfig.test.json`. Electron and its friends are stubbed by `moduleNameMapper` → `src/__mocks__/` (`electron`, `electron-log`, `electron-store` — an in-memory `Map` so `AuthService` needs no disk I/O — plus a CSS stub so `.tsx` files can at least be imported).

22 suites live in `src/__tests__/`, covering: `validators`, `serverUrl`, `docker-service` (Jetson mounts, `getStatus`, native argv), `auth.service`, `httpAuthInterceptor`, `nativeClientHeader`, `clientAuthEnv`, `trainingArmPropagation`, `client-projects.service`, `datasetConsent`, `deviceCapabilities.collector`, `evaluateEligibility`, `bundleVariants`, `generateChecksums`, `updater`, `renderer-csp`, `webpack-app-version`, `trainFlow`, `logView`, `runNotifications`, `sectionsRender`, `appTrainWiring`.

Two structural limits worth knowing before you rely on a green run:

- **No jsdom/RTL harness.** `.tsx` components are excluded from coverage by design and are only exercised indirectly. A component-level regression will not be caught here.
- **The suite tests functions, not wiring.** `trainingArmPropagation.test.ts` calls `buildContainerEnv` and the argv builder directly and passes — while the value never reaches them in the running app, because it is dropped at the preload bridge. If you are adding a field that crosses IPC, a unit test on the receiving function is not evidence that it arrives.

### Coverage gate (TE-11)

`jest.config.js` sets a **modest regression floor**, a few points under the measured baseline (stmts 36.7 / branch 35.0 / funcs 28.9 / lines 36.7):

```javascript
coverageThreshold: { global: { statements: 33, branches: 31, functions: 25, lines: 33 } },
```

It guards against regressions rather than demanding a target. `collectCoverageFrom` is `src/**/*.ts` only — `.tsx` excluded for the reason above.

### Anti-rot gate (TE-10)

CI runs `scripts/check_no_skipped_tests.sh fedlearn-desktop` **before** installing. Jest has no forbid-skip switch, so the script statically rejects `.skip` / `.only` / `xit` / `fit` and friends — a skipped test cannot ride a green run.

### What CI actually runs

```yaml
- run: bash scripts/check_no_skipped_tests.sh fedlearn-desktop
- run: npm ci && npm run lint && npm run test:coverage
```

That's the whole gate. In particular:

- **No `tsc --noEmit`.** `frontend/` and `mobile_client/` both have one; this unit does not. Types are checked only incidentally by `ts-loader` (at build time, which CI never runs for this unit) and by `ts-jest` on test-reachable sources. Run `npx tsc --noEmit` yourself.
- **No `npm run build`.** A webpack-level breakage lands green.
- **No `npm audit`.** See [02 → Dependency Vulnerability Posture](./02-security-model.md#dependency-vulnerability-posture).

A separate, **unfiltered** job runs `scripts/check_design_tokens.sh`, so a hand-edit of the generated `src/renderer/tokens.css` fails the build no matter which unit changed.

### Manual checklist

The suite does not cover the end-to-end flows, so verify these by hand before each release:

### Authentication Tests

```
[ ] Login with valid credentials → shell appears
[ ] Login with invalid credentials → error message shown
[ ] Login with unreachable server → connection failed error
[ ] Logout → auth modal reappears
[ ] App restart → auto-login via saved JWT (if keychain available)
[ ] JWT expiry (24h) → re-auth required on next launch
[ ] Settings: change server URL → new URL persisted after restart
[ ] Settings: change server URL → CURRENT SESSION IS CLEARED and the login screen returns
[ ] Remote http:// URL → refused with the plaintext warning; "Use HTTP anyway" accepts it and keeps the warning
[ ] Loopback http://localhost URL → accepted with no warning
[ ] "Save password" ticked → credentials pre-filled on next launch; unticked → form empty
[ ] Force a 401 mid-session (revoke server-side) → app returns to the login screen, not an opaque error
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
[ ] Docker not running + Jetson profile → actionable error in the log panel
    (NOT a startup banner — the eager ping was removed)
[ ] Non-Jetson profile + Docker not running → no Docker mention anywhere
[ ] Jetson profile + Docker running + image exists → container starts
[ ] Jetson profile + image missing → helpful error naming the build command
[ ] Stop training → container removed
[ ] Start training twice → old container cleaned up before the new one starts
[ ] Quit mid-run → container is stopped and removed, not orphaned (before-quit drain)
```

### Security Tests

```
[ ] Renderer cannot access Node.js: typeof require === 'undefined' in DevTools
[ ] window.fedLearnAPI is the only bridge: no other Node APIs on window
[ ] Dataset path with ../.. → rejected by sanitizeDatasetPath
[ ] A valid directory NOT chosen via the dialog → rejected by the consent allowlist
[ ] Hardware profile 'invalid' → rejected at preload AND at ipc.handlers
[ ] trainingArm 'BOGUS' → start FAILS with a message (never silently downgraded to FULL)
[ ] Log output containing <script> tags → rendered as plain text, not executed
[ ] Packaged build: no 'unsafe-eval' and no 'unsafe-inline' style-src in the shipped <meta> CSP
[ ] Packaged build: console.* stripped from the renderer and preload bundles
```

### Packaging Tests

```
[ ] Preflight fails loudly when the native bundle is absent: npm run check:bundle
[ ] Type check is clean (CI will NOT do this for you): npx tsc --noEmit
[ ] Build completes without errors: npm run build
[ ] Package completes: npm run package:mac (or platform equivalent)
[ ] Produced DMG/EXE can be installed and launched
[ ] App loads from the packaged file:// origin (not localhost)
[ ] JWT auth works in packaged mode
[ ] Native bundle is found at <resources>/fedlearn-client/
[ ] release/SHA256SUMS.txt exists and covers every installer artifact
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

The allowlist is defined separately in `preload.ts` and `validators.ts`. If you add a new profile to one but not the other, calls are rejected at the second validation layer with no obvious error message. The `HardwareProfile` union in `docker.service.ts` is a third copy.

### 3a. Adding a Field to `TrainingConfigInput` Without Adding It to the Invoke Payload

The single most repeatable bug in this codebase's IPC layer, and it type-checks perfectly at every layer.

`preload.ts`'s `startTraining` deliberately **reconstructs** the object it forwards rather than spreading `config`, so an unexpected renderer property cannot ride along. The cost is that a *new* field must be added in two places, not one:

```typescript
export interface TrainingConfigInput {
  …
  trainingArm?: string;        // ← added here…
}

return ipcRenderer.invoke('docker:start-training', {
  …
  strategy: config.strategy,
  // …and NOT added here ⇒ silently dropped at the bridge.
});
```

`trainingArm` is in this state today: declared on the interface, typed in `App.tsx`, populated by `TrainSection`, validated and consumed in Main — and never forwarded, so `--training-arm` / `TRAINING_ARM` is never emitted. A green suite proves nothing here, because the tests call the receiving functions directly. **Grep the invoke object, not the interface.**

### 4. Forgetting `PYTHONUNBUFFERED=1`

Without this env var, Python buffers stdout when not connected to a TTY. Real-time logs will appear only after the buffer fills (8KB by default) or the process ends. The app sets this automatically, but if you spawn a Python process manually for testing, remember to add it.

### 5. Trusting the Jetson `--runtime nvidia` Comment

`docker.service.ts` states that `--runtime nvidia` is "PROHIBITED on Jetson" because it hangs. **That was measured wrong on JetPack 6 and the ban is withdrawn.** On an AGX Orin at L4T R36.5.0 / JetPack 6.2, `docker run --runtime nvidia` worked (7.9 s, `torch.cuda.is_available()` True) while the device-mount path without it failed with `cuInit → 801` and a segfault — and `/dev/nvhost-ctrl`, the first entry in `JETSON_DEVICE_MOUNTS`, does not even exist on that L4T, which makes Docker hard-error.

The original hang was plausibly real on the JetPack 5 / `nvidia-container-runtime` the comment was written against, but that was **not** re-tested (no JetPack 5 hardware available) — treat it as inference. What is measured is that the ban does not hold on JetPack 6.

Do not "fix" either the code or the comment from memory. Read [07 → the correction](./07-hardware-profiles.md#the---runtime-nvidia-prohibition-was-measured-wrong-on-jetpack-6), then verify on the L4T version the target device actually runs.

### 6. Electron Reload Doesn't Reload Main Process

When you modify a Main Process file (`main.ts`, `docker.service.ts`, etc.) and webpack rebuilds it, **Electron does not automatically reload**. You must quit and restart Electron to pick up Main Process changes. Only renderer changes are hot-reloaded.

### 7. `electron-store` Schema Mismatch After Upgrades

If you change the schema of data stored in `electron-store` between versions, the old stored data will fail to parse. The `clearInvalidConfig: true` option handles this gracefully by resetting the store, but users will lose their saved server URL and will need to re-authenticate.

If you make breaking schema changes, consider implementing a migration in the `AuthService` constructor.

### 8. Hand-Editing `src/renderer/tokens.css`

It carries a `GENERATED by design/build-tokens.mjs — DO NOT EDIT` header, and a **CI job that runs on every change to any unit** (`scripts/check_design_tokens.sh`) compares it against `design/tokens.json`. Edit the JSON and regenerate. The same applies to the frontend and mobile token outputs.

The one literal that legitimately lives outside the token system is `backgroundColor: '#F6F3EE'` in `main.ts` — the main process cannot read CSS variables, so that value must be updated by hand on a palette swap.

### 9. Assuming CI Type-Checks This Unit

It does not. There is no `tsc --noEmit` step in the desktop job and CI never runs `npm run build`, so a type error outside test-reachable code, or a webpack-level breakage, can land green. Run `npx tsc --noEmit` and `npm run build` locally before pushing anything structural.

### 10. Re-Mounting `UpdateBanner`

`onUpdateAvailable`, `onUpdateProgress`, `onUpdateDownloaded`, `onUpdateNotAvailable` and `onUpdateError` have **no removal API** in the preload. `UpdateBanner` is therefore mounted once at shell level and never unmounted; moving it inside a section (or anything else that can unmount) stacks listeners with no way to clear them, and each update event then fires N times. The same reasoning is why `updater.ts` guards its registration with a module-level `updaterInitialized` flag — `createWindow()` can run more than once per process on macOS.

---

*Previous: [07 — Hardware Profiles & Training Execution](./07-hardware-profiles.md)*  
*Back to: [Desktop Wiki Index](./README.md)*
