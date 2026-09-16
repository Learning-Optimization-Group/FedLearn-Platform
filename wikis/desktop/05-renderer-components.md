# FedLearn Desktop — Renderer & React Components

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Renderer Architecture Overview](#renderer-architecture-overview)
2. [App.tsx — Shell & State](#apptsx--shell--state)
3. [AuthModal — Login & Server Configuration](#authmodal--login--server-configuration)
4. [TrainSection — The Guided Training Flow](#trainsection--the-guided-training-flow)
5. [HardwareProfilePicker — Profile Cards](#hardwareprofilepicker--profile-cards)
6. [LogPanel — Real-Time Log Stream](#logpanel--real-time-log-stream)
7. [ModelPlayground — "Use a Model"](#modelplayground--use-a-model)
8. [SettingsSection — Server, Updates, About](#settingssection--server-updates-about)
9. [StatusBar & StatusIndicator](#statusbar--statusindicator)
10. [UpdateBanner — Auto-Update Layer](#updatebanner--auto-update-layer)
11. [Styles Architecture](#styles-architecture)
12. [Performance Patterns](#performance-patterns)

---

## Renderer Architecture Overview

The renderer is a single-page React application mounted in the Electron Chromium window. It communicates exclusively through `window.fedLearnAPI` (provided by the preload bridge).

```
src/renderer/
├── index.tsx          ← React root (createRoot, StrictMode)
├── index.html         ← HTML shell; CSP <meta> baked in at build time
├── tokens.css         ← GENERATED from design/tokens.json — do not edit
├── fonts.css          ← Self-hosted @fontsource imports (Ledger typefaces)
├── styles.css         ← Global styles; imports tokens.css then fonts.css
├── client.types.ts    ← ClientProject / ProjectConnection
├── inference.types.ts ← InferableModel / InferenceResult / GenerationResult
├── App.tsx            ← Shell: rail, section outlet, StatusBar, all IPC wiring
└── components/
    ├── AuthModal.tsx         ← Login (shown before authentication)
    ├── TrainSection.tsx      ← Guided setup ⇄ running-run layout
    ├── trainFlow.ts          ← Pure phase / readiness / formatting logic
    ├── HardwareSelector.tsx  ← HardwareProfilePicker (controlled card grid)
    ├── LogPanel.tsx          ← Filtered, severity-coloured, follow-tail log view
    ├── logView.ts            ← Incremental log-line cache + filter (pure)
    ├── runNotifications.ts   ← Run completed/failed notifications (pure)
    ├── ModelPlayground.tsx   ← Inference + chat
    ├── SettingsSection.tsx   ← Server URL, updates, about
    ├── StatusBar.tsx         ← Persistent bottom strip
    ├── StatusIndicator.tsx   ← Status badge (rendered inside StatusBar)
    ├── UpdateBanner.tsx      ← Auto-update layer
    └── sections.css          ← Section-scoped styles
```

Three of these are deliberately **React-free and DOM-free** — `trainFlow.ts`, `logView.ts`, `runNotifications.ts`. The jest suite runs in a `node` environment with no jsdom/RTL harness, so extracting the logic is what makes it testable at all (`trainFlow.test.ts`, `logView.test.ts`, `runNotifications.test.ts`, `sectionsRender.test.ts`). Keep new renderer logic that can be pure, pure.

### Application States

```
App start
    │
    ▼
isAuthChecking = true                       → loading spinner
    │
    ▼  (checkAuth IPC call resolves)
    ├── authenticated = false → <AuthModal />
    │
    └── authenticated = true  → Shell
                                  ├── shell-titlebar  (drag strip; macOS inset)
                                  ├── shell-update-layer → <UpdateBanner />
                                  ├── shell-body
                                  │     ├── rail  (Train · Models · Settings · account)
                                  │     └── shell-outlet
                                  │           ├── <TrainSection />     (Cmd/Ctrl+1)
                                  │           ├── <ModelPlayground />  (Cmd/Ctrl+2)
                                  │           └── <SettingsSection />  (Cmd/Ctrl+3)
                                  └── <StatusBar />
```

**All three sections stay mounted.** The outlet toggles `shell-section-hidden` with CSS rather than unmounting, so the Model Playground's chat thread and streaming state, and the training log buffer, survive a section switch. `UpdateBanner` sits above the outlet at shell level and never unmounts, because its preload listeners have no removal API.

The account item in the rail shows a generic "Signed in" identity: the frozen preload surface exposes no username readback, and adding one purely for a label would mean a new IPC channel.

---

## App.tsx — Shell & State

`App.tsx` is the root of the application. It owns all shared state and every IPC subscription; the sections receive data and callbacks as props.

### State

```typescript
const [isAuthenticated, setIsAuthenticated] = useState(false);
const [isAuthChecking, setIsAuthChecking] = useState(true);        // loading spinner
const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
const [logs, setLogs] = useState<string[]>([]);                    // log buffer
const [section, setSection] = useState<Section>('train');          // 'train' | 'models' | 'settings'
const [serverHost, setServerHost] = useState('');                  // StatusBar readout
const [hardwareLabel, setHardwareLabel] = useState('');            // StatusBar chip
const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);// label + startedAt
```

### ContainerStatus Type

```typescript
// Shared type — imported by StatusIndicator and StatusBar
export type ContainerStatus =
  | 'idle'        // No training running
  | 'pulling'     // Image pull or process startup in progress
  | 'running'     // Container/process actively training
  | 'completed'   // Training finished successfully
  | 'error'       // Training failed
  | 'restarting'  // Docker restart policy triggered
  | 'paused'      // Container paused
  | 'stopped';    // Manually stopped
```

`trainFlow.ts` declares a structurally identical `TrainRunStatus` rather than importing this one — `TrainSection` must not import from `App.tsx` (the shell owns that file), so the contract is duplicated and stays assignable in both directions.

### Startup Effects

```typescript
// Effect 1: Auth check on startup
useEffect(() => {
  const checkAuth = async () => {
    try {
      const result = await window.fedLearnAPI.checkAuth();
      setIsAuthenticated(result.authenticated === true);
    } catch {
      setIsAuthenticated(false);
    } finally {
      setIsAuthChecking(false);  // Always exit the loading state
    }
  };
  checkAuth();
}, []);

// Effect 2: Session expiry (DE-8) — registered UNCONDITIONALLY on mount
useEffect(() => {
  window.fedLearnAPI.onSessionExpired(() => {
    setIsAuthenticated(false);
    setLogs([]); setContainerStatus('idle'); setSection('train');
    setActiveRun(null); setServerHost(''); setHardwareLabel('');
  });
  return () => window.fedLearnAPI.removeSessionExpiredListener();
}, []);
```

The `finally` block in Effect 1 is critical — even if `checkAuth()` throws, `isAuthChecking` must become `false` or the app is stuck on the spinner forever.

Effect 2's empty dependency array is equally deliberate. Unlike the log and status subscriptions below, it is **not** gated on `isAuthenticated`: it is exactly what detects the authenticated → expired transition, so gating it on the state it exists to change would make it dead code. Without it a 401 mid-session leaves the dashboard up showing opaque per-call "Not authenticated" errors.

Two further advisory effects populate the StatusBar — `detectHardware()` for the hardware chip and `getServerUrl()` for the host readout. Both are best-effort: a failure just leaves the chip hidden or shows "Not connected". The server-URL effect re-runs on `section` change so a URL saved in Settings is reflected when navigating away.

### Keyboard Section Switching

```typescript
useEffect(() => {
  if (!isAuthenticated) return;
  const onKeyDown = (e: KeyboardEvent) => {
    if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
    const target: Section | null =
      e.key === '1' ? 'train' : e.key === '2' ? 'models' : e.key === '3' ? 'settings' : null;
    if (target) { e.preventDefault(); setSection(target); }
  };
  window.addEventListener('keydown', onKeyDown);
  return () => window.removeEventListener('keydown', onKeyDown);
}, [isAuthenticated]);
```

A pure renderer listener — no menu accelerators, no IPC. `main.ts` keeps the application menu to standard roles only precisely so no menu item needs a channel to reach the renderer.

### Log Batching with requestAnimationFrame

Container output can arrive at very high rates (hundreds of lines per second during model training). Without batching, each incoming log line would trigger a `setState` → reconcile → re-render cycle.

The solution: batch all logs received within a single animation frame into one `setState`:

```typescript
const logBufferRef = useRef<string[]>([]);
const rafIdRef = useRef<number | null>(null);

useEffect(() => {
  if (!isAuthenticated) return;

  window.fedLearnAPI.onTrainingLog((logLine: string) => {
    logBufferRef.current.push(logLine);              // 1. buffer (no re-render)

    if (rafIdRef.current === null) {                 // 2. schedule one flush
      rafIdRef.current = requestAnimationFrame(() => {
        const batch = logBufferRef.current;
        logBufferRef.current = [];
        rafIdRef.current = null;

        setLogs((prev) => {                          // 3. one setState per frame
          const merged = [...prev, ...batch];
          return merged.length > MAX_LOG_LINES       // 4. cap at 10K entries
            ? merged.slice(merged.length - MAX_LOG_LINES)
            : merged;
        });
      });
    }
  });

  return () => {
    window.fedLearnAPI.removeTrainingLogListener();
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
  };
}, [isAuthenticated]);
```

**Why `useRef` instead of `useState` for the buffer?** `useRef` mutations don't trigger re-renders. A `useState` buffer would re-render on every `push`, defeating the batching entirely.

**Why `MAX_LOG_LINES = 10_000`?** Each entry can be several hundred characters. Without the cap, a training run lasting hours accumulates unbounded string data. Note this caps *buffer entries*, not rendered DOM lines — `LogPanel` applies its own, much lower, render cap.

### Status Polling

```typescript
useEffect(() => {
  if (!isAuthenticated) return;
  const pollStatus = async () => {
    try {
      const result = await window.fedLearnAPI.getDockerStatus();
      if (result.success && result.status) setContainerStatus(result.status as ContainerStatus);
    } catch { /* Silently ignore polling failures */ }
  };
  pollStatus();                                    // Immediate first poll
  const interval = setInterval(pollStatus, 3000);  // Then every 3s
  return () => clearInterval(interval);
}, [isAuthenticated]);
```

### Event Handlers

```typescript
const handleStartTraining = useCallback(async (config) => {
  // Anchor the elapsed timer at the moment Start was pressed — NEVER derived
  // from the 3s status poll. Label falls back to the model type until the
  // project name resolves.
  setActiveRun({ projectLabel: config.modelType || config.projectId.slice(0, 8), startedAt: Date.now() });
  window.fedLearnAPI.listTrainableProjects()
    .then((result) => {
      const project = result.projects?.find((p) => p.projectId === config.projectId);
      if (project) setActiveRun((prev) => (prev ? { ...prev, projectLabel: project.name } : prev));
    })
    .catch(() => { /* keep the model-type fallback label */ });

  setLogs([]);
  setContainerStatus('pulling');                   // Optimistic
  setLogs((prev) => [...prev, '[System] Starting training container...\n']);

  const result = await window.fedLearnAPI.startTraining(config);
  if (!result.success) {
    setContainerStatus('error');
    setLogs((prev) => [...prev, `[System] Failed to start: ${result.error || 'Unknown error'}\n`]);
  }
  // Success: the polling effect picks up 'running' within 3 s
}, []);
```

`handleStartTraining` uses **optimistic updates** — status becomes `'pulling'` before the IPC call resolves, so the user gets immediate feedback rather than waiting up to 3 s for the next poll. Anchoring `startedAt` at the click, rather than at the first poll that reports `running`, keeps the elapsed clock honest.

`handleLogout` clears the same six pieces of state the session-expiry handler does.

---

## AuthModal — Login & Server Configuration

`AuthModal` is rendered as a full-screen overlay when the user is not authenticated. It is the only place the auth gate lives — the shell is not rendered at all until `isAuthenticated`.

### Component State

```typescript
const [serverUrl, setServerUrl] = useState('http://localhost:8081');
const [username, setUsername] = useState('');
const [password, setPassword] = useState('');
const [error, setError] = useState('');
const [isLoading, setIsLoading] = useState(false);
const [showServerConfig, setShowServerConfig] = useState(false);  // collapsible section
const [serverSaved, setServerSaved] = useState(false);            // ✓ confirmation
const [insecureWarning, setInsecureWarning] = useState('');       // DE-13 refusal/override text
const [allowInsecure, setAllowInsecure] = useState(false);        // scoped to the current URL
const [showPassword, setShowPassword] = useState(false);          // eye toggle
const [savePassword, setSavePassword] = useState(false);          // "Save password" opt-in
```

### Mount Effects

Two independent effects run on mount:

1. **Server URL** — `getServerUrl()`, with the `/api` suffix stripped for display (`result.url.replace(/\/api$/, '')`). Settings deliberately shows the raw stored value *with* `/api`; the login screen strips it for a cleaner first-run experience.
2. **Saved credentials** — `getSavedCredentials()`. When a blob exists, username and password are pre-filled and the "Save password" checkbox is pre-checked. Nothing stored ⇒ the form stays empty.

### Insecure-Transport Flow (DE-13)

The login screen must save the server URL before it can log in, so it is the first place the plaintext-HTTP policy bites. `AuthModal` imports `isPlaintextRemoteUrl` from `src/shared/urlSecurity.ts` — the same module Main's policy uses — so the on-screen copy and the actual decision cannot drift.

The flow: save → if Main returns `code: 'INSECURE_HTTP'`, show the refusal plus a "Use HTTP anyway" button → on click, re-save with `{ allowInsecureHttp: true }` and keep the returned `warning` visible. Editing the URL field resets `allowInsecure`, so an acknowledgement never carries over to a different host.

### Login Flow

Server URL is saved first (in case the user changed it before clicking Sign In), then `login(username, password)` is called. On success, the "Save password" checkbox drives `saveCredentials(...)` or `clearSavedCredentials()`. `onLoginSuccess()` notifies the parent, which flips `isAuthenticated`.

Failure returns `{ success: false }` with no detail — the backend's reason is deliberately not surfaced, to avoid credential enumeration.

### UX Design Choices

- **Collapsible server config** — hidden under a toggle by default; most users never change it.
- **Server URL shown in the toggle label** — even collapsed, the user can confirm which backend they are about to authenticate against.
- **Save confirmation** — a `✓` replaces the Save button text briefly after saving.
- **Show/hide password** — an eye toggle (`Eye` / `EyeOff`), added with the save-password opt-in in `46aea4d`.
- **Loading state disables all inputs** — prevents double-submit during the async login.

---

## TrainSection — The Guided Training Flow

`TrainSection` replaced the old composite `HardwareSelector` (profile cards + project picker + dataset row + start/stop buttons in one form). It is a **two-state flow keyed on the run status `App` already owns**, and it derives everything from existing renderer state and existing preload IPC — no new channels, no new `App` state.

```
SETUP    (idle | stopped | completed | error)
         One guided card: model picker → dataset folder (choose or explicitly
         skip) → detected-hardware chip with a details disclosure → an Advanced
         disclosure hiding the profile override → readiness checklist → a single
         primary Start button, enabled when nothing is pending or blocked.
         Completed/error additionally show an outcome banner ("Run again"
         becomes the primary) and keep the previous run's log alongside.

RUNNING  (pulling | running | restarting | paused)
         Logs are the dominant surface: a compact run header (project, phase,
         hardware, elapsed, Stop) over a full-height LogPanel.
```

The `status` prop carries the **full** `ContainerStatus`, not a collapsed boolean. A boolean would make the completed/error banners and the finish notifications unreachable, and would show the setup card mid-run during `restarting`/`paused`. (`isRunning` survives as a compatibility prop for hosts that only have the boolean; it can only express running/idle.)

### Pure Logic in `trainFlow.ts`

| Export | Purpose |
|---|---|
| `ACTIVE_STATUSES` | `{ pulling, running, restarting, paused }` — a run is in flight |
| `derivePhase(status)` | → `'setup' \| 'running' \| 'completed' \| 'error'` |
| `describeDetection(d)` | Compact chip label, e.g. `Apple Silicon · CUDA — RTX 4090` |
| `deriveReadiness(input)` | The four checklist rows |
| `isReadyToStart(items)` | `every(state === 'ok' \|\| state === 'warn')` — warnings don't gate |
| `formatElapsed(ms)` | `m:ss` under an hour, `h:mm:ss` above; never negative |

### The Readiness Checklist

Four rows, each `ok` / `warn` / `pending` / `blocked`:

| Row | Blocking rule |
|---|---|
| **Server reachable** | Proxied by the trainable-projects fetch: loading ⇒ `pending`, error ⇒ `blocked`. |
| **Model selected** | `blocked` when no projects, none selected, or the selected project's `status !== 'RUNNING'` ("not accepting clients yet — ask the owner to start it, then refresh"). An advisory eligibility result downgrades it to `warn`, never `blocked`. |
| **Hardware detected** | Never blocking. `warn` on detection failure, and `warn` on `nativeBundleAvailable === false` ("reinstall to enable training"). |
| **Training data** | `ok` when a folder is chosen **or** the "Skip — train with the model's built-in dataset" checkbox is ticked; `pending` otherwise. This explicit choice is the one gate the redesign added. |

Start is enabled when nothing is `pending` or `blocked`. The blocking rules deliberately match the pre-redesign gating; eligibility and hardware issues stay advisory exactly as before.

### Starting a Run

```typescript
const res = await window.fedLearnAPI.getProjectConnection(selectedProject.projectId);
const c = res.connection;
onStart({
  hardwareProfile: selectedProfile,
  projectId:     c.projectId,
  serverAddress: c.serverAddress,
  partitionId:   String(c.partitionId),   // connection payload types it as a number
  modelType:     c.modelType,
  datasetPath:   datasetPath.trim(),
  connectionToken: c.connectionToken,
  strategy:      c.strategy,
  trainingArm:   c.trainingArm,
});
```

Every field except `hardwareProfile` and `datasetPath` comes from the backend. `trainingArm` is passed here but does not currently survive the preload bridge — see [04 → the `startTraining` validation flow](./04-preload-ipc-bridge.md#the-starttraining-validation-flow).

### Dataset Path Selection

The dataset input is `readOnly`; the only way to set it is the native dialog via `selectDatasetPath()`. That is not just UX — Main will refuse to bind-mount a path that was not returned by that dialog (`dataset-consent.ts`). A "Clear" button appears once a path is set, and the skip checkbox is disabled while a path is chosen.

### Run Notifications

`runNotifications.ts` classifies a status transition (`classifyRunTransition(prev, next)`) and fires an HTML5 `Notification` on completion or failure. It is permission-guarded inside `notifyRunOutcome`, renderer-only, and uses the project name captured when Start was pressed.

### Elapsed Clock

Anchored when the status first enters `ACTIVE_STATUSES`, ticked once a second, **frozen** (not reset) on `completed`/`error`, and reset on the next run. The `StatusBar` runs its own clock from the `startedAt` `App` recorded at click time; `TrainSection`'s is local to the run header.

---

## HardwareProfilePicker — Profile Cards

What remains of the old `HardwareSelector`: a controlled card grid, pure presentation. Detection and preselection live in the consumer.

```typescript
export const HARDWARE_PROFILES: HardwareProfileOption[] = [
  { id: 'discrete', label: 'Discrete GPU',   icon: MonitorCog,
    description: 'NVIDIA workstation with a dedicated PCIe GPU (CUDA). Runs the bundled native client.',
    dockerConfig: 'Native process (bundled client)' },
  { id: 'jetson',   label: 'Jetson SoC',     icon: CircuitBoard,
    description: 'NVIDIA Jetson edge device with an integrated Tegra GPU. Runs in a Docker container with direct /dev device mounts.',
    dockerConfig: 'Docker container (direct /dev device mounts)' },
  { id: 'mps',      label: 'Apple Silicon',  icon: Command,
    description: 'Mac M1/M2/M3/M4 with Metal (MPS) acceleration. Runs the bundled native client.',
    dockerConfig: 'Native process (bundled client)' },
  { id: 'cpu',      label: 'CPU Only',       icon: Cpu,
    description: 'Standard CPU training without GPU acceleration. Runs the bundled native client.',
    dockerConfig: 'Native process (bundled client)' },
];
```

The `dockerConfig` line states how the profile **actually** executes. It used to read `DeviceRequests: --gpus all` for `discrete`, which was untrue — `discrete` has never taken the Docker path in `DockerService.startTraining()`; it runs the bundled native client with CUDA torch. `2b02173` corrected the card copy and the `startDockerTraining` switch together.

`icon` is a lucide component reference, not an emoji string (`74cda60` replaced the emoji set).

---

## LogPanel — Real-Time Log Stream

`LogPanel` was once a single `<pre>{logs.join('')}</pre>`. It has grown display features — per-line severity colouring, arrival timestamps, a filter box, follow-tail with a "Jump to latest" pill — while keeping the XSS guarantee and staying fast.

### The data flow is unchanged

`App` still owns the `string[]` buffer. Each entry is an IPC chunk, which may contain several newline-separated lines.

### Incremental parsing (`logView.ts`)

Per-line severity and timestamps require per-line nodes, and re-splitting the whole buffer every batch would be O(n) per frame. `logView.ts` keeps a per-mount cache:

- `createLogLineCache()` / `updateLogLineCache(cache, logs)` split and classify **only** the entries appended since the last call, and return a new array reference only when content actually changed — safe as a `useMemo` dependency despite living in a ref.
- The buffer is append-only except on clear; a *shrinking* buffer resets the cache.
- Line objects keep their identity, so `React.memo`'d `LogLineRow`s with stable keys are reused rather than re-rendered.
- `filterLogLines(lines, query)` runs a case-insensitive filter over **everything**, not just what is rendered.

### Render cap

```typescript
/** Upper bound on DOM log lines; the App-side buffer (10K entries) is larger. */
const MAX_RENDERED_LINES = 2000;
```

When the filtered set exceeds it, the panel renders the most recent 2000 and shows a "Showing the last 2,000 of N lines" note. The full buffer stays searchable.

### Follow-tail

```typescript
const FOLLOW_EPSILON_PX = 50;

const handleScroll = useCallback(() => {
  const el = containerRef.current; if (!el) return;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_EPSILON_PX;
  setFollowing(atBottom);
}, []);
```

Scrolling up pauses the tail and reveals a "Jump to latest" pill; clicking it scrolls to the bottom and resumes. The 50 px tolerance handles fractional scroll positions.

`following` is state here (it drives the pill's visibility), unlike the earlier ref-based implementation.

### Empty state

Zero entries renders an empty-state card ("No output yet · Start a training session to see live logs here") rather than a blank box.

### Security: no `dangerouslySetInnerHTML`

```typescript
<pre className="log-content">
  {/*
    SECURITY: Every piece of log output below is a plain React text
    node — React escapes all content, so no HTML from container output
    is ever interpreted.
  */}
  {visible.map((line) => (
    <LogLineRow key={line.lineIndex} line={line} arrivedAt={entryTimesRef.current[line.entryIndex] ?? 0} />
  ))}
</pre>
```

Severity only selects a CSS class (`log-line-error` / `log-line-warn`); `line.text` is always rendered as a child. `<script>alert(1)</script>` in a log line appears as literal text.

### Timestamp caveat, stated in the code

Arrival timestamps are stamped renderer-side when an entry first appears. If `App`'s 10K cap trims the head while the buffer is full, older stamps drift by one entry — an accepted approximation, since these are arrival times rather than the container's own timestamps.

---

## ModelPlayground — "Use a Model"

The Models section: pick one of the signed-in user's trained models and run it. `InferableModel.inputKind` selects the input widget:

| `inputKind` | Widget | IPC |
|---|---|---|
| `image` | File picker + preview | `runInference(projectId, { imageBase64 })` |
| `vector` | Comma/whitespace-separated numeric field | `runInference(projectId, { values })` |
| `text` | Text area (classification) | `runInference(projectId, { text })` |
| `generation` | Multi-turn chat thread + max-tokens / temperature sliders | `runGeneration(...)` / `stopGeneration(...)` |

Inference runs **server-side** — the backend loads the real PyTorch model; the desktop app only marshals input and renders the result (predicted label, class probabilities).

### Generation streaming

Tokens arrive on the `inference:token` push channel from the Main-process STOMP bridge, and the `runGeneration` promise resolves with the complete result. Three details:

- A **"Generating…" bubble** appears before the first token, matching the web client.
- The assistant bubble is finalized through a `streamingRef`, not state, to avoid a StrictMode double-append (`74c80ba`).
- **Stop** cancels in flight (`ed95d4f`); the streamed partial is kept regardless of what the stop call returns.

### Input guard

```typescript
// Reject oversized files before FileReader pulls them fully into renderer memory.
const MAX_IMAGE_FILE_BYTES = 10 * 1024 * 1024;
```

The preload, IPC handler and backend all bound the encoded size too — this stops a multi-hundred-MB pick from freezing the UI before those checks ever run.

---

## SettingsSection — Server, Updates, About

The settings surface as a page section (it supersedes the old `SettingsModal`). Three cards:

**Server** — the URL field with label and help text, the DE-13 insecure-HTTP acknowledge flow, and success/error banners. Save is the one primary action. Editing the field resets both `allowInsecure` and the warning, because a different URL needs a fresh transport decision.

Unlike `AuthModal`, this shows the URL **with** the `/api` suffix — the raw stored value. That is intentional: a settings panel shows what is actually stored.

**Updates** — a manual "Check for updates" trigger. The progress UI itself lives in the shell-mounted `UpdateBanner` (its preload listeners have no removal API, so that component must stay at shell level); this card only fires the same `checkForUpdates` IPC and reports whether the request was accepted.

**About** — app version from the webpack `DefinePlugin` (`__APP_VERSION__`, guarded with a `typeof` check since it is undefined under jest) and the detected-device summary from `describeDetection`.

`SettingsSectionProps.onClose` is optional: dismissable hosts get a callback 1.5 s after a successful save; as a page section it is omitted and the success banner simply stays.

---

## StatusBar & StatusIndicator

`StatusBar` is a persistent full-width bottom strip rendered by the shell **outside** the section outlet, so it survives Train/Models/Settings switches. It shows:

- backend connection — a dot plus the host parsed out of the configured server URL, or "Not connected"
- the detected hardware profile chip (hidden while detection is pending or failed)
- the run state: the `StatusIndicator` badge plus `"<project> · 12:34 elapsed"` while a run is active
- the app version, right-aligned

All data is App-owned and arrives via props; the only local state is the 1 s elapsed tick, which runs **only** while `containerStatus` is `running` or `pulling`.

`StatusIndicator` is unchanged in shape — a stateless map from `ContainerStatus` to `{ label, colorClass, animate }`, with a `|| STATUS_CONFIG.idle` fallback for unexpected values and the `#status-indicator` id preserved as a test/automation hook. It is no longer rendered in a header; its only consumer is `StatusBar`.

```typescript
const STATUS_CONFIG: Record<ContainerStatus, { label: string; colorClass: string; animate: boolean }> = {
  idle:       { label: 'Idle',           colorClass: 'status-idle',      animate: false },
  pulling:    { label: 'Pulling image',  colorClass: 'status-pulling',   animate: true  },
  running:    { label: 'Training',       colorClass: 'status-running',   animate: true  },
  completed:  { label: 'Completed',      colorClass: 'status-completed', animate: false },
  error:      { label: 'Error',          colorClass: 'status-error',     animate: false },
  restarting: { label: 'Restarting',     colorClass: 'status-pulling',   animate: true  },
  paused:     { label: 'Paused',         colorClass: 'status-idle',      animate: false },
  stopped:    { label: 'Stopped',        colorClass: 'status-idle',      animate: false },
};
```

`--running` is its own design token (`73ed288` unified the "running" status colour across surfaces and added a token-sync CI guard) rather than a reused accent.

---

## UpdateBanner — Auto-Update Layer

A single component covering the whole `electron-updater` lifecycle: `idle → checking → upToDate | available → downloading → ready`, plus `error`.

It is mounted **once, at shell level, forever**, in a `shell-update-layer` that overlays the section outlet. The reason is a preload constraint, not a layout preference: `onUpdateAvailable`, `onUpdateProgress`, `onUpdateDownloaded`, `onUpdateNotAvailable` and `onUpdateError` have **no removal API**, so unmounting and re-mounting would stack listeners with no way to clear them.

Updates download automatically (`autoUpdater.autoDownload = true`), so there is no separate download prompt — the banner narrates "downloading in background → progress → restart to install". `upToDate` auto-dismisses after a few seconds; the banner is otherwise user-dismissable.

State semantics map onto tokens: available/checking/downloading → accent, ready/upToDate → success, error → danger.

---

## Styles Architecture

Three stylesheets, in load order: `tokens.css` → `fonts.css` → `styles.css` (plus `components/sections.css`, imported by the section components).

### `tokens.css` is generated, not written

```css
/* GENERATED by design/build-tokens.mjs — DO NOT EDIT. Edit design/tokens.json and re-run. */
:root {
  --canvas: #F6F3EE;
  --surface-1: #FFFFFF;
  --fg: #191A1C;
  --fg-muted: #6B6760;
  --accent: #1C314D;
  --accent-hover: #14243A;
  --running: #3A5A76;
  --font-sans: 'Hanken Grotesk', ui-sans-serif, system-ui, …;
  --font-display: 'Hanken Grotesk', …;
  --font-mono: 'JetBrains Mono', ui-monospace, …;
  /* …radii, spacing, shadows, durations, easings, the --text-* ramp, --series-1..8 */
}
```

`design/tokens.json` is the single source of truth for the whole platform; `design/build-tokens.mjs` generates the per-platform outputs (this file for desktop, the equivalent for frontend and mobile). A CI job runs `scripts/check_design_tokens.sh` **unconditionally** — a hand-edit of a generated file, or a `tokens.json` change without regenerating, fails the build rather than drifting.

This is the **Ledger** system: navy structural ink on quiet paper surfaces, light-first. Depth comes from the surface ladder plus 1 px hairline borders plus the shadow tokens — no glows, no gradients, no decorative ambience. The dark family (`--canvas: #0B1622`, `--accent: #4F8AC9`) is generated alongside it.

Two earlier systems appear in this repo's history and should not be mistaken for current: **Ember** (burnt orange on warm paper, Bricolage Grotesque display) and, before it, **Instrument**. Bricolage Grotesque is Ember-era and survives only in `design/brand/*.html` comparison assets — the desktop font stack is Hanken Grotesk for both sans and display.

### `styles.css` and `sections.css`

`styles.css` (~35 KB) holds the reset, base typography and the shared component classes. `components/sections.css` (~11 KB) holds the section-scoped styles (setup card, readiness rows, log toolbar, chat bubbles, status strip). Every colour/radius/space/type value resolves to a `var(--token)`; type sizes come from the `--text-*` ramp only.

### Fonts

`fonts.css` imports the `@fontsource` packages locally. No remote font host appears anywhere in the renderer, which is what lets the CSP be `font-src 'self'` in both dev and production (DE-14). The file's own comment notes that Bricolage Grotesque was retired with Ember.

### Key Component Classes

| Class | Component | Purpose |
|---|---|---|
| `.app-container` | Root | Full-height flex column layout |
| `.shell-titlebar` | App | Drag strip; `.shell-titlebar-mac` adds the traffic-light inset |
| `.rail` / `.rail-item` / `.rail-item-active` | App | Left icon rail |
| `.shell-outlet` / `.shell-section` / `.shell-section-hidden` | App | Section outlet + CSS visibility toggle |
| `.panel` / `.panel-header` / `.panel-title` | shared | Card container |
| `.setup-card` / `.train-setup-grid` | TrainSection | Guided setup layout |
| `.readiness-item` + `.readiness-{ok,warn,pending,blocked}` | TrainSection | Checklist rows |
| `.run-header` / `.run-banner-success` / `.run-banner-danger` | TrainSection | Running header, outcome banners |
| `.profile-card` / `.profile-card-active` | HardwareProfilePicker | Profile selection |
| `.log-panel` / `.log-toolbar` / `.log-line-error` / `.log-jump-pill` | LogPanel | Log viewer |
| `.statusbar` / `.statusbar-chip` / `.statusbar-dot-ok` | StatusBar | Bottom strip |
| `.status-indicator` / `.status-dot-pulse` | StatusIndicator | Badge + pulse animation |
| `.auth-overlay` / `.auth-modal` / `.auth-warning` | AuthModal | Login overlay, DE-13 warning |
| `.form-input` / `.form-help` / `.form-label` | shared | Form primitives |
| `.btn-primary` / `.btn-secondary` / `.btn-ghost` / `.btn-danger` | shared | Button variants |
| `.validation-error` / `.validation-success` | shared | Inline feedback |

---

## Performance Patterns

### 1. `useCallback` for Event Handlers

All handlers passed to child components are wrapped in `useCallback`, so a new log line re-rendering `App` does not force every section to re-render.

### 2. `useRef` for High-Frequency Values

Values that change frequently but must not trigger re-renders:
- `logBufferRef` — accumulating log lines between animation frames
- `rafIdRef` — the pending `requestAnimationFrame` id
- `lineCacheRef`, `entryTimesRef` (LogPanel) — incremental parse cache and arrival stamps
- `runStartRef`, `prevStatusRef`, `lastStartedNameRef` (TrainSection) — clock anchor, transition detection, notification label
- `streamingRef` (ModelPlayground) — the in-flight assistant message

### 3. Two-Level Log Capping

`App` caps the **buffer** at `MAX_LOG_LINES = 10_000` entries; `LogPanel` caps the **DOM** at `MAX_RENDERED_LINES = 2_000` lines. The two are independent on purpose — the buffer bounds memory, the render cap bounds reconciliation, and filtering still runs across the full buffer.

### 4. Memoized Log Rows

`LogLineRow` is `React.memo`'d and keyed on a stable `lineIndex`. Combined with the incremental cache's stable line-object identity, appending a batch re-renders only the new rows instead of up to 2000 spans per frame.

### 5. Sections Mounted Once

Toggling `shell-section-hidden` instead of unmounting avoids re-fetching the model list, re-running detection, and destroying the chat thread on every section switch. The cost is that all three sections' effects stay live — which is why the advisory ones are written to be idempotent and failure-tolerant.

### 6. Cancellation Flags on Async Effects

Every mount effect that calls IPC (`detectHardware`, `getServerUrl`, `getDeviceCapabilities`, `listTrainableProjects`) uses a `cancelled` flag in its cleanup, so a resolve after unmount is a no-op rather than a React warning.

### 7. Timer Cleanup

The 3 s status poll, the 1 s elapsed ticks in `TrainSection` and `StatusBar`, and the `keydown` listener are all cleared in their effect's return function, so nothing accumulates when `isAuthenticated` flips.

---

*Next: [06 — Build, Packaging & Distribution](./06-build-and-packaging.md)*  
*Previous: [04 — Preload & IPC Bridge](./04-preload-ipc-bridge.md)*
