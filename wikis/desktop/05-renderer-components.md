# FedLearn Desktop — Renderer & React Components

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Renderer Architecture Overview](#renderer-architecture-overview)
2. [App.tsx — Root Component & State Machine](#apptsx--root-component--state-machine)
3. [AuthModal — Login & Server Configuration](#authmodal--login--server-configuration)
4. [HardwareSelector — Training Configuration UI](#hardwareselector--training-configuration-ui)
5. [LogPanel — Real-Time Log Stream](#logpanel--real-time-log-stream)
6. [StatusIndicator — Container State Badge](#statusindicator--container-state-badge)
7. [SettingsModal — Server URL Management](#settingsmodal--server-url-management)
8. [Styles Architecture](#styles-architecture)
9. [Performance Patterns](#performance-patterns)

---

## Renderer Architecture Overview

The renderer is a single-page React application mounted in the Electron Chromium window. It communicates exclusively through `window.fedLearnAPI` (provided by the preload bridge).

```
src/renderer/
├── index.tsx          ← React root (ReactDOM.createRoot)
├── index.html         ← HTML shell (CSP meta tag for packaged builds)
├── styles.css         ← Global CSS (dark theme, component styles, animations)
├── App.tsx            ← Root component, application state machine
└── components/
    ├── AuthModal.tsx       ← Login form (shown before authentication)
    ├── HardwareSelector.tsx ← Training config + start/stop controls
    ├── LogPanel.tsx        ← Streaming log viewer
    ├── StatusIndicator.tsx ← Training status badge in header
    └── SettingsModal.tsx   ← Server URL settings overlay
```

### Application States

The app has three top-level rendering states:

```
App start
    │
    ▼
isAuthChecking = true
    │
    ▼  (checkAuth IPC call resolves)
    ├── authenticated = false → <AuthModal />
    │
    └── authenticated = true  → Dashboard
                                  ├── Header (logo + StatusIndicator + Settings + Logout)
                                  ├── Docker Warning Banner (conditional)
                                  ├── Left Panel: <HardwareSelector />
                                  ├── Right Panel: <LogPanel />
                                  └── Footer
```

---

## App.tsx — Root Component & State Machine

`App.tsx` is the root of the application. It owns all shared state and orchestrates communication between components.

### State

```typescript
const [isAuthenticated, setIsAuthenticated] = useState(false);
const [isAuthChecking, setIsAuthChecking] = useState(true);     // Show loading spinner
const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
const [logs, setLogs] = useState<string[]>([]);                 // Log buffer
const [showSettings, setShowSettings] = useState(false);         // Settings overlay
const [dockerWarning, setDockerWarning] = useState<string | null>(null);  // Banner
```

### ContainerStatus Type

```typescript
// Shared type — also imported by StatusIndicator
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

### Startup Effects

```typescript
// Effect 1: Docker daemon warning listener (registers on mount, never re-registers)
useEffect(() => {
  window.fedLearnAPI.onDockerUnavailable((msg: string) => {
    setDockerWarning(`Docker is not running: ${msg}`);
  });
}, []);

// Effect 2: Auth check on startup
useEffect(() => {
  const checkAuth = async () => {
    try {
      const result = await window.fedLearnAPI.checkAuth();
      setIsAuthenticated(result.authenticated === true);
    } catch {
      setIsAuthenticated(false);
    } finally {
      setIsAuthChecking(false);  // Always exit loading state
    }
  };
  checkAuth();
}, []);
```

The `finally` block is critical — even if `checkAuth()` throws, `isAuthChecking` must be set to `false` to exit the loading screen. Otherwise the app is stuck on the spinner indefinitely.

### Log Batching with requestAnimationFrame

Container output can arrive at very high rates (hundreds of lines per second during model training). Without batching, each incoming log line would trigger a `setState` → reconcile → re-render cycle, which at 60fps would cap at 60 updates/second and still thrash the layout engine.

The solution: batch all logs received within a single animation frame into one `setState`:

```typescript
const logBufferRef = useRef<string[]>([]);
const rafIdRef = useRef<number | null>(null);

useEffect(() => {
  if (!isAuthenticated) return;

  window.fedLearnAPI.onTrainingLog((logLine: string) => {
    // 1. Push to the buffer ref (no re-render)
    logBufferRef.current.push(logLine);

    // 2. Schedule a flush on the next frame (if not already scheduled)
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(() => {
        const batch = logBufferRef.current;
        logBufferRef.current = [];      // Clear buffer
        rafIdRef.current = null;        // Mark as unscheduled

        // 3. Single setState with entire batch
        setLogs((prev) => {
          const merged = [...prev, ...batch];
          // 4. Cap log buffer at 10K lines (prevents unbounded memory growth)
          if (merged.length > MAX_LOG_LINES) {
            return merged.slice(merged.length - MAX_LOG_LINES);
          }
          return merged;
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

**Why `useRef` instead of `useState` for the buffer?** `useRef` mutations don't trigger re-renders. If we used `useState` for the buffer, every `push` would cause a re-render — defeating the entire purpose of batching.

**Why `MAX_LOG_LINES = 10_000`?** Each log line can be up to several hundred characters. At 10K lines that's potentially several MB of strings in memory. Without this cap, a training run that lasts hours could accumulate gigabytes of log data, eventually causing OOM or severe slowdowns.

### Status Polling

Container status is polled every 3 seconds via IPC:

```typescript
useEffect(() => {
  if (!isAuthenticated) return;

  const pollStatus = async () => {
    try {
      const result = await window.fedLearnAPI.getDockerStatus();
      if (result.success && result.status) {
        setContainerStatus(result.status as ContainerStatus);
      }
    } catch {
      // Silently ignore polling failures (network hiccup, etc.)
    }
  };

  pollStatus();                           // Immediate first poll
  const interval = setInterval(pollStatus, 3000); // Then every 3s
  return () => clearInterval(interval);   // Cleanup
}, [isAuthenticated]);
```

3 seconds is a reasonable interval — fast enough to reflect state changes promptly, slow enough to not hammer the IPC layer.

### Event Handlers

```typescript
const handleLogin = useCallback(() => {
  setIsAuthenticated(true);
}, []);

const handleLogout = useCallback(async () => {
  await window.fedLearnAPI.logout();
  setIsAuthenticated(false);
  setLogs([]);                           // Clear log buffer
  setContainerStatus('idle');            // Reset status
}, []);

const handleStartTraining = useCallback(async (config) => {
  setLogs([]);                           // Clear previous logs
  setContainerStatus('pulling');         // Optimistic status update
  setLogs((prev) => [...prev, '[System] Starting training container...\n']);

  const result = await window.fedLearnAPI.startTraining(config);

  if (!result.success) {
    setContainerStatus('error');
    setLogs((prev) => [...prev, `[System] Failed to start: ${result.error}\n`]);
  }
  // Success: status will be updated by the polling effect
}, []);
```

`handleStartTraining` uses **optimistic updates** — it sets status to `'pulling'` immediately before the IPC call completes. This gives the user immediate visual feedback rather than waiting for the next status poll to detect the running state.

---

## AuthModal — Login & Server Configuration

`AuthModal` is rendered as a full-screen overlay when the user is not authenticated.

### Component State

```typescript
const [serverUrl, setServerUrl] = useState('http://localhost:8081');
const [username, setUsername] = useState('');
const [password, setPassword] = useState('');
const [error, setError] = useState('');
const [isLoading, setIsLoading] = useState(false);
const [showServerConfig, setShowServerConfig] = useState(false);  // Collapsible section
const [serverSaved, setServerSaved] = useState(false);            // ✓ confirmation
```

### Server URL Loading

On mount, the component fetches the persisted server URL from Main and strips the `/api` suffix for display:

```typescript
useEffect(() => {
  const loadUrl = async () => {
    try {
      const result = await window.fedLearnAPI.getServerUrl();
      if (result.success && result.url) {
        // Strip /api suffix — show user-friendly URL without the API path
        const displayUrl = result.url.replace(/\/api$/, '');
        setServerUrl(displayUrl);
      }
    } catch { /* Use default */ }
  };
  loadUrl();
}, []);
```

### Login Flow

```typescript
const handleSubmit = useCallback(async (e: React.FormEvent) => {
  e.preventDefault();
  setError('');

  if (!username.trim() || !password.trim()) {
    setError('Please enter both username and password.');
    return;
  }

  setIsLoading(true);
  try {
    // Save server URL first (in case user changed it before clicking Sign In)
    await window.fedLearnAPI.setServerUrl(serverUrl.trim());
    
    // Attempt login
    const result = await window.fedLearnAPI.login(username, password);
    
    if (result.success) {
      onLoginSuccess();  // Notify parent → setIsAuthenticated(true)
    } else {
      setError('Invalid credentials. Please try again.');
    }
  } catch (err) {
    setError(`Connection failed: ${err.message}`);
  } finally {
    setIsLoading(false);  // Always reset loading state
  }
}, [username, password, serverUrl, onLoginSuccess]);
```

### UX Design Choices

- **Collapsible server config:** The server URL is hidden under a toggle button by default. Most users don't need to change it, and hiding it reduces visual noise.
- **Server URL shown in toggle button:** Even when collapsed, the current server URL is visible in the toggle label, so the user can confirm which backend they're connecting to.
- **Save confirmation:** A `✓` replaces the "Save" button text for 2 seconds after saving, providing subtle feedback.
- **Loading state disables all inputs:** Prevents double-submit and UX confusion during the async login call.

---

## HardwareSelector — Training Configuration UI

`HardwareSelector` is the most complex UI component. It displays four hardware profile cards and a set of form inputs for configuring a training job.

### Hardware Profiles

```typescript
const HARDWARE_PROFILES: HardwareProfileOption[] = [
  {
    id: 'discrete',
    label: 'Discrete GPU',
    description: 'NVIDIA workstation with dedicated PCIe GPU. Uses --gpus all via DeviceRequests.',
    icon: '🖥️',
    dockerConfig: 'DeviceRequests: --gpus all',
  },
  {
    id: 'jetson',
    label: 'Jetson SoC',
    description: 'NVIDIA Jetson edge device with integrated Tegra GPU.',
    icon: '🔧',
    dockerConfig: 'Devices: /dev/nvhost-ctrl, nvhost-ctrl-gpu, ...',
  },
  {
    id: 'mps',
    label: 'Apple Silicon',
    description: 'Mac M1/M2/M3/M4 with Metal GPU. Runs natively (no Docker) for MPS acceleration.',
    icon: '🍎',
    dockerConfig: 'Native process (no Docker)',
  },
  {
    id: 'cpu',
    label: 'CPU Only',
    description: 'Standard CPU training without GPU acceleration.',
    icon: '💻',
    dockerConfig: 'No GPU configuration',
  },
];
```

### Auto-Detection on Mount

```typescript
useEffect(() => {
  let cancelled = false;  // Cancellation flag for component unmount
  (async () => {
    try {
      const result = await window.fedLearnAPI.detectHardware();
      if (cancelled || !result.success || !result.detection) return;

      const d = result.detection;
      setSelectedProfile(d.recommendedProfile);  // Auto-select best profile

      // Build human-readable detection summary
      const parts: string[] = [];
      if (d.platform === 'darwin' && d.arch === 'arm64') parts.push('Apple Silicon');
      else if (d.platform === 'win32') parts.push('Windows x64');
      else parts.push(`${d.platform}/${d.arch}`);

      if (d.cudaAvailable) parts.push(`CUDA — ${d.cudaInfo || 'NVIDIA GPU'}`);
      if (!d.nativeBundleAvailable) parts.push('native bundle missing — falling back to Docker');

      setDetectionLabel(parts.join(' · '));  // e.g., "Apple Silicon · CUDA — RTX 4090"
    } catch { /* Detection is best-effort */ }
  })();
  return () => { cancelled = true; };  // Prevent setState after unmount
}, []);
```

The cancellation flag (`cancelled`) prevents a `setState` call after the component has unmounted — which would be a no-op but would generate a React warning.

### Dataset Path Selection

```typescript
const handleSelectDataset = async () => {
  try {
    const result = await window.fedLearnAPI.selectDatasetPath();
    if (result.success && result.path) {
      setDatasetPath(result.path);
      setValidationError('');
    } else if (result.error) {
      setValidationError(`Dataset selection failed: ${result.error}`);
    }
  } catch (err: any) {
    setValidationError(`Error opening dialog: ${err.message}`);
  }
};
```

The dataset path input is `readOnly` — users can only set it via the native dialog, never by typing. This prevents path injection via keyboard input and guarantees the path shown in the UI is the same path passed through the native dialog.

### Client-Side Validation Before IPC

Before calling `startTraining`, the component validates all fields locally. This is the **first** of three validation layers (preload and Main being the other two):

```typescript
const handleStart = useCallback(() => {
  setValidationError('');

  // Presence checks
  if (!projectId.trim())    { setValidationError('Project ID is required.'); return; }
  if (!serverAddress.trim()) { setValidationError('Server address is required.'); return; }
  if (!partitionId.trim())  { setValidationError('Partition ID is required.'); return; }
  if (!modelType.trim())    { setValidationError('Model Architecture is required.'); return; }
  if (!datasetPath.trim())  { setValidationError('Local Dataset Path is required.'); return; }

  // Pattern validation — mirrors preload allowlists for consistent UX error messages
  if (!/^[a-zA-Z0-9_-]{1,128}$/.test(projectId))
    { setValidationError('Project ID must be alphanumeric (max 128 chars).'); return; }
  
  if (!/^[a-zA-Z0-9._:/-]{1,256}$/.test(serverAddress))
    { setValidationError('Invalid server address format.'); return; }
  
  if (!/^[0-9]{1,10}$/.test(partitionId))
    { setValidationError('Partition ID must be a number.'); return; }

  // All valid — call the API bridge
  onStart({
    hardwareProfile: selectedProfile,
    projectId: projectId.trim(),
    serverAddress: serverAddress.trim(),
    partitionId: partitionId.trim(),
    modelType: modelType.trim(),
    datasetPath: datasetPath.trim(),
  });
}, [/* deps */]);
```

### Props Interface

```typescript
interface HardwareSelectorProps {
  onStart: (config: {
    hardwareProfile: string;
    projectId: string;
    serverAddress: string;
    partitionId: string;
    modelType: string;
    datasetPath: string;
  }) => void;
  onStop: () => void;
  isRunning: boolean;  // Disables all inputs and profile cards when true
}
```

When `isRunning` is `true`, all profile cards and input fields are disabled, and the "Start Training" button is replaced by "Stop Training".

---

## LogPanel — Real-Time Log Stream

`LogPanel` is intentionally minimal by design — its simplicity is a security feature.

```typescript
const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const isAutoScrollRef = useRef(true);  // Track whether user has scrolled up

  // Detect user scroll position
  const handleScroll = useCallback(() => {
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current!;
    // "At bottom" = within 50px of the scroll end
    isAutoScrollRef.current = scrollHeight - scrollTop - clientHeight < 50;
  }, []);

  // Auto-scroll when new logs arrive (only if user is at the bottom)
  useEffect(() => {
    if (isAutoScrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="log-panel" ref={containerRef} onScroll={handleScroll}>
      <pre className="log-content">
        {logs.join('')}  {/* Plain text — no HTML interpretation */}
      </pre>
    </div>
  );
};
```

### Auto-Scroll with User Override

The auto-scroll behavior uses a ref (`isAutoScrollRef`) rather than state because:
1. Changing it should not trigger a re-render
2. The scroll handler fires frequently and needs to be fast

The 50px threshold for "at bottom" is intentional — it handles the case where the user is at the bottom but a pixel or two off due to fractional scroll positions.

### Performance: Single `join('')` vs. Many `<span>` Elements

An alternative implementation would map each log line to a `<span>`:
```typescript
{logs.map((line, i) => <span key={i}>{line}</span>)}
```

At 10,000 log lines this creates **10,000 DOM nodes**, each requiring a React fiber, reconciliation, and layout. With `join('')`, there is **exactly one DOM node** (`<pre>`) containing all the text. This makes log rendering O(1) in DOM nodes regardless of log count.

### Security: No `dangerouslySetInnerHTML`

The `LogPanel` explicitly documents its XSS-safe design:
```typescript
{/*
  SECURITY: Each log line is rendered as a plain text node.
  React's default behavior escapes all content — no HTML is interpreted.
  This prevents any XSS payload from container output from executing.
*/}
{logs.join('')}
```

React escapes all JSX text content by default. `<script>alert(1)</script>` in a log line becomes the literal escaped text `&lt;script&gt;alert(1)&lt;/script&gt;` in the DOM.

---

## StatusIndicator — Container State Badge

A simple stateless component that maps `ContainerStatus` to a visual badge.

```typescript
const STATUS_CONFIG: Record<ContainerStatus, { label: string; colorClass: string; animate: boolean }> = {
  idle:       { label: 'Idle',          colorClass: 'status-idle',     animate: false },
  pulling:    { label: 'Pulling Image', colorClass: 'status-pulling',  animate: true  },
  running:    { label: 'Training',      colorClass: 'status-running',  animate: true  },
  completed:  { label: 'Completed',     colorClass: 'status-completed', animate: false },
  error:      { label: 'Error',         colorClass: 'status-error',    animate: false },
  restarting: { label: 'Restarting',    colorClass: 'status-pulling',  animate: true  },
  paused:     { label: 'Paused',        colorClass: 'status-idle',     animate: false },
  stopped:    { label: 'Stopped',       colorClass: 'status-idle',     animate: false },
};

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.idle;  // Safe fallback

  return (
    <div className={`status-indicator ${config.colorClass}`} id="status-indicator">
      <span className={`status-dot ${config.animate ? 'status-dot-pulse' : ''}`} />
      <span className="status-label">{config.label}</span>
    </div>
  );
};
```

The `status-dot-pulse` CSS class applies a CSS `@keyframes` animation that creates a pulsing glow effect for active states (`running`, `pulling`, `restarting`).

The `|| STATUS_CONFIG.idle` fallback handles any unexpected status string values gracefully, always rendering a valid badge.

---

## SettingsModal — Server URL Management

`SettingsModal` is a settings overlay available from the main dashboard (after login). It allows users to update the backend server URL without logging out.

```typescript
const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const [serverUrl, setServerUrl] = useState('');
  
  // Load current URL on mount
  useEffect(() => {
    window.fedLearnAPI.getServerUrl().then((result) => {
      if (result.success && result.url) setServerUrl(result.url);
    });
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await window.fedLearnAPI.setServerUrl(serverUrl);
    if (result.success) {
      setSuccessMsg('Server URL updated successfully.');
      setTimeout(() => onClose(), 1500);  // Auto-close after confirmation
    } else {
      setError(result.error || 'Failed to update server URL.');
    }
  };
};
```

**Note:** Unlike `AuthModal`, `SettingsModal` shows the URL **with** the `/api` suffix (the raw stored value). This is intentional — settings panels typically show the actual stored value for transparency. `AuthModal` strips it for a cleaner first-time user experience.

---

## Styles Architecture

All styles are in a single `styles.css` file (18KB). The architecture uses:

### CSS Custom Properties (Design Tokens)

```css
:root {
  --color-bg-primary: #0a0a0f;
  --color-bg-secondary: #12121a;
  --color-accent: #6366f1;
  --color-accent-hover: #4f46e5;
  --color-text-primary: #e2e8f0;
  --color-text-muted: #64748b;
  --color-success: #34d399;
  --color-error: #f87171;
  /* ... */
}
```

### Key Component Classes

| Class | Component | Purpose |
|---|---|---|
| `.app-container` | Root | Full-height flex column layout |
| `.app-header` | Header | Fixed top bar with logo + controls |
| `.main-grid` | Main | Two-column panel layout |
| `.panel` | Panels | Glassmorphism card with border |
| `.profile-card` | HardwareSelector | Hardware profile selection button |
| `.profile-card-active` | HardwareSelector | Selected profile highlight |
| `.log-panel` | LogPanel | Scrollable log output area |
| `.status-indicator` | StatusIndicator | Status badge container |
| `.status-dot-pulse` | StatusIndicator | CSS keyframe pulse animation |
| `.auth-overlay` | AuthModal/SettingsModal | Full-screen backdrop |
| `.auth-modal` | AuthModal/SettingsModal | Centered modal card |
| `.auth-glow` | AuthModal | Decorative radial gradient elements |
| `.form-input` | Form inputs | Consistent dark input styling |
| `.btn-primary` | Buttons | Accent-colored action button |
| `.btn-ghost` | Buttons | Transparent border button |
| `.docker-warning` | App | Orange warning banner |
| `.validation-error` | HardwareSelector | Red error message box |

---

## Performance Patterns

### 1. `useCallback` for Event Handlers

All event handlers passed to child components are wrapped in `useCallback`:
```typescript
const handleStartTraining = useCallback(async (config) => { /* ... */ }, []);
const handleStopTraining  = useCallback(async () => { /* ... */ }, []);
const handleLogout        = useCallback(async () => { /* ... */ }, []);
```

This prevents child components from re-rendering when the parent re-renders (for unrelated reasons like new log lines).

### 2. `useRef` for High-Frequency Values

Values that change frequently but don't need to trigger re-renders are stored in refs:
- `logBufferRef` — accumulating log lines between animation frames
- `rafIdRef` — tracking the pending `requestAnimationFrame` ID
- `isAutoScrollRef` (in LogPanel) — tracking scroll position

### 3. Log Buffer Cap

The `MAX_LOG_LINES = 10_000` cap prevents the log array from growing indefinitely. Old lines are evicted from the front when the cap is exceeded:

```typescript
setLogs((prev) => {
  const merged = [...prev, ...batch];
  return merged.length > MAX_LOG_LINES
    ? merged.slice(merged.length - MAX_LOG_LINES)
    : merged;
});
```

### 4. Single `<pre>` Text Node for Logs

Using `logs.join('')` in a single `<pre>` instead of mapping to `<span>` elements keeps the DOM size constant at O(1) nodes regardless of log count.

### 5. Status Poll Cleanup

The `setInterval` for status polling is cleaned up in the effect's return function:
```typescript
const interval = setInterval(pollStatus, 3000);
return () => clearInterval(interval);
```

This prevents interval accumulation if `isAuthenticated` flips multiple times.

---

*Next: [06 — Build, Packaging & Distribution](./06-build-and-packaging.md)*  
*Previous: [04 — Preload & IPC Bridge](./04-preload-ipc-bridge.md)*
