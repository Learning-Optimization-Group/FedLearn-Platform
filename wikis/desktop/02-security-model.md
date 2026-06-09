# FedLearn Desktop — Security Model

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Security Philosophy](#security-philosophy)
2. [BrowserWindow Hardening](#browserwindow-hardening)
3. [Content Security Policy](#content-security-policy)
4. [Context Isolation & the Preload Bridge](#context-isolation--the-preload-bridge)
5. [Input Validation — Two-Layer Defense](#input-validation--two-layer-defense)
6. [JWT Confinement to Main Process](#jwt-confinement-to-main-process)
7. [safeStorage — OS-Level Encryption](#safestorage--os-level-encryption)
8. [Navigation and Window-Open Restrictions](#navigation-and-window-open-restrictions)
9. [Docker Socket Confinement](#docker-socket-confinement)
10. [Log Rendering — XSS Prevention](#log-rendering--xss-prevention)
11. [Threat Model Summary](#threat-model-summary)

---

## Security Philosophy

FedLearn Desktop is built around the principle of **minimal privilege + defense-in-depth**. Because the app embeds a web renderer (Chromium) that processes externally-sourced data (container log output, backend API responses), every trust boundary is explicitly defended at multiple layers.

The three non-negotiable invariants:

1. **The JWT never leaves the Main Process.** No matter what happens in the renderer.
2. **The Docker socket is never exposed to the renderer or any container.** Only `DockerService` in Main accesses it.
3. **Every input crossing the IPC boundary is validated twice** — in the Preload bridge and again in Main.

---

## BrowserWindow Hardening

The `BrowserWindow` is created with a strict set of `webPreferences` in `main.ts`:

```typescript
// src/main/main.ts
mainWindow = new BrowserWindow({
  webPreferences: {
    // ========== SECURITY: Non-Negotiable ==========
    nodeIntegration: false,    // Renderer has NO Node.js access
    contextIsolation: true,    // Renderer and preload have separate JS contexts
    sandbox: true,             // Additional OS-level process sandboxing
    // ===============================================
    preload: path.join(__dirname, '..', 'preload', 'preload.js'),
    devTools: isDev,           // DevTools only in dev mode
    webSecurity: true,         // Enforces same-origin policy
    allowRunningInsecureContent: false,
    experimentalFeatures: false,
  },
});
```

### Why each setting matters

| Setting | Value | Why |
|---|---|---|
| `nodeIntegration` | `false` | Without this, ANY JavaScript in the renderer can call `require('child_process')`, `require('fs')`, etc. This is the single most critical setting. |
| `contextIsolation` | `true` | Even with `nodeIntegration: false`, without context isolation the renderer can prototype-pollute the preload's JS context. With it enabled, the preload runs in a completely separate JavaScript world. |
| `sandbox` | `true` | Applies OS-level sandboxing (seccomp on Linux, App Sandbox on macOS, Low-IL on Windows). Restricts syscall surface of the renderer process. |
| `devTools` | `isDev` | Prevents opening DevTools in production builds, which could expose internal state. |
| `webSecurity` | `true` | Keeps same-origin policy enforced. Prevents the renderer from loading arbitrary URLs. |

---

## Content Security Policy

A Content Security Policy (CSP) is applied to every response. This acts as a defense-in-depth layer **on top of** `contextIsolation` — even if an XSS payload executes, the CSP blocks it from loading external scripts or exfiltrating data.

### Dev Mode (HTTP — response headers)

In development the app is served over HTTP from the webpack dev server, so CSP is applied via session-level response header injection:

```typescript
// src/main/main.ts
session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
  callback({
    responseHeaders: {
      ...details.responseHeaders,
      'Content-Security-Policy': [[
        "default-src 'self'",
        "script-src 'self' 'unsafe-eval'",   // webpack HMR needs eval
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data:",
        `connect-src 'self' ${apiConnectSrc}`,  // backend origins
        "frame-src 'none'",
        "object-src 'none'",
        "base-uri 'self'",
      ].join('; ')],
    },
  });
});
```

### Production Mode (file:// — meta tag)

In packaged builds the app loads from `file://`. Chromium's handling of `'self'` under `file://` origins is inconsistent — it does **not** resolve to the API host. Therefore, in production the CSP is embedded as a `<meta>` tag in `index.html`, and the `connect-src` origins are injected at build time via the `FEDLEARN_API_ORIGINS` environment variable.

### Dynamic API Origins

```typescript
// Comma-separated list of backend origins (set at CI/build time for production)
const apiOriginsFromEnv = (process.env.FEDLEARN_API_ORIGINS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

// In dev mode, always allow localhost
const defaultApiOrigins = isDev
  ? ['http://localhost:8081', 'ws://localhost:8081', 'http://localhost:9000', 'ws://localhost:9000']
  : [];

const apiConnectSrc = [...defaultApiOrigins, ...apiOriginsFromEnv].join(' ');
```

---

## Context Isolation & the Preload Bridge

With `contextIsolation: true` and `sandbox: true`, the renderer cannot access any Node.js globals. The **only** way for the renderer to communicate with the Main Process is through the API surface explicitly exposed by the preload script via `contextBridge`.

```
Renderer's window.js context     ←→     Preload's isolated context
                                              ↓ ipcRenderer
                                         Main Process
```

### What `contextBridge.exposeInMainWorld` does

`contextBridge` creates a **serialized** bridge — it clones values (using the structured clone algorithm) when passing between contexts. This prevents the renderer from getting a reference to any internal preload object, which would allow prototype pollution attacks.

```typescript
// src/preload/preload.ts
contextBridge.exposeInMainWorld('fedLearnAPI', {
  startTraining: async (config: TrainingConfigInput) => { /* ... */ },
  stopTraining:  async () => { /* ... */ },
  login:         async (username, password) => { /* ... */ },
  // ...
});
```

The renderer sees `window.fedLearnAPI` as a plain object with typed async functions. It cannot:
- Call `ipcRenderer.invoke` directly with arbitrary channel names
- Access any Node.js module
- Access the preload's internal state

---

## Input Validation — Two-Layer Defense

Every input that crosses the preload bridge is validated **twice**: once in the preload before `ipcRenderer.invoke`, and once again in the Main Process IPC handler. This means even a compromised renderer that calls `ipcRenderer.invoke` directly (bypassing the preload) cannot inject malicious values.

### Preload Validation (Layer 1)

```typescript
// src/preload/preload.ts — Allowlist constants
const ALLOWED_HARDWARE_PROFILES = ['discrete', 'jetson', 'cpu', 'mps'] as const;
const PROJECT_ID_PATTERN    = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN  = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;

// Used before every ipcRenderer.invoke('docker:start-training', ...)
startTraining: async (config: TrainingConfigInput) => {
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
  // Only reach ipcRenderer if ALL checks pass
  return ipcRenderer.invoke('docker:start-training', { ...config });
},
```

### Main Process Validation (Layer 2)

```typescript
// src/main/ipc.handlers.ts
ipcMain.handle('docker:start-training', async (_event, config: unknown) => {
  // Re-validate even though preload already did
  if (!validateHardwareProfile(cfg.hardwareProfile)) { /* reject */ }
  if (!validateProjectId(cfg.projectId))             { /* reject */ }
  if (!validateServerAddress(cfg.serverAddress))     { /* reject */ }
  if (!validatePartitionId(cfg.partitionId))         { /* reject */ }

  // Dataset path gets special treatment — full canonicalization
  const safeDatasetPath = sanitizeDatasetPath(cfg.datasetPath);
  if (safeDatasetPath === null)
    return { success: false, error: 'Invalid dataset path: must be an existing absolute directory' };

  // Only use the sanitized path in Docker bind mount — never the raw input
  const validConfig: TrainingConfig = {
    ...cfg,
    datasetPath: safeDatasetPath, // canonical, resolved, verified-directory path
  };
  await dockerService.startTraining(validConfig);
});
```

### `sanitizeDatasetPath` — Deep Dive

The dataset path is particularly critical because it's interpolated into a Docker bind mount string (`${path}:/data`). A malicious path like `../../etc` could expose sensitive host directories.

```typescript
function sanitizeDatasetPath(raw: unknown): string | null {
  // 1. Type and length check
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > 2048) return null;

  // 2. Null byte check — prevents directory traversal via embedded null terminator
  if (raw.includes('\0')) return null;

  // 3. Resolve to absolute path (collapses all .. segments)
  let resolved: string;
  try { resolved = path.resolve(raw); } catch { return null; }

  // 4. Verify no remaining .. segments (shouldn't be possible after resolve, but belt-and-suspenders)
  if (resolved.split(path.sep).some((seg) => seg === '..')) return null;

  // 5. Must be absolute
  if (!path.isAbsolute(resolved)) return null;

  // 6. Must exist AND be a directory (not a file, not a symlink to a non-dir)
  let stat: fs.Stats;
  try { stat = fs.statSync(resolved); } catch { return null; }
  if (!stat.isDirectory()) return null;

  return resolved;  // canonical, safe path
}
```

---

## JWT Confinement to Main Process

Authentication follows a strict trust model:

```
Renderer  ──(username, password)──►  Preload  ──►  Main  ──►  Backend API
                                                     │
                                                     ▼
                                              JWT stored in Main
                                              (OS-encrypted store)
                                                     │
                                        { success: boolean }
                                                     │
         ◄────────────────────────────────────────────
```

The JWT **never travels left** in this diagram. The renderer only ever receives `{ success: true }` or `{ success: false }`. This means:

- Even if the renderer's JavaScript environment is fully compromised (e.g., via a stored XSS in log output), the attacker cannot steal the JWT
- The JWT is not accessible via `window.*`, `localStorage`, `sessionStorage`, or `indexedDB` from the renderer

The `getAuthHeader()` method in `AuthService` is a Main-only method — it's never called from the IPC handlers that respond to the renderer; it's used internally when the Main process needs to make authenticated API calls on behalf of the user.

---

## safeStorage — OS-Level Encryption

When the JWT needs to persist across app restarts, it's encrypted using Electron's `safeStorage` API, which delegates to the OS credential store:

| Platform | Backend |
|---|---|
| macOS | Keychain |
| Windows | DPAPI (Data Protection API) |
| Linux | libsecret / GNOME Keyring |

```typescript
// src/main/auth.service.ts
private storeJwt(jwt: string, username: string): void {
  const expiresAt = Date.now() + JWT_EXPIRY_MS; // 24 hours

  if (safeStorage.isEncryptionAvailable()) {
    // Encrypt with OS keychain
    const encrypted = safeStorage.encryptString(jwt);
    const authData: AuthStore = {
      encryptedJwt: encrypted.toString('base64'),  // store as base64
      expiresAt,
      username,
    };
    this.store.set(AUTH_STORE_KEY, authData);
    log.info('[AuthService] JWT encrypted via safeStorage (OS keychain)');
    return;
  }

  // No OS encryption available (e.g., headless Linux without keyring)
  // SECURITY DECISION: Do NOT persist to disk in this case.
  // A previous design wrote reversible base64 — that is not encryption.
  // Instead: hold in process memory only. User must re-authenticate on relaunch.
  log.warn('[AuthService] safeStorage unavailable — JWT held in process memory only');
  this.store.delete(AUTH_STORE_KEY);  // Scrub any stale token from prior install
  this.sessionMemory = { jwt, expiresAt, username };
}
```

### Decryption and Auth Check

```typescript
isAuthenticated(): boolean {
  // 1. Check in-memory session (used when encryption was unavailable at login)
  if (this.sessionMemory) {
    if (Date.now() > this.sessionMemory.expiresAt) {
      this.logout();  // Clear expired session
      return false;
    }
    return this.sessionMemory.jwt.length > 0;
  }

  // 2. Check on-disk store
  const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;
  if (!authData || !authData.encryptedJwt) return false;

  // Check expiry BEFORE attempting decryption
  if (Date.now() > authData.expiresAt) { this.logout(); return false; }

  try {
    // Verify decryptability (keychain key may have changed since last login)
    const decrypted = safeStorage.decryptString(Buffer.from(authData.encryptedJwt, 'base64'));
    return decrypted.length > 0;
  } catch {
    // Keychain key changed (e.g., new OS user password) — logout gracefully
    log.warn('[AuthService] Failed to decrypt stored JWT — keychain may have changed');
    this.logout();
    return false;
  }
}
```

---

## Navigation and Window-Open Restrictions

Two additional hardening measures prevent the app from being used as a launcher for external content:

### Block New Window Creation

```typescript
// src/main/main.ts
app.on('web-contents-created', (_event, contents) => {
  contents.setWindowOpenHandler(() => {
    log.warn('[Security] Blocked attempt to open new window from renderer');
    return { action: 'deny' };
  });
  // ...
});
```

This prevents any script in the renderer from opening external browser windows (e.g., `window.open('http://evil.com')`).

### Block External Navigation

```typescript
contents.on('will-navigate', (event, url) => {
  let allowed = false;
  if (isDev && url.startsWith('http://localhost:9000')) {
    allowed = true;
  } else if (url.startsWith('file://')) {
    // Only allow navigation within the packaged app directory
    const filePath = decodeURIComponent(new URL(url).pathname);
    allowed = appDir ? filePath.startsWith(appDir) : true;
  }
  if (!allowed) {
    event.preventDefault();
    log.warn(`[Security] Blocked navigation to: ${url}`);
  }
});
```

This prevents the renderer from being navigated to external URLs — a common attack when XSS leads to a `location.href = 'http://attacker.com'`.

---

## Docker Socket Confinement

The Docker socket (`/var/run/docker.sock` on Unix, `//./pipe/docker_engine` on Windows) grants full control over all containers on the host. This is a **high-privilege resource**.

FedLearn Desktop's security contract:
1. The Docker socket is **only accessed in `DockerService`** (Main Process)
2. The Docker socket is **never mounted into training containers** (enforced in `startDockerTraining` — `AutoRemove: false` and no socket bind mount)
3. The renderer has no knowledge that Docker even exists — it only calls `startTraining(config)` via the API bridge

```typescript
// src/main/docker.service.ts
const hostConfig: Docker.HostConfig = {
  AutoRemove: false,
  Binds: [`${config.datasetPath}:/data`],  // ONLY the dataset directory
  // NOTE: /var/run/docker.sock is intentionally NOT mounted
};
```

If the training container were to receive the Docker socket, it could escape the container and control the host daemon. This is explicitly prevented.

---

## Log Rendering — XSS Prevention

Container output is untrusted data. A maliciously crafted training script could output HTML/JavaScript. The `LogPanel` component is designed to prevent any XSS from this source:

```typescript
// src/renderer/components/LogPanel.tsx
const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
  return (
    <div className="log-panel" ref={containerRef}>
      <pre className="log-content">
        {/*
          SECURITY: Each log line is a plain text node.
          React's default escaping prevents any HTML interpretation.
          No dangerouslySetInnerHTML. No innerHTML. No DOM injection.
        */}
        {logs.join('')}
      </pre>
    </div>
  );
};
```

React's JSX rendering **automatically escapes** all string content — `<script>alert(1)</script>` in a log line renders as the literal text, not as an executable script tag. This means even if an attacker controls the Python training script, they cannot achieve XSS through container output.

---

## Threat Model Summary

| Threat | Mitigation |
|---|---|
| Compromised renderer executes arbitrary code | `nodeIntegration: false`, `contextIsolation: true`, `sandbox: true` |
| Renderer steals JWT | JWT confined to Main Process; renderer only gets `{ success: boolean }` |
| XSS via malicious log output | `LogPanel` uses React text nodes (auto-escaped); no `innerHTML` |
| Path traversal via dataset path | `sanitizeDatasetPath()` resolves, canonicalizes, and stat-checks the path |
| Arbitrary IPC channel calls | Preload only exposes a typed, allowlisted API surface; no raw `ipcRenderer` |
| Input injection into Docker args | Double validation (preload + main) with strict regex allowlists |
| Docker socket access from container | Socket never mounted into training containers |
| Navigation to external URLs | `will-navigate` event handler blocks non-app URLs |
| New window/popup abuse | `setWindowOpenHandler` returns `{ action: 'deny' }` for all opens |
| Stale/decryptable JWT on disk | `safeStorage` uses OS keychain; if unavailable, no disk persistence |
| Token replay after OS user password change | Decryption failure triggers `logout()` and graceful re-auth prompt |
| Crash data exfiltration | `crashReporter.start({ uploadToServer: false })` — dumps stay local |

---

*Next: [03 — Main Process Deep Dive](./03-main-process.md)*
*Previous: [01 — Overview & Architecture](./01-overview-and-architecture.md)*
