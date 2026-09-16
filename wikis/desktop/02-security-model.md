# FedLearn Desktop — Security Model

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Security Philosophy](#security-philosophy)
2. [BrowserWindow Hardening](#browserwindow-hardening)
3. [Content Security Policy](#content-security-policy)
4. [Context Isolation & the Preload Bridge](#context-isolation--the-preload-bridge)
5. [Input Validation — Two-Layer Defense](#input-validation--two-layer-defense)
6. [Dataset-Path Consent](#dataset-path-consent-dataset-consentts)
7. [JWT Confinement to Main Process](#jwt-confinement-to-main-process)
8. [safeStorage — OS-Level Encryption](#safestorage--os-level-encryption)
9. [Session Expiry and Server-URL Rebinding](#session-expiry-and-server-url-rebinding)
10. [Transport Policy — Refusing Remote Plaintext HTTP](#transport-policy--refusing-remote-plaintext-http)
11. [Navigation and Window-Open Restrictions](#navigation-and-window-open-restrictions)
12. [Docker Socket Confinement](#docker-socket-confinement)
13. [Log Rendering — XSS Prevention](#log-rendering--xss-prevention)
14. [Threat Model Summary](#threat-model-summary)
15. [Dependency Vulnerability Posture](#dependency-vulnerability-posture)

---

## Security Philosophy

FedLearn Desktop is built around the principle of **minimal privilege + defense-in-depth**. Because the app embeds a web renderer (Chromium) that processes externally-sourced data (container log output, backend API responses), every trust boundary is explicitly defended at multiple layers.

The four non-negotiable invariants:

1. **The JWT never leaves the Main Process.** No matter what happens in the renderer.
2. **The Docker socket is never exposed to the renderer or any container.** Only `DockerService` in Main accesses it.
3. **Every input crossing the IPC boundary is validated twice** — in the Preload bridge and again in Main.
4. **A host directory is only bind-mounted if the user physically picked it** in the native dialog — validation alone is not consent.

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

A Content Security Policy (CSP) is applied in both runtime modes. This acts as a defense-in-depth layer **on top of** `contextIsolation` — even if an XSS payload executes, the CSP blocks it from loading external scripts or exfiltrating data.

There are two delivery mechanisms and they are **not** two views of the same policy — read both.

### The `<meta>` policy (both modes) — `webpack.csp.js`

`webpack.csp.js` is the single source of truth for the CSP baked into `index.html` by `HtmlWebpackPlugin`'s `templateParameters`. Both webpack renderer configs call it; they differ only in two flags:

```javascript
// fedlearn-desktop/webpack.csp.js
function buildRendererCsp({ allowEval, allowInlineStyle }) {
  const scriptSrc = allowEval ? "script-src 'self' 'unsafe-eval'" : "script-src 'self'";
  const styleSrc = allowInlineStyle ? "style-src 'self' 'unsafe-inline'" : "style-src 'self'";
  return [
    "default-src 'self'",
    scriptSrc,
    styleSrc,
    "font-src 'self'",
    "img-src 'self' data:",
    "connect-src 'self' http://localhost:* https://localhost:* ws://localhost:* wss://localhost:*",
    "frame-src 'none'",
    "object-src 'none'",
    "base-uri 'self'",
  ].join('; ');
}
```

| Flag | Dev (`webpack.renderer.config.js`) | Packaged (`webpack.prod.config.js`) | Why |
|---|---|---|---|
| `allowEval` | `true` | **`false`** | Webpack's development build uses the `eval` devtool. The production config sets `devtool: false`, so nothing at runtime needs it (DE-14, `8f43018`). |
| `allowInlineStyle` | `true` | **`false`** | Dev uses `style-loader`, which injects runtime `<style>` tags (required for HMR). Production extracts CSS with `MiniCssExtractPlugin` into a real file loaded via `<link rel="stylesheet">`, so `'unsafe-inline'` can be dropped (`18b3b59`). |

`font-src 'self'` with **no remote font host** in either mode: Hanken Grotesk and JetBrains Mono are self-hosted through `@fontsource` (`src/renderer/fonts.css`). Any doc showing `https://fonts.googleapis.com` / `https://fonts.gstatic.com` in this policy is describing the pre-DE-14 build.

### The dev-only response header — `main.ts`

Separately, and **only when `isDev`**, `main.ts` injects a `Content-Security-Policy` response header via `session.defaultSession.webRequest.onHeadersReceived`. This covers the HTTP dev-server origin. Its `connect-src` is assembled from two sources:

```typescript
// src/main/main.ts — dev only
const apiOriginsFromEnv = (process.env.FEDLEARN_API_ORIGINS || '')
  .split(',').map((s) => s.trim()).filter(Boolean);
const defaultApiOrigins = isDev
  ? ['http://localhost:8081', 'ws://localhost:8081', 'http://localhost:9000', 'ws://localhost:9000']
  : [];
const apiConnectSrc = [...defaultApiOrigins, ...apiOriginsFromEnv].join(' ');
```

> **Correction worth flagging:** `FEDLEARN_API_ORIGINS` reaches **only this dev-mode header**. The packaged build registers no `onHeadersReceived` handler at all, and its `<meta>` policy's `connect-src` is the fixed localhost wildcard list above — the env var does not widen it. An earlier version of this page claimed production origins were injected from `FEDLEARN_API_ORIGINS` at build time; that is not what the code does. A packaged build pointed at a non-localhost backend therefore depends on that fixed `connect-src` list, which is a real limitation, not a documented feature.

`src/__tests__/renderer-csp.test.ts` guards this: it calls `buildRendererCsp({ allowEval: false, allowInlineStyle: false })` and asserts the result carries no `unsafe-eval`, no `unsafe-inline` in `style-src`, and no `googleapis`/`gstatic` host — then loads both renderer webpack configs and checks the CSP each one hands to `HtmlWebpackPlugin`.

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
const PROJECT_ID_PATTERN      = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN    = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN  = /^[a-zA-Z0-9._:/-]{1,256}$/;
const MAX_STRING_LENGTH       = 256;
const MAX_IMAGE_BASE64_LEN    = 14 * 1024 * 1024;  // ~10 MB decoded
const MAX_VECTOR_LEN          = 100_000;
const MAX_TEXT_LEN            = 10_000;            // matches the backend's @Size(max = 10_000)

// Used before every ipcRenderer.invoke('docker:start-training', ...)
startTraining: async (config: TrainingConfigInput) => {
  if (!isValidHardwareProfile(config.hardwareProfile))  return { success: false, error: 'Invalid hardware profile' };
  if (!isValidProjectId(config.projectId))              return { success: false, error: 'Invalid project ID' };
  if (!isValidServerAddress(config.serverAddress))      return { success: false, error: 'Invalid server address' };
  if (!isValidPartitionId(config.partitionId))          return { success: false, error: 'Invalid partition ID' };
  if (!isValidModelType(config.modelType))              return { success: false, error: 'Invalid model type' };
  if (!isValidDatasetPath(config.datasetPath))          return { success: false, error: 'Invalid dataset path' };
  if (!isValidConnectionToken(config.connectionToken))  return { success: false, error: 'Invalid connection token' };
  if (config.strategy !== undefined && !/^[a-zA-Z0-9_\-.]{1,64}$/.test(config.strategy))
    return { success: false, error: 'Invalid strategy' };

  // Only reach ipcRenderer if ALL checks pass — and forward an EXPLICIT object,
  // never a spread of the renderer's own config.
  return ipcRenderer.invoke('docker:start-training', {
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

`isValidConnectionToken` accepts `undefined`/`null` (the legacy no-auth flow) but otherwise requires a bounded JWT-charset string (`/^[A-Za-z0-9._-]+$/`, ≤ 8192 chars) before it is forwarded as the FL `FEDLEARN_CONNECTION_TOKEN`.

> **Known gap in this list.** `TrainingConfigInput` declares `trainingArm?: string` and `TrainSection` sends it, but the explicit object above does **not** carry `trainingArm` — so the field is dropped at the preload boundary and Main always sees `undefined`. The explicit-reconstruction pattern is the right security posture; adding a field to the interface without adding it to the forwarded object is the failure mode it invites. Treat it as the worked example in [04 → Common Pitfalls](./04-preload-ipc-bridge.md#common-pitfalls).

### Main Process Validation (Layer 2)

Main's validators live in `src/main/validators.ts` — a deliberately `electron`-free, side-effect-free module so the *shipped* predicates can be unit-tested directly (`src/__tests__/validators.test.ts`). `ipc.handlers.ts` previously carried a diverged inline copy of them, which is exactly how the empty-dataset-path case ended up behaving differently on the two layers.

```typescript
// src/main/ipc.handlers.ts
ipcMain.handle('docker:start-training', async (_event, config: unknown) => {
  // Re-validate even though preload already did
  if (!validateHardwareProfile(cfg.hardwareProfile)) { /* reject */ }
  if (!validateProjectId(cfg.projectId))             { /* reject */ }
  if (!validateServerAddress(cfg.serverAddress))     { /* reject */ }
  if (!validatePartitionId(cfg.partitionId))         { /* reject */ }
  if (typeof cfg.modelType !== 'string' || !/^[a-zA-Z0-9_\-.]{1,128}$/.test(cfg.modelType)) { /* reject */ }

  // Dataset path gets special treatment — full canonicalization…
  const safeDatasetPath = sanitizeDatasetPath(cfg.datasetPath);
  if (safeDatasetPath === null)
    return { success: false, error: 'Invalid dataset path: must be an existing absolute directory' };

  // …and then a CONSENT check: only a directory the user picked in the native
  // dialog may be bind-mounted. '' means "no mount, use the built-in dataset".
  if (safeDatasetPath !== '' && !isDatasetPathConsented(safeDatasetPath))
    return { success: false, error: 'Dataset path must be selected with the "Select dataset" button' };

  // Only use the sanitized path — never the raw input.
  await dockerService.startTraining({ ...cfg, datasetPath: safeDatasetPath });
});
```

Two fields are treated asymmetrically on purpose:

- **`strategy`** falls back to `undefined` on a malformed value. Most strategy strings are no-ops on the client (only DeComFL changes its behaviour), so a bad one costs nothing and should never block a start.
- **`trainingArm`** is validated **strictly** and an unrecognised value **throws**, failing the start with an explanatory message. Silently downgrading to `FULL` against a `FROZEN_HEAD` server *is* the bug the field exists to prevent — the client would upload every parameter to a server holding only the head. Main currently accepts exactly `'FULL'` and `'FROZEN_HEAD'`.

### `sanitizeDatasetPath` — Deep Dive

The dataset path is particularly critical because it is interpolated into a Docker bind mount (`${path}:/data`) and passed to the native client as `--dataset-path`. A malicious path like `../../etc` could expose sensitive host directories.

```typescript
// src/main/validators.ts
export function sanitizeDatasetPath(raw: unknown): string | null {
  // 0. Optional field: empty / whitespace means "use the container's default
  //    dataset". Normalize to '' rather than rejecting. (The inline copy this
  //    module replaced got exactly this case wrong.)
  if (typeof raw === 'string' && raw.trim() === '') return '';

  // 1. Type and length check
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > MAX_DATASET_PATH_LEN) return null;

  // 2. Null byte check — prevents traversal via an embedded null terminator
  if (raw.includes('\0')) return null;

  // 3. Resolve to an absolute path (collapses all .. segments)
  let resolved: string;
  try { resolved = path.resolve(raw); } catch { return null; }

  // 4. Verify no remaining .. segments (belt-and-suspenders after resolve)
  if (resolved.split(path.sep).some((seg) => seg === '..')) return null;

  // 5. Must be absolute
  if (!path.isAbsolute(resolved)) return null;

  // 6. Must exist AND be a directory
  let stat: fs.Stats;
  try { stat = fs.statSync(resolved); } catch { return null; }
  if (!stat.isDirectory()) return null;

  return resolved;  // canonical, safe path
}
```

`MAX_DATASET_PATH_LEN` is 2048, matching the preload's own length bound.

---

## Dataset-Path Consent (`dataset-consent.ts`)

`sanitizeDatasetPath` proves a path is *well-formed and real*. It does not prove the **user** chose it — and under the "compromised renderer" threat model that distinction is the whole point: a headless renderer compromise could hand over `~/.ssh`, which passes every check above.

`src/main/dataset-consent.ts` closes that gap with an in-memory allowlist:

```typescript
const consented = new Set<string>();

/** Record a directory the user picked via the native dialog as consented for mounting. */
export function recordConsentedDatasetPath(p: string): void { /* consented.add(path.resolve(p)) */ }

/** True iff `resolvedPath` (already path.resolve'd) was user-selected. */
export function isDatasetPathConsented(resolvedPath: string): boolean {
  return consented.has(resolvedPath);
}
```

`dialog:open-directory` calls `recordConsentedDatasetPath` on the path the OS picker returned; `docker:start-training` refuses any non-empty path that is not on the list. Getting onto the list requires a real dialog invocation and a physical user selection, which a compromised renderer cannot forge.

The set is per-app-run and deliberately not persisted: the renderer always re-selects the dataset within the same session, because `TrainSection` holds `datasetPath` in component state rather than on disk.

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

`AuthService.getAuthHeader()` returns `Bearer <jwt>` and is Main-only. It is called by `ClientProjectService`, `InferenceService` and `InferenceStreamService` — all of which *are* reachable from the renderer over IPC — but the header itself is attached inside Main and only the response data crosses back. No IPC handler ever returns the header or the token to the renderer.

The one asymmetry worth knowing: the backend scopes `Authorization: Bearer` acceptance to native clients (SE-9). A Bearer token is honoured only when the request also carries the `X-FedLearn-Client: fedlearn-desktop` marker header, which `src/main/http.ts` sets as an instance-wide default so every service gets it without per-call wiring. The marker is an intent signal, **not** a secret; browsers stay strictly cookie-only.

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

### Saved credentials ("Save password")

`46aea4d` added an explicit opt-in that persists the *login credentials*, not just the session. It follows the same posture as the JWT and shares the same store file (`fedlearn-auth`, key `savedCredentials`):

- `saveCredentials()` refuses to write anything when `safeStorage.isEncryptionAvailable()` is false, and scrubs any prior blob — a reversible secret is never written to disk.
- When encryption is available it stores `safeStorage.encryptString(JSON.stringify({ username, password }))` as base64.
- `getSavedCredentials()` decrypts on demand for the login form's pre-fill; a decrypt failure (keychain key rotated) scrubs the stale blob and returns `null`.
- Unchecking the box calls `clearSavedCredentials()`.

The plaintext credentials do cross the bridge in both directions here — inherent to pre-filling a login form — but the *encrypted blob* never leaves Main.

---

## Session Expiry and Server-URL Rebinding

Two related mechanisms, both DE-8 era, both living in `AuthService`.

**401 ⇒ session expired (`http.ts` + `auth.service.ts`).** `installUnauthorizedHandler` puts exactly one response interceptor on the shared axios instance (it ejects any previous one, so handlers never stack). A 401 on any authenticated call — inference, client projects, generation — fires `handleSessionExpired()`, which clears both storage backends and pushes `auth:session-expired` to the renderer so it returns to the login screen instead of showing opaque per-call errors. The handler fires whether the 401 arrives as a resolved response (every current service uses a permissive `validateStatus`) or as a rejected promise.

The auth handshake itself is excluded by `isAuthHandshakeRequest`, matching `/auth/(login|me)`: a 401 there means "wrong credentials" or "not logged in yet", not "an existing session went stale". Without that exclusion the app would loop — show login → submit → 401 → re-show login. The frontend's `axiosConfig.ts` carries the equivalent exclusion.

`getAuthHeader()` also checks `expiresAt` *proactively* and routes through the same `handleSessionExpired()` rather than arming a request with a token already doomed to 401. `handleSessionExpired()` is guarded on `hasStoredSession()`, so a burst of concurrent in-flight failures clears once and signals once.

**Changing the server URL invalidates the session.** A JWT and the credentials are minted by *one* backend and must never be sent to another. `setApiUrl()` therefore calls `handleSessionExpired()` whenever the URL actually changes — covering both a legitimate switch and a compromised renderer calling `setServerUrl('https://attacker…')` in the hope that the next authenticated call ships the Bearer token to the new host. Startup deliberately loads `apiBaseUrl` directly in the constructor rather than through `setApiUrl`, so a persisted session survives a normal relaunch (`77cb95e`).

---

## Transport Policy — Refusing Remote Plaintext HTTP

Credentials and the session token flow to whatever URL the user configures, so DE-13 (`2de75a7`) made plaintext `http://` to a **non-loopback** host a refusal rather than a warning.

The rule lives in `src/shared/urlSecurity.ts` so Main's policy and the renderer's on-screen copy cannot drift:

```typescript
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (host === 'localhost' || host === '::1') return true;
  return /^127(\.\d{1,3}){3}$/.test(host);   // the whole 127.0.0.0/8 block
}
```

Suffix lookalikes such as `localhost.evil.com` do **not** qualify, and an `http://` URL that fails to parse is treated as remote (fail closed).

`validators.ts → evaluateServerUrl(raw, allowInsecureHttp)` is the whole decision for `auth:set-server-url`:

| Input | Result |
|---|---|
| Not a string / empty / > 512 chars | `{ ok: false, error: 'Invalid server URL' }` |
| No `http(s)://` prefix | `{ ok: false, error: 'URL must start with http:// or https://' }` |
| Remote `http://`, no override | `{ ok: false, code: 'INSECURE_HTTP', error: PLAINTEXT_HTTP_REFUSAL }` |
| Remote `http://` **with** `allowInsecureHttp` | `{ ok: true, url, warning: PLAINTEXT_HTTP_WARNING }` |
| `https://` or loopback `http://` | `{ ok: true, url }` — warning-free |

Accepted URLs are normalized: trailing slashes stripped, `/api` appended if absent. The renderer keys off the machine-readable `code` to show the "Use HTTP anyway" affordance, and the acknowledgement is scoped to the current URL only — editing the field resets it (`AuthModal.tsx`, `SettingsSection.tsx`). The preload forwards only the single known flag (`{ allowInsecureHttp: true }`), never an arbitrary renderer object.

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
2. The Docker socket is **never mounted into training containers** — the only bind in `HostConfig` is the dataset directory
3. The renderer has no knowledge that Docker even exists — it only calls `startTraining(config)` via the API bridge
4. The socket is not even opened unless the Jetson path is taken. `dockerode` connects lazily and the constructor performs no ping, so a macOS/Windows user on a native profile never touches it.

```typescript
// src/main/docker.service.ts
const hostConfig: Docker.HostConfig = {
  // Principle of least privilege — never mount the host Docker socket
  // into the training container.
  AutoRemove: false,
  Binds: [`${config.datasetPath}:/data`],  // ONLY the dataset directory
};
```

If the training container were to receive the Docker socket, it could escape the container and control the host daemon. This is explicitly prevented.

`AutoRemove: false` in that block is **not** a socket defence — it is a log-drain choice. With `AutoRemove: true` Docker deletes the container the instant it exits, before the attached log stream has drained; with `false` the container survives until `stopDockerContainer()` explicitly removes it, so the final bytes reach the log panel.

---

## Log Rendering — XSS Prevention

Container output is untrusted data. A maliciously crafted training script could output HTML/JavaScript. The `LogPanel` component is designed to prevent any XSS from this source.

The panel has grown display features since it was a single `<pre>{logs.join('')}</pre>` — per-line severity colouring, arrival timestamps, a filter box, follow-tail — so each line is now its own memoized row. The XSS property is unchanged, because every one of those pieces is still a **React text node**:

```typescript
// src/renderer/components/LogPanel.tsx
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

`LogLineRow` renders `{line.text}` as a child, never as `dangerouslySetInnerHTML`; severity only selects a CSS class. React's JSX rendering **automatically escapes** all string content — `<script>alert(1)</script>` in a log line renders as literal text, not an executable script tag. So even an attacker who controls the Python training script cannot achieve XSS through container output.

---

## Threat Model Summary

| Threat | Mitigation |
|---|---|
| Compromised renderer executes arbitrary code | `nodeIntegration: false`, `contextIsolation: true`, `sandbox: true` |
| Renderer steals JWT | JWT confined to Main Process; renderer only gets `{ success: boolean }` |
| XSS via malicious log output | `LogPanel` renders React text nodes (auto-escaped); no `innerHTML` |
| Path traversal via dataset path | `sanitizeDatasetPath()` resolves, canonicalizes, and stat-checks the path |
| Renderer mounts a host directory the user never chose | `dataset-consent.ts` allowlist — only a path returned by the native dialog can be bind-mounted |
| Arbitrary IPC channel calls | Preload only exposes a typed, allowlisted API surface; no raw `ipcRenderer` |
| Input injection into Docker args | Double validation (preload + `validators.ts`) with strict regex allowlists |
| Silent training-arm downgrade (client uploads full state dict to a head-only server) | `trainingArm` is validated strictly in Main and an unrecognised value **throws** instead of defaulting to `FULL` |
| Docker socket access from container | Socket never mounted into training containers |
| Navigation to external URLs | `will-navigate` event handler blocks non-app URLs |
| New window/popup abuse | `setWindowOpenHandler` returns `{ action: 'deny' }` for all opens |
| Stale/decryptable JWT on disk | `safeStorage` uses OS keychain; if unavailable, no disk persistence |
| Token replay after OS user password change | Decryption failure triggers `logout()` and graceful re-auth prompt |
| Token shipped to a different backend after a URL change | `setApiUrl()` clears the session and signals the renderer whenever the URL changes |
| Credentials/token over the wire in the clear | Remote plaintext `http://` refused with `code: 'INSECURE_HTTP'` unless explicitly overridden (DE-13) |
| Stale session left usable after server-side revocation | Single 401 interceptor on the shared axios instance → `auth:session-expired` push |
| Console leakage of secrets in packaged builds | `TerserPlugin` with `drop_console` on the renderer **and** preload bundles (Main keeps `console` for electron-log) |
| Crash data exfiltration | `crashReporter.start({ uploadToServer: false })` — dumps stay local |
| Known Chromium/Electron CVEs in the runtime | Electron pinned to `^42.4.0` (bumped from `^34.5.8` in `0be70dc`). See the posture section below — the pin is no longer sufficient on its own. |

---

## Dependency Vulnerability Posture

**This section states a measurement, and measurements of `npm audit` drift as the advisory database moves. Re-run it before relying on it.**

Measured on **2026-08-13** (`cd fedlearn-desktop && npm audit`, lockfile as committed):

```
31 vulnerabilities (2 low, 8 moderate, 19 high, 2 critical)
```

So the earlier claim on this page — "clean at `--audit-level=high`, only four moderate `uuid` advisories remain" — **no longer holds**. That was true immediately after the Electron 34 → 42 bump (`0be70dc`); the tree has since accumulated new advisories without any dependency change on our side.

What the current run reports, grouped by how it should be treated:

| Group | Severity | Packages (and where they come from) | Notes |
|---|---|---|---|
| Shipped runtime deps | high / moderate | `axios` (direct), `electron-updater` (direct) → `builder-util-runtime` — *cross-origin redirect leaks `PRIVATE-TOKEN` and mixed-case `Authorization`*; `form-data` ← `axios`; `@grpc/grpc-js` and `protobufjs` ← `dockerode` | These matter most: `electron-updater` and `axios` sit on the shipped auto-update and REST paths. All report `fixAvailable: true`. |
| Build/packaging toolchain | high / critical | `electron-builder` → `app-builder-lib`, `builder-util`, `builder-util-runtime`, `dmg-builder`, `electron-publish`; plus `js-yaml`, `tar` (critical), `shell-quote`, `nanoid`, `postcss`, `brace-expansion`, `ip-address`, `fast-uri`; `undici` ← `electron` → `@electron/get` | Dev-time only — they run on the release machine, not on a user's install. `fixAvailable: true` for all. |
| Dev server | moderate / critical | `webpack-dev-server` (direct) → `sockjs` → `websocket-driver` (critical); plus `http-proxy-middleware`, `launch-editor` | Local dev only; never shipped. |
| Electron itself | moderate | `electron` (direct) — `ProtocolResponse.url` reuses the default session cache instead of the registering session | `fixAvailable: true`. |
| The long-standing `uuid` case | moderate | `uuid` ← `dockerode` 4.0.x, and separately ← `sockjs`/`webpack-dev-server` | [GHSA-w5hq-g745-h8pq](https://github.com/advisories/GHSA-w5hq-g745-h8pq), missing buffer bounds check in v3/v5/v6. Still the one advisory whose only published fix is a **breaking major** (`dockerode@5.0.1`, `isSemVerMajor: true`), so it remains deliberately deferred. |

The honest reading: everything except the `dockerode` → `uuid` chain has a non-breaking fix available and is simply un-applied. The `overrides` block in `package.json` currently pins only `protobufjs >= 7.5.5` (which resolves to 8.6.1 and is *still* flagged, so the override no longer clears that advisory).

**CI does not gate on `npm audit`.** The desktop job (`.github/workflows/ci.yml`) runs `check_no_skipped_tests.sh`, `npm ci`, `npm run lint`, and `npm run test:coverage` — nothing else. Nothing in the pipeline will fail when this list grows, which is why the number above went stale unnoticed.

---

*Next: [03 — Main Process Deep Dive](./03-main-process.md)*
*Previous: [01 — Overview & Architecture](./01-overview-and-architecture.md)*
