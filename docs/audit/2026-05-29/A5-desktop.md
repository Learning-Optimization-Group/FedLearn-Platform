# A5 — Desktop Orchestrator Audit (`fedlearn-desktop`)

**Date:** 2026-05-29
**Auditor:** A5 (desktop)
**Target:** `fedlearn-desktop/` — Electron 34, dockerode 4, TypeScript, webpack
**Scope:** Electron security model, unsigned auto-install supply-chain risk, `safeStorage` keychain JWT handling, renderer IPC-bridge fail-open fallback, packaged-build CSP, dockerode Jetson device-mount path.
**Builds on:** `docs/audit/2026-05-27/02-frontend-desktop.md` (cited as `[27-Cx]`). This report **verifies, escalates, and extends** those findings with backend cross-checks the prior audit did not perform.

> Strategic direction (Electron vs Tauri vs native-C++-shell-over-mobile-core) is owned by **B5**. This report assesses the unit **as built** and defers the rebuild-vs-port decision to B5. Where I say "rebuild", I mean *this code as currently written cannot ship to production*, not *Electron is the wrong runtime*.

---

## Executive summary

The desktop orchestrator is the **single highest-severity unit in the platform from a supply-chain standpoint**. It auto-downloads and silently installs **unsigned, ad-hoc binaries from a public GitHub repo on every app quit**, with no signature or checksum verification possible because nothing is signed. One write to `anurag2796/FedLearn-Platform` releases (or a GitHub token compromise) is remote code execution on every installed machine. This is `[27-C5]`, and it is **not yet fixed** — `updater.ts:13-14` still sets `autoDownload=true` + `autoInstallOnAppQuit=true`.

Beyond the updater, I verified two findings the prior audit could only assert: (1) the desktop's `auth.service.ts` **breaks the platform's cookie-only JWT invariant** — but its *primary* extraction path is **dead code** (the backend login body has no `accessToken` field; confirmed at `AuthController.java:144-148`), and its working path extracts the HttpOnly cookie value into app storage and replays it as a `Bearer` header, which the backend filter accepts (`JwtAuthenticationFilter.java:43-51`); and (2) the renderer's IPC-bridge fallback **fails open to a fake authenticated `preview-user`** if preload fails to load (`App.tsx:100-105`).

The Main-process hardening is genuinely good (sandbox + contextIsolation + nodeIntegration off, allowlist IPC validation in both preload and main, dataset-path traversal sanitization, least-privilege Docker config with no socket mount). The Jetson device-mount path is **correct and matches the documented invariant** (no `--runtime nvidia`; direct `/dev/nvhost-*` mounts). The damage is concentrated in three places: the updater, the auth-token handling, and the fail-open renderer bridge.

**Net verdict: REFACTOR.** The architecture and Main-process security model are salvageable and above-average for an Electron app. Three subsystems must be rebuilt before any public distribution: the **auto-updater** (kill auto-install, add signing + checksum), the **auth token model** (stop laundering the cookie into Bearer storage), and the **fail-open bridge** (fail closed in packaged builds).

---

## What's actually good (salvage as-is)

Calibrated to a startup, do not waste cycles rewriting these:

- **Main-process window hardening** (`main.ts:45-56`): `nodeIntegration:false`, `contextIsolation:true`, `sandbox:true`, `webSecurity:true`, `allowRunningInsecureContent:false`, `experimentalFeatures:false`, `devTools` gated to dev. This is the correct Electron baseline.
- **Navigation/window lockdown** (`main.ts:170-195`): `setWindowOpenHandler → deny`; `will-navigate` restricted to `file://` rooted in the app dir (packaged) or the dev server origin. Correct.
- **Two-layer IPC validation**: preload allowlists every input (`preload.ts:32-115`) *and* Main re-validates (`ipc.handlers.ts:204-222`) — genuine defense-in-depth, not theater.
- **Dataset-path traversal hardening** (`ipc.handlers.ts:47-82`): NUL-byte rejection, `path.resolve` + residual `..` check, `statSync` existence + directory check before it's interpolated into the Docker bind `${path}:/data`. This closes a real container-escape vector.
- **Least-privilege Docker** (`docker.service.ts:336-341`): explicitly does **not** mount the host Docker socket into the training container; `AutoRemove:false` for post-mortem inspection.
- **Lifecycle cleanup** (`docker.service.ts:132-164`, `main.ts:144-159`): `before-quit` drains native processes and containers, SIGTERM→SIGKILL escalation guarded on actual `exitCode`/`signalCode` (not the misleading `proc.killed`). This is a class of bug most Electron apps get wrong.
- **`safeStorage` posture on disk** (`auth.service.ts:341-369`): when OS encryption is unavailable it refuses to persist (in-memory only) rather than writing reversible base64. The comment explicitly documents fixing a prior base64-is-not-encryption bug. Correct trade-off.

---

## Critical findings

### A5-C1 — Unsigned auto-install supply-chain RCE *(verified, escalated, UNFIXED)*

**Evidence:**
- `electron-builder.yml:67-69` — macOS `identity: null`, `hardenedRuntime: false`, `gatekeeperAssess: false`. No Windows `signtoolOptions`/`certificateFile` either.
- `electron-builder.yml:119-123` — `publish.provider: github`, `owner: anurag2796`, `repo: FedLearn-Platform` (a **public** repo).
- `updater.ts:13-14` — `autoUpdater.autoDownload = true; autoUpdater.autoInstallOnAppQuit = true;`
- `updater.ts:51` — `checkForUpdatesAndNotify()` runs on every launch.
- `main.ts:144-159` — `before-quit` is intercepted; with `autoInstallOnAppQuit=true` electron-updater stages the downloaded artifact and installs on the *next* quit. The UI `UpdateBanner` is **advisory only** — install does not require user consent.

**Why this is the top risk:** electron-updater on macOS/Windows verifies the **publisher code signature** of the downloaded artifact against the running app's signature. With `identity:null` there is **no signature to verify** — Squirrel.Mac / NSIS will install an ad-hoc / unsigned package. The `latest.yml`/`latest-mac.yml` SHA512 only guarantees the bytes match what's in the GitHub release; it does **not** establish that the release author is legitimate. Threat actors who can publish a release (compromised PAT, compromised maintainer account, malicious contributor with release rights, or a repo-transfer/typosquat of the public `anurag2796/FedLearn-Platform`) get **silent RCE on every installed machine on next quit**, with the app's full local privileges including the Docker socket and the user's stored JWT.

**Comment/code contradiction (intent was violated):** `updater.ts:12` comment reads *"Disable auto-download so we can prompt the user first"* immediately above `autoDownload = true`. Someone intended consent-gated updates and the opposite shipped.

**Dev footgun (extends `[27-M6]`):** `updater.ts:18` sets `forceDevUpdateConfig = (NODE_ENV === 'development')` — a developer running `npm run dev` will pull and (on quit) install a *real production release* over their dev build.

**Severity:** Critical. **Major.**

**Recommendation (phased):**
1. **Phase 0 (today, one line):** `autoInstallOnAppQuit = false` and `autoDownload = false`. Updates become check-only + user-initiated download/install. This removes the silent-RCE-on-quit primitive immediately. Also gate the whole updater behind `if (app.isPackaged)` and delete `forceDevUpdateConfig`.
2. **Phase 1 (before any public distribution):** macOS Developer ID signing + notarization (set `identity`, `hardenedRuntime:true`, the entitlements plist already exists at `build/entitlements.mac.plist`, wire `afterSign: scripts/notarize.js` — the hook is already stubbed at `electron-builder.yml:83`); Windows Authenticode (EV or OV cert). electron-updater then enforces publisher-signature match automatically.
3. **Phase 1+:** move release hosting off the public mono-repo. Either a dedicated private releases repo or an S3/`generic` provider behind a CDN, so "can push code" ≠ "can push an update to every user."

---

### A5-C2 — Auth model launders the HttpOnly cookie into Bearer storage; primary path is dead code *(verified against backend, extends `[27-C1]`)*

The prior audit flagged `auth.service.ts:122-125` and asked whether the `accessToken` path is a token leak or dead code. **I cross-checked the backend and resolved it:**

- **The `accessToken` path is DEAD CODE.** The login response body is `Map.of("username", "email", "role")` — there is **no `accessToken` field** (`AuthController.java:144-148`). The backend comment is explicit: *"Cookie-only auth: the JWT lives exclusively in the HttpOnly cookie so it can never be read by JS"* (`AuthController.java:138-143`). So `auth.service.ts:122` (`response.data.accessToken`) **never fires**.
- **The working path defeats the cookie contract anyway.** `auth.service.ts:128-140` reads the `Set-Cookie` header, regex-extracts the `jwtToken` value, and stores it. `getAuthHeader()` then sends it as `Authorization: Bearer <jwt>` (`auth.service.ts:259, 273, 297, 326`). The backend filter accepts Bearer **before** cookie (`JwtAuthenticationFilter.java:43-51`), so this works — but it means the desktop is the **one client that extracts the HttpOnly token into application-readable storage**, the exact thing the `HttpOnly` flag exists to prevent.

**Why it matters for production:** the platform's stated invariant is "no JS-readable token, cookies only." The desktop violates the *spirit* (token now lives in `electron-store`/process memory and is replayed as a Bearer header). For an Electron app this is arguably *acceptable* because the renderer is sandboxed and never sees the token — but it (a) couples the desktop to a backend that must keep accepting Bearer headers forever (if A1/B4 hardens the backend to cookie-only, the desktop breaks), and (b) the dead `accessToken` branch is a latent footgun: if anyone "helpfully" makes the backend return `accessToken` in the body to fix the dead code, every non-Electron client can suddenly exfil the token.

**Severity:** High (correctness + invariant drift; not an active leak today). **Major.**

**Recommendation:**
- Delete the dead `accessToken` branch (`auth.service.ts:117-125`) now — it is a trap.
- Decide the v2 contract explicitly (hand to **B4**): either (i) bless a desktop-specific Bearer path with a documented short-lived token + refresh, or (ii) keep the cookie in a Main-process cookie jar (`session.defaultSession.cookies` / a Node cookie jar on the axios instance) and **never** extract the value — the token stays opaque to the app, matching the web SPA. Option (ii) restores the invariant.
- The 24h hardcoded `JWT_EXPIRY_MS` (`auth.service.ts:36`) is a client-side guess at the cookie maxAge — if the backend changes `cookieMaxAgeSeconds`, the desktop's expiry drifts. Source it from `/auth/me` or a token-introspection response, don't hardcode.

---

### A5-C3 — Renderer IPC-bridge fails OPEN to a fake authenticated user *(verified, escalated from `[27-H10]` to Critical)*

**Evidence:** `App.tsx:83-154` `ensureFedLearnBridge()`. If `window.fedLearnAPI` is undefined (preload failed to load), it installs a no-op fallback that returns:
- `App.tsx:105` — `checkAuth: async () => ({ success: true, authenticated: true, username: 'preview-user' })`
- `App.tsx:100-103` — `login: async (u) => ({ success: true, username: u || 'preview-user' })`

The auth gate at `App.tsx:178-180` reads `r.authenticated === true` → renders the full authenticated app shell. **If the preload script fails to load in a packaged build** (signing/asar/path regression, a future `sandbox` config change, or a packaging bug), the renderer believes the user is authenticated and exposes the full UI. The prior audit rated this H10; given the desktop is an *end-user distributed binary* (not a dev tool), I escalate to **Critical** — fail-open auth in shipped software is a release blocker.

The intent (a browser "preview mode" for UI iteration via `webpack serve`) is legitimate for dev, but the gate is "is `window.fedLearnAPI` present", which is also false in a *broken packaged build*, not just in browser preview.

**Severity:** Critical (fail-open auth in distributed binary). **Major.**

**Recommendation:** Fail closed in packaged builds. Gate the fallback on an explicit dev signal that cannot be true in production — e.g. only install the fallback when `process.env.NODE_ENV !== 'production'` is compiled in via webpack `DefinePlugin`, or when a `__FEDLEARN_PREVIEW__` build flag is set. In a packaged build with no bridge, render a hard error screen ("Desktop bridge failed to initialize — reinstall"), never an authenticated shell. This is a <30-min fix and a hard release gate.

---

## High findings

### A5-H1 — Packaged-renderer CSP keeps `'unsafe-eval'` *(verified, `[27-C3]`)*
`src/renderer/index.html:8` (the **shipped** CSP) and `main.ts:87` (dev) both allow `script-src 'self' 'unsafe-eval'`. Only webpack dev source-map evaluation needs `eval`; the production bundle does not. Keeping it widens the RCE blast radius of any renderer XSS (e.g. a malicious log line — though logs render as text, see A5-M2). **Recommendation:** ship two CSPs — dev with `unsafe-eval`, packaged without. Verify the production bundle runs without it via `npm run package:mac`. Quick win. **Major** (it's the one renderer-side hole left after the good hardening).

### A5-H2 — `font-src https://frontend-cdn.perplexity.ai` in the shipped CSP *(new finding)*
`src/renderer/index.html:8` allowlists `https://frontend-cdn.perplexity.ai` in `font-src` (`[27-Low]` noted it as "unclear why"; I flag it harder). This is almost certainly **copy-paste leakage from an unrelated project** — there is no Perplexity dependency anywhere in this unit. It is a live network egress allowance to a third party in a security-sensitive packaged app. **Recommendation:** remove it; self-host the two Google fonts (Inter, JetBrains Mono) into the asar so the packaged app makes **zero** outbound font/CDN requests (also removes `fonts.googleapis.com`/`fonts.gstatic.com` from CSP and the `preconnect` at `index.html:19-24`). A desktop FL client for healthcare verticals (the pneumonia demo) should not be phoning third-party CDNs.

### A5-H3 — `demuxDockerStream` discards partial multiplexed frames → corrupted/lost log lines *(new finding)*
`docker.service.ts:474-502`. The function reads the 8-byte Docker stream header and, on a frame that spans chunk boundaries, appends the **partial payload to output and returns it immediately**, then unconditionally sets `state.partial = ''` (line 500). The `state.partial` carried in is read into `output` at line 475 but **never repopulated** with the leftover bytes — so the next chunk's continuation is emitted as if it were a fresh frame, and any mid-header split (line 479-481) is treated as final payload. For the Jetson path this means **the user's training logs can be silently corrupted or interleaved with header bytes** under load. The native path (`docker.service.ts:299-300`) is unaffected (raw stdout, no demux). **Severity:** High (observability of the FL run — the explicit platform goal — is compromised on exactly the hardware path that needs Docker). **Recommendation:** carry the unconsumed tail bytes (including sub-8-byte header fragments) into `state.partial` as a `Buffer`, not a string, so multi-byte UTF-8 sequences split across chunks also survive. Add a unit test with a frame split mid-header and mid-payload.

### A5-H4 — Zero tests on the three highest-risk modules *(extends `[27-M7]`)*
4 test files (`api.test.ts`, `ipc.client-handlers.test.ts`, `usePolling.test.ts`, `validators.test.ts`, 341 LOC total). **No test touches** `auth.service.ts`, `updater.ts`, `docker.service.ts`, the `demuxDockerStream` parser, or the `App.tsx` fail-open bridge — i.e. every module in this report's Critical/High list is untested. The validation allowlists are well-tested; the security-critical logic is not. **Recommendation:** as a CI gate, require tests for: (a) `demuxDockerStream` frame splitting, (b) `auth.service` token extraction + expiry + safeStorage-unavailable fallback, (c) `ensureFedLearnBridge` fail-closed in packaged mode, (d) updater config asserts `autoInstallOnAppQuit===false`.

### A5-H5 — No dependency / Electron-specific security scanning in CI *(extends `[27-M12]`)*
No `npm audit`, `osv-scanner`, or `electronegativity` step. Electron 34 is pinned (`package.json:60`) and Electron ships ~monthly Chromium security releases; a pinned-and-forgotten Electron is a growing CVE surface in a binary that auto-updates *itself* but whose Chromium is frozen at build time. **Recommendation:** add `npm audit --omit=dev` + Doyensec `electronegativity` (catches CSP/IPC/nodeIntegration regressions) to PR-time CI (hand release-engineering to **B7**), and a Renovate/Dependabot policy for Electron majors.

---

## Medium findings

- **A5-M1 — Updater `info` payloads typed `any` across the bridge** (`preload.ts:297-315`, `App.tsx:60-64`). `electron-updater` ships `UpdateInfo`/`ProgressInfo` types; use them. Low-risk but `any` on data crossing the trust boundary is exactly where to be strict.
- **A5-M2 — Training logs rendered into the renderer** (`docker.service.ts:526-529` → `docker:training-log` → `App.tsx:208`). Logs are user/process-controlled text and the preload forwards only `string` (`preload.ts:220-227`); confirm the `LogDrawer`/`LogPanel` render as text nodes (not `dangerouslySetInnerHTML`) — the prior audit found no `dangerouslySetInnerHTML` in the repo, so this is likely safe, but with `'unsafe-eval'` still in CSP (A5-H1) a regression here would be exploitable. Removing `unsafe-eval` is the real mitigation.
- **A5-M3 — `electron-store` cast to `any`** (`auth.service.ts:42`, `ipc.handlers.ts:92-93`) — ESM-interop workaround, but loses `Store<AuthStore>` typing on the module that holds the JWT. (`[27-M10]`.)
- **A5-M4 — `electron-store` pinned to 8.2.0** (`package.json:40`) — 8.x is the last CommonJS line (9.x is ESM-only), which is *why* the `any` casts exist. Fine for now; document the pin so no one "upgrades" it and breaks the webpack/CJS main build (cf. memory: `"type":"module"` was removed from this package.json to restore webpack compat).
- **A5-M5 — `disableHardwareAcceleration()`** (`main.ts:199`) hurts rendering of the 10k-line log buffer + any future charts. The comment's rationale ("GPU compute happens in Docker") conflates *training* GPU with *UI* GPU. (`[27-M9]`.) Reconsider once the log view scales.
- **A5-M6 — `MAX_LOG_LINES = 10_000`** (`App.tsx:81`) is a single ever-growing array sliced on every batch; for a multi-hour FL run this is fine for memory but the slice-on-every-RAF (`App.tsx:215-218`) is O(n) per frame. Use a ring buffer if log volume grows.
- **A5-M7 — No artifact integrity for the shipped PyInstaller client bundle.** `electron-builder.yml:40-44` ships `../client-docker/packaging/dist/fedlearn-client` as `extraResources` and `docker.service.ts:218-235` spawns `<resources>/fedlearn-client/fedlearn-client` directly. Inside the signed app bundle this inherits the app signature (once A5-C1 signing lands), but **today, unsigned**, the native client binary is as tamper-able as the updater artifact. Signing (A5-C1) covers this; call it out so it isn't forgotten.

---

## Low / hygiene

- `package.json:8` author email is the literal placeholder `[EMAIL_ADDRESS]`.
- `updater.ts:29` `update-not-available` handler is registered but the renderer's `onUpdateNotAvailable` only fires via the *manual* `updater:check` relay (`ipc.handlers.ts:478-482`), not the startup check — minor UX inconsistency.
- `hardware.probe.ts:38` runs `nvidia-smi` via `execFile` (no shell, args arrayified) — safe; 2s timeout is sensible.
- `ipc.handlers.ts:168` derives `jetson` purely from `process.arch === 'arm64'` on non-darwin — an Apple-Silicon Asahi-Linux or ARM server would be misclassified as Jetson. Edge case; document.
- Native bundle is single-Windows-variant (`electron-builder.yml:39` "only one Windows variant at a time") — CPU vs CUDA ship as separate installers, which complicates the update channel (a CPU user could be offered a CUDA build). Coordinate with the updater channel design.

---

## Jetson device-mount path — verification (per assignment)

**Verdict: correct, matches the documented invariant.** `docker.service.ts:41-48` defines the full `/dev/nvhost-*` + `/dev/nvmap` + `/dev/nvhost-gpu` device set; `docker.service.ts:343-350` mounts them via `HostConfig.Devices` for the `jetson` profile and the comment correctly states the `--runtime nvidia` flag is prohibited (it hangs on Jetson searching for PCIe discrete-GPU metadata — matches `CLAUDE.md`). The `discrete` profile correctly uses `DeviceRequests` (`--gpus all`) instead. This is the one place the code demonstrates real hardware-specific domain knowledge and it is right. Salvage as-is. One nit: the device list is hardcoded; JetPack r36 (Orin, newer L4T) may expose a different node set — pin it to the tested JetPack and document the version coupling.

---

## Decision table

| Module / file | Verdict | Rationale |
|---|---|---|
| `updater.ts` + `electron-builder.yml` signing | **rebuild** | Unsigned auto-install-on-quit from a public repo = silent RCE on every machine; cannot ship. Kill auto-install now, add signing + checksum before distribution. |
| `auth.service.ts` (token model) | **refactor** | Working path launders HttpOnly cookie into Bearer storage; primary `accessToken` path is dead code. Delete dead branch; move to opaque cookie-jar or a blessed desktop token contract (B4). |
| `App.tsx` `ensureFedLearnBridge` fail-open | **refactor** | Fails open to a fake authenticated `preview-user` if preload fails — must fail closed in packaged builds. Small, surgical fix. |
| `main.ts` window/Nav hardening | **salvage** | Correct Electron baseline (sandbox + isolation + nodeIntegration off + nav lockdown). |
| `preload.ts` + `ipc.handlers.ts` validation | **salvage** | Two-layer allowlist validation + dataset-path traversal sanitization is genuinely solid. |
| `docker.service.ts` Jetson/native dispatch + lifecycle | **salvage** | Jetson device mounts correct; least-privilege Docker; clean SIGTERM→SIGKILL shutdown. |
| `docker.service.ts` `demuxDockerStream` | **refactor** | Drops partial frames → corrupted Jetson-path logs; isolated parser bug with a clear fix + test. |
| Renderer CSP (`index.html`) | **refactor** | Drop `'unsafe-eval'` in packaged build; remove stray Perplexity CDN; self-host fonts. |
| `hardware.probe.ts` | **salvage** | Safe, bounded, best-effort detection. Minor arm64-Linux misclassification nit. |
| Test suite | **refactor** | Validators well-covered; the three Critical modules have zero tests — add as CI gates. |
| CI security scanning | **rebuild** (absent) | No `npm audit`/`electronegativity`/Renovate; build it (with B7). |

**Unit-level verdict: REFACTOR** — strong Main-process security foundation; three subsystems (updater, auth-token model, fail-open bridge) must be fixed before public distribution. The Electron-vs-native question is B5's; nothing here forces a runtime change.

---

## Prioritized recommendations

**P0 — release blockers (do before any binary leaves a developer's machine):**
1. `updater.ts:13-14` → `autoDownload=false`, `autoInstallOnAppQuit=false`; gate updater behind `app.isPackaged`; delete `forceDevUpdateConfig` (A5-C1 Phase 0). One line of real risk removed.
2. `App.tsx` fail-open bridge → fail closed in packaged builds; render a hard error instead of an authenticated shell (A5-C3).
3. Delete the dead `accessToken` branch (`auth.service.ts:117-125`) so no one resurrects it into a real token leak (A5-C2).

**P1 — before public distribution:**
4. macOS Developer ID signing + notarization (entitlements + `afterSign` hook already stubbed); Windows Authenticode (A5-C1 Phase 1). Move releases off the public mono-repo.
5. Drop `'unsafe-eval'` from the packaged CSP; remove the Perplexity CDN entry; self-host fonts (A5-H1, A5-H2).
6. Decide the v2 desktop auth contract with B4 (opaque cookie jar vs blessed Bearer + refresh) (A5-C2).

**P2 — quality / observability:**
7. Fix `demuxDockerStream` partial-frame handling + add a split-frame test (A5-H3).
8. Add tests for updater config, auth service, and the fail-closed bridge; wire `npm audit` + `electronegativity` + Renovate into PR-time CI (A5-H4, A5-H5; coordinate with B7).
9. Type the updater payloads; reconsider `disableHardwareAcceleration` once the log view scales (A5-M1, A5-M5).

---

## Open questions / uncertainty (flagged, not fabricated)

- **electron-updater signature semantics with `identity:null`:** I am confident the `latest*.yml` SHA512 does not establish author authenticity, and that macOS/Windows refuse to *silently* install across a signing-identity change *when the app is signed*. I have **not** empirically built and run the unsigned auto-install on a clean machine to confirm exactly what Squirrel.Mac does with an ad-hoc-signed delta — the precise failure/success mode of the unsigned install is **untested here**. The conclusion (do not ship unsigned auto-install) holds regardless of that detail.
- **Whether the backend will keep accepting `Authorization: Bearer`** after A1/B4 hardening. If the backend goes cookie-only, the desktop auth path breaks — this is a cross-unit dependency, not a desktop-local decision.
- The native PyInstaller bundle's own supply-chain (its Python deps) is **A4's** scope; I only flag that it ships unsigned alongside the app (A5-M7).
