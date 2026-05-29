# A2 — Web Frontend Audit (v2 Greenfield Design Input)

**Unit:** `frontend/` — React 19, Vite 6, TypeScript (strict), Tailwind v4
**Date:** 2026-05-29
**Branch:** `main-clean`
**Auditor role:** Senior frontend/systems engineer, calibrating for a production-grade startup.
**Builds on:** [`../2026-05-27/02-frontend-desktop.md`](../2026-05-27/02-frontend-desktop.md) (prior combined frontend+desktop review) and the cross-cutting themes in [`../2026-05-27/README.md`](../2026-05-27/README.md). This report **extends** that work — it does not re-derive the same findings. Where the prior audit already flagged an issue, I cite it (e.g. *prior M1*) and add what the v2 rebuild decision requires: severity re-grading, root-cause depth, and a concrete target architecture.

---

## Executive summary

The frontend is a competent React 19 + Vite 6 SPA with a genuinely good auth posture (cookie-only HttpOnly JWT, `withCredentials`, silent `/auth/me` probe, fail-fast on missing API URL) and a thoughtfully engineered live-log subsystem (module-level `logStore` with stable monotonic keys, telemetry sparkline cache). That core is **salvageable and worth keeping**. But three classes of problem make it not production-grade: (1) a **live functional bug** — the entire admin and permission UI is gated on `role === 'ADMIN'` while the backend now emits the V5 `PLATFORM_ADMIN` string, so every admin surface is silently dead; (2) **zero automated tests** against the most complex code in the repo (STOMP races, log merge, auth bootstrap); and (3) **no Content-Security-Policy anywhere** combined with eager-loaded third-party chart/animation libraries, which turns any vendor XSS into a full credentialed-session compromise. Add a single 1.0 MB unsplit JS bundle, ~18 `any`-typed escape hatches gated only at `warn`, and a 60fps render-storm animation fed entirely mock data, and the verdict is **refactor, not rebuild** — keep React 19 + Vite, repair the identity contract, add CSP + Vitest + Playwright, code-split, and harden the STOMP layer. Next.js is **not** warranted: there is no SEO/SSR requirement, the cookie-first-party constraint is already solved by the Vite proxy, and an SSR layer adds a server tier this single-EC2/CloudFront deployment does not need.

---

## What this app actually is

- **Pure client-rendered SPA.** `main.tsx` mounts `BrowserRouter`; everything is client routing (`App.tsx:60-101`). Deployment is static (S3 + CloudFront per the `vite.config.ts` trailing comment, lines 70-95).
- **Provider stack** (`main.tsx:25-42`): `ErrorBoundary → ThemeProvider → BrowserRouter → AuthProvider → NotificationProvider → ToastProvider → App`. Clean, no prop-drilling.
- **State** is entirely React Context + local `useState`/`useRef`. No Redux/Zustand/TanStack Query. Server state is hand-rolled with `useEffect` + Axios.
- **Three real-time channels over one STOMP/WS endpoint** (`/ws-logs`): `/topic/logs/{id}` (LogViewer), `/topic/status/*` + `/topic/results/*` (DashboardV2), `/user/queue/notifications` (NotificationContext).
- **6,166 lines** of TS/TSX across 47 files. Largest: `DashboardV2.tsx` (548), `ProjectDetailPage.tsx` (436), `LogViewer.tsx` (358), `apiServices.ts` (279).

---

## Architecture assessment

### Component & state architecture — **salvage**

The shape is sound for an app this size. Context for cross-cutting concerns (auth, theme, toast, notifications) and local state for views is the right call; reaching for Redux here would be over-engineering. Two structural weaknesses matter for v2:

1. **Server-state is hand-rolled and inconsistent.** Every view re-implements the same `useState([]) + useEffect(fetch) + try/catch(setError)` triad: `DashboardV2.tsx:92-120`, `TrainingInsightsView.tsx:9-22`, `ModelsView.tsx:46-56`, `DatasetsView.tsx:39-49` all call `api.fetchProjects()` independently with no shared cache, no dedup, no background revalidation. Navigating dashboard → training → models → datasets fires `fetchProjects()` four times. There is no request cancellation on unmount (`AuthContext` uses a `cancelled` flag at line 41/58, but the views do not), so a fast nav can `setState` on an unmounted view.

2. **`DashboardV2.tsx` is a 548-line god component** owning 14 `useState` hooks, two STOMP subscriptions, project CRUD, results aggregation, and five modals. It is the single hardest file to test and the one with the most `any` leaks (lines 181, 187, 210, 218, 223, 240, 259). This is the natural seam for extraction.

> **v2:** Adopt **TanStack Query** for all server state (projects, results, memberships, access requests). It gives caching, dedup, background refetch, and `AbortSignal` cancellation for free and deletes the four duplicate fetch triads. Keep Context only for truly global, low-churn state (auth identity, theme, toast). Split `DashboardV2` into a container + `<ProjectGrid>` + the modal set.

### Cookie-auth + Axios `withCredentials` contract — **salvage**

This is the strongest part of the codebase and the v2 design should preserve it verbatim. Evidence it is correct:

- `axiosConfig.ts:27-30` — single Axios instance, `withCredentials: true`, no Bearer header, no `localStorage`. Matches the platform invariant exactly.
- `axiosConfig.ts:5-7` — fail-fast on missing `VITE_FEDLEARN_API_URL` in prod builds.
- `axiosConfig.ts:42-62` — the 401 interceptor correctly distinguishes the silent `/auth/me` probe (line 35, 49-51) and the explicit login attempt from a "hard logout" 401 on data routes (fires `authError` → `App.tsx:45-52` → `logout()`). 403 is intentionally **not** a logout (comment at 55-58). This is exactly the right semantics.
- Backend confirms the contract: `AuthController.java:131-152` sets `jwtToken` as `HttpOnly`, `secure`, `SameSite` (default `Strict`), and the body returns only identity. `/auth/me` (lines 161-181) is a clean 401 probe.

**Carry forward unchanged.** The only change v2 needs here is the role type (next finding) and adding `VITE_SERVER_ROOT_URL` to the fail-fast set (*prior C2* — still valid; `NotificationContext.tsx:14` and `DashboardV2.tsx:21` both fall back to a hardcoded `http://...:8081` that breaks under HTTPS).

### V5 identity TYPE mismatch — **rebuild the type contract (live functional bug, not cosmetic)**

The prior audit graded this **M1/Medium** ("Frontend will mis-render permission UI *as the backend migrates*"). **I re-grade it Critical** because the migration has already happened on the backend and the bug is live today:

- The backend `/auth/login` and `/auth/me` return `"role", appUser.getPlatformRole()` (`AuthController.java:147` and `:179`).
- `getPlatformRole()` returns the V5 `platform_role` column, whose admin value is the literal string **`"PLATFORM_ADMIN"`** — confirmed by `BootstrapRunner.java:122` (`admin.setPlatformRole("PLATFORM_ADMIN")`) and `:92` (`existsByPlatformRole("PLATFORM_ADMIN")`), and by the backend `@PreAuthorize("hasRole('PLATFORM_ADMIN')")` at `TestEmailController.java:35`.
- The frontend type is `role: 'USER' | 'ADMIN'` (`apiServices.ts:59`, `AuthContext.tsx:15`, `User.role` `apiServices.ts:195`, `AdminUser.role` `:239`).
- Every consumer checks the wrong literal: `Sidebar.tsx:99` (`currentUser?.role === 'ADMIN'` gates the admin nav link), `AdminUsersPage.tsx:11,41,75,121`, `AdminProjectsPage.tsx:25`, `ProjectDetailPage.tsx:218` (`const isAdmin = currentUser?.role === 'ADMIN'`, then lines 219-221 derive `canManageMembers`/`canManageClients`/`canSeeManagement`).

**Consequence:** a real `PLATFORM_ADMIN` logs in, the backend returns `role: "PLATFORM_ADMIN"`, and `"PLATFORM_ADMIN" === "ADMIN"` is `false` everywhere. The admin nav link never renders, the admin pages bounce (`AdminUsersPage.tsx:41` `if (currentUser?.role !== 'ADMIN')`), and every `isAdmin`-derived permission in `ProjectDetailPage` is `false`. The admin surface is **dead from the browser** — mirroring the backend's own dead-admin finding (*prior README P0 #2*, the `hasRole('ADMIN')` vs `PLATFORM_ADMIN` mismatch). TypeScript cannot catch it because the value arrives as an untyped JSON string cast to the declared union — the compiler trusts the lie.

This is **deeper than a relabel.** The V5 model has three orthogonal role layers (platform / org / project — see project `CLAUDE.md` "Identity layers (V5)"), but the frontend `AuthIdentity` only models a single flat `role`. There is no representation of org membership/role or the user's org context at all, so multi-tenant UI (org switcher, org-admin views) cannot be built on the current type.

> **v2:** Model the identity contract to match V5 exactly:
> ```ts
> type PlatformRole = 'USER' | 'PLATFORM_ADMIN';
> type OrgRole = 'OWNER' | 'ADMIN' | 'MEMBER';
> type ProjectRole = 'OWNER' | 'MEMBER' | 'CLIENT';
> interface AuthIdentity {
>   username: string; email: string;
>   platformRole: PlatformRole;
>   orgs: { orgId: string; orgRole: OrgRole }[];   // backend must surface this
> }
> ```
> Validate the wire payload with **Zod** at the Axios boundary so a backend/frontend contract drift fails loudly in dev instead of silently disabling the admin UI. This is the single highest-leverage correctness fix in the frontend.

### STOMP-over-WebSocket live logs — **refactor**

The live-log subsystem is the most thoughtfully engineered part of the app, and the `logStore` design (`logStore.ts`) is genuinely good: module-level cache survives modal unmount, monotonic never-reused IDs (lines 47-54) used as React keys (documented rationale at 19-24), per-project subscription fan-out, trim at `MAX_LOGS_PER_PROJECT = 2000`. Keep the store.

But the wiring around it has real defects that a production startup cannot ship:

1. **Three independent STOMP clients to one endpoint** — `NotificationContext.tsx:26`, `DashboardV2.tsx:127`, `LogViewer.tsx:91` each open their own `new StompClient({ brokerURL })`. A user on the dashboard with the log modal open holds **3 concurrent WS connections** to the same `/ws-logs` broker. This does not scale: N browser tabs × 3 = connection multiplication against a single Spring STOMP broker that also fronts the FL-server stdout fan-out.

2. **Auth is implicit-cookie only, with a broken HTTP fallback** (*prior C2*, still valid). `NotificationContext.tsx:14` / `DashboardV2.tsx:21` default `VITE_SERVER_ROOT_URL` to `http://${hostname}:8081`; served over HTTPS this yields a `ws://` (not `wss://`) upgrade → mixed-content block → cookie loss. The handshake relies on the browser attaching the cookie to the WS upgrade, which works first-party (the `ec2demo` Vite-proxy trick) but is fragile cross-origin.

3. **History/live race** (*prior H4/H5*, confirmed). `LogViewer.tsx:67-86` (historical fetch) and `:89-145` (live subscribe) run as **parallel `useEffect`s with no ordering**. The dedup key is `timestamp+message` (`logStore.ts:99`); backend log timestamps are second-resolution (`new Date().toLocaleTimeString` at `LogViewer.tsx:106,134`), so two distinct lines in the same second with identical text collide and one is silently dropped — or a live line that arrives before history merges produces a duplicate that the prepend-merge cannot dedup. `isConnected` is set in `onConnect` (line 97) with no mounted-guard, so StrictMode double-mount can flip it on a torn-down client.

4. **Wildcard topic subscriptions** — `DashboardV2.tsx:133` subscribes `/topic/status/*` and `:153` `/topic/results/*`. The user receives status/results for **every project on the platform**, not just their own, then filters client-side. That is both a minor info-leak and wasted bandwidth; the topic should be scoped server-side to the authenticated user's projects.

> **v2:** One shared STOMP connection via a `WebSocketProvider` context (single client, ref-counted subscriptions per topic). Force `wss://` derivation and fail-fast if `VITE_SERVER_ROOT_URL` is missing in prod. Serialize history-then-live with a sequence number or monotonic server-side log ID (stop deduping on second-resolution timestamps — have the backend emit a stable `id`). Scope topics to the user. Consider whether SSE (one-directional server→client) is a better fit than full STOMP for log streaming — logs are pure downstream and SSE auto-reconnects with `Last-Event-ID` for gap-free resume, which directly solves the history/live race.

### Performance-observability of FL runs (the product's reason to exist) — **refactor**

This is the dimension the assignment calls out specifically, and it is half-built:

- The frontend **data plane is wired end-to-end**: `DashboardV2.tsx:153-168` subscribes `/topic/results/*` and merges `ProjectResult` (`{serverRound, loss, accuracy, gpuUtilization}`, `apiServices.ts:42-48`) into `resultsMap`; `LogViewer.tsx:115-129` parses inline `RoundResultDto` into the telemetry sparkline; `TrainingInsightsView`, `ModelsView`, `DatasetsView` all fetch real `/results` (confirmed: `api.fetchProjects()` + `fetchProjectResults()` at `TrainingInsightsView.tsx:17-22`, etc.).
- **But per the prior framework audit (`../2026-05-27/README.md` Theme 3, `04-observability.md`), no Python caller POSTs to `RoundResult`** — so `/topic/results/*` is silent and these panels render `'---'`/empty in practice. The frontend is rendering an empty telemetry surface waiting for data the framework never sends.
- **The flagship FederationOrrery is fed hardcoded mock data.** `DashboardV2.tsx:375-386` passes literal clients (`jetson-orin-1`, `lab-mac-m2`, `gpu-server-x`, `clinic-node`, `research-pc`) with fixed `round={28} totalRounds={50}`. None of it is real. A user watching their federation sees a fictional animation.
- **The orrery is a 60fps render storm.** `FederationOrrery.tsx:31-36` runs an unconditional `requestAnimationFrame` loop calling `setT(performance.now()/1000)` every frame whenever `spin` is true (default `true`, `:26`). That triggers a full React re-render of the SVG ~60×/sec, permanently, on the dashboard — for purely decorative motion over fake data. On a battery-constrained client (Jetson/laptop) this is wasted CPU/GPU all day.
- **DeComFL — the platform's actual research differentiator (zeroth-order, O(K·P) communication) — is not surfaced in the UI at all.** `StartProjectModal.tsx:115-118` offers only `FedAvg / FedAdam / FedAdagrad`. The whole point of the platform (the DeComFL paper) has no representation in the run-configuration UI or in any observability panel.

> **v2:** Drive the orrery and all telemetry from the **real** `/topic/results` + `/topic/status` streams (this also requires the framework fix from the prior audit — flag the dependency). Replace the rAF re-render loop with CSS/SVG keyframe animation (the codebase already does this for the round-pulse at `FederationOrrery.tsx:108`, `animation: 'pulse-ring 2.6s ...'`) or `framer-motion`'s declarative transforms — zero React re-renders. Add a `DeComFL` strategy option and an observability panel for its convergence (loss vs. communication-rounds, since cost is dimension-independent — that is the story to tell). Gate decorative animation behind `prefers-reduced-motion`.

### Missing Content-Security-Policy — **rebuild the security-header layer (Critical)**

*Prior C4* flagged this; I confirm and sharpen it. Verified:

- `frontend/index.html` ships **no** `<meta http-equiv="Content-Security-Policy">` (full file read; only a theme-init inline script and Google Fonts `<link>`s).
- The backend sets **no** CSP either. `SecurityConfig.java:128-144` configures CORS, CSRF-disable, and `frameOptions(sameOrigin)` — but **no `contentSecurityPolicy(...)`, no HSTS, no `Referrer-Policy`, no `X-Content-Type-Options`**. The only header is frame-options, and it is `sameOrigin` (not `DENY`) so the SPA is iframe-embeddable by same-origin pages.

**Why this is Critical, not Medium, for a startup:** the auth model is cookie-based. There is no `localStorage` token to steal (good), but a CSP-less page means any injected script — a compromised npm dep in the eagerly-loaded `recharts`/`framer-motion`/`lucide-react`/`react-icons` graph, or a future `dangerouslySetInnerHTML` (none today — verified) — runs with the user's session and can fire credentialed POSTs to the API. The `index.html` already loads Google Fonts cross-origin and the build inlines third-party chart libs, so the attack surface is non-trivial. `SameSite=Strict`/`Lax` cookies blunt CSRF but do **nothing** against same-origin XSS.

> **v2 CSP strategy (defense in depth, set at BOTH layers):**
> - **Backend is authoritative.** Add a CSP via Spring Security `headers.contentSecurityPolicy(...)` (or `helmet` if a proxy fronts it — *prior suggested-deps* lists `helmet`). Vite/React do not need `unsafe-inline` for scripts (React renders via JS, not inline event handlers); the one inline theme script in `index.html` should move to a hashed/nonce'd script or an external module so the policy can be `script-src 'self'`.
> - Baseline: `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self' https://<api-host> wss://<api-host>; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`.
> - Add `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`. Flip `frameOptions` to `DENY` (the SPA is never legitimately framed).
> - Tailwind v4 produces a static stylesheet, so `style-src 'unsafe-inline'` is only needed for the inline `style={{...}}` attributes used heavily throughout (`DashboardV2`, `StartProjectModal`, etc.) — note these are **attribute** styles, which CSP `style-src` governs; migrating hot paths to Tailwind classes would let you drop `unsafe-inline` from `style-src` later.
> - **Verify with a report-only rollout first** (`Content-Security-Policy-Report-Only`) to catch violations before enforcing.

### Bundle size & code-splitting — **refactor**

- The committed prod bundle is a **single ~1.0 MB JS chunk**: `frontend/dist/assets/index-BfVmNIer.js` = 1,002,991 bytes (+ 71 KB CSS), built May 19. No vendor split, no route split.
- `vite.config.ts` sets `minify: 'esbuild'` and `sourcemap: false` (good — the sourcemap rationale comment at lines 38-49 is correct security thinking) but has **no `build.target`** and **no `manualChunks`** (confirmed — full config read). `build.target` defaults to Vite's `'modules'`/baseline-widely-available, acceptable, but the lack of chunking is the problem.
- **Zero `React.lazy` / dynamic `import()` anywhere** (grep returned empty). `App.tsx:6-21` statically imports every route component including `DashboardV2` (which transitively pulls `FederationOrrery`, `LogViewer`, all five modals, `recharts`, `framer-motion`). The login/landing path therefore downloads the entire authenticated app — `recharts` (~3.4 MB on disk) and `framer-motion` (~3.6 MB on disk) ship to a user who only wants the login form.
- **Two icon libraries both shipped** (*prior depcheck suggestion*): `lucide-react` (23 MB on disk) **and** `react-icons` (39 MB on disk) are both in `package.json` dependencies. Tree-shaking helps, but shipping two icon systems is pure waste — pick one.

> **v2:** `React.lazy` the authenticated shell and the heavy modals (`DashboardV2`, `LogViewer`, `ResultsModal`, the chart-bearing views) behind `<Suspense>`; the landing/login bundle should be tiny. Add `manualChunks` to split `recharts` and `framer-motion` into a vendor chunk so they cache across deploys. Standardize on **one** icon library (`lucide-react` — it is already the dominant import). Add `rollup-plugin-visualizer` to CI and a bundle-size budget gate (*prior Phase 3*). Target: < 150 KB gzipped for the initial (unauthenticated) load.

### Zero unit tests — **rebuild the test layer (Critical gap)**

Confirmed: there is **no test file, no Vitest config, no `test` script** in the frontend (`package.json` scripts are `dev/build/lint/format/preview` only; no `vitest`/`jest`/`@testing-library` in devDependencies). *Prior M7* graded this Medium for the combined frontend+desktop unit; **for the web frontend specifically I grade it High** because the untested code is exactly the code most likely to break silently:

- the auth bootstrap + 401 interceptor branching (`axiosConfig.ts`, `AuthContext.tsx`) — a regression here logs everyone out or loops redirects;
- the `logStore` merge/dedup/trim invariants (`logStore.ts`) — the prepend-key correctness it documents at length is untested;
- the STOMP history/live race (`LogViewer.tsx`);
- the role-gating logic that is **currently shipping broken** (the `'ADMIN'` bug above would have been caught by one test asserting `PLATFORM_ADMIN` unlocks the admin nav).

A startup shipping a paid multi-tenant product with zero frontend tests is carrying unbounded regression risk on every deploy.

> **v2 testing stack:** **Vitest** (reuses the Vite pipeline, near-zero config) + **@testing-library/react** + **MSW** (mock the Axios/REST layer) for unit/integration; **Playwright** for one critical e2e: login → create project → start server → watch live logs. Wire `vitest --coverage` into the PR CI that *prior README P0 #4* mandates (there is currently no PR-time CI at all). First-10-tests target: auth interceptor branches, `logStore` invariants, role-gate rendering, the STOMP dedup.

### `no-explicit-any` at `warn`, ~18 sites — **refactor**

*Prior H2* flagged this; I re-counted: **18 `any` sites** (grep across `src`), gated at `warn` not `error` (`eslint.config.ts:30` — note the prior audit cited `eslint.config.js`; the file is now `eslint.config.ts`, same rule). The dangerous ones are not the `catch (err: any)` blocks (annoying but low-risk) — they are the **payload types**:

- `DashboardV2.tsx:181` `handleCreateProject(projectData: any)` — defeats the `ProjectData` interface (`apiServices.ts:16-22`); a missing required field compiles clean and 400s at runtime.
- `DashboardV2.tsx:218` `handleStartSubmit(projectId, config: any)`.
- `EditProjectModal.tsx:46` `(modelOptions as any)[project.modelType]` and `:71` submit payload `as any` — a field-name typo is invisible until runtime.
- `apiServices.ts:122-123` `fetchProjectLogs` returns `AxiosResponse<any[]>` then `LogViewer.tsx:74` maps `entry: any` — the log shape (`level/message/timestamp/stackTrace`) is known and is already typed as `LogEntryInput` in `logStore.ts:27`; reuse it.

> **v2:** `@typescript-eslint/no-explicit-any: 'error'`, fix the payload sites with real types (the interfaces already exist), and adopt `@typescript-eslint/strict-type-checked` (requires `parserOptions.project`). Validate untyped wire data (`fetchProjectLogs`, `/auth/me`, STOMP message bodies) with **Zod** at the boundary so `any` is never reintroduced through JSON.

### Vite-mode ↔ Spring-profile mapping & `strictPort:5173` CORS coupling — **salvage**

This is load-bearing and **correct** — do not "clean it up" in v2. Verified:

- `vite.config.ts:46-53` sets `strictPort: true` on `:5173` with an inline comment explaining the backend CORS allowlist is keyed on `:5173` and a silent shift to `:5174` produces opaque `Access-Control-Allow-Credentials` failures. Confirmed against `SecurityConfig.java:90-108` (CORS uses `allowedOriginPatterns` from the `app.cors.allowed-origins` CSV with credentials enabled).
- `.env.{development,ec2demo,production}` map 1:1 to Spring profiles, and the `ec2demo` proxy trick (`vite.config.ts:14-30` + `.env.ec2demo`) keeps cookies first-party to `localhost:5173` to sidestep Safari third-party-cookie blocking. This is a genuinely clever solution to the cross-origin cookie problem and is the **reason Next.js/SSR is unnecessary** — the hard part (first-party cookies to a remote backend) is already solved without a server tier.

**Keep this verbatim.** Two small v2 hardenings: (a) `VITE_SERVER_ROOT_URL` should join `VITE_FEDLEARN_API_URL` in the prod fail-fast (`axiosConfig.ts:5-7` only checks the API URL today; the WS root silently falls back to broken `http://`), and (b) the committed `VITE_PROXY_TARGET=https://fedlearn.duckdns.org` (*prior C6*) is a DNS-squat risk on a free service — move to a controlled domain before going more public.

---

## Minor findings (carried/confirmed from prior audit, still open)

- **NotificationContext leaks across login boundary** (*prior H9*) — `NotificationContext.tsx:35` keeps last 50 notifications, never cleared when `currentUser` becomes null (effect at `:23-50` re-runs on user change but stale notifications persist). PII leak across a logout→login on a shared machine. **Fix:** clear `notifications`/`unreadCount` when `currentUser` is null.
- **`logStore` cache `Map` is unbounded over project count** (*prior H8*) — `MAX_LOGS_PER_PROJECT` is per-project, but `cache` (`logStore.ts:43`) never evicts dead projects; navigating 100 projects accumulates up to 200k entries in memory. **Fix:** LRU eviction over projectIds.
- **`(window as any).global = window`** (`main.tsx:16`) + `define: { global: {} }` (*prior H7*) — one of these is dead/conflicting; STOMP's old SockJS needed `global`, but `@stomp/stompjs` v7 over native WebSocket does not. Remove both and verify.
- **Modals are `<div>` overlays, not `<dialog>`** (*prior M5*) — no focus trap, no `aria-modal`, inconsistent Escape-close across `LogViewer`, `StartProjectModal` (closes on backdrop click `:57-59` but no Escape), the redesign modals. **a11y + keyboard-nav gap.**
- **No `eslint-plugin-jsx-a11y`** (*prior M4*) — confirmed absent from `eslint.config.ts`.
- **AuthContext swallows all `/auth/me` errors** (*prior H6*) — `AuthContext.tsx:52-54` treats a network failure identically to "unauthenticated"; a transient backend blip silently logs the user out.
- **`stale dist/` committed** — `frontend/dist/` is checked in (the May-19 1 MB bundle). Build artifacts should be `.gitignore`d, not version-controlled.

---

## Decision table

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| React 19 + Vite 6 base stack | **salvage** | Right tool; no SSR/SEO need that would justify Next.js; the cookie-first-party problem is already solved by the Vite proxy. |
| Cookie-auth + Axios `withCredentials` + 401 interceptor | **salvage** | Textbook-correct posture; preserve verbatim, only fix the role type. |
| Vite-mode ↔ Spring-profile + `strictPort:5173` CORS coupling | **salvage** | Load-bearing and correct; clever first-party-cookie solution — keep, add `VITE_SERVER_ROOT_URL` fail-fast. |
| `logStore` (module cache + monotonic keys) | **salvage** | Genuinely well-engineered; add LRU over projects and test it. |
| V5 identity type contract (`role: 'USER'\|'ADMIN'`) | **rebuild** | Live bug — backend emits `PLATFORM_ADMIN`, killing all admin/permission UI; model the 3-layer V5 roles + Zod validation. |
| Security-header layer (CSP/HSTS/headers) | **rebuild** | Nonexistent at both layers; cookie-auth + eager third-party libs makes any XSS a full session compromise. |
| Frontend test layer | **rebuild** | Zero tests against the riskiest code (auth, STOMP, role-gates); stand up Vitest + Playwright + MSW. |
| Server-state management (hand-rolled `useEffect` fetch) | **refactor** | Replace with TanStack Query — kills 4 duplicate fetch triads, adds cache/dedup/cancellation. |
| STOMP/WS wiring (3 clients, races, wildcard topics) | **refactor** | Consolidate to one shared connection; serialize history/live; scope topics server-side; consider SSE for logs. |
| Bundle / code-splitting | **refactor** | 1 MB single chunk, zero `React.lazy`, two icon libs; split routes + vendor, drop one icon lib. |
| `DashboardV2.tsx` god component | **refactor** | 548 lines, 14 hooks, 2 subscriptions, 5 modals — extract container + grid + modal set. |
| FederationOrrery (mock data + 60fps rAF) | **refactor** | Wire to real `/topic/results`+`/status`; replace rAF setState storm with CSS/declarative animation; honor `prefers-reduced-motion`. |
| `no-explicit-any` (18 sites at `warn`) | **refactor** | Flip to `error`, type the payload sites (interfaces already exist), Zod the wire boundary. |
| `framer-motion` dependency | **salvage** | Keep, but use declarative transforms instead of rAF; valuable for the product's "live federation" feel. |
| `react-icons` dependency | **kill** | Redundant second icon library (39 MB) alongside `lucide-react`; standardize on `lucide-react`. |
| `frontend/dist/` committed | **kill** | Build artifact in version control; `.gitignore` it. |

---

## Recommended v2 stack (verdict: keep React 19 + Vite, do NOT move to Next.js)

| Concern | v2 choice | Why |
|---|---|---|
| Framework | **React 19 + Vite 6 (keep)** | Static SPA, no SSR/SEO requirement; CloudFront+S3 deploy is cheapest and simplest; the first-party-cookie problem is already solved by the Vite proxy. Next.js would add a server tier (cost, ops) for zero benefit here. |
| Server state | **TanStack Query** | Cache, dedup, background refetch, `AbortSignal` cancellation; deletes 4 duplicate `fetchProjects` triads. |
| Global state | **React Context (keep, trimmed)** | Auth/theme/toast are low-churn cross-cutting; no Redux/Zustand needed at this size. |
| Real-time | **One shared STOMP client (or SSE for logs)** | Collapse 3 connections to 1; SSE auto-reconnect with `Last-Event-ID` cleanly solves the log history/live race. |
| Wire-type safety | **Zod at the Axios + STOMP boundary** | Fails loudly on backend contract drift (would have caught the `PLATFORM_ADMIN` bug); kills `any` reintroduction. |
| Testing | **Vitest + @testing-library/react + MSW + Playwright** | Vitest reuses Vite; MSW mocks REST; one Playwright e2e covers the golden path. |
| Security headers | **Spring Security CSP (authoritative) + report-only rollout** | Set at the backend so it covers API + SPA; report-only first to avoid breakage. |
| Bundle | **`React.lazy` routes + `manualChunks` vendor split + bundle budget in CI** | Initial load < 150 KB gzipped; one icon library. |
| Lint | **`no-explicit-any: error` + `strict-type-checked` + `jsx-a11y`** | Close the type and a11y holes the current config leaves at `warn`/absent. |

---

## Prioritized recommendations

**P0 — correctness & security (this sprint)**
1. **Fix the V5 identity contract.** Type `platformRole: 'USER' | 'PLATFORM_ADMIN'` + org/project role layers; update all 9 `role === 'ADMIN'` call sites; Zod-validate `/auth/me`. *Restores the entire admin UI, which is dead today.*
2. **Add CSP + security headers at the backend** (`SecurityConfig.java`), report-only first, then enforce; flip `frameOptions` to `DENY`.
3. **Add `VITE_SERVER_ROOT_URL` to the prod fail-fast** and force `wss://` derivation (`axiosConfig.ts`, `NotificationContext.tsx:14`, `DashboardV2.tsx:21`).
4. **Stand up Vitest + first tests** on the auth interceptor, `logStore` invariants, and role-gate rendering; wire into the (still-missing) PR CI.

**P1 — performance & maintainability (next 2-4 weeks)**
5. **Code-split** (`React.lazy` the authenticated shell + heavy modals; `manualChunks` for `recharts`/`framer-motion`); drop `react-icons`; `.gitignore` `dist/`.
6. **Adopt TanStack Query**; delete the duplicate fetch triads; add unmount cancellation.
7. **Consolidate STOMP to one shared connection**; serialize history/live (stable server-side log IDs); scope `/topic/status`+`/topic/results` to the user.
8. **Wire FederationOrrery + telemetry to real streams** (depends on the framework `RoundResult` POST fix from `../2026-05-27/03-framework.md`); replace the rAF render storm with CSS/declarative animation + `prefers-reduced-motion`.
9. **Flip `no-explicit-any` to `error`**, type the payload sites, Zod the boundaries.

**P2 — polish & multi-tenant maturity (Phase 3 horizon)**
10. Add `eslint-plugin-jsx-a11y`, convert modals to `<dialog>` with focus trap, clear notifications on logout, LRU-evict `logStore` over projects, add a **DeComFL** strategy option + convergence-vs-communication-rounds observability panel (the product's actual story), Playwright e2e + bundle-size/a11y CI gates.
