# Frontend Architecture & State Management

## Technology Stack

The frontend is built for performance, security, and developer experience. Declared ranges live in
`frontend/package.json`; the **resolved** versions below are what `frontend/package-lock.json`
actually installs, which is what CI builds against.

| Concern | Library | Declared | Resolved |
|---|---|---|---|
| Framework | React + React DOM | `^19.0.0` | 19.2.4 |
| Language | TypeScript | `^5.7.2` | 5.9.3 |
| Build tool | Vite | `^6.3.1` | 6.4.2 |
| Styling | Tailwind CSS (v4, `@tailwindcss/vite` plugin) | `^4.1.12` | 4.2.2 |
| Routing | react-router-dom | `^7.5.2` | 7.17.0 |
| Data fetching | Axios | `^1.11.0` | 1.17.0 |
| Real-time | `@stomp/stompjs` | `^7.1.1` | 7.3.0 |
| Charting | Recharts | `^2.15.2` | 2.15.4 |
| Icons | `lucide-react` | `^0.487.0` | — |
| Class handling | `clsx`, `tailwind-merge`, `class-variance-authority` | — | — |
| Tests | Vitest + Testing Library + jsdom | `^3.2.6` | 3.2.6 |

Both `axios` and `react-router-dom` were bumped *within* their existing major to clear npm-audit
highs (`07b369f`, "bump axios/react-router-dom to patched versions (0 npm-audit high)") — neither
was a breaking upgrade.

There is **no state-management library** (no Redux, no Zustand, no React Query) and no UI kit
dependency: the primitives in `src/components/ui/` are local, and the design tokens are generated
(see [UI & Components](./UI_and_Components.md)).

## Directory Structure

The `src/` directory is logically partitioned to separate concerns:

```text
src/
├── api/             # The single Axios instance + its interceptors (axiosConfig.ts)
├── assets/          # Static images, icons, SVGs
├── components/
│   ├── brand/       # BrandMark / Wordmark / HeroNetwork
│   ├── redesign/    # Every routed view + the project modals (see note below)
│   ├── ui/          # Design-system primitives (Button, Modal, StatusPill, …)
│   ├── DiskLoader.tsx      ErrorBoundary.tsx
│   └── ProtectedRoute.tsx  RoleRoute.tsx
├── context/         # AuthContext (the only React Context in the app)
├── hooks/           # useStompClient · useProjectStatus · useFocusTrap
├── lib/             # serverConfig · connectionStatus · logger · utils (cn)
├── pages/           # Unauthenticated route-level views: Landing, Login, Register
├── services/        # Typed API clients + stores: apiServices, artifactService, logStore
├── styles/          # tailwind.css, theme.css, tokens.css (generated), brand.css, fonts.css
└── test/            # Vitest setup + shared auth fixtures
```

`components/redesign/` is a **historical name, not a second UI**. There is no "legacy" component
tree any more: `redesign/` holds the only shipped views, and the old `/v2/*` URLs are pure
redirects (see [Routing & Auth](./Routing_and_Auth.md)). Several files still carry an
`(Ember design system)` banner comment from the previous design cycle — the code itself is on
Ledger tokens.

## State Management Approach

Instead of using a heavy state management library, the FedLearn frontend uses three tiers:

### 1. React Context (auth identity only)

`AuthContext` is the single Context in the app. It owns `currentUser`, `isLoading`, the derived
`isAdmin` / `isOwner` booleans, `setSession` and `logout`. Everything else is either local state or
a service store. Details in [Routing & Auth](./Routing_and_Auth.md#the-authcontext).

### 2. Service-Level Stores (domain state that outlives a component)

For state that must survive unmounts and be writable from a WebSocket callback outside the React
tree, the app uses small module-scoped stores with a publish/subscribe API.

`src/services/logStore.ts` is the canonical one — a `Map<projectId, StoredLogEntry[]>` capped at
**2000 entries per project**:

```typescript
// src/services/logStore.ts
const MAX_LOGS_PER_PROJECT = 2000;
const cache = new Map<string, StoredLogEntry[]>();
let nextId = 1;                                  // monotonic, never reused

append(projectId: string, entry: LogEntryInput): void {
    // A NEW array every time — listeners feed this straight into setState, and an
    // identical reference hits React's Object.is bailout, freezing the log pane.
    const prev = cache.get(projectId) ?? [];
    const next = [...prev, stamp(entry)];
    if (next.length > MAX_LOGS_PER_PROJECT) {
        next.splice(0, next.length - MAX_LOGS_PER_PROJECT);
    }
    cache.set(projectId, next);
    emit(projectId);
}
```

Two non-obvious invariants are load-bearing and documented in the file itself:

- **Never mutate the cached array in place.** Listeners pass the emitted reference directly into
  `setState`; an in-place `push` returns the same reference, React bails out, and the live pane
  freezes until some unrelated re-render.
- **Every entry gets a store-assigned monotonic `id`, and consumers must key on it.**
  `mergeHistorical()` *prepends* the REST-fetched history, which shifts every array index — an
  index key would reconcile the wrong DOM node to the wrong entry (garbled timestamps, repeated
  lines, broken auto-scroll).

`logStore` also tracks `hasLoadedHistorical(projectId)` so `GET /projects/{id}/logs` is fetched at
most once per project per tab. `LogViewer` keeps a parallel `telemetryCache` (last 30 loss/accuracy
points per project) with the same "survive the modal close" motivation.

### 3. Local Component State

Standard `useState` / `useReducer` for isolated concerns (form inputs, modal visibility, filter
toggles, pagination). Admin directory views additionally mirror their filter + page state into the
URL via `useSearchParams`, so a filtered view is linkable and survives refresh/back.

## Hooks layer

Three hooks carry the cross-cutting behaviour that used to be copy-pasted per view:

| Hook | Owns |
|---|---|
| `useStompClient` | One STOMP client's whole lifecycle: activate, (re)subscribe on every CONNECTED frame, unsubscribe + deactivate on unmount, and the honest `{ isConnected, isReconnecting, lastError }` state. See [API & Services](./API_and_Services.md#real-time-logs--websockets). |
| `useProjectStatus` | The `/topic/status/{id}` + `/topic/results/{id}` pair, JSON-decoded, with the owning project id resolved from the message's `destination` header rather than from the subscribed pattern. `OwnerDashboard` passes `'*'` for that segment, but the backend rejects wildcard SUBSCRIBE destinations outright — see [API & Services](./API_and_Services.md#destinations). |
| `useFocusTrap` | Keeps focus inside an open dialog and returns it to the trigger on close. Used by `ui/Modal` and by the `LogViewer` overlay. |

`useStompClient` reads `subscriptions` and `onConnect` through refs that are refreshed on every
render, and derives its effect dependency from `subscriptions.map(s => s.topic).join(' ')` rather
than the array identity. Every call site builds that array inline, so without this the socket would
be torn down and rebuilt on every render.

## Environments, Vite modes and the dev proxy

Vite modes map **1:1** onto Spring profiles, and the committed `.env.*` files say so in their own
headers.

| Vite mode | Command | Spring profile | `VITE_FEDLEARN_API_URL` | Proxy |
|---|---|---|---|---|
| `development` | `npm run dev` | `dev` | `http://localhost:8081/api` | none — the browser calls `:8081` directly |
| `ec2demo` | `npm run dev:ec2demo` | `ec2demo` | `http://localhost:5173/api` | `/api` **and** `/ws-logs` → `VITE_PROXY_TARGET` |
| `production` | `npm run build` | `production` | injected out-of-band (see below) | none |

`.env.development`, `.env.ec2demo` and `.env.production` are committed; `.env.local` and
`.env.[mode].local` are gitignored personal overrides and win under Vite's precedence
(`.env.[mode].local > .env.local > .env.[mode] > .env`). `frontend/.env.example` documents every
variable the app reads.

**Why `ec2demo` proxies through Vite rather than pointing at the demo host directly.** The auth
contract is an HttpOnly cookie. If the browser loaded the SPA from `localhost:5173` and called a
remote API origin directly, that cookie would be a *third-party* cookie — which Safari blocks
outright and other browsers increasingly restrict. Routing `/api` and `/ws-logs` through the Vite
dev server keeps the cookie first-party to `localhost:5173` while the traffic still reaches the
real backend. `vite.config.ts` only installs the proxy block when `VITE_PROXY_TARGET` is set, so it
is dead weight in full-local dev:

```ts
// vite.config.ts — proxy is undefined unless VITE_PROXY_TARGET is set
const proxy = proxyTarget ? {
  '/api':     { target: proxyTarget,                          changeOrigin: true, secure: true },
  '/ws-logs': { target: proxyTarget.replace(/^http/, 'ws'), ws: true, changeOrigin: true, secure: true },
} : undefined;
```

**`strictPort: true` on 5173.** Vite is configured to *fail* rather than shift to 5174 when the
port is taken. The reason is the backend's CORS allowlist: `app.cors.allowed-origins` is fed into
`CorsConfiguration.setAllowedOriginPatterns(...)` with `allowCredentials(true)`, and a cookie-bearing
request from an unlisted origin fails with a missing `Access-Control-Allow-Credentials` — a
confusing symptom whose real cause is a stuck process on 5173. Worth being precise about the blast
radius: the shipped **`dev`** default is a port *wildcard*
(`CORS_ALLOWED_ORIGINS:http://localhost:*,http://127.0.0.1:*,http://[::1]:*,file://` in
`application-dev.properties`), so a silent shift to 5174 would still pass CORS there. The base
profile has no default at all and refuses to boot without an explicit `CORS_ALLOWED_ORIGINS`, so
any environment that lists origins literally — and the `ec2demo` first-party-cookie trick, which
depends on the SPA actually being on 5173 — does break on a port shift. Failing fast is the cheap
way to never have to tell those two cases apart.

## Build, guards and quality gates

- **`npm run build` is `tsc && vite build`** — the type check is part of the build, not a separate
  step, and CI additionally runs `npx tsc --noEmit`.
- **Production host guard (FE-7).** `.env.production` deliberately ships the placeholder host
  `REPLACE_WITH_YOUR_API_HOST`. Two independent guards stop it reaching a bundle: the npm
  `prebuild` hook (`scripts/check-prod-env.mjs`, which reuses Vite's own `loadEnv` so it inspects
  exactly what the bundle would inline), and three throws at module scope in `src/api/axiosConfig.ts`
  for a missing, placeholder, or non-`https://` origin. The real host is injected out-of-band via a
  gitignored `.env.local` or CI-exported env vars — never by editing the committed file.
- **No sourcemaps in the bundle.** `build.sourcemap` is `false`; the previous `true` emitted
  `dist/assets/index-*.js.map`, which anyone could fetch and de-minify back to readable TypeScript.
  Switch to `'hidden'` if an error tracker ever needs server-side de-minification.
- **ESLint 9 flat config** (`eslint.config.mjs`). Three rules are `error`, not `warn`, deliberately:
  `@typescript-eslint/no-explicit-any`, `@typescript-eslint/no-unused-vars`, and
  `react-refresh/only-export-components`. The one legitimate exception (the `useAuth` hook
  co-located with its provider) carries a scoped, documented disable.
- **Vitest** (`vitest.config.ts`) runs the component/unit suite on jsdom. It is kept separate from
  `vite.config.ts` so tests don't pull in the dev proxy or the Tailwind pipeline. `globals: false` —
  specs import `describe`/`it`/`expect`/`vi` explicitly, so the app tsconfig and lint config need no
  test-only changes. `src/test/setup.ts` stubs `matchMedia`, `ResizeObserver` and `scrollIntoView`,
  which jsdom lacks and Recharts/PlaygroundView touch on mount.
- **Coverage is a gate, not a report.** `vitest.config.ts` sets thresholds (lines/statements 54,
  functions 34, branches 66) a few points under the measured baseline — the intent is to catch a
  drop, not to pin an aspirational target.
- **CI** (`.github/workflows/ci.yml`, path-filtered on `frontend/**`) runs, in order:
  `npm ci` → `npm run lint` → `npx tsc --noEmit` → `npm run test:coverage` → `npm run build`.
  A separate, *unfiltered* `design-tokens` job runs `scripts/check_design_tokens.sh` on every push —
  see [UI & Components](./UI_and_Components.md#design-tokens-are-generated-not-hand-written).

## Cross-cutting utilities

- **`src/lib/logger.ts`** — `createLogger(scope)` returns a leveled logger. `debug` and `info` are
  stripped from production bundles (`import.meta.env.PROD`), every line is tagged `[Scope]` so it is
  greppable, and the whole app funnels through one chokepoint for a future Sentry/Datadog RUM wiring.
  Use `console.*` directly nowhere.
- **`src/components/ErrorBoundary.tsx`** wraps the entire tree in `main.tsx`, outside
  `BrowserRouter` and `AuthProvider`.
- **`src/components/DiskLoader.tsx`** is the standard async spinner (app bootstrap, auth check).
- **`main.tsx`** aliases `window.global = window` before render — some dependencies expect a
  Node-style `global`; `vite.config.ts` mirrors that with `define: { global: {} }`.
