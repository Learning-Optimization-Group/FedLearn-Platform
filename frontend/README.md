# FedLearn Frontend

React 19 + Vite + TypeScript SPA for the FedLearn Platform. The dashboard talks to the Spring Boot backend over REST and STOMP-over-WebSocket; auth is cookie-based.

## Stack

- **React 19** — function components, hooks, modern JSX runtime
- **Vite 6** — dev server + build (mode-aware env loading, HMR, native ESM)
- **TypeScript** (strict) — typed components, services, and API responses
- **Tailwind v4** — utility-first styling via `@tailwindcss/vite`
- **Axios** — HTTP client; `withCredentials: true` so cookies flow on every request
- **`@stomp/stompjs`** — native WebSocket STOMP client (no SockJS shim)
- **`recharts`** — telemetry sparklines on the dashboard
- **`lucide-react`** — icon set
- **React Router v7** — routing
- **Vitest + Testing Library** — component/unit suite, jsdom environment

**Node 24** is the pinned toolchain (`.nvmrc` → `24`, `.tool-versions` → `nodejs 24.4.0`), and CI runs the frontend job on 24.

## Project layout

```
frontend/
├── src/
│   ├── api/axiosConfig.ts       # Axios singleton, interceptors, env-driven baseURL
│   ├── components/
│   │   ├── redesign/            # Role dashboards, LogViewer, PlaygroundView, RegistryView, ...
│   │   ├── ui/                  # Ledger primitives (Button, Card, Modal, StatusPill, ...)
│   │   ├── brand/               # BrandMark, HeroNetwork
│   │   ├── ProtectedRoute.tsx   # Gate for authenticated routes
│   │   └── RoleRoute.tsx        # Gate for PROJECT_OWNER / PLATFORM_ADMIN routes
│   ├── pages/                   # Public routes only (Landing, Login, Register)
│   ├── services/
│   │   ├── apiServices.ts       # Typed wrappers around every backend endpoint
│   │   ├── artifactService.ts   # Registry / model-artifact calls
│   │   └── logStore.ts          # Cross-component WebSocket log cache
│   ├── context/AuthContext.tsx  # Bootstrap auth state from the /api/auth/me probe
│   ├── hooks/useStompClient.ts  # Owns one STOMP client's lifecycle + honest connection state
│   ├── lib/                     # logger, serverConfig (API + WS origin), connectionStatus, cn()
│   ├── styles/tokens.css        # GENERATED from design/tokens.json — do not hand-edit
│   ├── test/                    # Vitest setup + shared fixtures
│   └── main.tsx
├── scripts/check-prod-env.mjs   # `prebuild` guard: refuses to build on the placeholder host
├── .env.development             # full-local mode (committed)
├── .env.ec2demo                 # frontend-local + remote backend (committed)
├── .env.production              # prod build (committed; placeholder host)
├── .env.local                   # personal overrides (gitignored, as is .env.[mode].local)
├── .env.example                 # documents the env-var contract
├── vite.config.ts
├── vitest.config.ts
└── package.json
```

Dashboards are **not** in `pages/` — the authenticated surface lives in `components/redesign/` behind `ProtectedRoute` + `LayoutV2` (see `src/App.tsx` for the route table).

## Auth model

The backend issues an HttpOnly **`jwtToken` cookie** on `/api/auth/login`. `axiosConfig.ts` sets `withCredentials: true`; the browser handles everything else. There is no `localStorage`, no `Authorization: Bearer` header, no JS-readable token.

Bootstrap flow (`AuthContext.tsx`):

1. App mounts → silently probes `GET /api/auth/me`.
2. 200 → user is hydrated into context (`username`, `email`, `role`).
3. 401 → user is anonymous. The response interceptor swallows a 401 **only** for URLs containing `/auth/me` (and for the `/auth/login` attempt itself) — surfacing it would cause a redirect loop on the bootstrap probe.

A 401 on **any other** endpoint dispatches a `window` `authError` event. `App.tsx` listens for it and calls `logout()`, which clears the in-memory user; `ProtectedRoute` then `Navigate`s to `/login`, carrying the attempted location so the user lands back where they were after signing in.

A **403** is deliberately *not* treated as an auth failure — it means "authenticated but not allowed here", so the calling component renders the failure inline instead of logging the user out.

While authenticated, `AuthContext` re-polls `/api/auth/me` on window focus / tab visibility (debounced to 5 s), so a server-side role change surfaces without a re-login.

## Live training logs

Log streaming is STOMP over a native WebSocket — no SockJS.

- The broker URL is derived once in `lib/serverConfig.ts`: `VITE_SERVER_ROOT_URL` with the scheme swapped (`http`→`ws`, `https`→`wss`) plus `/ws-logs`.
- The handshake is authenticated by the **same HttpOnly cookie** (backend `JwtHandshakeInterceptor`); nothing token-shaped is attached client-side.
- `LogViewer` subscribes to **`/topic/logs/{projectId}`** and fetches the historical backlog once over REST, merging both into `services/logStore.ts` so reopening a project's logs doesn't reset the view.
- `hooks/useStompClient.ts` owns activate/deactivate, resubscribes on every CONNECTED frame (including automatic reconnects), and reports `isConnected` only from a real CONNECTED frame — a dropped socket can never present as a silent stall.

## Environment modes

Vite modes mirror the backend's Spring profiles 1:1:

| Mode | Spring profile | Script | Notes |
|---|---|---|---|
| `development` | `dev` | `npm run dev` | Full-local. Backend on `localhost:8081`, no proxy. CORS in `application-dev.properties` allows `http://localhost:*`. |
| `ec2demo` | `ec2demo` | `npm run dev:ec2demo` | Vite proxies `/api` and `/ws-logs` to `VITE_PROXY_TARGET` (committed default: `https://fedlearn.duckdns.org`). |
| `production` | `production` | `npm run build` | Static bundle. `VITE_FEDLEARN_API_URL` must be injected out-of-band; the `prebuild` guard fails the build while it is still the committed placeholder. |

`npm run build:ec2demo` also produces a bundle, in `ec2demo` mode — useful for shipping a build that talks to the demo host.

**Why `ec2demo` proxies through Vite rather than calling the EC2 host directly:** the browser only ever sees `http://localhost:5173/api/*` and `ws://localhost:5173/ws-logs`, so the session cookie stays **first-party to `localhost:5173`**. A direct cross-origin call would make `jwtToken` a third-party cookie, which Safari blocks outright — the login would appear to succeed and every subsequent request would 401.

The proxy target is env-driven (`VITE_PROXY_TARGET`, read by `vite.config.ts` via `loadEnv`), so pointing at a different backend is an `.env.local` change, not a `vite.config.ts` edit. The proxy block is only configured when that var is set, so full-local dev carries no dead proxy.

Vite env precedence (highest wins): `.env.[mode].local` > `.env.local` > `.env.[mode]` > `.env`. Full var contract: `.env.example`.

### Production env is injected, never committed

`.env.production` ships a **placeholder** host (`REPLACE_WITH_YOUR_API_HOST`) on purpose. Two guards enforce that a real origin is supplied out-of-band — a gitignored `.env.local`, or exported `VITE_FEDLEARN_API_URL` / `VITE_SERVER_ROOT_URL` in CI (both override the committed file via Vite's precedence):

- **`prebuild`** (`scripts/check-prod-env.mjs`) aborts `npm run build` while either var still contains the placeholder.
- **`axiosConfig.ts`** throws at module load in a `PROD` bundle if `VITE_FEDLEARN_API_URL` is missing, is still the placeholder, or is not an `https://` origin (an `http://` backend is blocked as mixed content once the SPA is served over HTTPS).

## Scripts

```bash
npm install
npm run dev               # Vite mode: development
npm run dev:ec2demo       # Vite mode: ec2demo (proxy → remote backend)
npm run build             # prebuild guard → tsc → vite build (mode: production)
npm run build:ec2demo     # tsc → vite build --mode ec2demo (npm's `prebuild` hook fires for `build` only)
npm run preview           # Local preview of the built bundle
npm run lint              # ESLint 9 flat config (eslint.config.mjs)
npm test                  # Vitest, watch mode
npm run test:run          # Vitest, single run
npm run test:coverage     # Vitest + v8 coverage; thresholds from vitest.config.ts
```

`vite.config.ts` enforces `strictPort: true` on `:5173`. If that port is busy, Vite refuses to start instead of silently shifting to `:5174`. This is load-bearing because the backend's CORS allowlist is what decides whether credentialed requests are accepted, and outside the local `dev` profile `app.cors.allowed-origins` has **no default** — it resolves to a bare `${CORS_ALLOWED_ORIGINS}` the operator has to supply, and both deploy scripts document `http://localhost:5173` as the value to set (`scripts/ec2-bootstrap.sh` writes it commented-out into the systemd unit; `scripts/deploy-to-aws.sh` prints it in the start-up instructions — neither sets it for you). Silently shifting ports would surface as a confusing missing-`Access-Control-Allow-Credentials` error rather than "your old dev server is still running".

One honest caveat: the local `dev` profile is port-wildcarded (`application-dev.properties` defaults `app.cors.allowed-origins` to `http://localhost:*,http://127.0.0.1:*,http://[::1]:*,file://`), so a port shift would *not* break full-local dev. Fail-fast still matters because `:5173` is the port every other environment, script, and `ec2demo` cookie assumption is written against.

Build output ships **no sourcemaps** (`build.sourcemap: false`) so the bundle can't be de-minified back to readable TypeScript from the CDN.

## Design system — Ledger

The visual contract is **Ledger**: navy structural ink on quiet paper surfaces, light-first (canvas `#F6F3EE`, surface `#FFFFFF`, ink `#191A1C`, muted `#6B6760`, accent `#1C314D` with `#14243A` hover; the dark family is navy-dark). Typefaces are Hanken Grotesk and JetBrains Mono.

`design/tokens.json` at the repo root is the single source of truth for **all three** surfaces (frontend, desktop, mobile). `design/build-tokens.mjs` generates the per-platform outputs; this app consumes `src/styles/tokens.css` as CSS custom properties, imported through `src/styles/tailwind.css`.

- Style new UI from the generated tokens. **Never hardcode a colour, radius, or spacing value in a component.**
- `src/styles/tokens.css` is generated — edit `design/tokens.json` and re-run `node design/build-tokens.mjs`.
- CI runs `scripts/check_design_tokens.sh` on **every** PR (no path filter): it regenerates from `tokens.json` and fails if any committed output has drifted.

## CI gate

The frontend job in `.github/workflows/ci.yml` runs on Node 24, only when `frontend/**` changes, and is a required part of the aggregate `ci-gate` check:

```bash
npm ci
npm run lint          # ESLint 9 — no-explicit-any / no-unused-vars / react-refresh are errors, not warnings
npx tsc --noEmit      # standalone typecheck
npm run test:coverage # vitest; coverage thresholds enforced from vitest.config.ts
npm run build         # includes the prebuild placeholder guard
```

Run the same four checks locally before opening a PR.

## Deployment

`npm run build` emits a static bundle to `dist/`. This repo does **not** wire a frontend hosting pipeline: the committed EC2 nginx config (`deploy/nginx/fedlearn.conf`) proxies `/`, `/api/internal/`, and `/ws-logs` to Spring Boot on `:8081` and never serves `dist/`. What *is* committed is host-agnostic SPA plumbing you can point at whatever you choose — `frontend/vercel.json` (rewrite every path to `/index.html`) and a CloudFront custom-error-response recipe in the trailing comment of `vite.config.ts`. Both exist so client-side routes don't 404 on a hard refresh; neither implies an active deployment.

## Conventions

- Strict TypeScript — interfaces for every component prop and API response.
- API calls go through `services/apiServices.ts` — never `axios.create()` ad hoc in a component.
- Never log cookies, request headers, or response bodies that could carry auth material.
- WebSocket surfaces use `useStompClient` rather than hand-rolling a `Client` — one place owns the lifecycle.
- The model-recipe picker is rendered from `GET /api/model-recipes`; the in-file fallback list exists only to keep the modal usable when that call fails. Add a model type in `fl-runtime/recipes.py` (and its `init_model.py` branch), not here.

## Adjacent docs

- Wiki / deeper architecture: [`wikis/frontend/`](../wikis/frontend/)
- Backend auth contract: [`backend/fl-platform-api/README.md`](../backend/fl-platform-api/README.md#auth-contract) and [`DEVELOPMENT.md`](../backend/fl-platform-api/DEVELOPMENT.md)
