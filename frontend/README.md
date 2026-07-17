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

## Project layout

```
frontend/
├── src/
│   ├── api/axiosConfig.ts       # Axios singleton, interceptors, env-driven baseURL
│   ├── components/              # Reusable + redesign/* (role dashboards, LogViewer, PlaygroundView, ...)
│   ├── pages/                   # Top-level routes (Login, Register, Dashboard, ...)
│   ├── services/
│   │   ├── apiServices.ts       # Typed wrappers around every backend endpoint
│   │   └── logStore.ts          # Cross-component WebSocket log cache
│   ├── context/AuthContext.tsx  # Bootstrap auth state from /api/auth/me probe
│   ├── lib/                     # logger, cn(), small util layer
│   └── main.tsx
├── .env.development             # full-local mode (committed)
├── .env.ec2demo                 # frontend-local + remote backend (committed)
├── .env.production              # prod build (committed; placeholder host)
├── .env.local                   # personal overrides (gitignored)
├── .env.example                 # documents the env-var contract
├── vite.config.ts
└── package.json
```

## Auth model

The backend issues an HttpOnly **`jwtToken` cookie** on `/api/auth/login`. `axiosConfig.ts` sets `withCredentials: true`; the browser handles everything else. There is no `localStorage`, no `Authorization: Bearer` header, no JS-readable token.

Bootstrap flow (`AuthContext.tsx`):

1. App mounts → silently probes `GET /api/auth/me`.
2. 200 → user is hydrated into context.
3. 401 → user is anonymous; the interceptor in `axiosConfig.ts` swallows this specific 401 (it's intentional, not an auth error to redirect on).

A 401 on any other endpoint dispatches an `authError` event, prompting the global redirect to `/login`.

## Environment modes

Vite modes mirror the backend's Spring profiles 1:1:

| Mode | Spring profile | Script | Notes |
|---|---|---|---|
| `development` | `dev` | `npm run dev` | Full-local. Backend on `localhost:8081`, no proxy. CORS in `application-dev.properties` allows `http://localhost:*`. |
| `ec2demo` | `ec2demo` | `npm run dev:ec2demo` | Vite proxies `/api` and `/ws-logs` to `https://fedlearn.duckdns.org`. Cookies stay first-party to `localhost:5173` — sidesteps Safari's third-party cookie traps. |
| `production` | `production` | `npm run build` / `npm run build:ec2demo` | Static bundle. `VITE_FEDLEARN_API_URL` is required at build time; `axiosConfig.ts` throws on boot if missing. |

Vite env precedence (highest wins): `.env.[mode].local` > `.env.local` > `.env.[mode]` > `.env`. Personal overrides go in the `*.local` files, which are gitignored. Full var contract: `.env.example`.

The Vite proxy target is env-driven (`VITE_PROXY_TARGET`), so a teammate pointing at a different EC2 can override it in `.env.local` without editing `vite.config.ts`.

## Scripts

```bash
npm install
npm run dev               # Vite mode: development
npm run dev:ec2demo       # Vite mode: ec2demo (proxy → live EC2)
npm run build             # tsc + vite build (mode: production)
npm run build:ec2demo     # tsc + vite build (mode: ec2demo)
npm run lint              # ESLint
npm run preview           # Local preview of the production bundle
```

`vite.config.ts` enforces `strictPort: true`. If `:5173` is busy, Vite refuses to start instead of silently shifting to `:5174` — that shift would break CORS, since the backend allowlist is keyed on `:5173`.

## Conventions

- Strict TypeScript — interfaces for every component prop and API response.
- Components in `components/redesign/*` are the current Apple-inspired UI; older flat components are kept until callers are migrated.
- API calls go through `services/apiServices.ts` — never `axios.create()` ad hoc in a component.
- Never log cookies, request headers, or response bodies that could carry auth material.
- WebSocket subscriptions clean up on unmount (`useRef<StompSubscription>` + cleanup).

## Adjacent docs

- Wiki / deeper architecture: [`wikis/frontend/`](../wikis/frontend/).
- Backend auth contract: [`backend/fl-platform-api/README.md`](../backend/fl-platform-api/README.md#cookie-auth-contract) and [`DEVELOPMENT.md`](../backend/fl-platform-api/DEVELOPMENT.md).
