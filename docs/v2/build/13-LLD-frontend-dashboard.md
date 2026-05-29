# 13 — Low-Level Design (LLD): Frontend Dashboard

**Document type:** Production build specification — Low-Level Design (LLD) for one deployable unit.
**Unit:** `frontend/` — the React Single-Page Application (SPA) dashboard.
**Audience:** a mid-sized (~30 billion parameter) local Large Language Model (LLM) implementing the build. Every contract, file path, version, and type signature is pre-decided. You implement the bodies; you do **not** redesign the contracts.
**Status:** build-authoritative for v2 (version 2). Conforms to and does not contradict `01-ARCHITECTURE-HLD.md`, `02-TECH-STACK.md`, `03-DATA-MODEL.md`, and `04-API-CONTRACTS.md`.
**Date authored:** 2026-05-29.

> **Numbering note:** this is the **Frontend LLD = document 13-** (file `docs/v2/build/13-LLD-frontend-dashboard.md`). The reconciled LLD numbering is recorded in `01-ARCHITECTURE-HLD.md` §3's "LLD numbering note," which lists 13- as the Frontend LLD (the original §3 table column once said 14-; the file numbering supersedes it). Treat "the Frontend LLD" and this document as the same artifact.

---

## 0. How to read this document

Every acronym is expanded in full the **first** time it appears, e.g. "SPA (Single-Page Application)". After first expansion the short form is used. This rule is repeated per the build-doc standard.

**Abbreviation key (first-use expansions, repeated here for self-containment):**
SPA (Single-Page Application), LLD (Low-Level Design), API (Application Programming Interface), REST (Representational State Transfer), JWT (JSON Web Token — JSON = JavaScript Object Notation), HTTP (HyperText Transfer Protocol), HTTPS (HTTP Secure), WS (WebSocket), STOMP (Simple Text Oriented Messaging Protocol), CORS (Cross-Origin Resource Sharing), CSP (Content-Security-Policy), HSTS (HTTP Strict Transport Security), XSS (Cross-Site Scripting), CSRF (Cross-Site Request Forgery), UUID (Universally Unique Identifier), DTO (Data Transfer Object), FL (Federated Learning), DeComFL (Dimension-Free Communication Federated Learning — the platform's zeroth-order FL strategy; the v1 wiki mis-expanded this as "Decomposed", which is wrong per the paper — see `docs/audit/2026-05-29/B1-paper-alignment.md:33`), ZO (Zeroth-Order), RNG (Random Number Generator), RBAC (Role-Based Access Control), RLS (Row-Level Security), SSR (Server-Side Rendering), TS (TypeScript), CSS (Cascading Style Sheets), OKLCH (Oklab Lightness-Chroma-Hue color space), SVG (Scalable Vector Graphics), rAF (requestAnimationFrame), MSW (Mock Service Worker), E2E (End-to-End), CI (Continuous Integration), URL (Uniform Resource Locator), SDK (Software Development Kit), OTel (OpenTelemetry), W3C (World Wide Web Consortium), a11y (accessibility), W&B (Weights & Biases), DLG (Deep Leakage from Gradients), CVA (class-variance-authority).

---

## 1. Purpose & single responsibility

The frontend is a **pure client-rendered SPA** that is the human control surface for the FedLearn platform: researchers and organization (org) admins log in, create projects, launch and monitor FL runs, and watch live convergence/communication-cost telemetry. Its single responsibility is **presentation + client-side orchestration of the control-plane contracts** — it owns no business logic that the backend does not also enforce, and it holds **no training data and no model weights**. It consumes the REST API, the STOMP-over-WebSocket live channel, and validates every wire payload at the boundary with Zod so backend/frontend contract drift fails loudly instead of silently disabling UI (the v1 failure mode, `docs/audit/2026-05-29/A2-frontend.md:52-59`).

---

## 2. Position in the system — dependencies and interfaces

### 2.1 What it depends on (consumes)

| Dependency | Contract source | What the frontend calls |
|---|---|---|
| Control plane REST API | `04-API-CONTRACTS.md` §2–§9, §12 | `/api/auth/*`, `/api/projects/*`, `/api/projects/{id}/runs`, `/api/runs/*`, `/api/orgs/*`, `/api/datasets/*`, `/api/artifacts/*`, `/api/admin/*`, `/api/users/me`. |
| STOMP-over-WebSocket live channel | `04-API-CONTRACTS.md` §11 | Subscribe `/topic/logs/{projectId}`, `/topic/results/{projectId}`, `/topic/status/{projectId}`, `/topic/runs/{projectId}`; handshake at `/ws-logs`. |
| Auth cookie | `04-API-CONTRACTS.md` §1 | The HttpOnly `jwtToken` cookie set by `POST /api/auth/login`; sent automatically via `withCredentials: true`. The frontend **never reads, stores, or transmits** the token itself. |
| Error envelope | `04-API-CONTRACTS.md` §12 | The single error shape `{ timestamp, status, code, message, path, traceId, fieldErrors? }`; the frontend switches on `code`. |
| Trace context | `04-API-CONTRACTS.md` §14 | The frontend OTel web SDK sets the `traceparent` header on the `POST .../runs` request (optional; the backend originates a root span if absent). |

### 2.2 What depends on it

| Consumer | How |
|---|---|
| The human operators (Researcher, Org Admin, Platform Admin) | `01-ARCHITECTURE-HLD.md` §2 actor table — browser → frontend (cookie JWT). |
| The Tauri v2 desktop renderer (`14-` Desktop LLD) | **Reuses this exact React renderer** (`02-TECH-STACK.md` §16.1, `README.md:54`). Every component built here must work unchanged in the desktop WebView. This means: no Node-only APIs in renderer code, and all backend URLs come from build-mode env vars (not hardcoded). |

### 2.3 What this unit EXPOSES

The frontend exposes **no network interface**. It is a static bundle served by nginx/CloudFront. Its "exposed surface" is internal: the shared `@fedlearn/tokens` OKLCH design-token package (consumed by web + desktop + mobile) and the shared Zod schema module (`src/api/schemas.ts`) that mirrors the wire contracts. Both are described in §5.

### 2.4 Load-bearing invariants inherited from v1 (do NOT change)

1. **Cookie-only auth.** `withCredentials: true` on every call; no `Authorization: Bearer`, no `localStorage`, no JS-readable token (`04-API-CONTRACTS.md` §1; `A2-frontend.md:41-48` rates this "textbook-correct, preserve verbatim").
2. **`strictPort: true` on `:5173`.** The backend CORS allowlist is keyed on `localhost:5173`; a silent fallback to `:5174` produces opaque `Access-Control-Allow-Credentials` failures (`A2-frontend.md:155-158`, `02-TECH-STACK.md` §12.2).
3. **Vite `--mode` ↔ Spring profile 1:1 mapping.** `.env.{development,ec2demo,production}` committed; `.env.local` gitignored (`02-TECH-STACK.md` §12.2).
4. **The `ec2demo` Vite-proxy first-party-cookie trick** sidesteps Safari third-party-cookie blocking and is *the reason SSR/Next.js is unnecessary* (`A2-frontend.md:158`). Keep it.

---

## 3. Tech stack for this unit (pinned; from `02-TECH-STACK.md`)

| Concern | Technology | Pinned version | One-line reasoning |
|---|---|---|---|
| Runtime | Node.js Active LTS | `24.x` (e.g. `24.4.0`) | One pinned Node across the JS/TS triangle; pin in `.nvmrc` + `.tool-versions` (`02-TECH-STACK.md` §1.3). |
| UI library | React + React-DOM | `19.x` (pin exact latest 19 patch) | Salvage; no SSR need → no Next.js (`02-TECH-STACK.md` §12.1). |
| Bundler / dev server | Vite + `@vitejs/plugin-react` | `6.x` (pin exact latest 6 patch) | Modes mirror Spring profiles; `strictPort:5173` load-bearing (§12.2). |
| Language | TypeScript (`tsc` strict) | `5.x` (pin exact latest 5 patch) | One TS version across all three TS surfaces (§12.3). |
| Server-state | TanStack Query (`@tanstack/react-query`) | `5.100.14` | Cache/dedup/cancellation; kills the v1 duplicate fetch triads (§13.1). |
| Wire validation | Zod | `4.4.3` | Runtime schema validation at the Axios + STOMP boundary (§13.2). |
| HTTP client | Axios | `^1.11.0` (carried from v1; pin exact) | One instance, `withCredentials`, the 401 interceptor (v1 `package.json`). |
| Live channel client | `@stomp/stompjs` | `7.1.1` (pin exact; carried from v1) | Native WebSocket STOMP client, no SockJS (§10). |
| Routing | `react-router-dom` | `7.x` (carried from v1 `^7.5.2`; pin exact) | Client routing for the SPA shell. |
| CSS framework | Tailwind CSS v4 (`@tailwindcss/vite` + `@tailwindcss/postcss`) | `4.x` (pin exact latest 4 patch) | `@theme` directive hosts the OKLCH tokens (§14.1). |
| Component system | shadcn/ui (copy-in, over Radix primitives) | CLI `npx shadcn@latest`; pin Radix primitive versions once generated | Owned-in-repo accessible components; one brand (§14.2). |
| Design tokens | `@fedlearn/tokens` (internal OKLCH package) | monorepo-versioned | One color source of truth; web + desktop + mobile (§14.3). |
| Charts | recharts | `2.x` (carried from v1 `^2.15.2`; pin exact) | Convergence + comm-cost charts; route series through `--chart-*` tokens (§3.1 C5). |
| Icons | `lucide-react` | `0.x` (carried from v1 `^0.487.0`; pin exact) | **One** icon library. `react-icons` is removed (`A2-frontend.md:194`). |
| Animation | `framer-motion` | pin exact latest 11/12.x (`verify-before-use`) | Declarative transforms only — no rAF setState storm (§6.6). |
| Unit/component tests | Vitest + `@testing-library/react` | Vitest `3.x`, RTL pin exact (`verify-before-use`) | Reuses the Vite pipeline (§15). |
| Network mocking | MSW (Mock Service Worker) | `2.14.6` | Mock the REST + STOMP boundary in tests (§15). |
| E2E tests | Playwright (`@playwright/test`) | `1.x` (`verify-before-use`) | One golden-path browser test (§15). |

**Removed vs v1 (verdict KILL):** `react-icons` (second icon library, 39 MB on disk, redundant — `A2-frontend.md:125,194`); the committed `frontend/dist/` build artifact (`A2-frontend.md:172,195`); the dead `(window as any).global = window` shim + `define: { global: {} }` (no longer needed by `@stomp/stompjs` v7 over native WebSocket — `A2-frontend.md:168`).

---

## 4. Module / file structure (exact tree, one-line responsibility each)

```
frontend/
├── .env.development              # mode=development → API on localhost:8081 (committed)
├── .env.ec2demo                  # mode=ec2demo → Vite proxies /api + /ws-logs to the demo host (committed)
├── .env.production               # mode=production → API + WS URLs required, fail-fast if missing (committed)
├── .env.local                    # gitignored personal overrides (NOT committed)
├── .nvmrc                        # "24" — pins Node for the whole frontend
├── index.html                    # SPA entry; carries the CSP <meta> fallback + module script (§8.3)
├── package.json                  # deps pinned per §3; scripts dev/build/lint/test/test:e2e
├── vite.config.ts                # strictPort:5173, mode→env mapping, manualChunks vendor split, ec2demo proxy
├── vitest.config.ts              # jsdom env, setupFiles → src/test/setup.ts, coverage v8
├── playwright.config.ts          # one project; baseURL from env; webServer = vite preview
├── tsconfig.json                 # strict; parserOptions.project for typed lint
├── eslint.config.ts              # tseslint strictTypeChecked + jsx-a11y; no-explicit-any:error
├── tailwind.config.ts            # Tailwind v4 config; consumes @fedlearn/tokens
├── components.json               # shadcn/ui generator config (style, aliases, tokens)
└── src/
    ├── main.tsx                  # mounts <App/> inside the provider stack (§6.1); inits OTel web SDK
    ├── App.tsx                   # <RouterProvider/>; route table with React.lazy code-split boundaries (§6.7)
    │
    ├── api/                      # the wire layer — the ONLY place network shapes are defined
    │   ├── axiosClient.ts        # single Axios instance, withCredentials, the 401 silent-probe interceptor (§5.1, §6.2)
    │   ├── schemas.ts            # ALL Zod schemas mirroring 04-API-CONTRACTS shapes (§5.5)
    │   ├── types.ts              # z.infer<> exported TS types (V5 role types + DTOs) (§5.4)
    │   ├── parse.ts              # parseOrThrow<T>(schema, data): T — boundary validation helper (§5.2)
    │   ├── endpoints.ts          # typed fn-per-endpoint: authApi, projectApi, runApi, orgApi, datasetApi, adminApi (§5.3)
    │   └── queryKeys.ts          # the TanStack Query key factory (single source of cache keys) (§5.6)
    │
    ├── query/                    # TanStack Query layer (hooks over api/endpoints.ts)
    │   ├── queryClient.ts        # the QueryClient singleton + default options (staleTime, retry) (§6.3)
    │   ├── useAuth.ts            # useMe(), useLogin(), useLogout(), useRegister() (§5.7)
    │   ├── useProjects.ts        # useProjects(), useProject(id), useCreateProject(), usePatchProject()
    │   ├── useRuns.ts            # useRuns(projectId), useRun(runId), useStartRun(), useStopRun(), useRunStatus()
    │   ├── useResults.ts         # useProjectResults(projectId) — seed for the live-merged results store
    │   ├── useOrgs.ts            # useOrgs(), useOrg(id), useOrgMembers(), useAddMember(), ...
    │   ├── useDatasets.ts        # useDatasets(), useDatasetVersions(), useCreatePartition()
    │   └── useAdmin.ts           # useAdminUsers(), useUpdateUserRole(), useAuditEvents() (PLATFORM_ADMIN-gated)
    │
    ├── realtime/                 # the ONE shared STOMP connection (§6.4, §6.5)
    │   ├── StompProvider.tsx     # single StompClient, ref-counted subscriptions, wss:// derivation, fail-fast
    │   ├── useStompTopic.ts      # useStompTopic<T>(destination, schema, onMessage) — validated subscribe hook
    │   ├── logStore.ts           # SALVAGED module-level cache: monotonic IDs, dedup, trim, LRU over projects (§6.5)
    │   └── liveResultsStore.ts   # per-(projectId,runId) live RoundResult array merged from /topic/results
    │
    ├── auth/
    │   ├── AuthContext.tsx       # exposes identity from useMe(); listens for the 'authError' event → logout (§6.2)
    │   ├── useIdentity.ts        # useIdentity() → AuthIdentity; useHasPlatformRole(), useOrgRole(orgId) (§5.4)
    │   ├── RequireAuth.tsx       # route guard: redirect to /login if unauthenticated
    │   └── RequirePlatformAdmin.tsx # route guard: 403 page if platformRole !== 'PLATFORM_ADMIN'
    │
    ├── context/
    │   ├── ThemeProvider.tsx     # light/dark; reads/writes OKLCH token class on <html>
    │   ├── OrgContext.tsx        # the active org (org switcher); persists selection per-user in memory
    │   └── ToastProvider.tsx     # SALVAGED toast system; typed levels, auto-dismiss (§6.x)
    │
    ├── pages/                    # route-level components (each is a React.lazy boundary)
    │   ├── LoginPage.tsx
    │   ├── RegisterPage.tsx
    │   ├── VerifyEmailPage.tsx
    │   ├── ForgotPasswordPage.tsx / ResetPasswordPage.tsx
    │   ├── DashboardPage.tsx     # container: org switcher + <ProjectGrid> (replaces the v1 548-line god component)
    │   ├── ProjectDetailPage.tsx # project tabs: Overview | Runs | Datasets | Members | Settings
    │   ├── RunObservabilityPage.tsx # the per-run telemetry surface (§6.8) — the product's reason to exist
    │   ├── DatasetsPage.tsx / ModelsPage.tsx
    │   ├── admin/AdminUsersPage.tsx / admin/AdminProjectsPage.tsx / admin/AuditLogPage.tsx
    │   └── NotFoundPage.tsx / ForbiddenPage.tsx
    │
    ├── features/                 # feature components owned by one domain
    │   ├── projects/ProjectGrid.tsx / ProjectCard.tsx / CreateProjectModal.tsx / EditProjectModal.tsx
    │   ├── runs/StartRunModal.tsx / RunStatusBadge.tsx / RunList.tsx
    │   ├── observability/
    │   │   ├── ConvergenceChart.tsx       # loss + accuracy vs round, per-run scoped (§6.8)
    │   │   ├── CommunicationCostPanel.tsx # the DeComFL bandwidth wedge (§6.8) — net-new
    │   │   ├── PerClientPanel.tsx         # small-multiples per client (contribution/loss/last-seen) (§6.8)
    │   │   └── FederationOrrery.tsx       # SALVAGED hero widget; declarative animation, real data (§6.9)
    │   ├── logs/LogViewer.tsx             # SALVAGED terminal; history+live merge via logStore (§6.5)
    │   └── orgs/OrgSwitcher.tsx / MemberList.tsx / AddMemberModal.tsx
    │
    ├── components/ui/            # shadcn/ui copy-in primitives (owned source)
    │   ├── dialog.tsx button.tsx input.tsx select.tsx badge.tsx card.tsx table.tsx
    │   ├── toast.tsx skeleton.tsx empty-state.tsx error-state.tsx status-badge.tsx
    │   └── ... (Radix-backed; focus-trap + aria-modal for free)
    │
    ├── lib/
    │   ├── env.ts               # reads + validates import.meta.env; fail-fast on missing prod URLs (§8.1)
    │   ├── wsUrl.ts             # derives wss:// from VITE_SERVER_ROOT_URL; throws on http:// in prod (§6.4)
    │   ├── errorCode.ts         # maps error-envelope `code` → user message + UI action (§9)
    │   └── logger.ts            # console chokepoint; the single OTel/RUM wiring point (B3 logger.ts:7)
    │
    ├── styles/
    │   └── theme.css            # @theme block importing @fedlearn/tokens OKLCH vars (§14)
    │
    └── test/
        ├── setup.ts             # RTL + jest-dom matchers; starts the MSW server (§15)
        ├── msw/handlers.ts      # MSW REST handlers per endpoint
        ├── msw/server.ts        # setupServer(...handlers) for node (Vitest)
        └── fixtures.ts          # canonical MeResponse / RunDto / RoundResult fixtures
```

---

## 5. Key interfaces & type signatures (FULL — you implement the bodies)

### 5.1 The single Axios instance (`src/api/axiosClient.ts`)

```ts
import axios, { type AxiosInstance, type AxiosError } from 'axios';
import { env } from '../lib/env';

// Custom event fired on a HARD-logout 401 (not the /auth/me probe). App.tsx listens.
export const AUTH_ERROR_EVENT = 'authError';

// One instance. withCredentials sends the HttpOnly jwtToken cookie. No Authorization header anywhere.
export const apiClient: AxiosInstance = axios.create({
  baseURL: env.apiUrl,            // VITE_FEDLEARN_API_URL; fail-fast validated in lib/env.ts
  withCredentials: true,          // INVARIANT: cookie auth (04-API-CONTRACTS §1)
  headers: { 'Content-Type': 'application/json' },
  timeout: 30_000,
});

// The 401 silent-probe interceptor. See §6.2 for the exact branching algorithm.
export function installAuthInterceptor(client: AxiosInstance): void;
```

### 5.2 Boundary validation helper (`src/api/parse.ts`)

```ts
import type { ZodType } from 'zod';

// Parse unknown wire data through a Zod schema at the boundary.
// On failure: log via logger.ts with the schema name + zod issues, then throw a typed ContractError.
export class ContractError extends Error {
  constructor(readonly schemaName: string, readonly issues: unknown) { super(`Contract drift: ${schemaName}`); }
}
export function parseOrThrow<T>(schema: ZodType<T>, data: unknown, schemaName: string): T;
```

### 5.3 Typed endpoint functions (`src/api/endpoints.ts`)

Each function calls `apiClient`, then `parseOrThrow(<Schema>, response.data, '<name>')`. The query hooks (§5.7) wrap these.

```ts
import type {
  MeResponse, LoginRequest, RegisterRequest, RegisterResponse,
  ProjectResponseDto, CreateProjectRequest, UpdateProjectRequest,
  RunDto, StartRunRequest, RunStatusDto, DeterminismManifestDto, CheckpointDto,
  RoundResultDto, ServerLogDto,
  OrgDto, CreateOrgRequest, OrgMemberDto, AddOrgMemberRequest, UpdateOrgMemberRoleRequest,
  DatasetDto, DatasetVersionDto, PartitionRecipeDto, CreatePartitionRecipeRequest,
  AdminUserDto, UpdateUserRoleRequest, AuditEventDto,
} from './types';

export const authApi = {
  register: (body: RegisterRequest) => Promise<RegisterResponse>,                 // POST /api/auth/register
  login:    (body: LoginRequest)    => Promise<MeResponse>,                        // POST /api/auth/login (sets cookie)
  me:       () => Promise<MeResponse>,                                             // GET  /api/auth/me (silent 401 probe)
  logout:   () => Promise<void>,                                                   // POST /api/auth/logout (204)
  verifyEmail:    (token: string) => Promise<{ verified: true }>,                  // POST /api/auth/verify-email
  forgotPassword: (email: string) => Promise<{ status: 'accepted' }>,              // POST /api/auth/password/forgot
  resetPassword:  (token: string, newPassword: string) => Promise<{ reset: true }>,// POST /api/auth/password/reset
};

export const projectApi = {
  list:    () => Promise<ProjectResponseDto[]>,                                    // GET  /api/projects
  get:     (projectId: string) => Promise<ProjectResponseDto>,                     // GET  /api/projects/{id}
  create:  (body: CreateProjectRequest) => Promise<ProjectResponseDto>,            // POST /api/projects (201)
  patch:   (projectId: string, body: UpdateProjectRequest) => Promise<ProjectResponseDto>, // PATCH
  remove:  (projectId: string) => Promise<{ projectId: string; message: string }>,// DELETE
  results: (projectId: string) => Promise<RoundResultDto[]>,                       // GET  /api/projects/{id}/results
  logs:    (projectId: string, page: number, size: number) => Promise<ServerLogDto[]>, // GET .../logs?page&size
};

export const runApi = {
  start:    (projectId: string, body: StartRunRequest) => Promise<RunDto>,         // POST /api/projects/{id}/runs (202)
  listForProject: (projectId: string, page: number, size: number, status?: string) => Promise<RunDto[]>,
  get:      (runId: string) => Promise<RunDto>,                                    // GET  /api/runs/{runId}
  stop:     (runId: string) => Promise<RunDto>,                                    // POST /api/runs/{runId}/stop (202)
  status:   (runId: string) => Promise<RunStatusDto>,                              // GET  /api/runs/{runId}/status
  manifest: (runId: string) => Promise<DeterminismManifestDto>,                    // GET  /api/runs/{runId}/manifest
  checkpoints: (runId: string) => Promise<CheckpointDto[]>,                        // GET  /api/runs/{runId}/checkpoints
};

export const orgApi = {
  list:        () => Promise<OrgDto[]>,                                            // GET  /api/orgs
  create:      (body: CreateOrgRequest) => Promise<OrgDto>,                        // POST /api/orgs (201; creator=OWNER)
  get:         (orgId: string) => Promise<OrgDto>,
  members:     (orgId: string) => Promise<OrgMemberDto[]>,
  addMember:   (orgId: string, body: AddOrgMemberRequest) => Promise<OrgMemberDto>,
  updateMember:(orgId: string, userId: number, body: UpdateOrgMemberRoleRequest) => Promise<OrgMemberDto>,
  removeMember:(orgId: string, userId: number) => Promise<void>,
};

export const datasetApi = {
  list:            (orgId?: string) => Promise<DatasetDto[]>,
  versions:        (datasetId: string) => Promise<DatasetVersionDto[]>,
  createPartition: (datasetId: string, versionId: string, body: CreatePartitionRecipeRequest) => Promise<PartitionRecipeDto>,
  partitions:      (datasetId: string, versionId: string) => Promise<PartitionRecipeDto[]>,
};

export const adminApi = {
  users:          (page: number, size: number, q?: string) => Promise<AdminUserDto[]>, // GET /api/admin/users
  updateUserRole: (userId: number, body: UpdateUserRoleRequest) => Promise<AdminUserDto>,
  auditEvents:    (params: { page: number; size: number; action?: string; targetType?: string; from?: string; to?: string }) => Promise<AuditEventDto[]>,
};
```

### 5.4 V5 role-type contract + AuthIdentity (the live-bug fix — `A2-frontend.md:50-74`)

These are the **single most important types in the unit.** v1 modeled a flat `role: 'USER' | 'ADMIN'` and gated every admin surface on `=== 'ADMIN'`, while the backend emits `'PLATFORM_ADMIN'` — so the entire admin UI was dead (`A2-frontend.md:52-59`). v2 models the three orthogonal V5 layers exactly, matching `04-API-CONTRACTS.md` §1.1.

```ts
// src/api/types.ts  (these are z.infer<> from src/api/schemas.ts — single source)

export type PlatformRole = 'USER' | 'PLATFORM_ADMIN';
export type OrgRole      = 'OWNER' | 'ADMIN' | 'MEMBER';
export type ProjectRole  = 'MEMBER' | 'CLIENT';   // implicit owner via projects.user_id

export interface OrgMembership {
  orgId:   string;        // UUID string (organizations.id is UUID — 04-API-CONTRACTS §1)
  orgName: string;
  orgRole: OrgRole;
}

// Mirrors MeResponse (04-API-CONTRACTS §2.1). userId is a JSON number (users.id is BIGINT/Long).
export interface AuthIdentity {
  userId:        number;
  username:      string;
  email:         string;
  platformRole:  PlatformRole;
  orgs:          OrgMembership[];
  emailVerified: boolean;
}
```

Permission helpers (`src/auth/useIdentity.ts`) — the ONLY place role checks live (never inline a string compare in a component again):

```ts
export function useIdentity(): AuthIdentity | null;                  // null while loading or unauthenticated
export function useIsPlatformAdmin(): boolean;                      // identity?.platformRole === 'PLATFORM_ADMIN'
export function useOrgRole(orgId: string): OrgRole | null;          // from identity.orgs
export function useIsOrgAdmin(orgId: string): boolean;             // OrgRole ∈ {OWNER,ADMIN} OR PLATFORM_ADMIN
```

### 5.5–5.6 Zod schemas + query-key factory

Zod schemas (`src/api/schemas.ts`) mirror **every** wire shape in `04-API-CONTRACTS.md`. Representative examples (implement the rest by the same pattern; one schema per DTO):

```ts
import { z } from 'zod';

export const PlatformRoleSchema = z.enum(['USER', 'PLATFORM_ADMIN']);
export const OrgRoleSchema      = z.enum(['OWNER', 'ADMIN', 'MEMBER']);

export const MeResponseSchema = z.object({
  userId:        z.number().int(),
  username:      z.string(),
  email:         z.string().email(),
  platformRole:  PlatformRoleSchema,                                  // <-- the v1 bug fix at the wire boundary
  orgs:          z.array(z.object({
                   orgId:   z.string().uuid(),
                   orgName: z.string(),
                   orgRole: OrgRoleSchema,
                 })),
  emailVerified: z.boolean(),
});

export const RunStatusSchema = z.enum([
  'PENDING','STARTING','RUNNING','STOPPING','SUCCEEDED','STOPPED','FAILED',  // 04-API-CONTRACTS §4.3
]);

export const RunDtoSchema = z.object({
  id:            z.string().uuid(),
  projectId:     z.string().uuid(),
  orgId:         z.string().uuid(),
  status:        RunStatusSchema,
  strategy:      z.enum(['FedAvg','DeComFL']),
  launcher:      z.enum(['KUBERNETES','ECS','LOCAL_PROCESS']),
  executorRef:   z.string(),
  grpcEndpoint:  z.string().nullable(),
  numRounds:     z.number().int(),
  minClients:    z.number().int(),
  roundDeadlineSeconds: z.number().int(),
  currentRound:  z.number().int(),
  datasetVersionId: z.string().uuid(),
  modelArtifactId:  z.string().uuid().nullable(),
  requestedByUserId: z.number().int(),
  seed:          z.number().int(),
  startedAt:     z.string().datetime().nullable(),
  finishedAt:    z.string().datetime().nullable(),
  createdAt:     z.string().datetime(),
  errorMessage:  z.string().nullable(),
});

// The DeComFL communication-cost wedge — nullable so FedAvg runs omit scalar fields (04-API-CONTRACTS §5.1)
export const RoundResultPayloadSchema = z.object({
  id:            z.string().uuid(),
  projectId:     z.string().uuid(),
  runId:         z.string().uuid(),
  serverRound:   z.number().int(),
  loss:          z.number().nullable(),
  accuracy:      z.number().nullable(),
  gpuUtilization: z.number().nullable(),
  uplinkBytes:        z.number().int().nullable(),
  downlinkBytes:      z.number().int().nullable(),
  scalarsTransmitted: z.number().int().nullable(),   // K*P scalars — the O(K*P) proof
  modelParamCount:    z.number().int().nullable(),   // model dimension d, for dimension-free comparison
  roundDurationSeconds: z.number().nullable(),
  aggregationSeconds:   z.number().nullable(),
  activeClients:        z.number().int().nullable(),
  timestamp:     z.string().datetime(),
});

// The standard error envelope (04-API-CONTRACTS §12)
export const ErrorEnvelopeSchema = z.object({
  timestamp: z.string().datetime(),
  status:    z.number().int(),
  code:      z.string(),
  message:   z.string(),
  path:      z.string(),
  traceId:   z.string(),
  fieldErrors: z.array(z.object({ field: z.string(), message: z.string() })).optional(),
});
```

The query-key factory (`src/api/queryKeys.ts`) is the single source of TanStack Query cache keys so invalidation is consistent:

```ts
export const qk = {
  me:        () => ['me'] as const,
  projects:  () => ['projects'] as const,
  project:   (id: string) => ['projects', id] as const,
  runs:      (projectId: string) => ['projects', projectId, 'runs'] as const,
  run:       (runId: string) => ['runs', runId] as const,
  runStatus: (runId: string) => ['runs', runId, 'status'] as const,
  results:   (projectId: string) => ['projects', projectId, 'results'] as const,
  orgs:      () => ['orgs'] as const,
  orgMembers:(orgId: string) => ['orgs', orgId, 'members'] as const,
  datasets:  (orgId?: string) => ['datasets', orgId ?? 'all'] as const,
  adminUsers:(q?: string) => ['admin', 'users', q ?? ''] as const,
  audit:     () => ['admin', 'audit'] as const,
};
```

### 5.7 Representative query/mutation hook signatures (`src/query/*`)

```ts
import type { UseQueryResult, UseMutationResult } from '@tanstack/react-query';

export function useMe(): UseQueryResult<MeResponse | null>;                       // retry:false; 401→null (the probe)
export function useLogin(): UseMutationResult<MeResponse, ContractError, LoginRequest>;
export function useLogout(): UseMutationResult<void, unknown, void>;

export function useProjects(): UseQueryResult<ProjectResponseDto[]>;              // org-scoped server-side
export function useProject(projectId: string): UseQueryResult<ProjectResponseDto>;
export function useCreateProject(): UseMutationResult<ProjectResponseDto, ContractError, CreateProjectRequest>;

export function useRuns(projectId: string): UseQueryResult<RunDto[]>;
export function useRun(runId: string): UseQueryResult<RunDto>;
export function useStartRun(projectId: string): UseMutationResult<RunDto, ContractError, StartRunRequest>;
export function useStopRun(): UseMutationResult<RunDto, ContractError, string /*runId*/>;
export function useRunStatus(runId: string, enabled: boolean): UseQueryResult<RunStatusDto>; // refetchInterval while non-terminal
```

### 5.8 The shared STOMP subscribe hook (`src/realtime/useStompTopic.ts`)

```ts
import type { ZodType } from 'zod';

// Subscribe to a STOMP destination via the ONE shared connection (§6.4).
// The hook ref-counts: many components subscribing to the same destination share one STOMP subscription.
// Every frame body is validated with `schema` before onMessage fires (drop + log on parse failure).
export function useStompTopic<T>(
  destination: string | null,        // null => no-op (e.g. before projectId known)
  schema: ZodType<T>,
  onMessage: (payload: T) => void,
): { connected: boolean };
```

### 5.9 Row types this unit reads (from `03-DATA-MODEL.md`, READ-ONLY over the wire)

The frontend owns no database tables. It consumes these row shapes **only as DTOs** (never the JPA entity — v1 leaked the `User` entity, `A2`/`A1-F3`). The relevant tables and the DTO they surface as:

| DB table (`03-DATA-MODEL.md`) | Surfaced DTO | Frontend usage |
|---|---|---|
| `users` (id BIGINT, platform_role) | `MeResponse` / `AdminUserDto` | identity + admin user list |
| `organizations` (id UUID), `organization_memberships` | `OrgDto` / `OrgMemberDto` | org switcher, member list |
| `projects` (id UUID, org_id NOT NULL) | `ProjectResponseDto` | project grid/detail |
| `fl_runs` (id UUID, status, current_round, seed) | `RunDto` / `RunStatusDto` | run list, live status |
| `round_results` (V7 table replacing v1 `round_result`; comm-cost cols `uplink_bytes`/`downlink_bytes`/`scalars_transmitted`, `03-DATA-MODEL.md §5.2`) | `RoundResultDto`/`RoundResultPayload` | convergence + comm-cost charts |
| `server_logs` | `ServerLogDto` | historical log fetch |
| `audit_events` (metadata JSONB) | `AuditEventDto` | admin audit log |
| `datasets` / `dataset_versions` / `partition_recipes` (`03-DATA-MODEL.md §5.1`) | `DatasetDto` etc. | dataset registry UI |

---

## 6. Core algorithms & flows (real code / precise pseudocode)

### 6.1 Provider stack (`src/main.tsx`)

Order is load-bearing; outer providers must mount before inner ones depend on them.

```
ErrorBoundary
  └─ QueryClientProvider (queryClient)        // server-state cache available to everything
       └─ ThemeProvider                        // OKLCH token class on <html>
            └─ AuthProvider                     // wraps useMe(); exposes identity, handles authError
                 └─ OrgContextProvider          // active org for org-scoped UI
                      └─ StompProvider           // ONE shared WS connection (mounted after auth so cookie exists)
                           └─ ToastProvider
                                └─ RouterProvider (App routes)
```

`main.tsx` also calls `initOtelWeb()` from `lib/logger.ts` once, before render, to register the web tracer that sets `traceparent` on the run-start request (`04-API-CONTRACTS.md` §14; optional — backend originates a root span if absent).

### 6.2 Cookie-auth + the 401 silent-probe interceptor (the exact branching — `A2-frontend.md:42-48`)

This is the salvaged v1 posture, preserved verbatim except the role type. Implement `installAuthInterceptor` exactly:

```text
function onResponseError(error):
    response = error.response
    requestUrl = error.config.url            // relative to baseURL, e.g. "/auth/me"

    if response is undefined:                 # network error / timeout, NOT an auth signal
        return Promise.reject(error)          # do NOT log out on a transient blip (fixes A2 H6)

    if response.status == 401:
        isMeProbe   = requestUrl matches /\/auth\/me$/
        isLoginCall = requestUrl matches /\/auth\/login$/
        if isMeProbe or isLoginCall:
            # SILENT: the SPA is probing "am I logged in?" or a login legitimately failed.
            # Swallow — useMe() maps this 401 to `null`; useLogin() surfaces BAD_CREDENTIALS.
            return Promise.reject(error)
        else:
            # HARD logout: a data route returned 401 => session is gone/expired.
            window.dispatchEvent(new CustomEvent(AUTH_ERROR_EVENT))
            return Promise.reject(error)

    if response.status == 403:
        # 403 is NEVER a logout (authn ok, authz denied). Surface FORBIDDEN to the caller. (A2:45)
        return Promise.reject(error)

    return Promise.reject(error)
```

`AuthProvider` (`src/auth/AuthContext.tsx`) adds a `window.addEventListener(AUTH_ERROR_EVENT, ...)` that calls `queryClient.clear()` + navigates to `/login`. `useMe()` is configured `retry: false` and, in its `queryFn`, catches a 401 and returns `null` (logged-out is a value, not an error) so the silent probe never throws into a render.

Sequence — boot / login / hard-logout:

```
Browser                         Axios instance                Spring Boot
  │ app boot                          │                            │
  │ useMe() ──GET /api/auth/me───────▶│ ──── (cookie attached) ───▶│
  │                                   │◀──── 401 NOT_AUTHENTICATED ─│   (no session)
  │ interceptor: isMeProbe → swallow  │                            │
  │ useMe() returns null → render /login                            │
  │                                                                 │
  │ submit login ─POST /api/auth/login {user,pass}─────────────────▶│
  │                                   │◀── 200 MeResponse + Set-Cookie jwtToken (HttpOnly) ─│
  │ useLogin onSuccess: setQueryData(qk.me(), me) + navigate('/')   │
  │                                                                 │
  │ later: GET /api/projects ─────────────────────────────────────▶│
  │                                   │◀──── 401 (cookie expired) ──│
  │ interceptor: NOT a probe → dispatch 'authError'                 │
  │ AuthProvider: queryClient.clear() + navigate('/login')          │
```

### 6.3 TanStack Query defaults (`src/query/queryClient.ts`)

```ts
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,            // 30s: projects/orgs change rarely; avoids the v1 4× refetch storm
      gcTime: 5 * 60_000,
      retry: (failureCount, err) => {
        // never retry auth/validation failures; retry transient 5xx up to 2x
        const status = (err as AxiosError)?.response?.status;
        if (status && status < 500) return false;
        return failureCount < 2;
      },
      refetchOnWindowFocus: false,
    },
    mutations: { retry: 0 },
  },
});
```

`useRunStatus(runId, enabled)` sets `refetchInterval: (q) => isTerminal(q.state.data?.status) ? false : 3000` so a non-terminal run polls every 3s and stops polling on `SUCCEEDED|STOPPED|FAILED`. The live STOMP `/topic/status` feed is the primary signal; polling is the fallback when the socket is down.

### 6.4 The ONE shared STOMP connection (`src/realtime/StompProvider.tsx`) — fixes the v1 3-connection problem (`A2-frontend.md:82`)

v1 opened **three** `StompClient`s to the same `/ws-logs` endpoint (NotificationContext, DashboardV2, LogViewer). v2 mounts exactly one and ref-counts subscriptions.

```text
class StompManager:
    client: StompClient | null = null
    refCounts: Map<destination, number> = {}
    callbacks: Map<destination, Set<(frame) => void>> = {}
    connected = false

    function connect():
        wsUrl = deriveWsUrl()                 # lib/wsUrl.ts — MUST be wss:// outside dev; throws on http:// in prod
        client = new StompClient({
            brokerURL: wsUrl,                  # cookie attached by the browser on the WS upgrade (first-party)
            reconnectDelay: 5000,              # auto-reconnect
            heartbeatIncoming: 10000,
            heartbeatOutgoing: 10000,
            onConnect: () => { connected=true; resubscribeAll() },   # re-subscribe every active destination
            onWebSocketClose: () => { connected=false },
            onStompError: (frame) => logger.error('stomp', frame.headers['message']),
        })
        client.activate()

    function subscribe(destination, onFrame):
        callbacks[destination].add(onFrame)
        refCounts[destination] = (refCounts[destination] ?? 0) + 1
        if refCounts[destination] == 1 and connected:
            client.subscribe(destination, frame => callbacks[destination].forEach(cb => cb(frame)))
        return function unsubscribe():
            callbacks[destination].delete(onFrame)
            refCounts[destination] -= 1
            if refCounts[destination] == 0:
                client.unsubscribe(destination); delete callbacks[destination]
```

`deriveWsUrl()` (`lib/wsUrl.ts`):
```text
function deriveWsUrl():
    root = env.serverRootUrl                  # VITE_SERVER_ROOT_URL — in prod fail-fast set (§8.1, A2:220)
    if env.mode == 'production' and root.startsWith('http://'):
        throw Error('VITE_SERVER_ROOT_URL must be https/wss in production (mixed-content would drop the cookie)')
    return root.replace(/^http/, 'ws') + '/ws-logs'    # http→ws, https→wss
```

### 6.5 Live-log history/live merge (`src/realtime/logStore.ts`) — SALVAGED, race fixed (`A2-frontend.md:84-90`)

Keep the genuinely good v1 store: module-level cache surviving modal unmount, **monotonic never-reused IDs** used as React keys (verified `frontend/src/services/logStore.ts:19,50` and `MAX_LOGS_PER_PROJECT=2000` at `:41`). Two v2 fixes:

1. **Dedup on a stable server-side `id`, not `timestamp+message`.** v1 deduped on second-resolution `timestamp+message`, so two distinct lines in the same second collided (`A2-frontend.md:86`). The v2 `LogLinePayload` carries no per-line server `id` in the contract today, so the store keys dedup on `(timestamp_ms, level, message, roundIdx)` where `timestamp` is the full millisecond ISO value from `04-API-CONTRACTS.md` §11 (`2026-05-29T16:18:02.123Z`) — millisecond resolution removes the same-second collision. **If** the backend later adds a monotonic line `id`, switch dedup to that (flag in code as a TODO referencing this section).
2. **Serialize history-then-live.** `LogViewer` must `await` the historical `projectApi.logs()` fetch and merge it **before** attaching the live subscription, or buffer live frames that arrive during the fetch and merge them after. Implement the buffer approach:

```text
function openLogViewer(projectId):
    liveBuffer = []
    subscribed = useStompTopic(`/topic/logs/${projectId}`, LogLinePayloadSchema, frame => {
        if (historyMerged) logStore.append(projectId, [frame])
        else liveBuffer.push(frame)              # buffer until history lands (no ordering race)
    })
    history = await projectApi.logs(projectId, 0, MAX)   # one shot
    logStore.replaceHistory(projectId, history)          # prepend-merge, dedup, trim to 2000
    historyMerged = true
    logStore.append(projectId, liveBuffer); liveBuffer = []
```

3. **LRU over project count.** v1's cache `Map` never evicted dead projects (`A2-frontend.md:167`). Add an LRU bound (e.g. 20 projects) over `projectId` keys; evict the least-recently-appended project's array.

### 6.6 (reserved — animation rule stated in §6.9)

### 6.7 Routing + code-splitting (`src/App.tsx`)

Use `react-router-dom` v7 with `React.lazy` on every authenticated/heavy route so the login bundle is tiny (`A2-frontend.md:120-127`; target < 150 KB gzipped initial). Each `lazy()` import is a code-split boundary; wrap routes in `<Suspense fallback={<Skeleton/>}>`.

```ts
const DashboardPage        = lazy(() => import('./pages/DashboardPage'));
const ProjectDetailPage    = lazy(() => import('./pages/ProjectDetailPage'));
const RunObservabilityPage = lazy(() => import('./pages/RunObservabilityPage'));   // pulls recharts
const AdminUsersPage       = lazy(() => import('./pages/admin/AdminUsersPage'));
// LoginPage / RegisterPage are eager (tiny, first paint) — NOT lazy.
```

Route table (guards from §5.4 helpers):

```
/login, /register, /verify-email, /forgot-password, /reset-password   → PUBLIC (eager)
/                       → RequireAuth → DashboardPage
/projects/:projectId    → RequireAuth → ProjectDetailPage
/runs/:runId            → RequireAuth → RunObservabilityPage
/datasets, /models      → RequireAuth → lazy pages
/admin/users            → RequirePlatformAdmin → AdminUsersPage
/admin/projects         → RequirePlatformAdmin → AdminProjectsPage
/admin/audit            → RequirePlatformAdmin → AuditLogPage
*                       → NotFoundPage
```

`vite.config.ts` `build.rollupOptions.output.manualChunks` splits `recharts` and `framer-motion` into a stable `vendor-charts` chunk so they cache across deploys (`A2-frontend.md:127`).

### 6.8 FL-run visualization — the per-run Observability surface (`RunObservabilityPage.tsx`)

This is the product's reason to exist (`C5-design-ux.md` §3, `B3-observability.md` §6). It is **per-run scoped** — never portfolio-flattened (the v1 `TrainingInsightsView` `.flat()` bug mixed rounds across projects, `C5-design-ux.md:73`, `A2-frontend.md:98`). Data flow:

```
useProjectResults(projectId)          # seed historical rounds via REST (TanStack Query)
        │
        ▼
liveResultsStore[(projectId,runId)]   # array of RoundResultPayload, merged from /topic/results (incremental)
        │  (useStompTopic '/topic/results/{projectId}', RoundResultPayloadSchema, filter to runId)
        ▼
┌─────────────────────────────┬──────────────────────────────┬─────────────────────────────┐
│ ConvergenceChart            │ CommunicationCostPanel        │ PerClientPanel              │
│ recharts dual-axis line:    │ DeComFL bandwidth wedge:      │ small-multiples per client: │
│ loss + accuracy vs          │ scalarsTransmitted + uplink/  │ contribution, local loss,   │
│ serverRound (scoped to runId)│ downlinkBytes vs modelParam-  │ last-seen (from /topic/runs │
│ smoothing toggle + target   │ Count → derived "bytes-per-   │ CLIENT_JOINED/LEFT events)  │
│ line; series via --chart-*  │ round vs equiv FedAvg" stat   │                             │
└─────────────────────────────┴──────────────────────────────┴─────────────────────────────┘
```

**`CommunicationCostPanel` (net-new, the differentiator) — exact derivation.** DeComFL's thesis is communication is O(K·P) scalars per round, **independent of model dimension d** (`B3-observability.md` §6.2). For each `RoundResultPayload`:

```text
actualUplinkBytes  = uplinkBytes                 # measured, ~ K*P*8 for DeComFL
equivFedAvgBytes   = modelParamCount * 4         # a dense FedAvg upload would move d float32 weights
savingsFactor      = equivFedAvgBytes / max(actualUplinkBytes, 1)
```
Render two series: cumulative `uplinkBytes` (DeComFL, near-flat) vs cumulative `equivFedAvgBytes` (the counterfactual, steep), on a log-y axis, plus a headline stat `"{savingsFactor}× less bandwidth than FedAvg"`. This is the W&B/Grafana-benchmarked "comm-cost panel" the audit calls the highest-value missing surface (`C5-design-ux.md:82,91`). Only show the scalar series when `strategy === 'DeComFL'` and `scalarsTransmitted != null` (FedAvg runs omit it — §5.5).

**`ConvergenceChart`.** recharts `LineChart`, X = `serverRound`, dual Y (loss left, accuracy right). All strokes come from `var(--chart-1)`…`var(--chart-5)` tokens, never raw hex (`C5-design-ux.md:42` — v1 leaked `#ef4444`/`#f43f5e` literals). A smoothing toggle (moving average window 1/3/5) and an optional target-accuracy reference line.

### 6.9 FederationOrrery (`features/observability/FederationOrrery.tsx`) — SALVAGED, made data-honest + performant

Keep it as the hero "live federation" widget (`C5-design-ux.md:212`), with three mandatory v2 fixes:

1. **Real data, not mock.** v1 passed literal clients (`jetson-orin-1`, fixed `round={28}`) — `A2-frontend.md:98`. v2 derives clients and the round from `liveResultsStore` + `/topic/runs` `CLIENT_JOINED/LEFT` events. If a run has no live data yet, render an explicit empty state ("Waiting for clients…"), never fake nodes.
2. **No rAF setState storm.** v1 ran an unconditional `requestAnimationFrame` calling `setT(...)` ~60×/sec, forcing a full React re-render of the SVG forever (`A2-frontend.md:99`). v2 animates with **CSS/SVG keyframes or `framer-motion` declarative transforms** — zero React re-renders for decorative motion (the codebase already does this for the round-pulse, `FederationOrrery.tsx:108`).
3. **`prefers-reduced-motion`.** Gate all spin/orbit animation behind `@media (prefers-reduced-motion: no-preference)` (WCAG 2.3.3; `C5-design-ux.md:153`). Encode client state with **text+icon+color**, not color alone (WCAG 1.4.1; `C5-design-ux.md:150`).

---

## 7. Data it owns

**This unit owns no persistent database tables.** Its only durable state is browser-local and in-memory:

| Store | Location | Shape | Lifetime |
|---|---|---|---|
| Server-state cache | TanStack Query `QueryClient` | keyed by `queryKeys.ts`; values are validated DTOs | session; `gcTime` 5 min; cleared on logout |
| `logStore` | module-level `Map<projectId, StoredLogEntry[]>` | `StoredLogEntry = LogLinePayload & { id: number }` (monotonic id) | app lifetime; LRU-bounded over 20 projects; trim 2000/project |
| `liveResultsStore` | module-level `Map<\`${projectId}:${runId}\`, RoundResultPayload[]>` | per-round results | app lifetime; cleared on logout |
| Active org | `OrgContext` React state | `orgId: string \| null` | session (in-memory only) |
| Theme | `ThemeProvider` + `<html class>` | `'light' \| 'dark'` | persisted (theme preference is non-sensitive; cookie/localStorage acceptable for theme only) |

**Invariant:** no JWT, no model weights, and no training data are ever stored client-side. The auth token lives only in the HttpOnly cookie the JS cannot read (`04-API-CONTRACTS.md` §1).

In-memory structures (TS):

```ts
interface StoredLogEntry extends LogLinePayload { id: number; }       // id = module monotonic counter (React key)
type LogCache    = Map<string /*projectId*/, StoredLogEntry[]>;       // LRU over projectId, trim 2000
type ResultCache = Map<string /*`${projectId}:${runId}`*/, RoundResultPayload[]>;
```

---

## 8. Configuration & environment variables

All frontend env vars are `VITE_`-prefixed (Vite only exposes those to the client). They live in committed `.env.{development,ec2demo,production}` and the gitignored `.env.local`. The `--mode` flag selects the file and maps 1:1 to a Spring profile (`02-TECH-STACK.md` §12.2).

| Variable | Type | dev (`.env.development`) | ec2demo | production | Validated where |
|---|---|---|---|---|---|
| `VITE_FEDLEARN_API_URL` | string (URL) | `http://localhost:8081` | (proxied → `/api`) | required, HTTPS | `lib/env.ts` — **fail-fast** if missing in prod (`A2-frontend.md:44`) |
| `VITE_SERVER_ROOT_URL` | string (URL) | `http://localhost:8081` | (proxied) | required, HTTPS | `lib/env.ts` — **fail-fast** if missing in prod (NEW v2; `A2-frontend.md:160,220`) |
| `VITE_PROXY_TARGET` | string (URL) | n/a | the demo backend host | n/a | `vite.config.ts` (ec2demo proxy only) |
| `MODE` (built-in) | `'development'\|'ec2demo'\|'production'` | auto | auto | auto | `import.meta.env.MODE` |

`lib/env.ts` (fail-fast contract):

```ts
function readEnv() {
  const mode = import.meta.env.MODE as 'development' | 'ec2demo' | 'production';
  const apiUrl = import.meta.env.VITE_FEDLEARN_API_URL as string | undefined;
  const serverRootUrl = import.meta.env.VITE_SERVER_ROOT_URL as string | undefined;
  if (mode === 'production') {
    if (!apiUrl) throw new Error('VITE_FEDLEARN_API_URL is required in production');
    if (!serverRootUrl) throw new Error('VITE_SERVER_ROOT_URL is required in production'); // v2 add
  }
  return { mode, apiUrl: apiUrl ?? 'http://localhost:8081', serverRootUrl: serverRootUrl ?? 'http://localhost:8081' };
}
export const env = readEnv();
```

`vite.config.ts` non-negotiables: `server.strictPort = true`, `server.port = 5173` (CORS allowlist keyed on it); the `ec2demo` proxy for `/api` and `/ws-logs` to `VITE_PROXY_TARGET` (keeps cookies first-party); `build.sourcemap = false`; `build.minify = 'esbuild'`; `manualChunks` splitting `recharts` + `framer-motion`.

---

## 9. Error handling & edge cases (enumerate real failure modes + exact handling)

| # | Failure mode | Exact handling |
|---|---|---|
| 1 | `GET /api/auth/me` 401 on boot | Interceptor swallows (isMeProbe); `useMe()` returns `null`; render `/login`. No redirect loop. (§6.2) |
| 2 | 401 on a data route (session expired mid-session) | Interceptor dispatches `authError`; `AuthProvider` clears the query cache + navigates `/login`. (§6.2) |
| 3 | 403 on any route | NOT a logout. Surface the error-envelope `message` via a toast keyed on `code=FORBIDDEN`; stay on the page. (`A2:45`) |
| 4 | Network error / timeout (no `response`) | Do NOT log out (fixes v1 `A2 H6`). Show a retry toast; TanStack Query retries 5xx/transient up to 2×. |
| 5 | Backend contract drift (Zod parse fails) | `parseOrThrow` throws `ContractError`; the query enters error state; `lib/logger.ts` logs schema name + issues. In dev this is loud (red error boundary); in prod it shows a generic "data error" state. This is the mechanism that would have caught the `PLATFORM_ADMIN` bug. (`A2:74`) |
| 6 | `409 RUN_ALREADY_ACTIVE` on start-run | `StartRunModal` maps `code` via `lib/errorCode.ts` → "A run is already active for this project." Disable the start button; refetch `useRuns`. |
| 7 | `409 ORG_QUOTA_EXCEEDED` | Map to "Your organization has reached its concurrent-run limit." Surface as a non-dismissible inline error on the modal. |
| 8 | `422 NO_DATASET_VERSION` / `422 UNSUPPORTED_LAUNCHER` | Inline field error in the run config form; for `UNSUPPORTED_LAUNCHER` hide `LOCAL_PROCESS` outside dev mode entirely. |
| 9 | `VALIDATION_FAILED` (400 with `fieldErrors`) | Map each `fieldError.field` to the form control; the field names match the Zod schema field names by design (`04-API-CONTRACTS.md` §1). |
| 10 | STOMP socket drops | `@stomp/stompjs` auto-reconnects (`reconnectDelay: 5000`); on `onConnect` re-subscribe every active destination; `useRunStatus` polling is the fallback while disconnected. (§6.4) |
| 11 | `wss://` derivation yields `ws://` in prod | `lib/wsUrl.ts` throws at startup (mixed-content would drop the cookie). (§6.4) |
| 12 | STOMP frame fails Zod validation | Drop the frame, log via `logger.ts`; never feed unvalidated data to a chart/store. (§5.8) |
| 13 | Subscribe to a topic the user can't access | Backend rejects the `SUBSCRIBE` frame at the channel interceptor (`04-API-CONTRACTS.md` §11); the client logs the STOMP error and shows "no access" — never silently shows another tenant's data (v1 wildcard-topic leak, `A2:88`). v2 subscribes only to the specific `projectId`, never `/topic/results/*`. |
| 14 | Notifications persist across logout | Clear all in-memory stores (logStore, liveResultsStore, query cache, notifications) when identity becomes `null` (fixes v1 PII leak `A2:166`). |
| 15 | Run reaches a terminal state | `useRunStatus` `refetchInterval` returns `false`; STOMP `/topic/status` `SUCCEEDED|FAILED|STOPPED` flips the badge; the orrery shows a final state, animation stops. |
| 16 | StrictMode double-mount (dev) | The shared STOMP manager is ref-counted and idempotent on `activate()`; subscribe/unsubscribe are balanced in `useEffect` cleanup so a double-mount does not leak a connection (fixes v1 `A2:86`). |

`lib/errorCode.ts` maps the full `code` registry (`04-API-CONTRACTS.md` §12.1) to `{ userMessage: string; action: 'toast' | 'inline' | 'logout' | 'silent' }`. The frontend switches on `code` (stable), never on `message` (human-facing, may change).

---

## 10. Testing strategy

Framework: **Vitest** (`3.x`) + `@testing-library/react` for unit/component; **MSW** (`2.14.6`) to mock the REST + STOMP boundary; **Playwright** (`1.x`) for one E2E golden path (`02-TECH-STACK.md` §15). v1 had **zero** frontend tests against the riskiest code (`A2-frontend.md:129-140`); v2 stands the layer up. Coverage starts by measuring, gates at 40% once auth/STOMP/role paths are covered (`02-TECH-STACK.md` §15).

| Test file | Asserts |
|---|---|
| `axiosClient.interceptor.test.ts` | `401 on /auth/me` → rejected, **no** `authError` event; `401 on /api/projects` → `authError` dispatched; `403` → rejected, **no** `authError`; network error (no response) → rejected, **no** `authError`. (the §6.2 branches) |
| `schemas.meResponse.test.ts` | Valid `MeResponse` with `platformRole:'PLATFORM_ADMIN'` parses; a payload with legacy `role:'ADMIN'` (no `platformRole`) **throws ContractError** (would have caught the v1 dead-admin bug). |
| `useIdentity.roles.test.tsx` | `useIsPlatformAdmin()` is `true` for `platformRole:'PLATFORM_ADMIN'`, `false` for `'USER'`; `useIsOrgAdmin(orgId)` is `true` for `OWNER`/`ADMIN`, `false` for `MEMBER`, `true` for any `PLATFORM_ADMIN`. |
| `RequirePlatformAdmin.test.tsx` | A `PLATFORM_ADMIN` sees the admin route; a `USER` is sent to `/forbidden`; the admin nav link renders **only** for `PLATFORM_ADMIN`. (regression-locks the v1 bug) |
| `logStore.test.ts` | Monotonic never-reused ids; dedup on `(timestamp_ms, level, message, roundIdx)` drops a true duplicate but keeps two same-second distinct lines; trim caps at 2000/project; LRU evicts the 21st project. |
| `logViewer.race.test.tsx` | Live frames arriving before history are buffered, then merged after history with correct ordering and no duplicate; `connected` never flips on a torn-down client under double-mount. |
| `stompProvider.refcount.test.tsx` | Two components subscribing to the same destination open ONE STOMP subscription; unmounting one keeps it; unmounting both unsubscribes. |
| `wsUrl.test.ts` | `https://h` → `wss://h/ws-logs`; `http://h` in prod **throws**; `http://localhost:8081` in dev returns `ws://...`. |
| `communicationCost.test.tsx` | Given `modelParamCount=66e6, uplinkBytes=240`, the savings stat ≈ `(66e6*4)/240`; the scalar series is hidden when `strategy='FedAvg'`. |
| `convergenceChart.scope.test.tsx` | Renders only the rounds for the given `runId` (regression-locks the v1 portfolio-flatten bug). |
| `errorCode.test.ts` | Every `code` in the §12.1 registry maps to a non-empty `userMessage` and a valid `action`. |
| Playwright `e2e/golden-path.spec.ts` | login → create project → start a run → see the run reach `RUNNING` and the convergence chart plot ≥1 round (MSW or a seeded backend supplies the run stream). |

MSW handlers (`src/test/msw/handlers.ts`) return fixtures that pass the Zod schemas; one negative handler per critical endpoint returns the error envelope to exercise §9 paths.

---

## 11. Build & run (exact commands, verify in isolation)

```bash
# from repo root, pin Node first
nvm use                       # reads frontend/.nvmrc → 24.x   (or: asdf install)

cd frontend
npm ci                        # install pinned deps from package-lock.json (NOT npm install)

# --- run locally against a local backend (Spring 'dev' profile on :8081) ---
npm run dev                   # vite --open --mode development → http://localhost:5173 (strictPort)

# --- run against the EC2 demo backend via the first-party-cookie proxy ---
npm run dev:ec2demo           # vite --mode ec2demo → proxies /api + /ws-logs to VITE_PROXY_TARGET

# --- typecheck + lint (CI gates these) ---
npx tsc --noEmit              # strict typecheck
npm run lint                  # eslint . (tseslint strictTypeChecked + jsx-a11y; no-explicit-any:error)

# --- unit/component tests + coverage ---
npm run test                  # vitest run
npm run test -- --coverage    # vitest run --coverage (v8)

# --- E2E (Playwright) ---
npx playwright install --with-deps   # one-time browser download
npm run test:e2e              # playwright test (webServer = vite preview of the prod build)

# --- production build + local verification of the static bundle ---
npm run build                 # tsc && vite build --mode production → dist/
npm run preview               # serve dist/ locally to smoke-test the built SPA
```

**Isolation verification (no backend):** `npm run test` runs entirely against MSW — no Spring Boot, no Postgres. The Playwright golden path can run against MSW (network-mocked) or a seeded local backend; default to MSW for CI determinism. `package.json` scripts to add: `"test": "vitest run"`, `"test:watch": "vitest"`, `"test:e2e": "playwright test"`.

**Definition of "this unit builds":** `tsc --noEmit` clean, `eslint .` clean, `vitest run` green, `vite build` produces `dist/` with the initial (unauthenticated) chunk < 150 KB gzipped (`A2-frontend.md:127`), and `react-icons` is absent from `package.json`.

---

## 12. Reasoning & alternatives (why this design; what was rejected; cite audit)

| Decision | Why | Rejected alternative (why) |
|---|---|---|
| React 19 + Vite 6, **not Next.js** | Static SPA; no SEO/SSR need; CloudFront+S3 deploy is cheapest; the first-party-cookie problem is already solved by the Vite proxy (`A2-frontend.md:13,158,203`). | Next.js/SSR — adds a server tier (cost, ops) for zero benefit; the hard part (first-party cookies to a remote backend) is already solved without a server (`A2:158`). |
| TanStack Query for server-state | Cache, dedup, background refetch, `AbortSignal` cancellation; deletes the v1 four duplicate `fetchProjects` triads that fired on every nav (`A2-frontend.md:33-37`, `README.md:123`). | Redux Toolkit Query (heavier, Redux-coupled); SWR (fewer cache-invalidation features for the run lifecycle) — `02-TECH-STACK.md` §13.1. Hand-rolled `useEffect+fetch` — the v1 anti-pattern being removed. |
| Zod at the Axios + STOMP boundary | Fails loudly on backend contract drift; would have caught the `PLATFORM_ADMIN` dead-admin bug at runtime in dev (`A2-frontend.md:74,151`). Kills `any`-reintroduction through JSON. | Yup/io-ts (weaker TS inference) — `02-TECH-STACK.md` §13.2. Trusting the declared TS union (v1's mistake: the compiler "trusts the lie" of an untyped JSON cast, `A2:59`). |
| Three V5 role-type layers + permission hooks | The single highest-leverage correctness fix; v1's flat `role:'USER'\|'ADMIN'` killed the entire admin UI because the backend emits `'PLATFORM_ADMIN'` (`A2-frontend.md:50-74`, `README.md:120`, `R7`). | A single flat `role` (v1) — structurally cannot model org/project context, blocks multi-tenant UI. |
| ONE shared STOMP connection, ref-counted | v1 opened 3 sockets to one endpoint per tab; N tabs × 3 multiplies connections against a single Spring broker (`A2-frontend.md:82`). | Three independent clients (v1) — does not scale; per-component `new StompClient` is the rejected pattern. SSE was considered for logs (`A2:90`) but STOMP is the locked bidirectional contract (`02-TECH-STACK.md` §10). |
| Subscribe to specific `projectId`, never `/topic/*` | v1 wildcard subscriptions delivered every project's telemetry to every user, then filtered client-side — an info-leak (`A2-frontend.md:88`). v2 backend also enforces topic-level authz (`04-API-CONTRACTS.md` §11). | Wildcard `/topic/results/*` (v1) — rejected: cross-tenant leak + wasted bandwidth. |
| Per-run scoped observability page (not portfolio) | The v1 `TrainingInsightsView` flattened rounds across all projects into one meaningless timeline (`C5-design-ux.md:73`, `A2:98`). | Portfolio-flatten `.flat()` (v1) — statistically meaningless; rejected. |
| CommunicationCostPanel as a first-class panel | DeComFL's entire wedge is O(K·P) communication independent of model dimension; v1 had no comm-cost UI and the field schema didn't exist (`C5-design-ux.md:82`, `B3-observability.md` §6.2). The `RoundResultPayload` now carries `scalarsTransmitted`/`uplinkBytes`/`modelParamCount` (`04-API-CONTRACTS.md` §5.1). | No comm-cost viz (v1) — the differentiator was unmeasured and invisible; rejected. |
| Orrery: real data + declarative animation + reduced-motion | v1 fed it mock clients and ran a 60fps rAF setState storm forever on a decorative SVG (`A2-frontend.md:98-99`); CSS keyframes/framer-motion remove the React re-renders; `prefers-reduced-motion` is a WCAG floor (`C5-design-ux.md:153`). | rAF `setT` per frame (v1) — wasted CPU/GPU on battery clients; rejected. Mock data (v1) — shows a fictional federation; rejected. |
| shadcn/ui + one OKLCH `@fedlearn/tokens` package | v1 had three disjoint palettes across surfaces and a coin-flip desktop accent (`C5-design-ux.md:28-40`); shadcn is already substrate-compatible (CVA + tailwind-merge + clsx in v1 `package.json`); Radix dialogs ship focus-trap + `aria-modal` for free, fixing the v1 `<div>`-overlay a11y gap (`C5-design-ux.md:121`). | MUI/Chakra (heavier, fights Tailwind v4 + the OKLCH approach, runtime weight — `C5:176`); three local palettes (v1) — rejected. |
| CSP/HSTS set authoritatively at the backend + meta fallback | Cookie-auth means any injected script runs with the session; a CSP-less page turns a vendor XSS into a full credentialed compromise (`A2-frontend.md:104-118`). Backend `headers.contentSecurityPolicy(...)` covers API+SPA; report-only rollout first. | No CSP (v1) — rejected; `SameSite` cookies do nothing against same-origin XSS (`A2:111`). |
| `React.lazy` route split + `manualChunks` vendor split | v1 shipped a single ~1.0 MB chunk with zero `React.lazy`, downloading the whole authenticated app (recharts+framer-motion) to a user who only wants the login form (`A2-frontend.md:120-127`). | Single bundle (v1) — rejected; target < 150 KB gzipped initial. |
| Vitest + Playwright + MSW | Vitest reuses the Vite pipeline; MSW mocks the boundary; one Playwright e2e covers the golden path (`A2-frontend.md:140`). | Jest (no native Vite integration); Cypress (Playwright has better multi-browser/parallelism) — `02-TECH-STACK.md` §15. Zero tests (v1) — unbounded regression risk on a paid multi-tenant product; rejected. |

**Uncertainty flagged (do not paper over):**
- The v2 `LogLinePayload` contract (`04-API-CONTRACTS.md` §11) does **not** include a monotonic server-side line `id`. The dedup key falls back to millisecond-resolution `timestamp` + fields. If the backend adds a stable line `id`, switch the dedup key to it (§6.5). This is a known soft spot, not a fabricated guarantee.
- `framer-motion`'s exact current major (11 vs 12) is `verify-before-use` — `02-TECH-STACK.md` does not pin the frontend animation lib version; resolve via `npm view framer-motion version` before pinning.
- The Tauri desktop renderer reuse (`14-` Desktop LLD) requires that no renderer code use Node/browser-only APIs unavailable in WebKitGTK on Linux; recharts/framer-motion parity in the WebView is a `verify-before-use` smoke test (`02-TECH-STACK.md` §16.1 open risk 1). This LLD keeps all backend URLs in env vars precisely to keep that reuse clean.

---

## 13. Build task checklist for the local model (ordered, dependency-first)

Each task is ~one file or one feature with a clear done-condition. Execute in order.

1. **Scaffold + pin deps.** Create `frontend/` with `package.json` per §3 (remove `react-icons`; add TanStack Query, Zod, Vitest, MSW, Playwright). Add `.nvmrc`=`24`. **Done:** `npm ci` resolves; `react-icons` absent.
2. **`lib/env.ts` + `.env.{development,ec2demo,production}`.** Implement the fail-fast reader (§8). **Done:** importing `env` in prod mode with missing `VITE_FEDLEARN_API_URL` or `VITE_SERVER_ROOT_URL` throws.
3. **`vite.config.ts` + `vitest.config.ts` + `tsconfig.json` + `eslint.config.ts`.** `strictPort:5173`, ec2demo proxy, `manualChunks`, jsdom test env, `no-explicit-any:error`, jsx-a11y. **Done:** `tsc --noEmit` + `eslint .` clean on an empty `src`.
4. **`@fedlearn/tokens` + `styles/theme.css` + `tailwind.config.ts` + `components.json`.** Seed OKLCH tokens from the v1 `theme.css`; wire the `@theme` block; init shadcn. **Done:** Tailwind builds; `--chart-1..5` resolve.
5. **`api/schemas.ts` + `api/types.ts`.** All Zod schemas mirroring `04-API-CONTRACTS.md`; export `z.infer<>` types incl. the V5 role types (§5.4–§5.5). **Done:** `schemas.meResponse.test.ts` passes (valid parses; legacy `role:'ADMIN'` throws).
6. **`api/parse.ts` + `api/axiosClient.ts` + interceptor.** `parseOrThrow`, the single instance, `installAuthInterceptor` per §6.2. **Done:** `axiosClient.interceptor.test.ts` passes all four branches.
7. **`api/endpoints.ts` + `api/queryKeys.ts`.** Typed fn-per-endpoint calling `apiClient` + `parseOrThrow`; the key factory. **Done:** every endpoint returns a validated type; no `any`.
8. **`query/queryClient.ts` + `query/useAuth.ts` + `auth/*`.** QueryClient defaults; `useMe/useLogin/useLogout`; `AuthContext`, `useIdentity`, `RequireAuth`, `RequirePlatformAdmin`. **Done:** `useIdentity.roles.test.tsx` + `RequirePlatformAdmin.test.tsx` pass (regression-locks the dead-admin bug).
9. **`lib/wsUrl.ts` + `realtime/StompProvider.tsx` + `realtime/useStompTopic.ts`.** Shared ref-counted connection; `wss://` derivation; validated subscribe. **Done:** `wsUrl.test.ts` + `stompProvider.refcount.test.tsx` pass.
10. **`realtime/logStore.ts` + `realtime/liveResultsStore.ts`.** Salvage logStore (monotonic id, dedup, trim, LRU); the live results store. **Done:** `logStore.test.ts` passes.
11. **`query/useProjects.ts` + `query/useRuns.ts` + `query/useResults.ts` + the rest.** All server-state hooks over endpoints. **Done:** hooks typecheck; `useRunStatus` stops polling on terminal status.
12. **`components/ui/*` (shadcn primitives) + `context/ThemeProvider/OrgContext/ToastProvider`.** Generate Dialog, Button, Input, Select, Badge, Card, Table, Toast, Skeleton, EmptyState, ErrorState, StatusBadge. **Done:** a modal opened via `Dialog` traps focus and closes on Escape (a11y).
13. **`main.tsx` + `App.tsx` routing.** Provider stack (§6.1); `React.lazy` route table + `<Suspense>` (§6.7). **Done:** login bundle code-split; `vite build` initial chunk < 150 KB gzipped.
14. **Auth pages.** `LoginPage`, `RegisterPage`, `VerifyEmailPage`, `Forgot/ResetPasswordPage`. **Done:** login flow drives `useLogin` → cookie → navigate `/`.
15. **`features/orgs/*` + `OrgSwitcher`.** Org create/switch, member list. **Done:** switching org re-scopes the project grid.
16. **`pages/DashboardPage.tsx` + `features/projects/*`.** Container + `ProjectGrid` + `ProjectCard` + Create/Edit modals (replaces the v1 548-line god component). **Done:** create-project flow round-trips; no `any` payloads.
17. **`features/runs/*` + `StartRunModal`.** Per-strategy hyperparameter form (FedAvg vs DeComFL fields per `04-API-CONTRACTS.md` §4.2); maps `409`/`422` codes. **Done:** start-run 202 → run appears in `RunList`; `LOCAL_PROCESS` hidden outside dev.
18. **`features/logs/LogViewer.tsx`.** History-then-live merge via logStore (§6.5); validated `/topic/logs` subscription. **Done:** `logViewer.race.test.tsx` passes.
19. **`features/observability/*` + `pages/RunObservabilityPage.tsx`.** `ConvergenceChart` (per-run scoped, `--chart-*` strokes), `CommunicationCostPanel` (DeComFL wedge), `PerClientPanel`. **Done:** `communicationCost.test.tsx` + `convergenceChart.scope.test.tsx` pass; charts render from `/topic/results`.
20. **`features/observability/FederationOrrery.tsx`.** Real data, declarative animation, `prefers-reduced-motion`, text+icon+color status. **Done:** no rAF setState; empty state when no clients.
21. **`pages/admin/*`.** AdminUsers/AdminProjects/AuditLog behind `RequirePlatformAdmin`; `useAdmin` hooks. **Done:** admin nav + pages render for `PLATFORM_ADMIN`, `/forbidden` for `USER`.
22. **`lib/errorCode.ts`.** Map the full §12.1 code registry. **Done:** `errorCode.test.ts` passes.
23. **CSP/HSTS meta fallback in `index.html`** (backend is authoritative; this is defense-in-depth) + remove the dead `global` shim. **Done:** CSP `<meta>` present; `(window as any).global` gone.
24. **Test layer wiring.** `src/test/setup.ts`, MSW handlers/server, fixtures; `playwright.config.ts`; the golden-path e2e. **Done:** `vitest run` green; `playwright test` green against MSW.
25. **CI hooks.** Ensure `tsc --noEmit`, `eslint .`, `vitest run --coverage`, `vite build` (with the bundle-size budget) run in the PR `ci.yml` paths-filter job. **Done:** all four green; coverage ≥ measured baseline.

---

*Every contract in this document traces to `04-API-CONTRACTS.md` (the wire shapes), `02-TECH-STACK.md` (the pinned versions), or a cited audit finding in `A2-frontend.md` / `C5-design-ux.md` / `B3-observability.md`. Where a fact about v1 code is asserted, it carries a `file:line` citation from those reports. Uncertainty is flagged in §12, not papered over.*
