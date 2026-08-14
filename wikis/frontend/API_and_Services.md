# API and Services

The frontend communicates with the FedLearn backend via REST APIs for operations and state, and
WebSockets (STOMP) for real-time telemetry and logs. Both transports authenticate with the **same
HttpOnly cookie** — there is no Bearer header and nothing in `localStorage` anywhere in this app.

## Axios Configuration

`src/api/axiosConfig.ts` exports **one** shared Axios instance. Every service module imports that
singleton; nothing else calls `axios.create()`.

### Base URL resolution

```typescript
const envBaseUrl = import.meta.env.VITE_FEDLEARN_API_URL;

// Local-dev fallback only; production builds fail fast (see below).
const baseURL = import.meta.env.PROD
    ? envBaseUrl
    : envBaseUrl || (typeof window !== 'undefined'
        ? `http://${window.location.hostname}:8081/api`
        : 'http://localhost:8081/api');

const api = axios.create({ baseURL, withCredentials: true });
```

`VITE_FEDLEARN_API_URL` comes from the committed `.env.[mode]` file for the active Vite mode — see
[Architecture § Environments](./Architecture.md#environments-vite-modes-and-the-dev-proxy) for the
mode ↔ Spring-profile table. The `window.location.hostname` fallback exists so a dev server reached
from another device on the LAN still finds the backend; it never fires in a production build.

### The production host guard (FE-7)

Three module-scope throws run only under `import.meta.env.PROD`, so a misconfigured production
bundle dies loudly at load instead of silently shipping a dead origin:

| Condition | Message |
|---|---|
| `VITE_FEDLEARN_API_URL` unset | must be set for production builds |
| value contains `REPLACE_WITH_YOUR_API_HOST` | still the placeholder from `.env.production` |
| value does not start with `https://` | mixed content — the browser will block it |

The npm `prebuild` hook (`scripts/check-prod-env.mjs`) catches the placeholder case earlier still,
before Vite ever runs.

### `withCredentials: true`

This is the whole auth contract on the REST side. The backend issues an HttpOnly `jwtToken` cookie
on `/auth/login`; `withCredentials` makes the browser attach it to every request, including
cross-origin ones (which is why the backend's CORS config sets `allowCredentials(true)` and uses
`setAllowedOriginPatterns` rather than a literal `*`). Nothing in the frontend reads, stores or
forwards the token.

### The response interceptor

```typescript
// Endpoints whose 401 means "no session" rather than "session expired".
const SILENT_401_ENDPOINTS = ['/auth/me'];

api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        const status = error.response?.status;
        if (status === 401
            && !error.config?.url?.includes('/auth/login')
            && !isSilent401(error.config)) {
            window.dispatchEvent(new Event('authError'));
        }
        return Promise.reject(error);
    }
);
```

Three behaviours here are deliberate and pinned by `src/api/axiosConfig.test.ts`:

- **401 on a data route → global logout.** The interceptor dispatches a bare `authError` window
  event; `App.tsx` listens for it and calls `logout()`. The interceptor itself knows nothing about
  React, routing or the auth context.
- **401 is swallowed for exactly two URLs.** `/auth/login` (a wrong password is not a session
  expiry, and the login form renders its own error) and `/auth/me` (the bootstrap probe — surfacing
  its 401 as `authError` would produce an immediate redirect loop on every anonymous page load).
  `SILENT_401_ENDPOINTS` is a substring match on the request URL.
- **403 is never a logout.** 403 means "authenticated, but not allowed *this* resource" — a
  non-admin hitting `/users`, a non-owner touching someone else's project. The interceptor leaves it
  alone so the calling component can render "permission denied" inline. Several service functions
  document this in their own doc comments.

The interceptor always re-rejects with the original error, so callers still see the status and body.

## Service Abstraction Layer

All API calls are named exports in `src/services/apiServices.ts`, which isolates Axios from the
components and is the single source of truth for request/response typings. `src/services/artifactService.ts`
is a second typed client over the *same* singleton for the registry/marketplace surface (FE-11/FE-12).

Three error helpers sit at the top of `apiServices.ts` and are used across the views, because the
backend's `GlobalExceptionHandler` returns `{message}` (sometimes `{error}`):

```typescript
errorStatus(err)             // the HTTP status, or undefined for a non-HTTP failure
errorMessage(err, fallback)  // the backend's human-readable message, or the fallback
isEmptyBody(data)            // a 204 / empty body carries no resource
```

`isEmptyBody` matters more than it looks: `GET /owner-requests/mine` and
`GET /projects/{id}/deletion-request` return **204 with an empty body** to mean "no request yet",
which is a normal result rather than an error.

### Endpoint groups

| Group | Functions → endpoints |
|---|---|
| Auth | `loginUser` POST `/auth/login` · `registerUser` POST `/auth/register` · `fetchCurrentUser` GET `/auth/me` · `logoutUser` POST `/auth/logout` |
| Recipe catalog | `fetchModelRecipes` GET `/model-recipes` |
| Projects | `fetchProjects` / `fetchOwnedProjects` GET `/projects` · `createProject` POST `/projects` · `fetchProject` GET `/projects/{id}` · `updateProject` PUT `/projects/{id}` · `updateProjectVisibility` PATCH `/projects/{id}` · `deleteProject` DELETE `/projects/{id}` · `startProjectServer` POST `/projects/{id}/start` · `stopProjectServer` POST `/projects/{id}/stop` · `fetchProjectResults` GET `/projects/{id}/results` · `fetchProjectLogs` GET `/projects/{id}/logs` |
| Membership & access | `fetchAccessRequests` / `requestProjectAccess` on `/projects/{id}/access-requests` · `decideAccessRequest` PUT `…/{requestId}` · `fetchMemberships` / `addMembership` / `removeMembership` on `/projects/{id}/memberships` · `fetchDiscoverableProjects` GET `/projects/discover` |
| Deletion workflow | `submitDeletionRequest` POST + `fetchProjectDeletionRequest` GET `/projects/{id}/deletion-request` |
| Owner promotion | `submitOwnerRequest` POST `/owner-requests` · `fetchMyOwnerRequest` GET `/owner-requests/mine` |
| Inference | `fetchInferableModels` GET `/inference/models` · `runInference` POST `/inference/{id}` · `runGeneration` POST `/inference/{id}/generate` · `stopGeneration` POST `/inference/{id}/generate/stop` |
| Profile | `fetchMyProfile` GET + `updateMyProfile` PATCH `/users/me/profile` |
| Admin | `fetchAdminOverview` `/admin/overview` · `fetchAdminUsers` / `searchAdminUsers` · `updateUserRole` PUT `/admin/users/{id}/role` · `updateUserStatus` PUT `/admin/users/{id}/status` · `fetchAdminProjects` / `searchAdminProjects` · `fetchAuditEvents` `/admin/audit-events` · `fetchOwnerRequests` / `decideOwnerRequest` · `fetchDeletionRequests` / `decideDeletionRequest` |
| Benchmarks (admin) | `fetchBenchmarkOverview` `/admin/benchmarks/overview` · `fetchProjectBenchmark` `/admin/benchmarks/projects/{id}` |
| Artifacts (`artifactService.ts`) | `listArtifacts` GET `/artifacts?projectId=` · `getArtifact` GET `/artifacts/{id}` · `getLineage` GET `/artifacts/{id}/lineage` · `downloadBlob` GET `/artifacts/{id}/blob` (`responseType: 'blob'`) · `listMarketplace` GET `/marketplace/adapters` · `publishAdapter` POST + `unpublishAdapter` DELETE `/marketplace/adapters/{id}/publish` |

The search endpoints return a `Paged<T>` envelope (`{ items, page, size, total }`); the admin
directories page against those rather than the unbounded legacy list endpoints.

### The model-recipe catalog contract

`GET /api/model-recipes` is served by the backend from `fl-runtime/recipes.py --describe`
(snake_case on the Python side, camelCase on the wire — Jackson bridges it with `@JsonAlias`). The
frontend's `ModelRecipe` type mirrors that shape:

```typescript
export interface ModelRecipe {
    key: string;                 // PNEUMONIA_CNN, CNN, CIFAR_RESNET18, MLP, TRANSFORMER, …
    displayName: string;
    inputKind: string;           // image | text | tabular | …
    classes: string[];
    baseModels: string[];
    optimizers: string[];
    supportedArms?: string[];    // absent for recipes that declare none
    armTradeoff?: ArmTradeoff;   // present ONLY when the recipe offers more than one arm
}
```

The catalog currently advertises **seven** keys, and only three of them offer a choice of arm — so
most projects never see the arm control at all:

| Recipe key | `supportedArms` |
|---|---|
| `PNEUMONIA_CNN` | `FULL`, `FROZEN_HEAD` |
| `CNN` | `FULL`, `FROZEN_HEAD` |
| `CIFAR_RESNET18` | `FULL`, `FROZEN_HEAD`, `OVA_LP` |
| `MLP`, `TRANSFORMER`, `LLM_LORA`, `TINYNET_GOLDEN` | `FULL` only |

(`fl-runtime/recipes.py` also defines `BLOOD_CNN` and `FROZEN_DEMO`, which are deliberately kept
out of `RECIPE_METADATA` — dispatchable but not selectable, so they never reach the picker.)

`ArmTradeoff` carries `headline`, optional `commRatio` / `ondeviceRatio`, a `measuredOn`
string map, per-arm `ArmFacts`, and a `caveats: string[]`. Two subtleties in `ArmFacts` are
contractual, not incidental:

- `accuracyAuc` and `accuracyPct` are **mutually exclusive** — binary tasks are reported as AUC,
  multi-class as top-1.
- `ondeviceFeasible` is `boolean | null`, and `null` means **not measured**, which is a different
  statement from a measured `false`. The UI must render `null` as absence, never as a claim in
  either direction.

How the picker renders all of this is documented in
[UI & Components § the training arm](./UI_and_Components.md#the-project-creation-picker-and-the-training-arm).

## Real-Time Logs & WebSockets

### Origin derivation (`src/lib/serverConfig.ts`)

```typescript
export const SERVER_ROOT_URL: string =
    import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;

export const WS_BROKER_URL: string = `${SERVER_ROOT_URL.replace(/^http/, 'ws')}/ws-logs`;
```

One module owns the HTTP origin and the broker URL derived from it (`http→ws`, `https→wss`, then
`/ws-logs`). Before FE-9 this derivation was copy-pasted across several views, and any one of them
drifting would silently desync a WebSocket surface from the REST origin it is supposed to share.
`VITE_SERVER_ROOT_URL` must stay on the same host as `VITE_FEDLEARN_API_URL` — in `ec2demo` both
point at `localhost:5173` so the Vite proxy (which has `ws: true` on `/ws-logs`) carries the socket
too, keeping the cookie first-party.

The backend's `JwtHandshakeInterceptor` authenticates the STOMP handshake from that **same HttpOnly
cookie**. Nothing token-related is attached here or anywhere upstream.

### Destinations

| Destination | Subscriber | Payload |
|---|---|---|
| `/topic/logs/{projectId}` | `LogViewer` | a log line, or a `RoundResultDto` carrying `loss` / `accuracy` / `serverRound` |
| `/topic/status/{projectId}` | `useProjectStatus` | `{ projectId, newStatus, serverPort? }` |
| `/topic/results/{projectId}` | `useProjectStatus` | one `ProjectResult` per completed round |
| `/topic/inference/{projectId}` | `PlaygroundView` | streamed generation tokens |

`OwnerDashboard` subscribes with `projectId: '*'`, intending one socket for every owned project, and
`useProjectStatus` therefore reads the owning id off `message.headers.destination` rather than off
the subscribed pattern.

> **That wildcard does not reach the broker — the backend rejects it.** There are no wildcard
> destinations server-side: `WebSocketService` only ever publishes concrete
> `/topic/{logs,status,results}/{projectId}`, and `StompSubscriptionInterceptor` (BA-5, hardened in
> `d6fb77d`) rejects *any* SUBSCRIBE destination containing `*` or `?` with
> `AccessDeniedException("Wildcard subscriptions are not permitted")` — the SimpleBroker matches
> destinations as Ant patterns, so an ungated wildcard would have received every tenant's
> broadcasts. `OwnerDashboard`'s live status/results push is therefore refused today (its list
> refreshes on mount and after its own mutations, not from the socket). This is a live
> frontend/backend contract mismatch, not a design; the source comment in `OwnerDashboard.tsx:107-109`
> carries the same stale belief. The destination-header id resolution above stays correct and is
> what a per-project fan-out would need.

### `useStompClient` — one lifecycle, honest state

Every WS surface used to hand-roll its own `new StompClient({ brokerURL, reconnectDelay: 5000 })`,
its own subscribe/unsubscribe bookkeeping, and its own approximate notion of "connected". The hook
now owns all of it and reports state derived **only** from the real `onConnect`,
`onWebSocketClose`, `onStompError` and `onWebSocketError` callbacks — so a dropped socket can never
present as a silent stall or a fake "connected".

It returns `{ isConnected, isReconnecting, lastError }`, and
`src/lib/connectionStatus.ts` maps that snapshot onto four honest phases plus a caller-chosen label:

| Phase | Meaning | `StatusPill` kind |
|---|---|---|
| `connecting` | never connected yet, no error observed | `idle` |
| `live` | the STOMP CONNECTED frame has arrived | `running` |
| `reconnecting` | was live, socket dropped, auto-retry under way | `pending` |
| `error` | never connected, and a STOMP/WebSocket error was observed | `error` |

`error` and `reconnecting` both mean "the client is retrying in the background"
(`reconnectDelay` defaults to 5000 ms); the distinction is purely whether the surface has *ever*
been live, so the label can honestly say "still trying" versus "hasn't connected yet".

### LogViewer's message handling

`LogViewer` supplies only the parsing; the socket belongs to `useStompClient`. Two details are
worth knowing:

- Frames are appended to `logStore` **even while paused**. "Pause" freezes auto-scroll only, so
  nothing received in the paused window is lost.
- A frame is treated as telemetry only when `Number.isFinite(parsed.loss) && Number.isFinite(parsed.accuracy)`.
  A merely "defined" check let a malformed or hostile frame with `null`/`"x"` into the telemetry
  cache, where `latest.loss.toFixed(...)` would throw and white-screen the whole SPA on render.
  Non-numeric telemetry is ignored; an unparseable body is stored as a plain INFO line.

### LogStore (telemetry cache)

`logStore.ts` is an intentionally framework-agnostic publish/subscribe cache so a WebSocket callback
can write to it from outside the React tree:

1. The subscription handler pushes entries via `logStore.append(projectId, entry)`.
2. `LogViewer` subscribes and re-renders from the emitted snapshot.
3. Historical lines from `GET /projects/{id}/logs` are **prepended** by `mergeHistorical()`, deduped
   on `timestamp|message`, and fetched at most once per project (`hasLoadedHistorical`).
4. Entries are trimmed to the last 2000 per project.

The store assigns each entry a monotonic `id` that consumers must use as the React key — see
[Architecture § Service-Level Stores](./Architecture.md#2-service-level-stores-domain-state-that-outlives-a-component)
for why an index key breaks under `mergeHistorical`.

The net effect: closing and reopening a project's log modal shows prior session output immediately,
with no refetch and no gap in the live stream.
