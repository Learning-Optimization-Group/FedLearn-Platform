# Authentication & Routing

## The Auth Contract: HttpOnly Cookies

The frontend employs a secure, modern authentication strategy. We **intentionally avoid** storing
JWT tokens in `localStorage` or `sessionStorage` to mitigate Cross-Site Scripting (XSS) attacks.

Instead, the authentication relies entirely on **HttpOnly Cookies** set by the backend. The frontend
does not see the token directly. Concretely, on this branch:

- The backend sets `jwtToken` as `HttpOnly` + `SameSite` + `Secure`. Both of the latter are
  profile-driven: `app.auth.cookie.same-site` is `Lax` in the base/`dev`/`ec2demo` profiles and
  `Strict` under `production`; `app.auth.cookie.secure` defaults to `false` in the base profile
  (so `dev` and `ec2demo`, which run over plain HTTP, keep it off) and `true` under `production`.
- Every Axios call carries `withCredentials: true`. There is **no** `Authorization: Bearer` header
  anywhere in `frontend/src/`, and nothing is written to web storage.
- The STOMP WebSocket at `/ws-logs` reuses the same cookie via the backend's
  `JwtHandshakeInterceptor` — the frontend never attaches anything to the handshake.
- The web token carries a distinct audience claim (SE-20), so a token minted for one surface cannot
  be replayed against another. That check lives entirely on the backend; the SPA is unaware of it.

### Interceptor Flow

We configure Axios to always send credentials and to intercept `401 Unauthorized` responses
globally.

```typescript
// src/api/axiosConfig.ts
const api = axios.create({ baseURL, withCredentials: true });

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

`/auth/me` is a **silent 401 probe** — a 401 there means "no session yet", not "session expired", so
letting it fire `authError` would produce a redirect loop on every anonymous page load. `/auth/login`
is excluded for the same reason in reverse: a wrong password is not a session expiry. A **403 is
deliberately not a logout** — see
[API & Services § the response interceptor](./API_and_Services.md#the-response-interceptor).

### The AuthContext

`src/context/AuthContext.tsx` encapsulates the current user and exposes
`{ currentUser, isLoading, isAdmin, isOwner, setSession, logout }`.

1. **Bootstrap.** On mount it calls `fetchCurrentUser()` (`GET /auth/me`). Success populates
   `{ username, email, role }`; failure is swallowed, leaving `currentUser = null` (anonymous). The
   effect flips `isLoading` false either way, and is cancellation-guarded so a fast unmount cannot
   set state on a dead component.
2. **Role normalisation.** `normalizeRole()` coerces anything that is not `PLATFORM_ADMIN` or
   `PROJECT_OWNER` to `USER`, so an unexpected backend value can never widen access in the UI.
3. **Identity refresh without a reload.** While authenticated, the context re-polls `/auth/me` on
   `window` focus and on `document.visibilitychange → visible`, debounced to at most one call per
   5 s and guarded against overlapping requests. Backend authorities are reloaded from the database
   on every request, so a server-side role change surfaces in the UI on the next tab focus — no
   re-login, no page reload. A failed refresh leaves the existing session intact rather than tearing
   it down.
4. **`setSession`** replaces the in-memory user; `LoginPage` calls it after a successful
   `POST /auth/login`.
5. **`logout`** is best-effort: it awaits `logoutUser()` inside a try/catch, logs a warning on
   failure, and clears local state unconditionally — the UI must never keep claiming a session that
   the backend has already dropped.

`App.tsx` is what connects the interceptor to the context:

```tsx
useEffect(() => {
    const handleAuthError = () => { void logout(); };
    window.addEventListener('authError', handleAuthError);
    return () => window.removeEventListener('authError', handleAuthError);
}, [logout]);
```

While `isLoading` is true, `App` renders a full-screen loader instead of the router, so no route
ever renders against an unresolved session.

### Roles & Backend RBAC

Authentication is **cookie-only** — the frontend never reads or sends a token; the role carried on
the session is the platform role (`USER` / `PROJECT_OWNER` / `PLATFORM_ADMIN`), typed as
`Role` in `AuthContext`.

The backend has three role layers; the web client only sees the first directly:

- **Platform role** (`users.platform_role`) — the one the SPA branches on. `USER` may join and train;
  `PROJECT_OWNER` may create and own projects (granted by an admin, via the owner-promotion
  workflow); `PLATFORM_ADMIN` manages everything. Authorities are reloaded from the DB on every
  backend request, which is exactly what makes the focus re-poll above sufficient.
- **Org role** (`organization_memberships`) — the multi-tenant layer. `OrgScopeFilter` only
  *populates* the request-scoped `OrgScope`; it makes no allow/deny decision, so it is not itself a
  security boundary. The SPA has no org UI today.
- **Project membership** (`OWNER` / `MEMBER` / `CLIENT`) plus the `project_access_requests` approval
  workflow — surfaced through `ProjectOwnerPanel` and `ClientDashboard`, not through routing.

Both approval workflows are wired end to end in the SPA: owner promotion (`ClientDashboard` submits,
`AdminDashboard` decides) and project deletion (`ProjectOwnerPanel` requests, `AdminDashboard`
approves — direct `DELETE /api/projects/{id}` is admin-only). Project visibility is three tiers,
`PUBLIC` / `RESTRICTED` / `PRIVATE`, with the plain-language copy for each centralised in
`VISIBILITY_HELP` in `apiServices.ts`.

See the backend's [Identity, Multi-Tenancy & Audit](../backend/06_identity_multitenancy_and_audit.md)
page for the server-side model.

## Routing Configuration

`react-router-dom` v7. `main.tsx` mounts `BrowserRouter`; `App.tsx` owns the whole route table.
There is **one** UI — the historical "legacy vs V2" split is gone, and the old `/v2/*` paths survive
only as redirects.

### The route map

| Path | Element | Guard |
|---|---|---|
| `/` | `LandingPage` | public |
| `/login` | `LoginPage` (redirects to `/dashboard` when already signed in) | public |
| `/register` | `RegisterPage` (same redirect) | public |
| `/dashboard` | `RoleDashboard` | `ProtectedRoute` |
| `/models` | `ModelsView` | `ProtectedRoute` |
| `/registry` | `RegistryView` | `ProtectedRoute` |
| `/marketplace` | `MarketplaceView` | `ProtectedRoute` |
| `/playground` | `PlaygroundView` | `ProtectedRoute` |
| `/settings` | `SettingsView` | `ProtectedRoute` |
| `/datasets` | `DatasetsView` | + `RoleRoute allow={['PROJECT_OWNER','PLATFORM_ADMIN']}` |
| `/nodes` | `NodeNetwork` (platform **user-account** management, not devices) | + `RoleRoute allow={['PLATFORM_ADMIN']}` |
| `/admin/projects` | `AdminProjectsView` | + `RoleRoute allow={['PLATFORM_ADMIN']}` |
| `/admin/projects/:projectId` | `AdminProjectDetail` | + `RoleRoute allow={['PLATFORM_ADMIN']}` |
| `/admin/audit` | `AuditLogView` (lazy, inside `Suspense`) | + `RoleRoute allow={['PLATFORM_ADMIN']}` |
| `/admin/benchmarks` | `BenchmarkDashboard` | + `RoleRoute allow={['PLATFORM_ADMIN']}` |
| `/v2`, `/v2/nodes`, `/v2/models`, `/v2/playground`, `/v2/datasets`, `/v2/settings` | `<Navigate replace>` to the canonical path | public (they are pure redirects) |
| `*` | `NotFoundPage` | public |

Every authenticated route is nested under `ProtectedRoute` → `LayoutV2` (sidebar + `<Outlet/>`), so
the shell is mounted once and each page owns its own scroll.

```tsx
<Route element={<ProtectedRoute />}>
    <Route element={<LayoutV2 />}>
        <Route path="/dashboard" element={<RoleDashboard />} />
        {/* … shared surfaces … */}

        <Route element={<RoleRoute allow={['PROJECT_OWNER', 'PLATFORM_ADMIN']} />}>
            <Route path="/datasets" element={<DatasetsView />} />
        </Route>

        <Route element={<RoleRoute allow={['PLATFORM_ADMIN']} />}>
            <Route path="/nodes" element={<NodeNetwork />} />
            {/* … admin surfaces … */}
        </Route>
    </Route>
</Route>
```

Two placement decisions worth not "fixing":

- **`/nodes` is admin-only, not owner-plus-admin.** Despite the name it is platform user-account
  management backed by an admin-only endpoint (`GET /admin/users/search`), so an owner reaching it
  would only ever see a 403. `App.routes.test.tsx` pins this: a plain `USER` *and* a
  `PROJECT_OWNER` are both bounced to `/dashboard`, and only `PLATFORM_ADMIN` gets through.
- **`AuditLogView` is the only lazily-loaded route.** It is admin-only and rarely hit, so it is code
  split to keep the main chunk lean; the `Suspense` fallback reuses the app loader.

### `/dashboard` is role-aware

`/dashboard` is the single canonical landing route for every role. `RoleDashboard` picks the
surface:

| Role | Component | What it offers |
|---|---|---|
| `PLATFORM_ADMIN` | `AdminDashboard` | bounded approval queues, platform stats that deep-link into the paginated directories |
| `PROJECT_OWNER` | `OwnerDashboard` | own projects + create/start/stop, plus `ProjectOwnerPanel` (visibility, join requests, memberships, request-deletion) |
| `USER` | `ClientDashboard` | request owner promotion, discover projects to join — with a note that actual training happens in the desktop app |

The sidebar mirrors the same gating (`navGroupsForRole` in `Sidebar.tsx`): the Workspace group grows
a **Data** item for owners and admins, an **Admin** group (Users / Projects / Audit log / Benchmarks)
appears only for `PLATFORM_ADMIN`, and every role gets Account → Settings.

### The `ProtectedRoute` Guard

```tsx
// src/components/ProtectedRoute.tsx
const ProtectedRoute: React.FC = () => {
    const { currentUser, isLoading } = useAuth();
    const location = useLocation();

    if (isLoading) return <DiskLoader message="Checking authentication..." />;
    if (!currentUser) {
        // Remember where the user was headed so LoginPage can send them back.
        return <Navigate to="/login" replace state={{ from: location }} />;
    }
    return <Outlet />;
};
```

Two things it does beyond the obvious redirect: it renders a loader while the session is still
resolving (so a refresh on a deep link does not flash the login page), and it stashes the attempted
location in the navigation state. `LoginPage` reads `location.state.from.pathname` after
`setSession` and navigates there instead of dumping the user on `/dashboard` (FE-10).

### The `RoleRoute` Guard

```tsx
// src/components/RoleRoute.tsx
const RoleRoute: React.FC<{ allow: Role[] }> = ({ allow }) => {
    const { currentUser } = useAuth();
    if (currentUser && allow.includes(currentUser.role)) return <Outlet />;
    return <Navigate to="/dashboard" replace />;
};
```

`RoleRoute` sits **inside** `ProtectedRoute`, so the user is already known to be authenticated; it
only gates a subtree by platform role and bounces a disallowed role to the role-aware landing route.

> **It is a UX guard, not a security boundary.** The bundle is public and the route table is
> readable; the backend still enforces RBAC and returns 403 for anything the role cannot do. The
> value of `RoleRoute` is that a user never lands on a page that can only render an error — not that
> it prevents anything.
