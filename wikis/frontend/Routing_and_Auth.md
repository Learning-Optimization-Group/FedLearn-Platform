# Authentication & Routing

## The Auth Contract: HttpOnly Cookies

The frontend employs a secure, modern authentication strategy. We **intentionally avoid** storing JWT tokens in `localStorage` or `sessionStorage` to mitigate Cross-Site Scripting (XSS) attacks. 

Instead, the authentication relies entirely on **HttpOnly Cookies** set by the backend. The frontend does not see the token directly.

### Interceptor Flow
We configure Axios to always send credentials (`withCredentials: true`), and intercept `401 Unauthorized` responses globally.

```typescript
// src/api/axiosConfig.ts
const api = axios.create({
    baseURL,
    withCredentials: true, // Crucial for cookie-based auth
});

api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        const status = error.response?.status;
        
        // Broadcast an auth error globally to trigger a logout
        if (status === 401 && !error.config?.url?.includes('/auth/login') && !isSilent401(error.config)) {
            window.dispatchEvent(new Event('authError'));
        }
        return Promise.reject(error);
    }
);
```

### The AuthContext

The `AuthContext` encapsulates the current user state and listens for the global `authError` to coordinate logouts.

1. **Bootstrap**: On application load, it pings `/auth/me` to check if a valid session cookie exists.
2. **Session Updates**: The `setSession` method is invoked upon a successful login.
3. **Logout**: Best-effort backend call to invalidate the cookie, followed by clearing the local UI state.

```typescript
// Flow in App.tsx
useEffect(() => {
    const handleAuthError = () => {
        void logout();
    };
    window.addEventListener('authError', handleAuthError);
    return () => window.removeEventListener('authError', handleAuthError);
}, [logout]);
```

### Roles & Backend RBAC

Authentication is **cookie-only** — the frontend never reads or sends a token; the role carried on the session is the layered `platform_role` (`USER` / `PROJECT_OWNER` / `PLATFORM_ADMIN`).

> ✅ **Branch reality.** The identity/RBAC endpoints (membership, admin, access-request, discover) and the three-layer `platform_role` model **are present on this branch** (backend `V4`–`V7` migrations; see the backend [Identity, Multi-Tenancy & Audit](../backend/06_identity_multitenancy_and_audit.md) page). The web client renders them through role-gated routes (`RoleRoute allow={['PLATFORM_ADMIN']}` / `['PROJECT_OWNER', …]`) and dashboards (`AdminDashboard`, `OwnerDashboard`, `ClientDashboard`) plus the owner-promotion / deletion-request approval flows — superseding the original coarse `USER` / `ADMIN` role. The web client ships the **Ember** design system.

## Routing Configuration

We use `react-router-dom` v7 to handle application routing. The router distinguishes between public and protected routes.

### Route Definitions
The `App.tsx` handles top-level routing, using `ProtectedRoute` to encapsulate sensitive areas.
Notice how we seamlessly support both Legacy and V2 UIs by nesting them under different Layouts:

```tsx
<Routes>
    {/* Public Routes */}
    <Route path="/" element={<LandingPage />} />
    <Route path="/login" element={currentUser ? <Navigate to="/dashboard" replace /> : <LoginPage />} />

    {/* Protected Routes */}
    <Route element={<ProtectedRoute />}>
        {/* Original UI */}
        <Route element={<Layout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            {/* ... */}
        </Route>

        {/* Redesigned UI (v2) */}
        <Route element={<LayoutV2 />}>
            <Route path="/dashboard" element={<RoleDashboard />} />   {/* role-aware: Admin/Owner/Client */}
            <Route path="/v2/nodes" element={<NodeNetwork />} />
            {/* ... */}
        </Route>
    </Route>
</Routes>
```

### The ProtectedRoute Guard
The `ProtectedRoute` acts as a barrier, checking `AuthContext` to determine if a user can access the child routes. If `currentUser` is null, it redirects to `/login`.

```tsx
// src/components/ProtectedRoute.tsx
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute() {
    const { currentUser } = useAuth();
    if (!currentUser) return <Navigate to="/login" replace />;
    return <Outlet />;
}
```
