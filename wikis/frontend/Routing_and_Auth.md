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

### Roles & Backend RBAC (UI deferred)

Authentication is **cookie-only** — the frontend never reads or sends a token; the single role carried on the session is `platform_role` (`USER` / `PLATFORM_ADMIN`).

The backend additionally exposes identity/RBAC endpoints (membership, admin, access-request, and discover surfaces). **These are not yet surfaced in the web UI** — the web client currently ships the Instrument design system unchanged, with no membership/admin/access-request/discover screens. They exist server-side only; building the corresponding UI is deferred.

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
            <Route path="/v2" element={<DashboardV2 />} />
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
