import React, { createContext, useState, useContext, useEffect, useCallback, ReactNode } from 'react';
import { fetchCurrentUser, logoutUser, AuthIdentity } from '../services/apiServices';
import { createLogger } from '../lib/logger';

const log = createLogger('AuthContext');

/**
 * The frontend never sees the JWT — it lives in an HttpOnly cookie that
 * the browser attaches automatically. The User shape here mirrors what the
 * /auth/me endpoint returns; nothing more.
 */
export type Role = 'USER' | 'PROJECT_OWNER' | 'PLATFORM_ADMIN';

interface User {
    username: string;
    email: string;
    role: Role;
}

interface AuthContextType {
    currentUser: User | null;
    isLoading: boolean;
    /** PLATFORM_ADMIN — can manage users/roles and approve platform-level requests. */
    isAdmin: boolean;
    /** PROJECT_OWNER or PLATFORM_ADMIN — can create/own projects. */
    isOwner: boolean;
    /** Replace the in-memory user (called by LoginPage on successful login). */
    setSession: (user: User) => void;
    /** Best-effort backend logout + clear local state. */
    logout: () => Promise<void>;
}

/** Coerce an arbitrary backend value into a known role; default USER. */
function normalizeRole(role: unknown): Role {
    return role === 'PLATFORM_ADMIN' || role === 'PROJECT_OWNER' ? role : 'USER';
}

const AuthContext = createContext<AuthContextType | null>(null);

interface AuthProviderProps {
    children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Bootstrap: ask the backend whether the cookie still grants us a session.
    // 401 here is silent (axios interceptor ignores it for this endpoint),
    // so an unauthenticated load just lands at currentUser=null.
    useEffect(() => {
        let cancelled = false;
        fetchCurrentUser()
            .then((res) => {
                if (cancelled) return;
                const identity: AuthIdentity = res.data;
                setUser({
                    username: identity.username,
                    email: identity.email,
                    role: normalizeRole(identity.role),
                });
            })
            .catch(() => {
                // Either no cookie or expired — anonymous mode, no action needed.
            })
            .finally(() => {
                if (!cancelled) setIsLoading(false);
            });
        return () => { cancelled = true; };
    }, []);

    // Keep the in-memory identity fresh while the tab is authenticated.
    // Authorities are reloaded from the DB on every backend request, so a
    // server-side role change only needs a re-poll of /auth/me to surface —
    // no full page reload. We re-poll when the tab regains focus or becomes
    // visible again, debounced so rapid focus/visibility toggles don't hammer
    // the endpoint. Listeners are registered only while authenticated and torn
    // down on logout/unmount. Uses the same cookie-backed call as bootstrap —
    // no token handling, no transport change.
    const isAuthenticated = user !== null;
    useEffect(() => {
        if (!isAuthenticated) return;

        let cancelled = false;
        let inFlight = false;
        let lastRefreshAt = 0;
        const MIN_INTERVAL_MS = 5000;

        const refresh = () => {
            if (cancelled || inFlight) return;
            const now = Date.now();
            if (now - lastRefreshAt < MIN_INTERVAL_MS) return;
            lastRefreshAt = now;
            inFlight = true;
            fetchCurrentUser()
                .then((res) => {
                    if (cancelled) return;
                    const identity: AuthIdentity = res.data;
                    setUser({
                        username: identity.username,
                        email: identity.email,
                        role: normalizeRole(identity.role),
                    });
                })
                .catch(() => {
                    // A failed refresh (transient network blip, silent 401)
                    // shouldn't tear down a working session — leave state as-is.
                })
                .finally(() => {
                    inFlight = false;
                });
        };

        const handleVisibility = () => {
            if (document.visibilityState === 'visible') refresh();
        };

        window.addEventListener('focus', refresh);
        document.addEventListener('visibilitychange', handleVisibility);
        return () => {
            cancelled = true;
            window.removeEventListener('focus', refresh);
            document.removeEventListener('visibilitychange', handleVisibility);
        };
    }, [isAuthenticated]);

    const setSession = useCallback((newUser: User) => {
        setUser({ ...newUser, role: normalizeRole(newUser.role) });
    }, []);

    const logout = useCallback(async () => {
        // Best-effort: clear the cookie server-side, but always clear local
        // state so the UI doesn't keep claiming we're logged in even if the
        // network call fails (e.g. backend already restarted, cookie purged).
        try {
            await logoutUser();
        } catch (err) {
            log.warn('Logout request failed; clearing local session anyway', err);
        }
        setUser(null);
    }, []);

    const isAdmin = user?.role === 'PLATFORM_ADMIN';
    const isOwner = user?.role === 'PLATFORM_ADMIN' || user?.role === 'PROJECT_OWNER';

    const value: AuthContextType = { currentUser: user, isLoading, isAdmin, isOwner, setSession, logout };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

// FE-6: the useAuth hook is intentionally co-located with its provider (the canonical
// context pattern). Splitting it into a separate module purely to satisfy fast-refresh
// would churn every import site for no runtime benefit, so scope the rule off here only.
// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
