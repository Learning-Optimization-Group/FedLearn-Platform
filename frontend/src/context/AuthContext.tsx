import React, { createContext, useState, useContext, useEffect, useCallback, ReactNode } from 'react';
import { fetchCurrentUser, logoutUser, AuthIdentity } from '../services/apiServices';
import { createLogger } from '../lib/logger';

const log = createLogger('AuthContext');

/**
 * The frontend never sees the JWT — it lives in an HttpOnly cookie that
 * the browser attaches automatically. The User shape here mirrors what the
 * /auth/me endpoint returns; nothing more.
 */
interface User {
    username: string;
    email: string;
    role: 'USER' | 'ADMIN';
}

interface AuthContextType {
    currentUser: User | null;
    isLoading: boolean;
    /** Replace the in-memory user (called by LoginPage on successful login). */
    setSession: (user: User) => void;
    /** Best-effort backend logout + clear local state. */
    logout: () => Promise<void>;
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
                    role: identity.role,
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

    const setSession = useCallback((newUser: User) => {
        setUser(newUser);
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

    const value: AuthContextType = { currentUser: user, isLoading, setSession, logout };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
