import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { api, setAuthLostHandler } from '../lib/restClient';
import { clearToken, getToken, setToken } from '../lib/authStore';

export interface Identity { username: string }
type Status = 'unknown' | 'unauthenticated' | 'authenticated';

/** POST /api/auth/login → store the body token → return identity. Throws if no token. */
export async function performLogin(username: string, password: string): Promise<Identity> {
  const res = await api.post('/api/auth/login', { username, password });
  const token = res?.data?.accessToken;
  if (!token) throw new Error('Login response did not include a token');
  await setToken(token);
  return { username: res.data.username ?? username };
}

/** GET /api/auth/me as the bootstrap session probe. Null if no token or the probe fails. */
export async function probeSession(): Promise<Identity | null> {
  const token = await getToken();
  if (!token) return null;
  try {
    const res = await api.get('/api/auth/me');
    return { username: res?.data?.username ?? '' };
  } catch {
    return null;
  }
}

interface AuthValue {
  status: Status;
  username: string | null;
  login: (u: string, p: string) => Promise<void>;
  logout: () => Promise<void>;
}
const AuthCtx = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<Status>('unknown');
  const [username, setUsername] = useState<string | null>(null);

  const logout = useCallback(async () => {
    await clearToken();
    setUsername(null);
    setStatus('unauthenticated');
  }, []);

  // A 401 anywhere (via restClient) routes back to Login.
  useEffect(() => { setAuthLostHandler(() => { void logout(); }); }, [logout]);

  // Bootstrap probe.
  useEffect(() => {
    probeSession().then((id) => {
      if (id) { setUsername(id.username); setStatus('authenticated'); }
      else setStatus('unauthenticated');
    });
  }, []);

  const login = useCallback(async (u: string, p: string) => {
    const id = await performLogin(u, p);
    setUsername(id.username);
    setStatus('authenticated');
  }, []);

  return <AuthCtx.Provider value={{ status, username, login, logout }}>{children}</AuthCtx.Provider>;
}

export function useAuth(): AuthValue {
  const v = useContext(AuthCtx);
  if (!v) throw new Error('useAuth must be used within AuthProvider');
  return v;
}
