import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { api, setAuthLostHandler } from '../lib/restClient';
import { clearToken, getToken, setToken } from '../lib/authStore';
import { clearSavedCredentials, saveCredentials } from '../lib/credentialStore';
import {
  beginAuthentication, beginLogout, captureAuthAdmission, getBackendScope, isCurrentAuthAdmission,
  isCurrentBackendScope, onBackendScopeChange, requireCurrentAuthAdmission, type AuthAdmission,
} from '../lib/backendScope';

export interface Identity { username: string }
type Status = 'unknown' | 'unauthenticated' | 'authenticated';

/** POST /api/auth/login → store the body token → return identity. Throws if no token. */
async function loginAtAdmission(username: string, password: string, admission: AuthAdmission, remember?: boolean): Promise<Identity> {
  requireCurrentAuthAdmission(admission);
  const res = await api.post('/api/auth/login', { username, password }, { baseURL: admission.scope.origin });
  requireCurrentAuthAdmission(admission);
  const token = res?.data?.accessToken;
  if (!token) throw new Error('Login response did not include a token');
  if (remember === true) await saveCredentials({ username, password }, admission.scope.origin, admission);
  else if (remember === false) await clearSavedCredentials(admission);
  await setToken(token, admission.scope.origin, admission);
  requireCurrentAuthAdmission(admission);
  return { username: res.data.username ?? username };
}

export async function performLogin(username: string, password: string, remember?: boolean): Promise<Identity> {
  return loginAtAdmission(username, password, beginAuthentication(), remember);
}

/** POST /api/auth/register, then immediately log in with the same credentials. */
export async function performRegister(
  username: string,
  email: string,
  password: string,
): Promise<Identity> {
  const admission = beginAuthentication();
  await api.post('/api/auth/register', { username, email, password }, { baseURL: admission.scope.origin });
  requireCurrentAuthAdmission(admission);
  return loginAtAdmission(username, password, admission, false);
}

/** GET /api/auth/me as the bootstrap session probe. Null if no token or the probe fails. */
export async function probeSession(): Promise<Identity | null> {
  if (!getBackendScope()) return null;
  const admission = captureAuthAdmission();
  const token = await getToken(admission.scope.origin);
  if (!token) return null;
  try {
    if (!isCurrentAuthAdmission(admission)) return null;
    const res = await api.get('/api/auth/me', { baseURL: admission.scope.origin });
    if (!isCurrentAuthAdmission(admission)) return null;
    return { username: res?.data?.username ?? '' };
  } catch {
    return null;
  }
}

interface AuthValue {
  status: Status;
  username: string | null;
  login: (u: string, p: string, remember?: boolean) => Promise<void>;
  register: (u: string, e: string, p: string) => Promise<void>;
  logout: () => Promise<void>;
}
const AuthCtx = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<Status>('unknown');
  const [username, setUsername] = useState<string | null>(null);
  const operation = useRef(0);

  const invalidate = useCallback(() => {
    operation.current++;
    setUsername(null);
    setStatus('unauthenticated');
  }, []);

  const logout = useCallback(async () => {
    const admission = beginLogout();
    invalidate();
    await Promise.allSettled([clearToken(admission), clearSavedCredentials(admission)]);
  }, [invalidate]);

  // A 401 anywhere (via restClient) routes back to Login.
  useEffect(() => {
    setAuthLostHandler(invalidate);
    return () => { setAuthLostHandler(null); };
  }, [invalidate]);
  useEffect(() => onBackendScopeChange(invalidate), [invalidate]);

  // Bootstrap probe.
  useEffect(() => {
    const attempt = operation.current;
    const scope = getBackendScope();
    let mounted = true;
    probeSession().then((id) => {
      if (!mounted || attempt !== operation.current || !scope || !isCurrentBackendScope(scope)) return;
      if (id) { setUsername(id.username); setStatus('authenticated'); }
      else setStatus('unauthenticated');
    });
    return () => { mounted = false; };
  }, []);

  const login = useCallback(async (u: string, p: string, remember?: boolean) => {
    const attempt = ++operation.current;
    const id = await performLogin(u, p, remember);
    if (attempt !== operation.current) throw new Error('The server session changed. Sign in again.');
    setUsername(id.username);
    setStatus('authenticated');
  }, []);

  const register = useCallback(async (u: string, e: string, p: string) => {
    const attempt = ++operation.current;
    const id = await performRegister(u, e, p);
    if (attempt !== operation.current) throw new Error('The server session changed. Sign in again.');
    setUsername(id.username);
    setStatus('authenticated');
  }, []);

  return (
    <AuthCtx.Provider value={{ status, username, login, register, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth(): AuthValue {
  const v = useContext(AuthCtx);
  if (!v) throw new Error('useAuth must be used within AuthProvider');
  return v;
}
