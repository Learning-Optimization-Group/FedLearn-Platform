import { normalizeBackendOrigin } from './backendOrigin';

export interface BackendScope { readonly origin: string; readonly epoch: number }
export interface AuthAdmission { readonly scope: BackendScope; readonly version: number }

let active: BackendScope | null = null;
let epoch = 0;
let authVersion = 0;
let readable = false;
const listeners = new Set<(scope: BackendScope) => void>();

export function getBackendScope(): BackendScope | null { return active; }

export function requireBackendScope(): BackendScope {
  if (!active) throw new Error('Server settings are not ready.');
  return active;
}

export function isCurrentBackendScope(scope: BackendScope): boolean {
  return active?.epoch === scope.epoch && active.origin === scope.origin;
}

/** Publish only after startup/switch persistence and native cookie clearing have succeeded. */
export function publishBackendOrigin(value: string): BackendScope {
  const origin = normalizeBackendOrigin(value);
  if (active?.origin === origin) return active;
  const first = active === null;
  active = Object.freeze({ origin, epoch: ++epoch });
  authVersion++;
  readable = first;
  for (const listener of listeners) listener(active);
  return active;
}

export function onBackendScopeChange(listener: (scope: BackendScope) => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function captureAuthAdmission(): AuthAdmission {
  return { scope: requireBackendScope(), version: authVersion };
}

export function isCurrentAuthAdmission(admission: AuthAdmission): boolean {
  return isCurrentBackendScope(admission.scope) && admission.version === authVersion;
}

export function beginAuthentication(): AuthAdmission {
  requireBackendScope();
  authVersion++;
  return captureAuthAdmission();
}

/** Invalidate pending logins and prevent credential reads before any asynchronous deletion. */
export function beginLogout(): AuthAdmission | null {
  authVersion++;
  readable = false;
  return active ? captureAuthAdmission() : null;
}

export function credentialReadsAllowed(origin: string): boolean {
  return readable && active?.origin === origin;
}

export function admitCredentialReads(admission: AuthAdmission): void {
  if (!isCurrentAuthAdmission(admission)) throw new Error('The server or sign-in attempt changed. Sign in again.');
  readable = true;
}

export function requireCurrentAuthAdmission(admission: AuthAdmission): void {
  if (!isCurrentAuthAdmission(admission)) throw new Error('The server or sign-in attempt changed. Sign in again.');
}
