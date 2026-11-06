import EncryptedStorage from 'react-native-encrypted-storage';
import { normalizeBackendOrigin } from './backendOrigin';
import {
  admitCredentialReads, beginLogout, captureAuthAdmission, credentialReadsAllowed,
  getBackendScope, isCurrentAuthAdmission, requireCurrentAuthAdmission, type AuthAdmission,
} from './backendScope';
import { mutateCredentials, waitForCredentialMutations } from './credentialMutation';

const KEY = 'fedlearn.authToken';
let revision = 0;

export interface AuthSession { readonly origin: string; readonly token: string; readonly revision: number }
export type AuthSessionIdentity = Pick<AuthSession, 'origin' | 'revision'>;

function decode(raw: string | null): AuthSession | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw);
    if (value?.schemaVersion !== 1 || typeof value.origin !== 'string'
        || normalizeBackendOrigin(value.origin) !== value.origin || typeof value.token !== 'string'
        || !value.token || /\s/.test(value.token) || !Number.isSafeInteger(value.revision)
        || value.revision < 1 || value.revision >= Number.MAX_SAFE_INTEGER - 1) return null;
    revision = Math.max(revision, value.revision);
    return { origin: value.origin, token: value.token, revision: value.revision };
  } catch { return null; }
}

/** A credential is usable only for the current, explicitly captured backend. */
export async function getSession(value: string): Promise<AuthSession | null> {
  try {
    const origin = normalizeBackendOrigin(value);
    await waitForCredentialMutations();
    if (!credentialReadsAllowed(origin)) return null;
    const session = decode(await EncryptedStorage.getItem(KEY));
    return credentialReadsAllowed(origin) && session?.origin === origin ? session : null;
  } catch { return null; }
}

export async function getToken(origin: string): Promise<string | null> {
  return (await getSession(origin))?.token ?? null;
}

/** The origin and token share one atomic encrypted write. */
export async function setToken(token: string, value: string, admission: AuthAdmission = captureAuthAdmission()): Promise<AuthSession> {
  const origin = normalizeBackendOrigin(value);
  if (!token || typeof token !== 'string' || /\s/.test(token) || origin !== admission.scope.origin) {
    throw new Error('The sign-in credential is not valid for this server.');
  }
  return mutateCredentials(async () => {
    requireCurrentAuthAdmission(admission);
    decode(await EncryptedStorage.getItem(KEY));
    requireCurrentAuthAdmission(admission);
    const session = { origin, token, revision: ++revision };
    await EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, ...session }));
    if (!isCurrentAuthAdmission(admission)) {
      // No other credential mutation can enter until this queued operation ends.
      try { await EncryptedStorage.removeItem(KEY); } catch { /* Bound stale records are never admitted. */ }
      requireCurrentAuthAdmission(admission);
    }
    admitCredentialReads(admission);
    return session;
  });
}

/** A delayed response may delete only the exact session it presented. */
export function clearSessionIfMatches(expected: AuthSessionIdentity): Promise<boolean> {
  return mutateCredentials(async () => {
    try {
      const current = decode(await EncryptedStorage.getItem(KEY));
      if (current?.origin !== expected.origin || current.revision !== expected.revision) return false;
      await EncryptedStorage.removeItem(KEY);
      return true;
    } catch { return false; }
  });
}

/** Explicit logout revokes pending admissions and reads before asynchronous deletion. */
export async function clearToken(admission?: AuthAdmission | null): Promise<void> {
  const guard = admission === undefined ? beginLogout() : admission;
  return mutateCredentials(async () => {
    if (guard ? !isCurrentAuthAdmission(guard) : getBackendScope() !== null) return;
    try { await EncryptedStorage.removeItem(KEY); } catch { /* Reads remain revoked until a new sign-in. */ }
  });
}

/** Recovery must durably invalidate an unknown issuer before accepting replacement settings. */
export function forgetTokenForRecovery(): Promise<void> {
  beginLogout();
  return mutateCredentials(() => EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, cleared: true })));
}

/** Called only after a strict startup read of the previously persisted backend URL. */
export function migrateLegacyToken(value: string): Promise<void> {
  const origin = normalizeBackendOrigin(value);
  return mutateCredentials(async () => {
    const raw = await EncryptedStorage.getItem(KEY);
    if (!raw || decode(raw)) return;
    if (!/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(raw)) return;
    await EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, origin, token: raw, revision: ++revision }));
  });
}
