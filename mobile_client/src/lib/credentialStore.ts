import EncryptedStorage from 'react-native-encrypted-storage';
import { normalizeBackendOrigin } from './backendOrigin';
import {
  captureAuthAdmission, getBackendScope, isCurrentAuthAdmission, isCurrentBackendScope,
  requireCurrentAuthAdmission, type AuthAdmission,
} from './backendScope';
import { mutateCredentials, waitForCredentialMutations } from './credentialMutation';

const KEY = 'fedlearn.savedCredentials';

export interface SavedCredentials {
  username: string;
  password: string;
}
interface StoredCredentials extends SavedCredentials { schemaVersion: 1; origin: string }

function validPair(value: unknown): value is SavedCredentials {
  if (typeof value !== 'object' || value === null) return false;
  const pair = value as Partial<SavedCredentials>;
  return typeof pair.username === 'string' && !!pair.username && typeof pair.password === 'string' && !!pair.password;
}

function decode(raw: string | null): StoredCredentials | null {
  try {
    const value: Partial<StoredCredentials> = raw ? JSON.parse(raw) : {};
    if (!validPair(value)) return null;
    const candidate = value as SavedCredentials & Partial<StoredCredentials>;
    return candidate.schemaVersion === 1 && typeof candidate.origin === 'string'
      && normalizeBackendOrigin(candidate.origin) === candidate.origin ? candidate as StoredCredentials : null;
  } catch { return null; }
}

/** Remembered passwords are available only to their issuer's current login form. */
export async function getSavedCredentials(value: string): Promise<SavedCredentials | null> {
  try {
    const origin = normalizeBackendOrigin(value);
    const scope = getBackendScope();
    if (!scope || scope.origin !== origin) return null;
    await waitForCredentialMutations();
    const credentials = decode(await EncryptedStorage.getItem(KEY));
    if (!isCurrentBackendScope(scope) || credentials?.origin !== origin) return null;
    return { username: credentials.username, password: credentials.password };
  } catch { return null; }
}

export async function saveCredentials(
  credentials: SavedCredentials, value: string, admission: AuthAdmission = captureAuthAdmission(),
): Promise<void> {
  const origin = normalizeBackendOrigin(value);
  if (!validPair(credentials) || origin !== admission.scope.origin) throw new Error('Saved credentials do not match this server.');
  return mutateCredentials(async () => {
    requireCurrentAuthAdmission(admission);
    await EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, origin, ...credentials }));
    if (!isCurrentAuthAdmission(admission)) {
      try { await EncryptedStorage.removeItem(KEY); } catch { /* Origin binding still prevents foreign autofill. */ }
      requireCurrentAuthAdmission(admission);
    }
  });
}

export async function clearSavedCredentials(admission?: AuthAdmission | null): Promise<void> {
  const guard = admission === undefined ? (getBackendScope() ? captureAuthAdmission() : null) : admission;
  return mutateCredentials(async () => {
    if (guard ? !isCurrentAuthAdmission(guard) : getBackendScope() !== null) return;
    try { await EncryptedStorage.removeItem(KEY); } catch { /* No unscoped password is ever returned. */ }
  });
}

export function forgetSavedCredentialsForRecovery(): Promise<void> {
  return mutateCredentials(() => EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, cleared: true })));
}

/** Migrate opt-in credentials with startup's successfully read legacy backend. */
export function migrateLegacySavedCredentials(value: string): Promise<void> {
  const origin = normalizeBackendOrigin(value);
  return mutateCredentials(async () => {
    const raw = await EncryptedStorage.getItem(KEY);
    if (!raw || decode(raw)) return;
    let legacy: unknown;
    try { legacy = JSON.parse(raw); } catch { return; }
    if (!validPair(legacy) || 'origin' in legacy || 'schemaVersion' in legacy) return;
    await EncryptedStorage.setItem(KEY, JSON.stringify({ schemaVersion: 1, origin, ...legacy }));
  });
}
