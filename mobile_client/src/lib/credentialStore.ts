// Opt-in "Save password" store for the login screen. Encrypted at rest via
// react-native-encrypted-storage (Android Keystore / iOS Secure Enclave) — the same backing
// as authStore.ts / serverConfig.ts. Persists ONLY when the user checks "Save password";
// cleared on an unchecked login or on sign-out. A saved password is a convenience, not a
// security boundary — it lives in the OS-encrypted store, never in plain app storage.
import EncryptedStorage from 'react-native-encrypted-storage';

const KEY = 'fedlearn.savedCredentials';

export interface SavedCredentials {
  username: string;
  password: string;
}

/** Load the saved username+password, or null if nothing valid is stored. */
export async function getSavedCredentials(): Promise<SavedCredentials | null> {
  try {
    const raw = await EncryptedStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<SavedCredentials>;
    if (typeof parsed?.username === 'string' && typeof parsed?.password === 'string') {
      return { username: parsed.username, password: parsed.password };
    }
    // Malformed / partial payload — treat as nothing saved.
    return null;
  } catch {
    // Keystore locked/corrupted after an OS upgrade, or unparseable JSON → logged-out state.
    return null;
  }
}

/** Persist the username+password (only call this when the user opted in). */
export async function saveCredentials(creds: SavedCredentials): Promise<void> {
  await EncryptedStorage.setItem(KEY, JSON.stringify(creds));
}

/** Forget any saved credentials (unchecked "Save password", or sign-out). */
export async function clearSavedCredentials(): Promise<void> {
  try {
    await EncryptedStorage.removeItem(KEY);
  } catch {
    // Already absent / keystore locked — nothing to do.
  }
}
