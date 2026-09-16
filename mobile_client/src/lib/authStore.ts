// Encrypted JWT store for the mobile Bearer-auth contract.
// Backed by Android Keystore / iOS Secure Enclave via react-native-encrypted-storage.
// Mirrors the clientId.ts pattern (same KEY + EncryptedStorage usage).
import EncryptedStorage from 'react-native-encrypted-storage';

const KEY = 'fedlearn.authToken';

export async function getToken(): Promise<string | null> {
  try {
    return await EncryptedStorage.getItem(KEY);
  } catch {
    // Keystore locked / corrupted after OS upgrade → treat as logged out.
    return null;
  }
}

export async function setToken(jwt: string): Promise<void> {
  await EncryptedStorage.setItem(KEY, jwt);
}

export async function clearToken(): Promise<void> {
  try {
    await EncryptedStorage.removeItem(KEY);
  } catch {
    // Already absent / keystore locked — nothing to do.
  }
}
