// Stable, persisted client id (fixes A6 §M-M4: v1 generated a new id every launch, so the server
// saw a fresh "client" each run). Stored in encrypted storage (Android Keystore-backed /
// iOS Data-Protection) per the on-device security model (15-LLD §7).
import EncryptedStorage from 'react-native-encrypted-storage';

const KEY = 'fedlearn.clientId';

// Minimal RFC-4122 v4 generator. A client id is an identifier, not a secret, so Math.random is
// acceptable; if a cryptographically-strong id is later required, swap in react-native-get-random-values.
function uuidv4(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export async function getOrCreateClientId(): Promise<string> {
  const existing = await EncryptedStorage.getItem(KEY);
  if (existing) return existing;
  const id = uuidv4();
  await EncryptedStorage.setItem(KEY, id);
  return id;
}
