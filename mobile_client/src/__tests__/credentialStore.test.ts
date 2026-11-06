import EncryptedStorage from 'react-native-encrypted-storage';
import { clearSavedCredentials, getSavedCredentials, migrateLegacySavedCredentials, saveCredentials } from '../lib/credentialStore';
import { beginAuthentication, beginLogout, publishBackendOrigin } from '../lib/backendScope';

jest.mock('react-native-encrypted-storage', () => {
  const values = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => values.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => { values.set(key, value); }),
    removeItem: jest.fn(async (key: string) => { values.delete(key); }),
    clear: jest.fn(async () => { values.clear(); }),
  };
});
const ORIGIN = 'https://issuer.example.invalid';
const OTHER = 'https://other.example.invalid';

const PAIR = { username: 'probe', password: 'placeholder-saved-password' };
beforeEach(async () => {
  jest.clearAllMocks();
  await EncryptedStorage.clear();
  publishBackendOrigin(ORIGIN);
});

test('remembered credentials are returned only to their issuing server form', async () => {
  await saveCredentials(PAIR, ORIGIN);
  await expect(getSavedCredentials(ORIGIN)).resolves.toEqual(PAIR);
  await expect(getSavedCredentials(OTHER)).resolves.toBeNull();
});

test('legacy opt-in credentials require strict startup origin binding', async () => {
  await EncryptedStorage.setItem('fedlearn.savedCredentials', JSON.stringify(PAIR));
  await expect(getSavedCredentials(ORIGIN)).resolves.toBeNull();
  await migrateLegacySavedCredentials(ORIGIN);
  await expect(getSavedCredentials(ORIGIN)).resolves.toEqual(PAIR);
  await expect(getSavedCredentials(OTHER)).resolves.toBeNull();
});

test.each(['{broken', '{"username":"probe"}', '{"origin":"https://other.example.invalid","username":"probe","password":"placeholder"}'])('malformed or unbound records never prefill a password: %s', async (raw) => {
  await EncryptedStorage.setItem('fedlearn.savedCredentials', raw);
  await migrateLegacySavedCredentials(ORIGIN);
  await expect(getSavedCredentials(ORIGIN)).resolves.toBeNull();
});

test('a stale saved-password read cannot survive an origin round trip', async () => {
  await saveCredentials(PAIR, ORIGIN);
  const read = EncryptedStorage.getItem as jest.Mock;
  const original = read.getMockImplementation()!;
  let signal!: () => void;
  let release!: () => void;
  const entered = new Promise<void>((done) => { signal = done; });
  const blocked = new Promise<void>((done) => { release = done; });
  read.mockImplementationOnce(async (key: string) => { signal(); const raw = await original(key); await blocked; return raw; });
  const credentials = getSavedCredentials(ORIGIN);
  await entered;
  publishBackendOrigin(OTHER);
  publishBackendOrigin(ORIGIN);
  release();
  await expect(credentials).resolves.toBeNull();
});

test('logout invalidates a pending saved-password write', async () => {
  const admission = beginAuthentication();
  beginLogout();
  await expect(saveCredentials(PAIR, ORIGIN, admission)).rejects.toThrow();
  await expect(getSavedCredentials(ORIGIN)).resolves.toBeNull();
});

test('explicitly forgetting the saved password removes it without touching a dataset', async () => {
  await EncryptedStorage.setItem('dataset-control', 'preserved');
  await saveCredentials(PAIR, ORIGIN);
  await clearSavedCredentials();
  await expect(getSavedCredentials(ORIGIN)).resolves.toBeNull();
  await expect(EncryptedStorage.getItem('dataset-control')).resolves.toBe('preserved');
});

test('a keystore read failure produces no prefilled credentials', async () => {
  await saveCredentials(PAIR, ORIGIN);
  (EncryptedStorage.getItem as jest.Mock).mockRejectedValueOnce(new Error('keystore locked'));
  await expect(getSavedCredentials(ORIGIN)).resolves.toBeNull();
});
