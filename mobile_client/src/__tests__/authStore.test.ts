import EncryptedStorage from 'react-native-encrypted-storage';
import { clearSessionIfMatches, clearToken, getSession, getToken, migrateLegacyToken, setToken } from '../lib/authStore';
import { admitCredentialReads, beginAuthentication, captureAuthAdmission, publishBackendOrigin } from '../lib/backendScope';

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

beforeEach(async () => {
  jest.clearAllMocks();
  await EncryptedStorage.clear();
  publishBackendOrigin(ORIGIN);
  admitCredentialReads(captureAuthAdmission());
});

test('an encrypted session is readable only for its current issuing origin', async () => {
  await setToken('placeholder-token', ORIGIN);
  await expect(getToken(ORIGIN)).resolves.toBe('placeholder-token');
  await expect(getToken(OTHER)).resolves.toBeNull();
});

test('a failed atomic write preserves the earlier token and its origin together', async () => {
  await setToken('placeholder-previous', ORIGIN);
  (EncryptedStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('keystore unavailable'));
  await expect(setToken('placeholder-replacement', ORIGIN)).rejects.toThrow('keystore unavailable');
  await expect(getToken(ORIGIN)).resolves.toBe('placeholder-previous');
  await expect(getToken(OTHER)).resolves.toBeNull();
});

test('legacy raw tokens are refused until bound by strict startup migration', async () => {
  await EncryptedStorage.setItem('fedlearn.authToken', 'header.payload.signature');
  await expect(getToken(ORIGIN)).resolves.toBeNull();
  await migrateLegacyToken(ORIGIN);
  await expect(getToken(ORIGIN)).resolves.toBe('header.payload.signature');
  await expect(getToken(OTHER)).resolves.toBeNull();
});

test.each(['{"token":"placeholder"}', '{broken', 'not-a-token'])('malformed records are not reinterpreted as legacy credentials: %s', async (raw) => {
  await EncryptedStorage.setItem('fedlearn.authToken', raw);
  await migrateLegacyToken(ORIGIN);
  await expect(getToken(ORIGIN)).resolves.toBeNull();
});

test('unavailable encrypted storage never yields a credential', async () => {
  await setToken('placeholder-token', ORIGIN);
  (EncryptedStorage.getItem as jest.Mock).mockRejectedValueOnce(new Error('keystore locked'));
  await expect(getToken(ORIGIN)).resolves.toBeNull();
});

test('logout prevents a pending admission from restoring its token', async () => {
  const admission = beginAuthentication();
  await clearToken();
  await expect(setToken('placeholder-stale', ORIGIN, admission)).rejects.toThrow();
  await expect(getToken(ORIGIN)).resolves.toBeNull();
});

test('a stale session response cannot delete a replacement login', async () => {
  await setToken('placeholder-first', ORIGIN);
  const first = await getSession(ORIGIN);
  if (!first) throw new Error('Missing first session fixture');
  await setToken('placeholder-second', ORIGIN, beginAuthentication());
  await expect(clearSessionIfMatches(first)).resolves.toBe(false);
  await expect(getToken(ORIGIN)).resolves.toBe('placeholder-second');
});

test('a delayed conditional delete cannot erase a queued replacement write', async () => {
  await setToken('placeholder-first', ORIGIN);
  const first = await getSession(ORIGIN);
  if (!first) throw new Error('Missing first session fixture');
  const remove = EncryptedStorage.removeItem as jest.Mock;
  const original = remove.getMockImplementation()!;
  let signal!: () => void;
  let release!: () => void;
  const entered = new Promise<void>((done) => { signal = done; });
  const blocked = new Promise<void>((done) => { release = done; });
  remove.mockImplementationOnce(async (key: string) => { signal(); await blocked; return original(key); });
  const deletion = clearSessionIfMatches(first);
  await entered;
  const replacement = setToken('placeholder-second', ORIGIN, beginAuthentication());
  release();
  await Promise.all([deletion, replacement]);
  await expect(getToken(ORIGIN)).resolves.toBe('placeholder-second');
});
