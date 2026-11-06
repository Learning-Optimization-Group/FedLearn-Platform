import EncryptedStorage from 'react-native-encrypted-storage';
import { api } from '../lib/restClient';
import { initServerConfig, setServerBaseUrl } from '../lib/serverConfig';
import { clearToken } from '../lib/authStore';
import { performLogin, performRegister } from '../context/AuthContext';
import type { AxiosAdapter, AxiosResponse, InternalAxiosRequestConfig } from 'axios';

jest.mock('../lib/nativeCookies', () => ({ clearNativeCookies: jest.fn(async () => {}) }));

jest.mock('react-native-encrypted-storage', () => {
  const values = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => values.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => { values.set(key, value); }),
    removeItem: jest.fn(async (key: string) => { values.delete(key); }),
    clear: jest.fn(async () => { values.clear(); }),
  };
});

const FIRST = 'https://first.example.invalid';
const SECOND = 'https://second.example.invalid';
const TOKEN = 'placeholder-first-server-token';
const originalAdapter = api.defaults.adapter;
let outbound: { target: string; authorization: unknown }[];
let nativeCookies: boolean | undefined;

beforeEach(async () => {
  jest.clearAllMocks();
  await EncryptedStorage.clear();
  outbound = [];
  nativeCookies = undefined;
  api.defaults.adapter = (async (config) => {
    nativeCookies = config.withCredentials;
    outbound.push({ target: api.getUri(config), authorization: config.headers.get('Authorization') });
    return { data: config.url === '/api/auth/login' ? { accessToken: TOKEN, username: 'probe' } : {},
      status: 200, statusText: 'OK', headers: {}, config };
  }) as AxiosAdapter;
  await setServerBaseUrl(FIRST);
  await performLogin('probe', 'placeholder-password');
  outbound = [];
});

afterEach(() => { api.defaults.adapter = originalAdapter; });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

function response(config: InternalAxiosRequestConfig, token?: string): AxiosResponse {
  return { data: token ? { accessToken: token, username: 'probe' } : {}, status: 200,
    statusText: 'OK', headers: {}, config };
}

test('changing backend does not send the previous backend token to its reachability probe', async () => {
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: FIRST + '/api/auth/me', authorization: 'Bearer ' + TOKEN }]);
  outbound = [];
  await setServerBaseUrl(SECOND);
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: SECOND + '/api/auth/me', authorization: undefined }]);
});

test.each([SECOND, FIRST + ':8444'])('an absolute request to %s cannot inherit the configured backend credential', async (origin) => {
  await api.get(origin + '/api/auth/me');
  expect(outbound).toEqual([{ target: origin + '/api/auth/me', authorization: undefined }]);
});

test('a per-request base URL override cannot inherit the configured backend credential', async () => {
  await api.get('/api/auth/me', { baseURL: SECOND });
  expect(outbound).toEqual([{ target: SECOND + '/api/auth/me', authorization: undefined }]);
});

test('a protocol-relative foreign target cannot inherit the configured backend credential', async () => {
  await api.get('//second.example.invalid/api/auth/me');
  expect(outbound).toEqual([{ target: '//second.example.invalid/api/auth/me', authorization: undefined }]);
});

test('failed startup settings do not bind the previous credential to a default backend', async () => {
  (EncryptedStorage.getItem as jest.Mock).mockRejectedValueOnce(new Error('settings locked'));
  await expect(initServerConfig()).rejects.toThrow('settings locked');
  expect(outbound).toEqual([]);
});

test('a cold backend switch binds a legacy token to its saved issuer before replacing the setting', async () => {
  await jest.isolateModulesAsync(async () => {
    const storage = jest.requireMock('react-native-encrypted-storage') as typeof EncryptedStorage;
    const config = jest.requireActual<typeof import('../lib/serverConfig')>('../lib/serverConfig');
    const auth = jest.requireActual<typeof import('../lib/authStore')>('../lib/authStore');
    await storage.clear();
    await storage.setItem('fedlearn.serverBaseUrl', FIRST);
    await storage.setItem('fedlearn.authToken', 'header.payload.signature');
    await config.setServerBaseUrl(SECOND);
    await config.initServerConfig();
    await expect(auth.getToken(SECOND)).resolves.toBeNull();
  });
});

test('a cold switch cannot reassign a legacy credential whose saved issuer is invalid', async () => {
  await jest.isolateModulesAsync(async () => {
    const storage = jest.requireMock('react-native-encrypted-storage') as typeof EncryptedStorage;
    const config = jest.requireActual<typeof import('../lib/serverConfig')>('../lib/serverConfig');
    await storage.clear();
    await storage.setItem('fedlearn.serverBaseUrl', FIRST + '/invalid-base-path');
    await storage.setItem('fedlearn.authToken', 'header.payload.signature');
    await expect(config.setServerBaseUrl(SECOND)).rejects.toThrow();
    await expect(storage.getItem('fedlearn.serverBaseUrl')).resolves.toBe(FIRST + '/invalid-base-path');
  });
});

test('explicit recovery forgets legacy sign-in without touching local datasets', async () => {
  await jest.isolateModulesAsync(async () => {
    const storage = jest.requireMock('react-native-encrypted-storage') as typeof EncryptedStorage;
    const config = jest.requireActual<typeof import('../lib/serverConfig')>('../lib/serverConfig');
    const auth = jest.requireActual<typeof import('../lib/authStore')>('../lib/authStore');
    const credentials = jest.requireActual<typeof import('../lib/credentialStore')>('../lib/credentialStore');
    await storage.clear();
    await storage.setItem('fedlearn.serverBaseUrl', FIRST + '/invalid-base-path');
    await storage.setItem('fedlearn.authToken', 'header.payload.signature');
    await storage.setItem('fedlearn.savedCredentials', JSON.stringify({ username: 'probe', password: 'placeholder-old' }));
    await storage.setItem('dataset-control', 'preserved');
    await config.recoverServerConfig(SECOND);
    await config.initServerConfig();
    await expect(auth.getToken(SECOND)).resolves.toBeNull();
    await expect(credentials.getSavedCredentials(SECOND)).resolves.toBeNull();
    await expect(storage.getItem('dataset-control')).resolves.toBe('preserved');
    await expect(config.getServerBaseUrl()).resolves.toBe(SECOND);
  });
});

test.each(['fedlearn.authToken', 'fedlearn.savedCredentials', 'fedlearn.serverBaseUrl'])('recovery refuses to publish replacement settings when %s cannot be written', async (failingKey) => {
  await jest.isolateModulesAsync(async () => {
    const storage = jest.requireMock('react-native-encrypted-storage') as typeof EncryptedStorage;
    const config = jest.requireActual<typeof import('../lib/serverConfig')>('../lib/serverConfig');
    await storage.clear();
    await storage.setItem('fedlearn.serverBaseUrl', FIRST + '/invalid-base-path');
    await storage.setItem('fedlearn.authToken', 'header.payload.signature');
    const write = storage.setItem as jest.Mock;
    const original = write.getMockImplementation()!;
    write.mockImplementation((key: string, value: string) => key === failingKey
      ? Promise.reject(new Error('storage unavailable')) : original(key, value));
    try {
      await expect(config.recoverServerConfig(SECOND)).rejects.toThrow('storage unavailable');
      await expect(storage.getItem('fedlearn.serverBaseUrl')).resolves.toBe(FIRST + '/invalid-base-path');
      await expect(config.initServerConfig()).rejects.toThrow();
    } finally { write.mockImplementation(original); }
  });
});

test('a pending login cannot authenticate a newly selected backend', async () => {
  const started = deferred<InternalAxiosRequestConfig>();
  const pending = deferred<AxiosResponse>();
  api.defaults.adapter = async (config) => { started.resolve(config); return pending.promise; };
  const login = performLogin('probe', 'placeholder-pending-password');
  const config = await started.promise;
  await setServerBaseUrl(SECOND);
  pending.resolve(response(config, 'placeholder-stale-first-server-token'));
  await expect(login).rejects.toThrow();
});

test('logout prevents a pending login from restoring its credential', async () => {
  const started = deferred<InternalAxiosRequestConfig>();
  const pending = deferred<AxiosResponse>();
  api.defaults.adapter = async (config) => { started.resolve(config); return pending.promise; };
  const login = performLogin('probe', 'placeholder-pending-password');
  const config = await started.promise;
  await clearToken();
  pending.resolve(response(config, 'placeholder-revoked-login-token'));
  await expect(login).rejects.toThrow();
});

test('registration does not forward its password through a later login after changing servers', async () => {
  const started = deferred<InternalAxiosRequestConfig>();
  const pending = deferred<AxiosResponse>();
  api.defaults.adapter = async (config) => {
    outbound.push({ target: api.getUri(config), authorization: config.headers.get('Authorization') });
    if (config.url === '/api/auth/register') { started.resolve(config); return pending.promise; }
    return response(config, 'placeholder-new-login-token');
  };
  const registration = performRegister('probe', 'probe@example.invalid', 'placeholder-registration-password');
  const config = await started.promise;
  await setServerBaseUrl(SECOND);
  pending.resolve(response(config));
  await expect(registration).rejects.toThrow();
  expect(outbound.some((request) => request.target === SECOND + '/api/auth/login')).toBe(false);
});

test.each([FIRST, SECOND])('an older 401 cannot remove the replacement session for %s', async (origin) => {
  const started = deferred<void>();
  const release = deferred<void>();
  const replacement = 'placeholder-replacement-session-token';
  api.defaults.adapter = async (config) => {
    if (config.url === '/api/projects') {
      started.resolve();
      await release.promise;
      throw { response: { status: 401 }, config };
    }
    outbound.push({ target: api.getUri(config), authorization: config.headers.get('Authorization') });
    return response(config, config.url === '/api/auth/login' ? replacement : undefined);
  };
  const oldRequest = api.get('/api/projects').catch(() => undefined);
  await started.promise;
  if (origin !== FIRST) await setServerBaseUrl(origin);
  await performLogin('replacement', 'placeholder-replacement-password');
  release.resolve();
  await oldRequest;
  outbound = [];
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: origin + '/api/auth/me', authorization: 'Bearer ' + replacement }]);
});

test('native requests cannot enable the ambient cookie jar', async () => {
  await api.get('/api/auth/me', { withCredentials: true });
  expect(nativeCookies).toBe(false);
});

test('canonical saves of the same origin retain its authenticated session', async () => {
  await setServerBaseUrl('HTTPS://FIRST.example.invalid:443/');
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: FIRST + '/api/auth/me', authorization: 'Bearer ' + TOKEN }]);
});

test.each([
  'https://first.example.invalid/api',
  'https://first.example.invalid@second.example.invalid',
  'https://first.example.invalid?next=other',
  'https://first.example.invalid#other',
  'https://first.example.invalid:0',
  'https://first.example.invalid:65536',
  'https://[::1',
  'https://first..example.invalid',
])('refuses ambiguous or invalid backend authority %s before changing the target', async (url) => {
  await expect(setServerBaseUrl(url)).rejects.toThrow();
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: FIRST + '/api/auth/me', authorization: 'Bearer ' + TOKEN }]);
});

test('saving the same backend retains its authenticated session', async () => {
  await setServerBaseUrl(FIRST + '/');
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: FIRST + '/api/auth/me', authorization: 'Bearer ' + TOKEN }]);
});

test('a failed backend-setting write leaves the old target and its credential together', async () => {
  (EncryptedStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('storage unavailable'));
  await expect(setServerBaseUrl(SECOND)).rejects.toThrow('storage unavailable');
  await api.get('/api/auth/me');
  expect(outbound).toEqual([{ target: FIRST + '/api/auth/me', authorization: 'Bearer ' + TOKEN }]);
});
