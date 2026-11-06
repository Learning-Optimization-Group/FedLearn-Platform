import EncryptedStorage from 'react-native-encrypted-storage';
import { api, configureApi, setAuthLostHandler } from '../lib/restClient';
import { clearToken, getToken, setToken } from '../lib/authStore';
import { stompAuthHeaders } from '../lib/stompClient';

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
const TOKEN = 'placeholder-session-token';

beforeEach(async () => {
  jest.clearAllMocks();
  await EncryptedStorage.clear();
  configureApi(ORIGIN);
  await setToken(TOKEN, ORIGIN);
});
afterEach(() => { setAuthLostHandler(null); });

test('outbound native requests carry the bound Bearer and native-client marker', async () => {
  let headers: Record<string, unknown> = {};
  await api.get('/api/client/projects', { adapter: async (config) => {
    headers = config.headers.toJSON();
    return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
  } });
  expect(headers.Authorization).toBe('Bearer ' + TOKEN);
  expect(headers['X-FedLearn-Client']).toBe('fedlearn-mobile');
});

test('a signed-out native request keeps its marker without inheriting Authorization', async () => {
  await clearToken();
  let headers: Record<string, unknown> = {};
  await api.get('/api/client/projects', { adapter: async (config) => {
    headers = config.headers.toJSON();
    return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
  } });
  expect(headers.Authorization).toBeUndefined();
  expect(headers['X-FedLearn-Client']).toBe('fedlearn-mobile');
});

test('401 for the presented current session removes that credential and signals auth loss', async () => {
  const onLost = jest.fn();
  setAuthLostHandler(onLost);
  await expect(api.get('/api/projects', { adapter: async (config) => {
    throw { response: { status: 401 }, config };
  } })).rejects.toBeDefined();
  await expect(getToken(ORIGIN)).resolves.toBeNull();
  expect(onLost).toHaveBeenCalledTimes(1);
});

test('a failed silent session probe leaves sign-in recovery to its caller', async () => {
  const onLost = jest.fn();
  setAuthLostHandler(onLost);
  await expect(api.get('/api/auth/me', { adapter: async (config) => {
    throw { response: { status: 401 }, config };
  } })).rejects.toBeDefined();
  await expect(getToken(ORIGIN)).resolves.toBe(TOKEN);
  expect(onLost).not.toHaveBeenCalled();
});

test('an unauthenticated foreign response cannot sign out the configured backend', async () => {
  const onLost = jest.fn();
  setAuthLostHandler(onLost);
  await expect(api.get('https://other.example.invalid/api/projects', { adapter: async (config) => {
    expect(config.headers.get('Authorization')).toBeUndefined();
    throw { response: { status: 401 }, config };
  } })).rejects.toBeDefined();
  await expect(getToken(ORIGIN)).resolves.toBe(TOKEN);
  expect(onLost).not.toHaveBeenCalled();
});

test('callers cannot restore ambient cookies or substitute an unscoped Authorization header', async () => {
  await api.get('/api/projects', { headers: { Cookie: 'jwtToken=placeholder-cookie', Authorization: 'Bearer placeholder-foreign' },
    withCredentials: true, adapter: async (config) => {
      expect(config.withCredentials).toBe(false);
      expect(config.headers.get('Cookie')).toBeUndefined();
      expect(config.headers.get('Authorization')).toBe('Bearer ' + TOKEN);
      return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
    } });
});

test('STOMP upgrade and CONNECT headers use the native-client authentication contract', () => {
  expect(stompAuthHeaders(TOKEN)).toEqual({
    'X-FedLearn-Client': 'fedlearn-mobile', Authorization: 'Bearer ' + TOKEN,
  });
  expect(stompAuthHeaders(null)).toEqual({ 'X-FedLearn-Client': 'fedlearn-mobile' });
});
