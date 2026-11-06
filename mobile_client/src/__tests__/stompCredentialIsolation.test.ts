import EncryptedStorage from 'react-native-encrypted-storage';
import { Client } from '@stomp/stompjs';
import { clearToken, setToken } from '../lib/authStore';
import { configureApi } from '../lib/restClient';
import { setServerBaseUrl } from '../lib/serverConfig';
import { connectStomp } from '../lib/stompClient';

jest.mock('@stomp/stompjs', () => ({ Client: jest.fn() }));
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

interface Options { webSocketFactory: () => WebSocket; connectHeaders: Record<string, string> }
interface FakeClient {
  onConnect: (() => void) | null;
  activate: jest.Mock;
  deactivate: jest.Mock;
  subscribe: jest.Mock;
}
const FIRST = 'https://issuer.example.invalid';
const SECOND = 'https://other.example.invalid';
const TOKEN = 'placeholder-websocket-token';
const originalWebSocket = globalThis.WebSocket;
let options: Options;
let client: FakeClient;
let deferActivation: boolean;
let activated: Promise<void>;

beforeEach(async () => {
  jest.clearAllMocks();
  await EncryptedStorage.clear();
  await EncryptedStorage.setItem('fedlearn.serverBaseUrl', FIRST);
  configureApi(FIRST);
  await setToken(TOKEN, FIRST);
  deferActivation = false;
  globalThis.WebSocket = jest.fn(function () { return { close: jest.fn() }; }) as unknown as typeof WebSocket;
  let signal!: () => void;
  activated = new Promise<void>((resolve) => { signal = resolve; });
  (Client as jest.MockedClass<typeof Client>).mockImplementation(function (configuration) {
    options = configuration as unknown as Options;
    client = { onConnect: null, activate: jest.fn(), deactivate: jest.fn(async () => {}),
      subscribe: jest.fn(() => ({ unsubscribe: jest.fn() })) };
    client.activate.mockImplementation(() => {
      signal();
      if (!deferActivation) { options.webSocketFactory(); client.onConnect?.(); }
    });
    return client as unknown as Client;
  });
});
afterEach(() => { globalThis.WebSocket = originalWebSocket; });

test('upgrade and CONNECT use the same captured issuing server and credential', async () => {
  await connectStomp();
  const construct = globalThis.WebSocket as unknown as jest.Mock;
  expect(construct.mock.calls[0]?.[0]).toBe('wss://issuer.example.invalid/ws-logs');
  expect(construct.mock.calls[0]?.[2]).toEqual({ headers: {
    'X-FedLearn-Client': 'fedlearn-mobile', Authorization: 'Bearer ' + TOKEN,
  } });
  expect(options.connectHeaders).toEqual({ 'X-FedLearn-Client': 'fedlearn-mobile', Authorization: 'Bearer ' + TOKEN });
});

test('an independently changed persisted URL cannot redirect a current server credential', async () => {
  await EncryptedStorage.setItem('fedlearn.serverBaseUrl', SECOND);
  await connectStomp();
  const construct = globalThis.WebSocket as unknown as jest.Mock;
  expect(construct.mock.calls[0]?.[0]).toBe('wss://issuer.example.invalid/ws-logs');
  expect(options.connectHeaders.Authorization).toBe('Bearer ' + TOKEN);
});

test.each(['server change', 'logout'])('a delayed WebSocket factory cannot outlive %s', async (transition) => {
  deferActivation = true;
  const connection = connectStomp();
  const settled = connection.then(() => 'connected', () => 'refused');
  await activated;
  if (transition === 'server change') await setServerBaseUrl(SECOND);
  else await clearToken();
  expect(() => options.webSocketFactory()).toThrow();
  client.onConnect?.();
  await expect(settled).resolves.toBe('refused');
  expect(globalThis.WebSocket).not.toHaveBeenCalled();
  expect(client.deactivate).toHaveBeenCalledTimes(1);
});
