import {
  api,
  setAuthLostHandler,
  NATIVE_CLIENT_HEADER,
  NATIVE_CLIENT_VALUE,
} from '../lib/restClient';
import { stompAuthHeaders } from '../lib/stompClient';
import * as authStore from '../lib/authStore';

jest.mock('../lib/authStore');
// stompClient pulls in serverConfig (native encrypted storage) — not needed for the
// pure header-builder under test here.
jest.mock('../lib/serverConfig', () => ({ getServerBaseUrl: jest.fn() }));
const store = authStore as jest.Mocked<typeof authStore>;

// Axios's public TS types don't expose InterceptorManager's internal `.handlers` array — reaching
// into it is the simplest way to invoke a registered interceptor directly, without a live network
// call. These describe only the shape restClient.ts's own interceptors actually consume/return.
interface RequestInterceptors {
  handlers: Array<{
    fulfilled: (config: {
      headers: Record<string, string>;
      url?: string;
    }) => Promise<{ headers: Record<string, string> }>;
  }>;
}
interface ResponseInterceptors {
  handlers: Array<{
    rejected: (error: {
      response?: { status: number };
      config?: { url?: string };
    }) => Promise<never>;
  }>;
}

describe('restClient interceptors', () => {
  beforeEach(() => jest.clearAllMocks());

  test('request interceptor attaches Bearer when a token exists', async () => {
    store.getToken.mockResolvedValue('jwt-xyz');
    // Run the request interceptor directly against a config object.
    const handler = (api.interceptors.request as unknown as RequestInterceptors).handlers[0]!.fulfilled;
    const cfg = await handler({ headers: {}, url: '/api/client/projects' });
    expect(cfg.headers.Authorization).toBe('Bearer jwt-xyz');
  });

  test('request interceptor omits Bearer when no token', async () => {
    store.getToken.mockResolvedValue(null);
    const handler = (api.interceptors.request as unknown as RequestInterceptors).handlers[0]!.fulfilled;
    const cfg = await handler({ headers: {}, url: '/api/client/projects' });
    expect(cfg.headers.Authorization).toBeUndefined();
  });

  test('401 (non-probe) clears token and signals auth-lost', async () => {
    const onLost = jest.fn();
    setAuthLostHandler(onLost);
    const rejected = (api.interceptors.response as unknown as ResponseInterceptors).handlers[0]!.rejected;
    await expect(
      rejected({ response: { status: 401 }, config: { url: '/api/projects' } }),
    ).rejects.toBeDefined();
    expect(store.clearToken).toHaveBeenCalled();
    expect(onLost).toHaveBeenCalled();
  });

  test('401 on /auth/me probe does NOT signal auth-lost', async () => {
    const onLost = jest.fn();
    setAuthLostHandler(onLost);
    const rejected = (api.interceptors.response as unknown as ResponseInterceptors).handlers[0]!.rejected;
    await expect(
      rejected({ response: { status: 401 }, config: { url: '/api/auth/me' } }),
    ).rejects.toBeDefined();
    expect(onLost).not.toHaveBeenCalled();
  });
});

// SE-9: the backend honors `Authorization: Bearer` only when the request also carries the
// X-FedLearn-Client marker (browsers stay cookie-only). Every mobile request must send it.
describe('native-client marker (SE-9)', () => {
  beforeEach(() => jest.clearAllMocks());

  test('X-FedLearn-Client is a shared default header on the api instance', () => {
    expect(NATIVE_CLIENT_HEADER).toBe('X-FedLearn-Client');
    expect(api.defaults.headers.common[NATIVE_CLIENT_HEADER]).toBe('fedlearn-mobile');
  });

  test('an outbound request carries the marker alongside the Bearer token', async () => {
    store.getToken.mockResolvedValue('jwt-xyz');
    let seen: { get(name: string): unknown } | null = null;
    // Stub adapter — captures the fully-merged headers without touching the network.
    await api.get('/api/client/projects', {
      adapter: async (config) => {
        seen = config.headers as unknown as { get(name: string): unknown };
        return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
      },
    });
    expect(seen!.get(NATIVE_CLIENT_HEADER)).toBe(NATIVE_CLIENT_VALUE);
    expect(seen!.get('Authorization')).toBe('Bearer jwt-xyz');
  });

  test('STOMP headers (WS upgrade + CONNECT frame) carry the marker with and without a token', () => {
    expect(stompAuthHeaders('jwt-xyz')).toEqual({
      [NATIVE_CLIENT_HEADER]: NATIVE_CLIENT_VALUE,
      Authorization: 'Bearer jwt-xyz',
    });
    expect(stompAuthHeaders(null)).toEqual({ [NATIVE_CLIENT_HEADER]: NATIVE_CLIENT_VALUE });
  });
});
