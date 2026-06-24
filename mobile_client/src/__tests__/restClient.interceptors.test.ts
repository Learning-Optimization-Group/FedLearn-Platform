import { api, setAuthLostHandler } from '../lib/restClient';
import * as authStore from '../lib/authStore';

jest.mock('../lib/authStore');
const store = authStore as jest.Mocked<typeof authStore>;

describe('restClient interceptors', () => {
  beforeEach(() => jest.clearAllMocks());

  test('request interceptor attaches Bearer when a token exists', async () => {
    store.getToken.mockResolvedValue('jwt-xyz');
    // Run the request interceptor directly against a config object.
    const handler = (api.interceptors.request as any).handlers[0].fulfilled;
    const cfg = await handler({ headers: {}, url: '/api/client/projects' });
    expect(cfg.headers.Authorization).toBe('Bearer jwt-xyz');
  });

  test('request interceptor omits Bearer when no token', async () => {
    store.getToken.mockResolvedValue(null);
    const handler = (api.interceptors.request as any).handlers[0].fulfilled;
    const cfg = await handler({ headers: {}, url: '/api/client/projects' });
    expect(cfg.headers.Authorization).toBeUndefined();
  });

  test('401 (non-probe) clears token and signals auth-lost', async () => {
    const onLost = jest.fn();
    setAuthLostHandler(onLost);
    const rejected = (api.interceptors.response as any).handlers[0].rejected;
    await expect(
      rejected({ response: { status: 401 }, config: { url: '/api/projects' } }),
    ).rejects.toBeDefined();
    expect(store.clearToken).toHaveBeenCalled();
    expect(onLost).toHaveBeenCalled();
  });

  test('401 on /auth/me probe does NOT signal auth-lost', async () => {
    const onLost = jest.fn();
    setAuthLostHandler(onLost);
    const rejected = (api.interceptors.response as any).handlers[0].rejected;
    await expect(
      rejected({ response: { status: 401 }, config: { url: '/api/auth/me' } }),
    ).rejects.toBeDefined();
    expect(onLost).not.toHaveBeenCalled();
  });
});
