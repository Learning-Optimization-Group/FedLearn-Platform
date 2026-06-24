import { api } from '../lib/restClient';
import * as authStore from '../lib/authStore';
import { performLogin, probeSession } from '../context/AuthContext';

jest.mock('../lib/restClient', () => ({ api: { post: jest.fn(), get: jest.fn() }, setAuthLostHandler: jest.fn() }));
jest.mock('../lib/authStore');
const store = authStore as jest.Mocked<typeof authStore>;
const mApi = api as unknown as { post: jest.Mock; get: jest.Mock };

describe('AuthContext logic', () => {
  beforeEach(() => jest.clearAllMocks());

  test('performLogin posts credentials, stores the body token, returns identity', async () => {
    mApi.post.mockResolvedValue({ data: { accessToken: 'jwt-1', username: 'alice' } });
    const id = await performLogin('alice', 'pw');
    expect(mApi.post).toHaveBeenCalledWith('/api/auth/login', { username: 'alice', password: 'pw' });
    expect(store.setToken).toHaveBeenCalledWith('jwt-1');
    expect(id.username).toBe('alice');
  });

  test('performLogin throws when the body has no accessToken', async () => {
    mApi.post.mockResolvedValue({ data: { username: 'alice' } });
    await expect(performLogin('alice', 'pw')).rejects.toThrow();
    expect(store.setToken).not.toHaveBeenCalled();
  });

  test('probeSession returns username on 200, null on failure', async () => {
    store.getToken.mockResolvedValue('jwt-1');
    mApi.get.mockResolvedValue({ data: { username: 'alice' } });
    await expect(probeSession()).resolves.toEqual({ username: 'alice' });

    store.getToken.mockResolvedValue('jwt-1');
    mApi.get.mockRejectedValue({ response: { status: 401 } });
    await expect(probeSession()).resolves.toBeNull();
  });

  test('probeSession returns null with no stored token (no network call)', async () => {
    store.getToken.mockResolvedValue(null);
    await expect(probeSession()).resolves.toBeNull();
    expect(mApi.get).not.toHaveBeenCalled();
  });
});
