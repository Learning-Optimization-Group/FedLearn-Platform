import { api } from '../lib/restClient';
import * as authStore from '../lib/authStore';
import { performLogin, performRegister, probeSession } from '../context/AuthContext';

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

  // MO-6: performRegister posts the sign-up, then immediately logs in with the same credentials.
  test('performRegister posts /register then logs in with the same credentials, storing the token', async () => {
    mApi.post
      .mockResolvedValueOnce({ data: {} }) // /api/auth/register
      .mockResolvedValueOnce({ data: { accessToken: 'jwt-2', username: 'bob' } }); // /api/auth/login
    const id = await performRegister('bob', 'bob@example.com', 'sekret1');
    expect(mApi.post).toHaveBeenNthCalledWith(1, '/api/auth/register', {
      username: 'bob',
      email: 'bob@example.com',
      password: 'sekret1',
    });
    expect(mApi.post).toHaveBeenNthCalledWith(2, '/api/auth/login', {
      username: 'bob',
      password: 'sekret1',
    });
    expect(store.setToken).toHaveBeenCalledWith('jwt-2');
    expect(id.username).toBe('bob');
  });

  test('performRegister surfaces a register failure and never attempts login', async () => {
    mApi.post.mockRejectedValueOnce({ response: { data: { message: 'username taken' } } });
    await expect(performRegister('bob', 'bob@example.com', 'sekret1')).rejects.toMatchObject({
      response: { data: { message: 'username taken' } },
    });
    expect(mApi.post).toHaveBeenCalledTimes(1); // register only; no login attempt
    expect(store.setToken).not.toHaveBeenCalled();
  });
});
