// fedlearn-desktop/src/__tests__/auth.service.test.ts
//
// DE-8: real session-expiry handling and the 401 -> re-auth path. Pins:
//   1. A 401 from a normal authenticated backend call (any main-process
//      service, since they all go through the shared `http` instance) clears
//      the stored session and fires exactly one 'auth:session-expired' push
//      to the renderer.
//   2. The auth handshake itself (login, the /auth/me probe) is excluded —
//      a 401 there must not fire the signal (no show-login -> submit -> 401
//      -> re-show-login loop).
//   3. getAuthHeader() proactively checks the stored expiresAt: an
//      already-past expiry is treated as an expired session (cleared +
//      signalled) instead of arming a request that would just 401 downstream.
//
// electron-store and safeStorage are swapped for in-memory mocks (see
// jest.config.js / src/__mocks__) so AuthService is constructible with no
// disk I/O and no real OS keychain.

import type { BrowserWindow } from 'electron';
import { safeStorage } from 'electron';
import { AuthService } from '../main/auth.service';
import { http } from '../main/http';

// Reaches past AuthService's public surface to seed/inspect stored session
// state directly — mirrors the PrivateDockerService cast pattern already
// used in docker-service.test.ts for the same reason (avoid a real network
// call just to arrange test state).
type AuthServiceInternals = {
  store: { set: (key: string, value: unknown) => void };
  storeJwt: (jwt: string, username: string) => void;
};

function asInternals(auth: AuthService): AuthServiceInternals {
  return auth as unknown as AuthServiceInternals;
}

function fakeWindow(): { win: BrowserWindow; send: jest.Mock } {
  const send = jest.fn();
  const win = {
    isDestroyed: () => false,
    webContents: { send },
  } as unknown as BrowserWindow;
  return { win, send };
}

describe('AuthService — 401 interceptor wiring (DE-8)', () => {
  beforeEach(() => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
  });

  it('clears the session and fires exactly one auth:session-expired event on a 401 from a normal call', async () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');
    expect(auth.isAuthenticated()).toBe(true);

    // Simulate the shared http instance observing a 401 on a normal endpoint —
    // exercised end to end via installUnauthorizedHandler's callback, which
    // AuthService wired to its own handleSessionExpired in its constructor.
    // validateStatus matches inference.service.ts's own override, under
    // which a 401 resolves rather than throws.
    await http.get(`${auth.getApiUrl()}/inference/models`, {
      headers: { Authorization: auth.getAuthHeader() as string },
      validateStatus: () => true,
      adapter: async (config) => ({ data: {}, status: 401, statusText: 'Unauthorized', headers: {}, config }),
    });

    expect(auth.isAuthenticated()).toBe(false);
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith('auth:session-expired');
  });

  it('does not clear or signal on a 401 from the login call itself (no loop)', async () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);

    await http.post(
      `${auth.getApiUrl()}/auth/login`,
      { username: 'alice', password: 'wrong' },
      { validateStatus: (status) => status < 500, adapter: async (config) => ({ data: {}, status: 401, statusText: 'Unauthorized', headers: {}, config }) },
    );

    expect(send).not.toHaveBeenCalled();
  });

  it('does not clear or signal on a 401 from the /auth/me probe (no loop)', async () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');

    await http.get(`${auth.getApiUrl()}/auth/me`, {
      validateStatus: () => true,
      adapter: async (config) => ({ data: {}, status: 401, statusText: 'Unauthorized', headers: {}, config }),
    });

    // Session untouched — /auth/me is a probe, not evidence of expiry.
    expect(auth.isAuthenticated()).toBe(true);
    expect(send).not.toHaveBeenCalled();
  });

  it('a second 401 after the session is already cleared does not signal again', async () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');

    const hit401 = () =>
      http.get(`${auth.getApiUrl()}/inference/models`, {
        validateStatus: () => true,
        adapter: async (config) => ({ data: {}, status: 401, statusText: 'Unauthorized', headers: {}, config }),
      });

    await hit401();
    await hit401();

    expect(send).toHaveBeenCalledTimes(1);
  });
});

describe('AuthService.getAuthHeader — proactive expiry (DE-8)', () => {
  beforeEach(() => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
  });

  it('treats an already-past on-disk expiresAt as expired: returns null, clears, and signals once', () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    // Seed a stored session whose expiresAt is already in the past — bypasses
    // storeJwt's own Date.now() + JWT_EXPIRY_MS so the test controls expiry
    // directly instead of waiting out a real 24h window.
    asInternals(auth).store.set('auth', {
      encryptedJwt: 'irrelevant-not-reached-before-expiry-check',
      expiresAt: Date.now() - 1_000,
      username: 'alice',
    });

    const header = auth.getAuthHeader();

    expect(header).toBeNull();
    expect(auth.isAuthenticated()).toBe(false);
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith('auth:session-expired');
  });

  it('treats an already-past in-memory expiresAt as expired (no-keyring fallback path)', () => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(false);
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-xyz', 'bob'); // falls back to in-memory since encryption is "unavailable"

    // Force the in-memory session's expiresAt into the past the same way —
    // reach past the private field rather than waiting out JWT_EXPIRY_MS.
    (auth as unknown as { sessionMemory: { expiresAt: number } }).sessionMemory.expiresAt = Date.now() - 1_000;

    const header = auth.getAuthHeader();

    expect(header).toBeNull();
    expect(auth.isAuthenticated()).toBe(false);
    expect(send).toHaveBeenCalledTimes(1);
  });

  it('does not clear or signal while the stored session is still within its expiry window', () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');

    const header = auth.getAuthHeader();

    expect(header).toBe('Bearer jwt-abc');
    expect(auth.isAuthenticated()).toBe(true);
    expect(send).not.toHaveBeenCalled();
  });
});

// DE-11: login/logout LIFECYCLE. The DE-8 tests above pin the 401/expiry edges; these pin the core
// state transitions — login stores the JWT (or fails cleanly), logout clears it, and the in-memory
// (no-keyring) backend round-trips the same way as the encrypted on-disk one.
describe('AuthService login/logout lifecycle (DE-11)', () => {
  beforeEach(() => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
  });

  it('login: a 200 with an accessToken stores the JWT and authenticates', async () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    const post = jest.spyOn(http, 'post').mockResolvedValue({
      status: 200, data: { accessToken: 'jwt-xyz', username: 'alice' }, headers: {},
    } as never);
    const ok = await auth.login('alice', 'pw');
    expect(ok).toBe(true);
    expect(post).toHaveBeenCalledWith(
      expect.stringContaining('/auth/login'), { username: 'alice', password: 'pw' }, expect.any(Object),
    );
    expect(auth.isAuthenticated()).toBe(true);
    expect(auth.getAuthHeader()).toBe('Bearer jwt-xyz');
    post.mockRestore();
  });

  it('login: a non-200 response returns false and stores no session', async () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    const post = jest.spyOn(http, 'post').mockResolvedValue({ status: 401, data: {}, headers: {} } as never);
    expect(await auth.login('alice', 'wrong')).toBe(false);
    expect(auth.isAuthenticated()).toBe(false);
    expect(auth.getAuthHeader()).toBeNull();
    post.mockRestore();
  });

  it('login: a 200 with no accessToken and no jwt cookie returns false', async () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    const post = jest.spyOn(http, 'post').mockResolvedValue({ status: 200, data: {}, headers: {} } as never);
    expect(await auth.login('alice', 'pw')).toBe(false);
    expect(auth.isAuthenticated()).toBe(false);
    post.mockRestore();
  });

  it('login: a thrown request error is caught and returns false (no crash)', async () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    const post = jest.spyOn(http, 'post').mockRejectedValue(new Error('ECONNREFUSED'));
    expect(await auth.login('alice', 'pw')).toBe(false);
    expect(auth.isAuthenticated()).toBe(false);
    post.mockRestore();
  });

  it('logout: clears the session so isAuthenticated() is false and getAuthHeader() is null', () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');
    expect(auth.isAuthenticated()).toBe(true);
    auth.logout();
    expect(auth.isAuthenticated()).toBe(false);
    expect(auth.getAuthHeader()).toBeNull();
  });

  it('logout is idempotent — a second logout does not throw', () => {
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');
    auth.logout();
    expect(() => auth.logout()).not.toThrow();
    expect(auth.isAuthenticated()).toBe(false);
  });

  it('in-memory fallback (safeStorage unavailable): login round-trips through memory, logout clears it', async () => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(false);
    const { win } = fakeWindow();
    const auth = new AuthService(win);
    const post = jest.spyOn(http, 'post').mockResolvedValue({
      status: 200, data: { accessToken: 'mem-jwt', username: 'bob' }, headers: {},
    } as never);
    expect(await auth.login('bob', 'pw')).toBe(true);
    expect(auth.isAuthenticated()).toBe(true);
    expect(auth.getAuthHeader()).toBe('Bearer mem-jwt');
    auth.logout();
    expect(auth.isAuthenticated()).toBe(false);
    post.mockRestore();
  });
});

// Client-audit HIGH: a JWT is minted by ONE backend and must never be sent to a different host. If the
// server URL is repointed (a compromised renderer calling setServerUrl, or a legitimate server switch),
// changing it must clear the session so the token/credentials are never delivered to the newly-set host.
describe('AuthService.setApiUrl — a server change clears the session', () => {
  beforeEach(() => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
  });

  it('clears the session and signals re-auth when the server URL changes', () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');
    expect(auth.isAuthenticated()).toBe(true);

    auth.setApiUrl('https://attacker.example/api'); // renderer repoints the API base to a foreign host

    expect(auth.isAuthenticated()).toBe(false);          // the old JWT is gone
    expect(auth.getAuthHeader()).toBeNull();             // nothing to leak to the new host
    expect(send).toHaveBeenCalledWith('auth:session-expired');
  });

  it('does not clear the session when setApiUrl keeps the same URL', () => {
    const { win, send } = fakeWindow();
    const auth = new AuthService(win);
    asInternals(auth).storeJwt('jwt-abc', 'alice');

    auth.setApiUrl(auth.getApiUrl()); // re-saving the unchanged URL must not drop the session

    expect(auth.isAuthenticated()).toBe(true);
    expect(send).not.toHaveBeenCalled();
  });
});

describe('AuthService — saved credentials ("Save password")', () => {
  beforeEach(() => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
    (safeStorage.encryptString as jest.Mock).mockImplementation((v: string) => Buffer.from(v, 'utf8'));
    (safeStorage.decryptString as jest.Mock).mockImplementation((b: Buffer) => b.toString('utf8'));
  });

  it('round-trips encrypted credentials through save/get', () => {
    const auth = new AuthService();
    expect(auth.saveCredentials('alice', 's3cret')).toBe(true);
    expect(auth.getSavedCredentials()).toEqual({ username: 'alice', password: 's3cret' });
  });

  it('returns null when nothing is saved', () => {
    const auth = new AuthService();
    expect(auth.getSavedCredentials()).toBeNull();
  });

  it('clearSavedCredentials forgets the saved pair', () => {
    const auth = new AuthService();
    auth.saveCredentials('alice', 's3cret');
    auth.clearSavedCredentials();
    expect(auth.getSavedCredentials()).toBeNull();
  });

  it('refuses to persist (and returns false) when OS encryption is unavailable', () => {
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(false);
    const auth = new AuthService();
    expect(auth.saveCredentials('alice', 's3cret')).toBe(false);
    // With encryption back on there is still nothing stored — it was never written.
    (safeStorage.isEncryptionAvailable as jest.Mock).mockReturnValue(true);
    expect(auth.getSavedCredentials()).toBeNull();
  });

  it('scrubs and returns null when the stored blob cannot be decrypted', () => {
    const auth = new AuthService();
    auth.saveCredentials('alice', 's3cret');
    (safeStorage.decryptString as jest.Mock).mockImplementationOnce(() => {
      throw new Error('keychain changed');
    });
    expect(auth.getSavedCredentials()).toBeNull();
    // The stale blob was scrubbed, so a subsequent read is also null.
    expect(auth.getSavedCredentials()).toBeNull();
  });
});
