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
