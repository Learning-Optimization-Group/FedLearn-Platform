// fedlearn-desktop/src/__tests__/httpAuthInterceptor.test.ts
//
// DE-8: the shared `http` axios instance must treat a 401 from any
// authenticated backend call as a session-expiry signal — except the auth
// handshake itself (POST /auth/login, GET /auth/me), where a 401 is an
// expected "wrong credentials" / "not logged in yet" outcome, not evidence of
// an existing session going stale. Getting that exclusion wrong loops:
// show login -> submit -> 401 -> re-show login.
//
// These tests exercise the interceptor in isolation (no AuthService, no
// electron-store) via a stub axios adapter — see nativeClientHeader.test.ts
// for the same no-network-adapter pattern used elsewhere in this suite.

import { AxiosError } from 'axios';
import { http, installUnauthorizedHandler, isAuthHandshakeRequest } from '../main/http';

/**
 * A stub adapter that mimics the one piece of real axios/adapter behaviour
 * these tests depend on: settling resolve-vs-reject by running the merged
 * `config.validateStatus` against `status`, exactly like the real http/xhr
 * adapters' internal `settle()` does. Without this, a bare stub that always
 * resolves can never exercise the interceptor's onRejected branch — axios
 * always hands `config.validateStatus` a concrete function by adapter time
 * (the instance default, merged in, when no per-call override is given).
 */
function stubAdapter(status: number) {
  return async (config: import('axios').InternalAxiosRequestConfig) => {
    const response = { data: {}, status, statusText: status === 200 ? 'OK' : 'Unauthorized', headers: {}, config };
    if (config.validateStatus?.(status)) {
      return response;
    }
    throw new AxiosError('Request failed', AxiosError.ERR_BAD_REQUEST, config, undefined, response);
  };
}

describe('isAuthHandshakeRequest', () => {
  it('matches the login endpoint and the /auth/me probe on any host', () => {
    expect(isAuthHandshakeRequest('http://localhost:8081/api/auth/login')).toBe(true);
    expect(isAuthHandshakeRequest('https://fedlearn.duckdns.org/api/auth/me')).toBe(true);
  });

  it('does not match other authenticated endpoints', () => {
    expect(isAuthHandshakeRequest('http://localhost:8081/api/inference/models')).toBe(false);
    expect(isAuthHandshakeRequest('http://localhost:8081/api/client/projects')).toBe(false);
    expect(isAuthHandshakeRequest('http://localhost:8081/api/auth/set-server-url')).toBe(false);
  });

  it('handles a missing url without throwing', () => {
    expect(isAuthHandshakeRequest(undefined)).toBe(false);
  });
});

describe('installUnauthorizedHandler', () => {
  afterEach(() => {
    // Leave no dangling handler for the next test file's import of `http`
    // (the instance is a module-level singleton for the whole process).
    installUnauthorizedHandler(() => {});
  });

  it('fires on a 401 resolved response from a normal authenticated endpoint', async () => {
    const onUnauthorized = jest.fn();
    installUnauthorizedHandler(onUnauthorized);

    await http.get('http://localhost:8081/api/inference/models', {
      validateStatus: () => true, // matches inference.service.ts's own override
      adapter: stubAdapter(401),
    });

    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it('fires exactly once when a 401 surfaces as a rejected promise (default validateStatus)', async () => {
    const onUnauthorized = jest.fn();
    installUnauthorizedHandler(onUnauthorized);

    await expect(
      http.get('http://localhost:8081/api/client/projects', { adapter: stubAdapter(401) }),
    ).rejects.toBeTruthy();

    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it('does NOT fire on a 401 from /auth/login (no loop on bad credentials)', async () => {
    const onUnauthorized = jest.fn();
    installUnauthorizedHandler(onUnauthorized);

    await http.post(
      'http://localhost:8081/api/auth/login',
      { username: 'x', password: 'wrong' },
      { validateStatus: (status) => status < 500, adapter: stubAdapter(401) },
    );

    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('does NOT fire on a 401 from the /auth/me probe (no loop on a not-yet-authenticated check)', async () => {
    const onUnauthorized = jest.fn();
    installUnauthorizedHandler(onUnauthorized);

    await http.get('http://localhost:8081/api/auth/me', {
      validateStatus: () => true,
      adapter: stubAdapter(401),
    });

    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('does not fire on a 200', async () => {
    const onUnauthorized = jest.fn();
    installUnauthorizedHandler(onUnauthorized);

    await http.get('http://localhost:8081/api/inference/models', { adapter: stubAdapter(200) });

    expect(onUnauthorized).not.toHaveBeenCalled();
  });

  it('replaces the previous handler instead of stacking (idempotent registration)', async () => {
    const first = jest.fn();
    const second = jest.fn();
    installUnauthorizedHandler(first);
    installUnauthorizedHandler(second);

    await http.get('http://localhost:8081/api/inference/models', {
      validateStatus: () => true,
      adapter: stubAdapter(401),
    });

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledTimes(1);
  });
});
