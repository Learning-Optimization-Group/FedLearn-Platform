// =============================================================================
// FedLearn Desktop — shared backend HTTP client (Main process)
// =============================================================================
// Single axios instance for every backend REST call. The backend scopes
// `Authorization: Bearer` acceptance to native clients (SE-9): a Bearer token
// is honored only when the request also carries the X-FedLearn-Client marker
// header — browsers stay strictly cookie-only. The marker is a plain client
// identifier (an intent signal), NOT a secret.
//
// All main-process services must call the backend through this instance so the
// marker rides on every request as a shared default, never per-call.
//
// DE-8: this instance also carries the single active 401 handler for the whole
// app (installUnauthorizedHandler, below) — session-expiry detection lives here
// so every service (inference, client-projects, inference-stream, ...) gets it
// for free just by calling through `http`, with no per-call wiring.
// =============================================================================

import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';

export const NATIVE_CLIENT_HEADER = 'X-FedLearn-Client';
export const NATIVE_CLIENT_VALUE = 'fedlearn-desktop';

export const http: AxiosInstance = axios.create();
http.defaults.headers.common[NATIVE_CLIENT_HEADER] = NATIVE_CLIENT_VALUE;

// The auth handshake itself — logging in, and the /auth/me session probe.
// A 401 from either is an expected "wrong credentials" / "not logged in yet"
// outcome, not evidence of an existing session going stale. Treating it as a
// session-expiry signal would loop: show login -> submit -> 401 -> re-show
// login. Matches the equivalent exclusion in frontend/src/api/axiosConfig.ts.
const AUTH_HANDSHAKE_PATH_PATTERN = /\/auth\/(login|me)(?:[/?#]|$)/;

/** True when `url` is the login endpoint or the /auth/me probe (any host). */
export function isAuthHandshakeRequest(url: string | undefined): boolean {
  if (!url) return false;
  return AUTH_HANDSHAKE_PATH_PATTERN.test(url);
}

let ejectPreviousInterceptor: (() => void) | null = null;

/**
 * Installs the single active 401 handler on the shared `http` instance.
 * `onUnauthorized` fires for a 401 response on any request EXCEPT the auth
 * handshake (see {@link isAuthHandshakeRequest}) — whether that 401 surfaces
 * as a resolved response (a call using a permissive `validateStatus`, as every
 * current service does) or as a rejected promise (axios's own default
 * `validateStatus`, 2xx-only).
 *
 * Idempotent: calling this again (e.g. AuthService re-constructed) ejects the
 * previous handler first, so exactly one is ever active — never stacked.
 */
export function installUnauthorizedHandler(onUnauthorized: () => void): void {
  if (ejectPreviousInterceptor) {
    ejectPreviousInterceptor();
    ejectPreviousInterceptor = null;
  }

  const maybeFire = (status: number | undefined, url: string | undefined): void => {
    if (status === 401 && !isAuthHandshakeRequest(url)) {
      onUnauthorized();
    }
  };

  const id = http.interceptors.response.use(
    (response: AxiosResponse) => {
      maybeFire(response.status, response.config?.url);
      return response;
    },
    (error: AxiosError) => {
      maybeFire(error.response?.status, error.config?.url);
      return Promise.reject(error);
    },
  );

  ejectPreviousInterceptor = () => http.interceptors.response.eject(id);
}
