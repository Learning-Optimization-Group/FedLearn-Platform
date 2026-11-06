// REST client for the control plane (/api/*). Bearer-token auth contract via authStore.
// The base URL comes from app config — call configureApi(...) once at startup.
// On 401 (excluding the silent /auth/me probe), clears the stored token and signals the
// registered auth-lost handler so AuthContext can redirect to Login.
import axios, { AxiosHeaders, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios';
import { clearSessionIfMatches, getSession, type AuthSessionIdentity } from './authStore';
import { normalizeBackendOrigin, requestOrigin } from './backendOrigin';
import {
  beginLogout, captureAuthAdmission, getBackendScope, isCurrentAuthAdmission,
  publishBackendOrigin, type AuthAdmission,
} from './backendScope';

interface SessionRequest extends InternalAxiosRequestConfig {
  __fedlearnSession?: { session: AuthSessionIdentity; admission: AuthAdmission };
}

// Native-client marker (SE-9): the backend accepts `Authorization: Bearer` only when the
// request also carries this header — browsers stay strictly cookie-only. The value is a
// plain client identifier (an intent signal), NOT a secret.
export const NATIVE_CLIENT_HEADER = 'X-FedLearn-Client';
export const NATIVE_CLIENT_VALUE = 'fedlearn-mobile';

export const api: AxiosInstance = axios.create({
  timeout: 15000,
  withCredentials: false,
  headers: { 'Content-Type': 'application/json' },
});
// Shared default — every outbound request identifies itself as the native mobile client.
api.defaults.headers.common[NATIVE_CLIENT_HEADER] = NATIVE_CLIENT_VALUE;

export function configureApi(baseUrl: string): void {
  const origin = normalizeBackendOrigin(baseUrl);
  api.defaults.baseURL = origin;
  publishBackendOrigin(origin);
}

let authLostHandler: (() => void) | null = null;
/** Registered by AuthContext so a 401 can route the app back to Login. */
export function setAuthLostHandler(fn: (() => void) | null): void {
  authLostHandler = fn;
}

// Attach the Bearer token (async getToken is allowed — axios v1.x supports Promise request interceptors).
api.interceptors.request.use(async (config) => {
  config.withCredentials = false;
  config.headers = AxiosHeaders.from(config.headers);
  config.headers.delete('Authorization');
  config.headers.delete('Cookie');
  const scoped = config as SessionRequest;
  delete scoped.__fedlearnSession;
  const scope = getBackendScope();
  const destination = requestOrigin(config.baseURL, config.url);
  if (scope && destination === scope.origin) {
    const admission = captureAuthAdmission();
    const session = await getSession(scope.origin);
    if (!isCurrentAuthAdmission(admission)) throw new Error('The server session changed. Try again.');
    if (session) {
      config.headers.set('Authorization', `Bearer ${session.token}`);
      scoped.__fedlearnSession = { session: { origin: session.origin, revision: session.revision }, admission };
    }
  }
  return config;
});

// On 401 (except the silent /auth/me probe), drop the token and signal auth-lost.
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const url: string = error?.config?.url ?? '';
    const presented = (error?.config as SessionRequest | undefined)?.__fedlearnSession;
    if (status === 401 && !url.endsWith('/auth/me') && presented) {
      const removed = await clearSessionIfMatches(presented.session);
      if (removed && isCurrentAuthAdmission(presented.admission)) {
        beginLogout();
        authLostHandler?.();
      }
    }
    return Promise.reject(error);
  },
);
