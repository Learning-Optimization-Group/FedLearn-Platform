// REST client for the control plane (/api/*). Bearer-token auth contract via authStore.
// The base URL comes from app config — call configureApi(...) once at startup.
// On 401 (excluding the silent /auth/me probe), clears the stored token and signals the
// registered auth-lost handler so AuthContext can redirect to Login.
import axios, { type AxiosInstance } from 'axios';
import { getToken, clearToken } from './authStore';

export const api: AxiosInstance = axios.create({
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

export function configureApi(baseUrl: string): void {
  if (!baseUrl) throw new Error('configureApi: FEDLEARN_API_URL is required');
  api.defaults.baseURL = baseUrl;
}

let authLostHandler: (() => void) | null = null;
/** Registered by AuthContext so a 401 can route the app back to Login. */
export function setAuthLostHandler(fn: () => void): void {
  authLostHandler = fn;
}

// Attach the Bearer token (async getToken is allowed — axios v1.x supports Promise request interceptors).
api.interceptors.request.use(async (config) => {
  const token = await getToken();
  if (token) {
    config.headers = config.headers ?? {};
    (config.headers as Record<string, string>).Authorization = `Bearer ${token}`;
  }
  return config;
});

// On 401 (except the silent /auth/me probe), drop the token and signal auth-lost.
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const url: string = error?.config?.url ?? '';
    if (status === 401 && !url.endsWith('/auth/me')) {
      await clearToken();
      authLostHandler?.();
    }
    return Promise.reject(error);
  },
);
