import axios, { AxiosError, AxiosRequestConfig } from 'axios';

const envBaseUrl = import.meta.env.VITE_FEDLEARN_API_URL;

if (import.meta.env.PROD && !envBaseUrl) {
    throw new Error('VITE_FEDLEARN_API_URL must be set for production builds');
}

// Local-dev fallback only; production builds fail fast above.
const baseURL =
    import.meta.env.PROD
        ? envBaseUrl
        : envBaseUrl || (typeof window !== 'undefined'
            ? `http://${window.location.hostname}:8081/api`
            : 'http://localhost:8081/api');

/**
 * The auth contract is cookie-only: the backend issues an HttpOnly
 * `jwtToken` cookie on /auth/login. {@code withCredentials: true} ensures
 * the browser sends it on every same-/cross-origin request (the latter
 * requires the backend's CORS to allow credentials, which it does).
 *
 * We intentionally do NOT attach a Bearer header from localStorage anymore —
 * keeping the token out of JavaScript-readable storage closes the XSS
 * exfiltration vector that plagued the previous design.
 */
const api = axios.create({
    baseURL,
    withCredentials: true,
});

// Endpoints whose 401 means "no session" rather than "session expired".
// We probe these from AuthContext on bootstrap; surfacing the 401 as an
// authError event would cause an immediate redirect loop.
const SILENT_401_ENDPOINTS = ['/auth/me'];

function isSilent401(config?: AxiosRequestConfig): boolean {
    if (!config?.url) return false;
    return SILENT_401_ENDPOINTS.some((path) => config.url!.includes(path));
}

api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        const status = error.response?.status;

        // 401 = no/invalid session → log the user out everywhere except for
        //       the explicit login attempt and the silent /me probe.
        if (status === 401
            && !error.config?.url?.includes('/auth/login')
            && !isSilent401(error.config)) {
            window.dispatchEvent(new Event('authError'));
        }

        // 403 = authenticated but unauthorized for THIS specific resource
        //       (e.g. a non-admin hitting /users, or a non-owner hitting
        //       another user's project). Don't log out — let the calling
        //       component decide how to render the failure.

        return Promise.reject(error);
    }
);

export default api;
export { baseURL };
