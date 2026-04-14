import axios, { AxiosError } from 'axios';

const envBaseUrl = import.meta.env.VITE_API_BASE_URL;

if (import.meta.env.PROD && !envBaseUrl) {
    throw new Error('VITE_API_BASE_URL must be set for production builds');
}

const baseURL =
    envBaseUrl ||
    (typeof window !== 'undefined'
        ? `http://${window.location.hostname}:8081/api`
        : 'http://localhost:8081/api');

const api = axios.create({
    baseURL,
    withCredentials: true, // Uses HttpOnly cookie for session auth
});

// Attach the access token as a Bearer header when available (Electron / LAN fallback).
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('jwtToken');
    if (token) {
        config.headers = config.headers ?? {};
        (config.headers as Record<string, string>).Authorization = `Bearer ${token}`;
    }
    return config;
});

// Response interceptor - handles auth errors globally
api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        if (error.response && (error.response.status === 401 || error.response.status === 403)) {
            // Do not force a page reload if the user is explicitly trying to log in
            if (error.config && !error.config.url?.includes('/auth/login')) {
                localStorage.removeItem('jwtToken');
                window.dispatchEvent(new Event('authError'));
                window.location.href = '/login';
            }
        }
        return Promise.reject(error);
    }
);

export default api;
export { baseURL };
