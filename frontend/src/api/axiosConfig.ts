import axios, { AxiosError } from 'axios';

const api = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081/api',
    withCredentials: true // Forces browser to use secure HttpOnly cookies
});


// Response interceptor - handles auth errors globally
api.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
        if (error.response && (error.response.status === 401 || error.response.status === 403)) {
            localStorage.removeItem('jwtToken');
            window.dispatchEvent(new Event('authError'));
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

export default api;
