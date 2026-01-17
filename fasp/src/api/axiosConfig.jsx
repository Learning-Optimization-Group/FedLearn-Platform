import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    },
});


console.log('API_BASE_URL:', API_BASE_URL);

// api.defaults.headers.common['ngrok-skil-browser-warning'] = 'true'

api.interceptors.request.use(
    (config) => {
        // 1. Get the token from localStorage at the moment the request is being made.
        const token = localStorage.getItem('jwtToken');

        // 2. If the token exists, add it to the Authorization header.
        console.log("Sending request with token:", token);
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }

        // 3. Return the modified config to be sent.
        return config;
    },
    (error) => {
        // Handle any request errors.
        return Promise.reject(error);
    }
);


// Optional but recommended: Add the response error interceptor for auto-logout
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && (error.response.status === 401 || error.response.status === 403)) {
            // If we get an auth error, clear the token and reload to the landing page.
            // This handles expired tokens automatically.
            localStorage.removeItem('jwtToken');
            window.location.href = '/login'; // Or just '/'
        }
        return Promise.reject(error);
    }
);


export default api;