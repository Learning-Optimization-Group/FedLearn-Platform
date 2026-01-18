// 1. Read the base URL for standard API calls from the .env file
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

// 2. Read the root server URL for other connections (like WebSockets)
const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || 'http://localhost:8081';

// 3. Derive the WebSocket URL from the server root URL
//    This replaces "http" with "ws" to create the correct protocol string.
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

// 4. Export the configured values for the rest of the application to use
export {
    API_BASE_URL,
    SERVER_ROOT_URL,
    WEBSOCKET_URL_BASE,
};