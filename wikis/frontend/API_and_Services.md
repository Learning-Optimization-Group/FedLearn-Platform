# API and Services

The frontend communicates with the FedLearn backend via REST APIs for operations and state, and WebSockets (STOMP) for real-time telemetry and logs.

## Axios Configuration

The global Axios instance is configured in `src/api/axiosConfig.ts`. It handles base URL resolution dynamically depending on the environment (dev vs prod) and injects interceptors.

```typescript
const baseURL = import.meta.env.PROD
        ? envBaseUrl
        : envBaseUrl || (typeof window !== 'undefined'
            ? `http://${window.location.hostname}:8081/api`
            : 'http://localhost:8081/api');

const api = axios.create({ baseURL, withCredentials: true });
```
This instance is exported and consumed by `apiServices.ts`, which abstracts all backend endpoints into strongly-typed functions.

## Service Abstraction Layer

All API calls are defined as named exports in `src/services/apiServices.ts`. This isolates Axios logic from the components and provides a single source of truth for Request/Response typings.

```typescript
// Example from apiServices.ts
export const fetchProjects = () => api.get('/projects');
export const createProject = (data: ProjectPayload) => api.post('/projects', data);
```

## Real-Time Logs & WebSockets

Federated learning involves distributed clients. Real-time updates are critical. We use `@stomp/stompjs` to connect to the backend's WebSocket broker.

### LogStore (Telemetry Cache)
The `logStore.ts` acts as an agnostic cache for WebSocket payloads. Because logs are pushed constantly, we decouple the React render cycle from the WebSocket listener using a publish-subscribe pattern.

1. The WebSocket listener pushes entries via `logStore.append(projectId, entry)`.
2. Components like `LogViewer` subscribe to `logStore`.
3. The store automatically trims logs to a maximum limit (e.g., 2000) to prevent memory leaks.

```typescript
// Pushing to the store
export const logStore = {
    append(projectId: string, entry: StoredLogEntry): void {
        const arr = cache.get(projectId) ?? [];
        arr.push(entry);
        if (arr.length > 2000) arr.splice(0, arr.length - 2000);
        cache.set(projectId, arr);
        emit(projectId); // Notifies React components
    }
}
```

This guarantees that if the user closes and reopens a project modal, the logs are immediately available without refetching from the backend.
