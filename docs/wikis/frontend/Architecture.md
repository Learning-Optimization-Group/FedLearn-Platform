# Frontend Architecture & State Management

## Technology Stack

The frontend is built for performance, security, and developer experience. The core technologies include:

- **Framework**: React 19 (Hooks, Functional Components)
- **Language**: TypeScript 5.7+
- **Build Tool**: Vite 6.3+ (Extremely fast HMR and optimized builds)
- **Styling**: Tailwind CSS 4.x (Utility-first CSS), standard CSS modules.
- **Routing**: React Router DOM 7.5+
- **Data Fetching**: Axios 1.11+
- **Real-time / Telemetry**: StompJS 7+ (WebSockets)
- **Charting**: Recharts 2.15+

## Directory Structure

The `src/` directory is logically partitioned to separate concerns:

```text
src/
├── api/             # Axios configuration and base API clients
├── assets/          # Static images, icons, SVGs
├── components/      # Reusable UI components
│   └── redesign/    # V2 components (modern Apple-inspired UI)
├── context/         # React Contexts (e.g., AuthContext)
├── lib/             # Utility libraries and helpers (logger, generic utils)
├── pages/           # Route-level components / Views
├── services/        # Service layer for API wrappers and stores (e.g., logStore)
└── styles/          # Global stylesheets and legacy component CSS
```

## State Management Approach

Instead of using a heavy state management library like Redux, the FedLearn frontend utilizes a combination of **React Context API** and **Service-Level Stores**:

### 1. React Context (Global UI & Auth State)
We use Context primarily for state that is required globally and changes infrequently, such as the current authenticated user.

### 2. Service-Level Stores (Domain State)
For state that needs to be accessed independently of the React component tree (e.g., by WebSocket listeners) and persists across unmounts, we use lightweight, in-memory stores. 

A prime example is the `logStore.ts`, which caches logs per project:
```typescript
// src/services/logStore.ts
const cache = new Map<string, StoredLogEntry[]>();

export const logStore = {
    get(projectId: string): StoredLogEntry[] {
        return cache.get(projectId) ?? [];
    },
    append(projectId: string, entry: StoredLogEntry): void {
        const arr = cache.get(projectId) ?? [];
        arr.push(entry);
        cache.set(projectId, arr.slice(-2000)); // Keep last 2000 logs
        emit(projectId);
    },
    // ... subscription logic
};
```
This pattern allows the `LogViewer` to unmount and remount instantly without losing the currently streaming logs.

### 3. Local Component State
Standard React `useState` and `useReducer` are used for isolated, component-specific state (e.g., form inputs, modal visibility, local toggles).
