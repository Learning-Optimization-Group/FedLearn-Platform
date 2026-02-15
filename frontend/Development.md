# Frontend Development Guide

Complete guide for developers working on the FedLearn frontend.

## Table of Contents

- [Project Overview](#project-overview)
- [File Structure Explained](#file-structure-explained)
- [Component Details](#component-details)
- [Page Details](#page-details)
- [Services & API](#services--api)
- [State Management](#state-management)
- [Styling](#styling)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)

---

## Project Overview

The frontend is a React single-page application (SPA) that provides a web interface for managing federated learning experiments.

**Core Technologies**:
- React 19
- Vite 6.3 (build tool)
- React Router v7 (routing)
- Axios (HTTP client)
- STOMP.js (WebSocket)
- React Icons

---

## File Structure Explained

### `/src/api/`

#### `axiosConfig.jsx`
Axios instance configuration with interceptors.

**Key Features**:
- Base URL configuration from environment variables
- JWT token injection on all requests
- 401 handling (auto-redirect to login)
- Request/response logging (dev mode)

**Code Structure**:
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080',
  timeout: 10000,
});

// Add token to headers
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Handle unauthorized
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

**When to Modify**:
- Add global error handling
- Implement request retry logic
- Add request/response transformers
- Configure timeout per endpoint

---

### `/src/assets/`

Static files like images, logos, and icons.

**Structure**:
```
assets/
└── images/
    ├── logo_black.png
    ├── logo.png
    ├── ai.png
    └── react.svg
```

**Usage**:
```javascript
import logo from '../assets/images/logo.png';

<img src={logo} alt="Logo" />
```

---

### `/src/components/`

Reusable React components used across multiple pages.

#### `CopyIcon.jsx`
Copy-to-clipboard button component.

**Props**:
- `textToCopy`: string - Text to copy when clicked

**Usage**:
```jsx
<CopyIcon textToCopy={project.id} />
```

**Features**:
- Click to copy
- Visual feedback (icon change)
- Timeout reset

---

#### `CreateProjectModal.jsx`
Modal dialog for creating new FL projects.

**Props**:
- `isOpen`: boolean
- `onClose`: () => void
- `onCreate`: (projectData) => Promise<void>

**Form State**:
```javascript
{
  projectName: '',
  type: 'CNN',
  model: '',
  optimizer: 'Adam',
  strategy: 'FedAvg',
  rounds: 10,
  minClients: 2,
  clientsPerRound: 5
}
```

**Validation Rules**:
- Project name: required, non-empty
- Rounds: > 0
- Min clients: > 0
- Clients per round: >= min clients

**API Call**:
```javascript
const handleSubmit = async (e) => {
  e.preventDefault();
  
  try {
    await onCreate({
      name: projectName,
      type,
      model,
      optimizer,
      strategy,
      rounds: parseInt(rounds),
      minClients: parseInt(minClients),
      clientsPerRound: parseInt(clientsPerRound)
    });
    
    onClose();
  } catch (error) {
    setError(error.message);
  }
};
```

**When to Modify**:
- Add new project types
- Add new strategies
- Add custom validation
- Add more configuration options

---

#### `DiskLoader.jsx`
Loading spinner component.

**Props**:
- `size?`: number (default: 50)
- `color?`: string (default: '#4a9eff')

**Usage**:
```jsx
{isLoading && <DiskLoader size={60} color="#00ff00" />}
```

---

#### `Layout.jsx`
Main application layout with header, navigation, and footer.

**Structure**:
```jsx
<Layout>
  <header>
    <nav>
      <Link to="/dashboard">Dashboard</Link>
      <Link to="/models">Models</Link>
      {user && <span>Welcome, {user.username}</span>}
      <button onClick={logout}>Logout</button>
    </nav>
  </header>
  
  <main>{children}</main>
  
  <footer>© 2024 NeuroSphere</footer>
</Layout>
```

**When to Modify**:
- Add new navigation items
- Change header/footer styling
- Add breadcrumbs
- Implement sidebar

---

#### `LogViewer.jsx`
Real-time server log viewer using WebSocket.

**Props**:
- `projectId`: string
- `isOpen`: boolean
- `onClose`: () => void

**State**:
```javascript
{
  logs: [],              // Array of log messages
  isConnected: false,    // WebSocket connection status
  autoScroll: true       // Auto-scroll to bottom
}
```

**WebSocket Setup**:
```javascript
useEffect(() => {
  if (!isOpen) return;
  
  const client = new Client({
    brokerURL: import.meta.env.VITE_WS_URL,
    onConnect: () => {
      setIsConnected(true);
      
      client.subscribe(`/topic/logs/${projectId}`, (message) => {
        setLogs(prev => [...prev, message.body]);
      });
    },
    onDisconnect: () => setIsConnected(false),
  });
  
  client.activate();
  
  return () => client.deactivate();
}, [isOpen, projectId]);
```

**Features**:
- Auto-scroll to latest log
- Connection status indicator
- Clear logs button
- Copy all logs
- Pause auto-scroll

**When to Modify**:
- Add log filtering
- Add log search
- Add log export (CSV/JSON)
- Add syntax highlighting

---

#### `ModelCard.jsx`
Display card for saved models.

**Props**:
```javascript
{
  model: {
    id: string,
    name: string,
    type: string,
    accuracy: number,
    createdAt: string,
    size: string
  },
  onDownload: (modelId) => void,
  onDelete: (modelId) => void
}
```

**Usage**:
```jsx
<ModelCard
  model={modelData}
  onDownload={handleDownload}
  onDelete={handleDelete}
/>
```

---

#### `ProjectCard.jsx`
Display card for FL projects with controls.

**Props**:
```javascript
{
  project: {
    id: string,
    name: string,
    type: string,
    model: string,
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED',
    optimizer: string,
    strategy: string,
    rounds: number,
    minClients: number
  },
  onToggleServer: (projectId) => void,
  onViewLogs: (projectId) => void,
  onViewResults: (projectId) => void
}
```

**Key Elements**:
- Status badge (color-coded)
- Project metadata
- Run configuration display
- Action buttons
- Server toggle switch

**Status Colors**:
```javascript
const statusColors = {
  RUNNING: '#00ff00',
  STOPPED: '#ff0000',
  COMPLETED: '#0000ff'
};
```

**When to Modify**:
- Add edit project functionality
- Add delete project
- Add clone project
- Show more metadata

---

#### `ProtectedRoute.jsx`
Route wrapper for authentication.

**Code**:
```jsx
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  
  if (loading) return <DiskLoader />;
  
  if (!user) return <Navigate to="/login" replace />;
  
  return children;
}
```

**Usage in App.jsx**:
```jsx
<Route 
  path="/dashboard" 
  element={
    <ProtectedRoute>
      <DashboardPage />
    </ProtectedRoute>
  } 
/>
```

---

#### `ResultsModal.jsx`
Modal displaying training results and metrics.

**Props**:
- `projectId`: string
- `isOpen`: boolean
- `onClose`: () => void

**Displays**:
- Training accuracy per round
- Loss curves
- Client statistics
- Convergence metrics

**API Call**:
```javascript
useEffect(() => {
  if (isOpen && projectId) {
    logsService.getResults(projectId)
      .then(data => setResults(data))
      .catch(error => setError(error.message));
  }
}, [isOpen, projectId]);
```

---

### `/src/context/`

#### `AuthContext.jsx`
Global authentication state management.

**Provides**:
- `user`: Current user object (from JWT)
- `login(token)`: Login function
- `logout()`: Logout function
- `loading`: Initial load state

**Implementation**:
```jsx
import { createContext, useContext, useState, useEffect } from 'react';
import { jwtDecode } from 'jwt-decode';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = jwtDecode(token);
        
        // Check if expired
        if (decoded.exp * 1000 > Date.now()) {
          setUser(decoded);
        } else {
          localStorage.removeItem('token');
        }
      } catch (error) {
        localStorage.removeItem('token');
      }
    }
    setLoading(false);
  }, []);

  const login = (token) => {
    localStorage.setItem('token', token);
    const decoded = jwtDecode(token);
    setUser(decoded);
  };

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
    window.location.href = '/';
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
```

**Usage**:
```jsx
import { useAuth } from '../context/AuthContext';

function MyComponent() {
  const { user, login, logout } = useAuth();
  
  return (
    <div>
      {user ? (
        <>
          <p>Welcome, {user.username}</p>
          <button onClick={logout}>Logout</button>
        </>
      ) : (
        <button onClick={() => navigate('/login')}>Login</button>
      )}
    </div>
  );
}
```

---

### `/src/pages/`

Each page is a full-screen view with its own route.

#### `LandingPage.jsx`
Public landing page (marketing).

**Route**: `/`

**Sections**:
- Hero section with CTA
- Feature cards
- Call-to-action buttons

#### `LoginPage.jsx`
User login.

**Route**: `/login`

**Form Fields**:
- Email
- Password

**Flow**:
1. Submit form
2. Call `authService.login(credentials)`
3. Save token via `login(token)`
4. Redirect to `/dashboard`

#### `RegisterPage.jsx`
New user registration.

**Route**: `/register`

**Form Fields**:
- Username
- Email
- Password
- Confirm Password

#### `DashboardPage.jsx`
Main dashboard with project list.

**Route**: `/dashboard` (protected)

**Features**:
- Fetch all projects on mount
- Display project cards
- Create new project button
- Toggle server
- View logs
- View results

#### `ModelsPage.jsx`
Saved models list.

**Route**: `/models` (protected)

#### `TrainingPage.jsx`
Training progress for specific project.

**Route**: `/training/:projectId` (protected)

#### `SettingsPage.jsx`
User settings.

**Route**: `/settings` (protected)

---

### `/src/services/`

#### `apiServices.jsx`
All API calls centralized in one file.

**Structure**:
```javascript
import api from '../api/axiosConfig';

// Authentication
export const authService = {
  login: async (credentials) => {
    const response = await api.post('/api/auth/login', credentials);
    return response.data;
  },
  
  signup: async (userData) => {
    const response = await api.post('/api/auth/signup', userData);
    return response.data;
  },
  
  logout: () => {
    localStorage.removeItem('token');
    window.location.href = '/';
  }
};

// Projects
export const projectService = {
  getAll: async () => {
    const response = await api.get('/api/projects');
    return response.data;
  },
  
  getById: async (id) => {
    const response = await api.get(`/api/projects/${id}`);
    return response.data;
  },
  
  create: async (projectData) => {
    const response = await api.post('/api/projects', projectData);
    return response.data;
  },
  
  update: async (id, updates) => {
    const response = await api.put(`/api/projects/${id}`, updates);
    return response.data;
  },
  
  delete: async (id) => {
    await api.delete(`/api/projects/${id}`);
  }
};

// Server Control
export const serverService = {
  start: async (projectId) => {
    const response = await api.post(`/api/projects/${projectId}/start`);
    return response.data;
  },
  
  stop: async (projectId) => {
    const response = await api.post(`/api/projects/${projectId}/stop`);
    return response.data;
  },
  
  getStatus: async (projectId) => {
    const response = await api.get(`/api/projects/${projectId}/status`);
    return response.data;
  }
};

// Logs and Results
export const logsService = {
  getLogs: async (projectId, params = {}) => {
    const response = await api.get(`/api/projects/${projectId}/logs`, { params });
    return response.data;
  },
  
  getResults: async (projectId) => {
    const response = await api.get(`/api/projects/${projectId}/results`);
    return response.data;
  }
};
```

**When to Add New Endpoints**:
1. Add function to appropriate service object
2. Use consistent naming (getX, createX, updateX, deleteX)
3. Handle errors appropriately
4. Return response.data

---

## State Management

Currently using **React Context API** for global state:

- `AuthContext` - User authentication

**For local state**, use `useState` and `useEffect`.

**Adding New Context**:
```jsx
// src/context/ProjectContext.jsx
import { createContext, useContext, useState } from 'react';

const ProjectContext = createContext();

export function ProjectProvider({ children }) {
  const [projects, setProjects] = useState([]);
  
  const refreshProjects = async () => {
    const data = await projectService.getAll();
    setProjects(data);
  };
  
  return (
    <ProjectContext.Provider value={{ projects, setProjects, refreshProjects }}>
      {children}
    </ProjectContext.Provider>
  );
}

export const useProjects = () => useContext(ProjectContext);
```

---

## Styling

Uses **plain CSS** with modular approach.

**Global Styles**: `src/index.css`
**Component Styles**: Inline or in `src/styles/`

**Color Palette**:
```css
--bg-dark: #1a2332;
--bg-light: #f5f5f5;
--primary: #4a9eff;
--accent: #00d4ff;
--success: #00ff00;
--error: #ff0000;
--warning: #ffaa00;
```

---

## Common Patterns

### Fetching Data on Mount

```javascript
function MyComponent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    async function fetchData() {
      try {
        const result = await apiService.getData();
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    
    fetchData();
  }, []);
  
  if (loading) return <DiskLoader />;
  if (error) return <div>Error: {error}</div>;
  
  return <div>{JSON.stringify(data)}</div>;
}
```

### Form Handling

```javascript
function MyForm() {
  const [formData, setFormData] = useState({
    field1: '',
    field2: ''
  });
  
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    await apiService.submitData(formData);
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input name="field1" value={formData.field1} onChange={handleChange} />
      <input name="field2" value={formData.field2} onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
}
```

---

## Best Practices

1. **Component Files**: One component per file, PascalCase naming
2. **API Calls**: Always in `apiServices.jsx`, never inline in components
3. **Error Handling**: Always use try-catch for async operations
4. **Loading States**: Show loading indicators during data fetch
5. **PropTypes**: Add for type safety (optional but recommended)
6. **Comments**: Document complex logic
7. **Formatting**: Use consistent indentation (2 spaces)

---

For deployment and production setup, see main [README.md](README.md).