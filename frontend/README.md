# FedLearn Platform - Frontend

Web-based dashboard for managing and monitoring federated learning experiments. Built with React and Vite, this frontend provides an intuitive interface for creating FL projects, configuring training parameters, and viewing real-time server logs.

## Features

- 🚀 **Project Management** - Create and manage FL training projects
- ⚙️ **Server Configuration** - Configure strategies, rounds, and client parameters
- 📊 **Live Monitoring** - Real-time server logs via WebSocket
- 🔐 **Authentication** - JWT-based user authentication
- 📱 **Responsive Design** - Works across desktop and tablet devices

## Tech Stack

- **Framework**: React 19
- **Build Tool**: Vite 6.3
- **Routing**: React Router v7
- **HTTP Client**: Axios
- **WebSocket**: STOMP.js (@stomp/stompjs v7.1)
- **Icons**: React Icons
- **Styling**: Custom CSS

## Project Structure

```
frontend/
├── public/                     # Static assets
├── src/
│   ├── api/                   # API configuration
│   │   └── axiosConfig.jsx   # Axios instance setup
│   ├── assets/               # Images and static files
│   │   └── images/           # Logo, icons
│   ├── components/           # Reusable React components
│   │   ├── CopyIcon.jsx
│   │   ├── CreateProjectModal.jsx
│   │   ├── DiskLoader.jsx
│   │   ├── Layout.jsx
│   │   ├── LogViewer.jsx
│   │   ├── ModelCard.jsx
│   │   ├── ProjectCard.jsx
│   │   ├── ProtectedRoute.jsx
│   │   └── ResultsModal.jsx
│   ├── context/              # React Context for state
│   │   └── AuthContext.jsx  # Authentication state
│   ├── pages/                # Page components
│   │   ├── DashboardPage.jsx
│   │   ├── HomePage.jsx
│   │   ├── LandingPage.jsx
│   │   ├── LoginPage.jsx
│   │   ├── ModelsPage.jsx
│   │   ├── RegisterPage.jsx
│   │   ├── SettingsPage.jsx
│   │   └── TrainingPage.jsx
│   ├── services/             # API service layer
│   │   └── apiServices.jsx  # All API calls
│   ├── styles/               # CSS stylesheets
│   ├── App.jsx               # Main app component with routes
│   ├── App.css               # App-level styles
│   ├── main.jsx              # Entry point
│   └── index.css             # Global styles
├── .gitignore
├── eslint.config.js
├── index.html
├── package.json
├── package-lock.json
├── README.md
├── vercel.json               # Vercel deployment config
└── vite.config.js            # Vite configuration
```

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend server running (Spring Boot API at `http://localhost:8080`)

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_API_BASE_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080/ws
VITE_APP_NAME=NeuroSphere
EOF

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Available Scripts

```bash
npm run dev      # Start development server (Vite)
npm run build    # Build for production
npm run preview  # Preview production build locally
npm run lint     # Run ESLint
```

## Configuration

### Environment Variables

Create `.env` in the frontend root:

```env
# Backend API URL
VITE_API_BASE_URL=http://localhost:8080

# WebSocket URL for live logs
VITE_WS_URL=ws://localhost:8080/ws

# App Configuration
VITE_APP_NAME=NeuroSphere
```

**Important**: All environment variables must be prefixed with `VITE_` to be accessible in the application.

### Axios Configuration

The Axios instance is configured in `src/api/axiosConfig.jsx`:

```javascript
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

```

## Key Pages

### 1. LandingPage.jsx
Marketing landing page with feature showcase and CTA buttons.

**Route**: `/`

**Features**:
- Hero section with platform description
- Feature cards (Federated Training, On-Demand Server, etc.)
- Navigation to Sign In/Sign Up

### 2. LoginPage.jsx & RegisterPage.jsx
Authentication pages for user login and registration.

**Routes**: `/login`, `/register`

**Features**:
- Form validation
- JWT token storage
- Error handling
- Redirect to dashboard on success

### 3. DashboardPage.jsx
Main dashboard showing all FL projects.

**Route**: `/dashboard` (protected)

**Features**:
- List all user projects
- Create new project button
- Project cards with configuration
- Server start/stop controls
- Live logs viewer
- Results modal

### 4. ModelsPage.jsx
View and manage saved models.

**Route**: `/models` (Under development)

**Features**:
- List trained models
- Download model checkpoints
- View model metadata

### 5. TrainingPage.jsx
Training progress and metrics visualization.

**Route**: `/training/:projectId` (Under development)

### 6. SettingsPage.jsx
User settings and preferences.

**Route**: `/settings` (Under development)

## Key Components

### ProjectCard.jsx
Displays individual FL project with controls.

**Props**:
```javascript
{
  project: {
    id: string,
    name: string,
    type: string,
    model: string,
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED',
    strategy: string,
    rounds: number,
    minClients: number
  },
  onToggleServer: (projectId) => void,
  onViewLogs: (projectId) => void,
  onViewResults: (projectId) => void
}
```

### CreateProjectModal.jsx
Modal for creating new FL projects.

<!-- **Props**:
```javascript
{
  isOpen: boolean,
  onClose: () => void,
  onCreate: (projectData) => void
}
``` -->

**Form Fields**:
- Project name
- Type (CNN, Transformer, etc.)
- Model architecture
- Optimizer
- Strategy (FedAvg, DeComFL)
- Number of rounds
- Minimum clients

### LogViewer.jsx
Real-time server log viewer using WebSocket.

<!-- **Props**:
```javascript
{
  projectId: string,
  isOpen: boolean,
  onClose: () => void
}
``` -->

**Features**:
- WebSocket connection to `/topic/logs/{projectId}`
- Auto-scroll to latest log
- Connection status indicator
- Clear logs button

### ResultsModal.jsx
Displays training results and metrics.

<!-- **Props**:
```javascript
{
  projectId: string,
  isOpen: boolean,
  onClose: () => void
} -->
```

### ProtectedRoute.jsx
Route wrapper for authentication.

**Usage**:
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

### Layout.jsx
Main layout component with header and navigation.

## Services

### apiServices.jsx

All API calls are centralized in `src/services/apiServices.jsx`:

```javascript
import api from '../api/axiosConfig';

export const authService = {
  login: (credentials) => api.post('/api/auth/login', credentials),
  signup: (userData) => api.post('/api/auth/signup', userData),
  logout: () => {
    localStorage.removeItem('token');
    window.location.href = '/';
  }
};

export const projectService = {
  getAll: () => api.get('/api/projects'),
  getById: (id) => api.get(`/api/projects/${id}`),
  create: (data) => api.post('/api/projects', data),
  update: (id, data) => api.put(`/api/projects/${id}`, data),
  delete: (id) => api.delete(`/api/projects/${id}`)
};

export const serverService = {
  start: (projectId) => api.post(`/api/projects/${projectId}/start`),
  stop: (projectId) => api.post(`/api/projects/${projectId}/stop`),
  getStatus: (projectId) => api.get(`/api/projects/${projectId}/status`)
};

export const logsService = {
  get: (projectId) => api.get(`/api/projects/${projectId}/logs`),
  getResults: (projectId) => api.get(`/api/projects/${projectId}/results`)
};
```

## Authentication Flow

### Context Setup (AuthContext.jsx)

```javascript
import { createContext, useState, useContext, useEffect } from 'react';
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
        setUser(decoded);
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
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
```

### Usage in Components

```javascript
import { useAuth } from '../context/AuthContext';

function DashboardPage() {
  const { user, logout } = useAuth();
  
  return (
    <div>
      <p>Welcome, {user?.username}</p>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

## Routing (App.jsx)

```javascript
// Your App.jsx with the fix applied

import React, { useState, useEffect } from 'react';
import { Routes, Route, Link, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ModelsPage from './pages/ModelsPage';
import TrainingPage from './pages/TrainingPage';
import SettingsPage from './pages/SettingsPage';
import './App.css'
// ... other imports
import Layout from './components/Layout';
import { useAuth } from './context/AuthContext.jsx';
import ProtectedRoute from './components/ProtectedRoute.jsx';
import LandingPage from './pages/LandingPage';

// You might want a dedicated loading component for a better user experience
const AppLoading = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <h2>Loading Application...</h2>
  </div>
);


function App() {
  // Destructure the new isLoading state from your context
  const { currentUser, logout, isLoading } = useAuth();

  useEffect(() => {
    const handleAuthError = () => {
      logout();
    };
    window.addEventListener('authError', handleAuthError);
    return () => {
      window.removeEventListener('authError', handleAuthError);
    };
  }, [logout]);

  // --- THE FIX ---
  // While the AuthContext is performing its initial check, show a loading screen.
  // This prevents the router from rendering prematurely.
  if (isLoading) {
    return <AppLoading />;
  }

  // Once isLoading is false, the rest of your app renders with the correct currentUser state.
  return (
    <div className="App">
      <Routes>
        {/* Your routing logic below remains IDENTICAL and is now free of race conditions */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={currentUser ? <Navigate to="/dashboard" /> : <LoginPage />} />
        <Route path="/register" element={currentUser ? <Navigate to="/dashboard" /> : <RegisterPage />} />

        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/models" element={<ModelsPage />} />
            {/* <Route path="/training" element={<TrainingPage />} /> */}
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Route>

        <Route path="*" element={<div><h2>404 Page Not Found</h2><Link to="/">Go Home</Link></div>} />
      </Routes>
    </div>
  );
}

export default App;
```

## Development Guide

### Adding a New Page

1. **Create page component**:
```bash
# Create new file in src/pages/
touch src/pages/NewPage.jsx
```

```jsx
// src/pages/NewPage.jsx
export default function NewPage() {
  return (
    <div>
      <h1>New Page</h1>
    </div>
  );
}
```

2. **Add route** in `App.jsx`:
```jsx
import NewPage from './pages/NewPage';

<Route path="/new-page" element={<NewPage />} />
```

3. **Add navigation** (if needed):
```jsx
<Link to="/new-page">New Page</Link>
```

### Adding a New Component

1. **Create component**:
```bash
touch src/components/NewComponent.jsx
```

```jsx
// src/components/NewComponent.jsx
export default function NewComponent({ prop1, prop2 }) {
  return (
    <div>
      {prop1} - {prop2}
    </div>
  );
}
```

2. **Import and use**:
```jsx
import NewComponent from '../components/NewComponent';

<NewComponent prop1="value" prop2="data" />
```

### Adding API Endpoints

1. **Add to `apiServices.jsx`**:
```javascript
export const newService = {
  getData: () => api.get('/api/new-endpoint'),
  postData: (data) => api.post('/api/new-endpoint', data)
};
```

2. **Use in component**:
```jsx
import { newService } from '../services/apiServices';

const data = await newService.getData();
```

## Deployment

### Vercel (Current Production)

The frontend is deployed on Vercel at: [Your Vercel URL]

**Automatic Deployment**:
- Push to `main` branch → Auto-deploy to production
- Pull requests → Deploy preview URLs

**Manual Deployment**:
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

**vercel.json** configuration:
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Build and run**:
```bash
docker build -t fedlearn-frontend .
docker run -p 80:80 fedlearn-frontend
```

### Manual Static Hosting

```bash
# Build
npm run build

# Deploy dist/ folder to:
# - Netlify
# - AWS S3 + CloudFront
# - GitHub Pages
# - Any static host
```

## Troubleshooting

### Issue: API calls failing (CORS)

**Check**:
1. Backend CORS configuration allows frontend origin
2. `VITE_API_BASE_URL` is correct in `.env`

**Backend CORS (Spring Boot)**:
```java
@Configuration
public class CorsConfig {
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                    .allowedOrigins("http://localhost:5173", "https://your-vercel-url.vercel.app")
                    .allowedMethods("*")
                    .allowCredentials(true);
            }
        };
    }
}
```

### Issue: WebSocket not connecting

**Solution**:
1. Verify backend WebSocket endpoint is running
2. Check `VITE_WS_URL` in `.env`
3. Ensure firewall/proxy allows WebSocket

### Issue: Token expired errors

**Solution**: Token expiration is handled automatically. User will be redirected to login.

### Issue: Build fails

```bash
# Clear cache
rm -rf node_modules package-lock.json

# Reinstall
npm install

# Build
npm run build
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Backend Dependencies

This frontend requires the Spring Boot backend with:

- `/api/auth/**` - Authentication endpoints
- `/api/projects/**` - Project management
- `/ws` - WebSocket endpoint for live logs

## Contributing

When modifying the frontend:

1. Follow existing component structure
2. Keep components in `src/components/`
3. Keep pages in `src/pages/`
4. Centralize API calls in `src/services/apiServices.jsx`
5. Use AuthContext for authentication state
6. Test with backend integration before PR

## Performance Tips

- Use React.lazy() for code splitting
- Optimize images in `src/assets/`
- Minimize bundle size with tree shaking
- Use production build for deployment

## Security

- JWT tokens in localStorage (consider httpOnly cookies for production)
- All API calls use HTTPS in production
- Input validation on forms
- XSS protection via React

## Next Steps

- **Backend Integration**: See [Backend README](../backend/README.md)
- **Framework Usage**: See [Framework README](../framework/README.md)

---

<!-- **Live Demo**: [Your Vercel URL] -->
**Repository**: [GitHub URL]