import React, { useEffect } from 'react';
import { Routes, Route, Link, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import './App.css';
import { useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import DiskLoader from './components/DiskLoader';
import { LayoutV2 } from './components/redesign/LayoutV2';
import { DashboardV2 } from './components/redesign/DashboardV2';
import { NodeNetwork } from './components/redesign/NodeNetwork';
import { ModelsView } from './components/redesign/ModelsView';
import { DatasetsView } from './components/redesign/DatasetsView';
import { SettingsView } from './components/redesign/SettingsView';

const AppLoading: React.FC = () => (
    <div className="flex items-center justify-center h-screen bg-canvas text-fg">
        <DiskLoader message="Loading Application..." />
    </div>
);

const NotFoundPage: React.FC = () => (
    <div className="flex flex-col items-center justify-center h-screen gap-3 bg-canvas text-fg p-8 text-center font-sans">
        <h2 className="text-h2 text-fg">404 — Page Not Found</h2>
        <p className="text-body text-fg-muted">The page you're looking for doesn't exist.</p>
        <Link to="/" className="text-label font-medium text-accent hover:text-accent-hover transition-colors">
            Go Home
        </Link>
    </div>
);

function App() {
    const { currentUser, logout, isLoading } = useAuth();

    useEffect(() => {
        // The axios 401 interceptor dispatches `authError` for any data-route
        // 401 (cookie missing or rejected). We swallow the logout's promise
        // because there's nothing useful to do if the backend logout call
        // itself fails — local state is cleared either way.
        const handleAuthError = () => {
            void logout();
        };
        window.addEventListener('authError', handleAuthError);
        return () => {
            window.removeEventListener('authError', handleAuthError);
        };
    }, [logout]);

    if (isLoading) {
        return <AppLoading />;
    }

    return (
        <div className="App">
            <Routes>
                {/* Public */}
                <Route path="/" element={<LandingPage />} />
                <Route
                    path="/login"
                    element={currentUser ? <Navigate to="/dashboard" replace /> : <LoginPage />}
                />
                <Route
                    path="/register"
                    element={currentUser ? <Navigate to="/dashboard" replace /> : <RegisterPage />}
                />

                {/* Authenticated — single tokenized UI */}
                <Route element={<ProtectedRoute />}>
                    <Route element={<LayoutV2 />}>
                        <Route path="/dashboard" element={<DashboardV2 />} />
                        <Route path="/nodes" element={<NodeNetwork />} />
                        <Route path="/models" element={<ModelsView />} />
                        <Route path="/datasets" element={<DatasetsView />} />
                        <Route path="/settings" element={<SettingsView />} />
                    </Route>
                </Route>

                {/* Retired /v2 split — keep old links working */}
                <Route path="/v2" element={<Navigate to="/dashboard" replace />} />
                <Route path="/v2/nodes" element={<Navigate to="/nodes" replace />} />
                <Route path="/v2/models" element={<Navigate to="/models" replace />} />
                <Route path="/v2/datasets" element={<Navigate to="/datasets" replace />} />
                <Route path="/v2/settings" element={<Navigate to="/settings" replace />} />

                <Route path="*" element={<NotFoundPage />} />
            </Routes>
        </div>
    );
}

export default App;
