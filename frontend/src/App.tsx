import React, { useEffect } from 'react';
import { Routes, Route, Link, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ModelsPage from './pages/ModelsPage';
import TrainingPage from './pages/TrainingPage';
import SettingsPage from './pages/SettingsPage';
import ClientsPage from './pages/ClientsPage';
import './App.css';
import Layout from './components/Layout';
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
import { TrainingInsightsView } from './components/redesign/TrainingInsightsView';
import DiscoverPage from './pages/DiscoverPage';
import MyRequestsPage from './pages/MyRequestsPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import AdminUsersPage from './pages/AdminUsersPage';
import AdminProjectsPage from './pages/AdminProjectsPage';

const AppLoading: React.FC = () => (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <DiskLoader message="Loading Application..." />
    </div>
);

const NotFoundPage: React.FC = () => (
    <div style={{ padding: '2rem', textAlign: 'center' }}>
        <h2>404 - Page Not Found</h2>
        <p>The page you're looking for doesn't exist.</p>
        <Link to="/">Go Home</Link>
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
                <Route path="/" element={<LandingPage />} />
                <Route
                    path="/login"
                    element={currentUser ? <Navigate to="/dashboard" replace /> : <LoginPage />}
                />
                <Route
                    path="/register"
                    element={currentUser ? <Navigate to="/dashboard" replace /> : <RegisterPage />}
                />

                <Route element={<ProtectedRoute />}>
                    {/* Redesigned UI is now the default protected experience. */}
                    <Route element={<LayoutV2 />}>
                        <Route path="/dashboard" element={<DashboardV2 />} />
                        <Route path="/clients" element={<NodeNetwork />} />
                        <Route path="/models" element={<ModelsView />} />
                        <Route path="/training" element={<TrainingInsightsView />} />
                        <Route path="/datasets" element={<DatasetsView />} />
                        <Route path="/settings" element={<SettingsView />} />
                        <Route path="/discover" element={<DiscoverPage />} />
                        <Route path="/my/requests" element={<MyRequestsPage />} />
                        <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
                        <Route path="/admin/users" element={<AdminUsersPage />} />
                        <Route path="/admin/projects" element={<AdminProjectsPage />} />

                        {/* Backward-compatible aliases for existing v2 deep links. */}
                        <Route path="/v2" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/v2/nodes" element={<Navigate to="/clients" replace />} />
                        <Route path="/v2/models" element={<Navigate to="/models" replace />} />
                        <Route path="/v2/training" element={<Navigate to="/training" replace />} />
                        <Route path="/v2/datasets" element={<Navigate to="/datasets" replace />} />
                        <Route path="/v2/settings" element={<Navigate to="/settings" replace />} />
                    </Route>

                    {/* Legacy UI remains available under /legacy/* routes. */}
                    <Route element={<Layout />}>
                        <Route path="/legacy/dashboard" element={<DashboardPage />} />
                        <Route path="/legacy/clients" element={<ClientsPage />} />
                        <Route path="/legacy/models" element={<ModelsPage />} />
                        <Route path="/legacy/training" element={<TrainingPage />} />
                        <Route path="/legacy/settings" element={<SettingsPage />} />
                    </Route>
                </Route>

                <Route path="*" element={<NotFoundPage />} />
            </Routes>
        </div>
    );
}

export default App;
