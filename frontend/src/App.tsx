import React, { useEffect } from 'react';
import { Routes, Route, Link, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ModelsPage from './pages/ModelsPage';
import TrainingPage from './pages/TrainingPage';
import SettingsPage from './pages/SettingsPage';
import './App.css';
import Layout from './components/Layout';
import { useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import DiskLoader from './components/DiskLoader';
import { LayoutV2 } from './components/redesign/LayoutV2';
import { DashboardV2 } from './components/redesign/DashboardV2';

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
        const handleAuthError = () => {
            logout();
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
                    {/* Original UI */}
                    <Route element={<Layout />}>
                        <Route path="/dashboard" element={<DashboardPage />} />
                        <Route path="/models" element={<ModelsPage />} />
                        <Route path="/training" element={<TrainingPage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                    </Route>

                    {/* Redesigned UI (v2) — Apple-inspired dark theme */}
                    <Route element={<LayoutV2 />}>
                        <Route path="/v2" element={<DashboardV2 />} />
                        <Route path="/v2/nodes" element={<div className="flex-1 flex items-center justify-center text-[#86868b]">Node Network — Coming Soon</div>} />
                        <Route path="/v2/models" element={<div className="flex-1 flex items-center justify-center text-[#86868b]">Models — Coming Soon</div>} />
                        <Route path="/v2/datasets" element={<div className="flex-1 flex items-center justify-center text-[#86868b]">Datasets — Coming Soon</div>} />
                        <Route path="/v2/settings" element={<div className="flex-1 flex items-center justify-center text-[#86868b]">Settings — Coming Soon</div>} />
                    </Route>
                </Route>

                <Route path="*" element={<NotFoundPage />} />
            </Routes>
        </div>
    );
}

export default App;
