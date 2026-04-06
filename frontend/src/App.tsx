import React, { useEffect } from 'react';
import { Routes, Route, Link, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
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
                    <Route element={<Layout />}>
                        <Route path="/dashboard" element={<DashboardPage />} />
                        <Route path="/models" element={<ModelsPage />} />
                        <Route path="/training" element={<TrainingPage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                    </Route>
                </Route>

                <Route path="*" element={<NotFoundPage />} />
            </Routes>
        </div>
    );
}

export default App;
