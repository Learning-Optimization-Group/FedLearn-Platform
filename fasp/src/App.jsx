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