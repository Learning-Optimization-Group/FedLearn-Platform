// in ProtectedRoute.jsx
import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ProtectedRoute = () => {
    const { currentUser, isLoading } = useAuth();

    // 1. If we are still doing the initial check, show a loading indicator.
    if (isLoading) {
        return <div>Loading authentication...</div>;
    }

    // 2. If the check is done and there's no user, redirect to login.
    if (!currentUser) {
        return <Navigate to="/login" />;
    }

    // 3. If the check is done and there IS a user, render the child routes.
    return <Outlet />;
};

export default ProtectedRoute;