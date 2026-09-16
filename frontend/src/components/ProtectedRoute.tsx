import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import DiskLoader from './DiskLoader';

const ProtectedRoute: React.FC = () => {
    const { currentUser, isLoading } = useAuth();
    const location = useLocation();

    if (isLoading) {
        return <DiskLoader message="Checking authentication..." />;
    }

    if (!currentUser) {
        // Remember where the user was headed so LoginPage can send them back
        // there after authenticating, instead of dumping them on /dashboard.
        return <Navigate to="/login" replace state={{ from: location }} />;
    }

    return <Outlet />;
};

export default ProtectedRoute;
