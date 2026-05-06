import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import DiskLoader from './DiskLoader';

const ProtectedRoute: React.FC = () => {
    const { currentUser, isLoading } = useAuth();

    if (isLoading) {
        return <DiskLoader message="Checking authentication..." />;
    }

    if (!currentUser) {
        return <Navigate to="/login" replace />;
    }

    return <Outlet />;
};

export default ProtectedRoute;
