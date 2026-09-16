import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth, type Role } from '../context/AuthContext';

interface RoleRouteProps {
    /** Roles permitted to view the nested routes. */
    allow: Role[];
}

/**
 * Sits inside ProtectedRoute (so the user is already known to be authenticated)
 * and gates a route subtree by platform role. A user whose role isn't in
 * `allow` is redirected to /dashboard — the role-aware landing route that
 * forwards each role to the dashboard it's allowed to see.
 *
 * This is a UX guard, not a security boundary: the backend still enforces RBAC
 * and returns 403 for anything the role can't do.
 */
const RoleRoute: React.FC<RoleRouteProps> = ({ allow }) => {
    const { currentUser } = useAuth();

    if (currentUser && allow.includes(currentUser.role)) {
        return <Outlet />;
    }

    return <Navigate to="/dashboard" replace />;
};

export default RoleRoute;
