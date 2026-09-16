// =============================================================================
// FedLearn Frontend — RoleDashboard (role-aware /dashboard landing)
// =============================================================================
// /dashboard is the single canonical "Overview" route every role lands on.
// This component picks the right surface for the signed-in role so the same
// URL + sidebar link works for everyone:
//   PLATFORM_ADMIN -> AdminDashboard   (manage users, requests, all projects)
//   PROJECT_OWNER  -> OwnerDashboard   (my projects + ownership controls)
//   USER           -> ClientDashboard  (request owner access + discover)

import { useAuth } from '../../context/AuthContext';
import { AdminDashboard } from './AdminDashboard';
import { OwnerDashboard } from './OwnerDashboard';
import { ClientDashboard } from './ClientDashboard';

export function RoleDashboard() {
    const { currentUser } = useAuth();

    if (currentUser?.role === 'PLATFORM_ADMIN') {
        return <AdminDashboard />;
    }
    if (currentUser?.role === 'PROJECT_OWNER') {
        return <OwnerDashboard />;
    }
    return <ClientDashboard />;
}
