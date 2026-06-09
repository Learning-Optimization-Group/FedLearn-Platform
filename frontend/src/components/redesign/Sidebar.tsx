// =============================================================================
// FedLearn Frontend — Redesigned Sidebar (Instrument design system)
// =============================================================================
// Wired to existing AuthContext for user profile and logout.

import { Brain, LayoutDashboard, Settings, Boxes, Network, Database, LogOut } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { cn } from '../../lib/utils';
import { useAuth } from '../../context/AuthContext';

const navItems = [
  { icon: LayoutDashboard, label: 'Overview', path: '/dashboard' },
  { icon: Network, label: 'Node Network', path: '/nodes' },
  { icon: Boxes, label: 'Models', path: '/models' },
  { icon: Database, label: 'Datasets', path: '/datasets' },
  { icon: Settings, label: 'Settings', path: '/settings' },
];

export function Sidebar() {
  const { currentUser, logout } = useAuth();

  const initials = currentUser?.username ? currentUser.username.slice(0, 2).toUpperCase() : 'U';

  return (
    <div className="w-64 bg-canvas border-r border-hairline h-screen flex flex-col text-fg font-sans flex-shrink-0 relative z-10 selection:bg-accent selection:text-accent-fg">
      <div className="h-20 flex items-center gap-3 px-8">
        <div className="w-8 h-8 rounded-md bg-surface-1 border border-hairline flex items-center justify-center">
          <Brain className="w-5 h-5 text-fg" strokeWidth={1.5} />
        </div>
        <span className="text-h4 tracking-tight text-fg">FedLearn</span>
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-4 flex flex-col gap-1">
        <div className="text-caption font-medium tracking-widest uppercase text-fg-muted mb-2 px-4 mt-4">
          Menu
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/dashboard'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-4 py-2.5 rounded-md text-label font-medium transition-colors',
                isActive ? 'bg-surface-2 text-fg' : 'text-fg-muted hover:bg-surface-1 hover:text-fg'
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon
                  className={cn('w-[18px] h-[18px]', isActive ? 'text-fg' : 'text-fg-muted')}
                  strokeWidth={1.5}
                />
                {item.label}
              </>
            )}
          </NavLink>
        ))}
      </div>

      <div className="p-4">
        <div className="rounded-card p-3 flex items-center gap-3">
          <div className="w-9 h-9 rounded-pill bg-surface-2 flex items-center justify-center text-label font-medium text-fg">
            {initials}
          </div>
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-label font-medium text-fg tracking-tight truncate">
              {currentUser?.username || 'User'}
            </span>
            <span className="text-caption text-fg-muted">Admin</span>
          </div>
          <button
            onClick={logout}
            className="text-fg-muted hover:text-danger transition-colors p-1"
            title="Logout"
          >
            <LogOut className="w-4 h-4" strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </div>
  );
}
