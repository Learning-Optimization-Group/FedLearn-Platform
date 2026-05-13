import { Brain, LayoutDashboard, Settings, Boxes, Network, Database, LogOut, ChartLine, Compass, Inbox, ShieldCheck, Users } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { cn } from '../../lib/utils';
import { useAuth } from '../../context/AuthContext';
import { ThemeToggle } from '../ThemeToggle';
import { NotificationBell } from './NotificationBell';

const baseNavItems = [
  { icon: LayoutDashboard, label: 'Overview', path: '/dashboard' },
  { icon: Network, label: 'Node Network', path: '/clients' },
  { icon: Boxes, label: 'Models', path: '/models' },
  { icon: ChartLine, label: 'Training', path: '/training' },
  { icon: Database, label: 'Datasets', path: '/datasets' },
  { icon: Compass, label: 'Discover', path: '/discover' },
  { icon: Inbox, label: 'My Requests', path: '/my/requests' },
  { icon: Settings, label: 'Settings', path: '/settings' },
];

const adminNavItems = [
  { icon: Users, label: 'Manage Users', path: '/admin/users' },
  { icon: ShieldCheck, label: 'All Projects', path: '/admin/projects' },
];

export function Sidebar() {
  const { currentUser, logout } = useAuth();

  const initials = currentUser?.username
    ? currentUser.username.slice(0, 2).toUpperCase()
    : 'U';

  return (
    <div className="w-64 h-screen shrink-0 flex flex-col relative z-10 font-sans border-r border-(--border-color) bg-(--background-secondary) text-(--text-primary) selection:bg-(--accent-primary) selection:text-white transition-colors duration-300">
      <div className="h-20 flex items-center gap-3 px-8">
        <div className="w-8 h-8 rounded-xl bg-(--background-card) border border-(--border-color) flex items-center justify-center shadow-(--shadow-soft)">
          <Brain className="w-5 h-5 text-(--accent-primary)" />
        </div>
        <span className="font-semibold text-lg tracking-tight text-(--text-primary) flex-1">FedLearn</span>
        <NotificationBell />
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-4 flex flex-col gap-1">
        <div className="text-[11px] font-medium tracking-widest uppercase text-(--text-secondary) mb-2 px-4 mt-4">Menu</div>
        {baseNavItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/dashboard'}
            className={({ isActive }) => cn(
              'flex items-center gap-3 px-4 py-2.5 rounded-xl text-[15px] font-medium transition-all duration-200',
              isActive
                ? 'bg-(--background-card) text-(--text-primary) shadow-(--shadow-soft)'
                : 'text-(--text-secondary) hover:bg-(--background-card) hover:text-(--text-primary)'
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn('w-4.5 h-4.5', isActive ? 'text-(--accent-primary)' : 'text-(--text-secondary)')} />
                {item.label}
              </>
            )}
          </NavLink>
        ))}

        {currentUser?.role === 'ADMIN' && (
          <>
            <div className="text-[11px] font-medium tracking-widest uppercase text-(--text-secondary) mb-2 px-4 mt-6">Admin</div>
            {adminNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) => cn(
                  'flex items-center gap-3 px-4 py-2.5 rounded-xl text-[15px] font-medium transition-all duration-200',
                  isActive
                    ? 'bg-(--background-card) text-(--text-primary) shadow-(--shadow-soft)'
                    : 'text-(--text-secondary) hover:bg-(--background-card) hover:text-(--text-primary)'
                )}
              >
                {({ isActive }) => (
                  <>
                    <item.icon className={cn('w-4.5 h-4.5', isActive ? 'text-(--accent-primary)' : 'text-(--text-secondary)')} />
                    {item.label}
                  </>
                )}
              </NavLink>
            ))}
          </>
        )}
      </div>

      <div className="p-4 space-y-3 border-t border-(--border-color) bg-(--surface-glass) backdrop-blur-xl">
        <ThemeToggle />
        <div className="rounded-2xl p-3 flex items-center gap-3 bg-(--background-card) border border-(--border-color) shadow-(--shadow-soft)">
          <div className="w-9 h-9 rounded-full bg-(--background-secondary) flex items-center justify-center text-sm font-medium text-(--text-primary)">
            {initials}
          </div>
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-[15px] font-medium text-(--text-primary) tracking-tight truncate">
              {currentUser?.username || 'User'}
            </span>
            <span className="text-[13px] text-(--text-secondary)">{currentUser?.role ?? 'User'}</span>
          </div>
          <button
            onClick={logout}
            className="text-(--text-secondary) hover:text-destructive transition-colors p-1"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
