import {
  LayoutDashboard,
  Settings,
  Boxes,
  Network,
  Database,
  LogOut,
  ChartLine,
  Compass,
  Inbox,
  ShieldCheck,
  Users,
  Search,
} from 'lucide-react';
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

  const initials = currentUser?.username ? currentUser.username.slice(0, 2).toUpperCase() : 'U';

  return (
    <div className="w-64 h-screen shrink-0 flex flex-col relative z-10 font-sans border-r border-(--border-color) bg-(--background-secondary) text-(--text-primary) selection:bg-(--accent-primary) selection:text-white transition-colors duration-300">
      <div className="h-[60px] flex items-center justify-start px-[18px] border-b border-(--border-color)">
        <span className="font-display font-semibold text-[19px] tracking-tight text-(--text-primary) flex-1 italic cursor-pointer">
          FedLearn
        </span>
        <div className="flex items-center gap-1">
          <ThemeToggle compact />
          <NotificationBell />
        </div>
      </div>

      <div className="p-[14px] pb-1.5">
        <div className="flex items-center gap-2 px-2.5 py-[7px] bg-(--background-card) border border-(--border-color) rounded-(--radius) text-[12.5px] text-(--text-secondary)">
          <Search className="w-3.5 h-3.5" />
          <span className="flex-1">Search projects, clients...</span>
          <span className="font-mono text-[11px] px-1.5 py-0.5 border border-(--border-color) border-b-2 rounded bg-(--background-secondary) text-(--text-secondary)">
            ⌘K
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2 flex flex-col gap-0.5">
        <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary) pt-3 pb-1.5 px-2">
          Workspace
        </div>
        {baseNavItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/dashboard'}
            className={({ isActive }) =>
              cn(
                'relative flex items-center gap-2.5 px-2.5 py-2 w-full rounded-md text-[13px] transition-all duration-150',
                isActive
                  ? 'bg-(--background-card) text-(--text-primary) font-medium'
                  : 'text-(--text-secondary) hover:bg-(--background-card) hover:text-(--text-primary)'
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-[-8px] top-2 bottom-2 w-0.5 bg-(--accent-primary) rounded-sm" />
                )}
                <item.icon
                  className={cn(
                    'w-4 h-4',
                    isActive ? 'text-(--accent-primary)' : 'text-(--text-secondary)'
                  )}
                />
                <span className="flex-1">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}

        {currentUser?.role === 'ADMIN' && (
          <>
            <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary) pt-5 pb-1.5 px-2">
              System
            </div>
            {adminNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  cn(
                    'relative flex items-center gap-2.5 px-2.5 py-2 w-full rounded-md text-[13px] transition-all duration-150',
                    isActive
                      ? 'bg-(--background-card) text-(--text-primary) font-medium'
                      : 'text-(--text-secondary) hover:bg-(--background-card) hover:text-(--text-primary)'
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span className="absolute left-[-8px] top-2 bottom-2 w-0.5 bg-(--accent-primary) rounded-sm" />
                    )}
                    <item.icon
                      className={cn(
                        'w-4 h-4',
                        isActive ? 'text-(--accent-primary)' : 'text-(--text-secondary)'
                      )}
                    />
                    <span className="flex-1">{item.label}</span>
                  </>
                )}
              </NavLink>
            ))}
          </>
        )}
      </div>

      <div className="p-3 border-t border-(--border-color) flex items-center gap-2.5">
        <div className="w-[34px] h-[34px] shrink-0 rounded-full bg-gradient-to-br from-(--accent-primary) to-pink-500 flex items-center justify-center text-[12px] font-bold text-white shadow-sm">
          {initials}
        </div>
        <div className="flex flex-col flex-1 min-w-0">
          <span className="text-[13px] font-medium text-(--text-primary) tracking-tight truncate">
            {currentUser?.username || 'User'}
          </span>
          <span className="text-[11px] text-(--text-secondary) truncate">
            {currentUser?.role ?? 'User'}
          </span>
        </div>
        <button
          type="button"
          onClick={logout}
          className="shrink-0 text-(--text-secondary) hover:text-(--text-primary) hover:bg-(--background-card) rounded-lg p-1.5 transition-colors"
          aria-label="Sign out"
          title="Sign out"
        >
          <LogOut className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
