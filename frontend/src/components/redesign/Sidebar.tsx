// =============================================================================
// FedLearn Frontend — Redesigned Sidebar (Apple-inspired dark theme)
// =============================================================================
// Wired to existing AuthContext for user profile and logout.

import { Brain, LayoutDashboard, Settings, Boxes, Network, Database, LogOut } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { cn } from '../../lib/utils';
import { useAuth } from '../../context/AuthContext';

const navItems = [
  { icon: LayoutDashboard, label: 'Overview', path: '/v2' },
  { icon: Network, label: 'Node Network', path: '/v2/nodes' },
  { icon: Boxes, label: 'Models', path: '/v2/models' },
  { icon: Database, label: 'Datasets', path: '/v2/datasets' },
  { icon: Settings, label: 'Settings', path: '/v2/settings' },
];

export function Sidebar() {
  const { currentUser, logout } = useAuth();

  const initials = currentUser?.username
    ? currentUser.username.slice(0, 2).toUpperCase()
    : 'U';

  return (
    <div className="w-64 bg-black border-r border-[#2c2c2e] h-screen flex flex-col text-[#f5f5f7] font-sans flex-shrink-0 relative z-10 selection:bg-[#0a84ff] selection:text-white">
      <div className="h-20 flex items-center gap-3 px-8">
        <div className="w-8 h-8 rounded-xl bg-[#1c1c1e] border border-[#2c2c2e] flex items-center justify-center">
          <Brain className="w-5 h-5 text-[#f5f5f7]" />
        </div>
        <span className="font-semibold text-lg tracking-tight text-[#f5f5f7]">FedLearn</span>
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-4 flex flex-col gap-1">
        <div className="text-[11px] font-medium tracking-widest uppercase text-[#86868b] mb-2 px-4 mt-4">Menu</div>
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/v2'}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-4 py-2.5 rounded-xl text-[15px] font-medium transition-all duration-200",
              isActive
                ? "bg-[#2c2c2e] text-[#f5f5f7]"
                : "text-[#86868b] hover:bg-[#1c1c1e] hover:text-[#f5f5f7]"
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon className={cn(
                  "w-[18px] h-[18px]",
                  isActive ? "text-[#f5f5f7]" : "text-[#86868b]"
                )} />
                {item.label}
              </>
            )}
          </NavLink>
        ))}
      </div>

      <div className="p-4">
        <div className="rounded-2xl p-3 flex items-center gap-3">
          <div className="w-9 h-9 rounded-full bg-[#2c2c2e] flex items-center justify-center text-sm font-medium text-[#f5f5f7]">
            {initials}
          </div>
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-[15px] font-medium text-[#f5f5f7] tracking-tight truncate">
              {currentUser?.username || 'User'}
            </span>
            <span className="text-[13px] text-[#86868b]">Admin</span>
          </div>
          <button
            onClick={logout}
            className="text-[#86868b] hover:text-[#ff453a] transition-colors p-1"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
