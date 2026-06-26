// =============================================================================
// FedLearn Frontend — Sidebar (Ember design system)
// =============================================================================
// Wired to existing AuthContext for user profile and logout. Plain-language
// nav labels grouped under quiet uppercase section headers.

import { LayoutDashboard, Settings, Boxes, Network, Database, LogOut, FlaskConical, Gauge } from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { cn } from '../../lib/utils';
import { useAuth } from '../../context/AuthContext';
import type { Role } from '../../context/AuthContext';
import { Wordmark } from '../brand';

type NavItem = { icon: typeof LayoutDashboard; label: string; path: string; end?: boolean };
type NavGroup = { heading: string; items: NavItem[] };

// Project-management surfaces (Devices / Data) are owner+admin only — a plain
// USER's sidebar shows Overview + the model surfaces it's allowed to use.
const OWNER_ITEMS: NavItem[] = [
    { icon: Network, label: 'Devices', path: '/nodes' },
    { icon: Database, label: 'Data', path: '/datasets' },
];

function navGroupsForRole(role: Role): NavGroup[] {
    const workspace: NavItem[] = [
        { icon: LayoutDashboard, label: 'Overview', path: '/dashboard', end: true },
        { icon: Boxes, label: 'Models', path: '/models' },
        { icon: FlaskConical, label: 'Use a model', path: '/playground' },
    ];
    if (role === 'PROJECT_OWNER' || role === 'PLATFORM_ADMIN') {
        // Devices sits next to Models; Data after.
        workspace.splice(1, 0, OWNER_ITEMS[0]);
        workspace.push(OWNER_ITEMS[1]);
    }
    const groups: NavGroup[] = [{ heading: 'Workspace', items: workspace }];
    if (role === 'PLATFORM_ADMIN') {
        groups.push({
            heading: 'Admin',
            items: [{ icon: Gauge, label: 'Benchmarks', path: '/admin/benchmarks' }],
        });
    }
    groups.push({ heading: 'Account', items: [{ icon: Settings, label: 'Settings', path: '/settings' }] });
    return groups;
}

const ROLE_LABEL: Record<Role, string> = {
    USER: 'Member',
    PROJECT_OWNER: 'Project owner',
    PLATFORM_ADMIN: 'Platform admin',
};

export function Sidebar() {
    const { currentUser, logout } = useAuth();

    const initials = currentUser?.username ? currentUser.username.slice(0, 2).toUpperCase() : 'U';
    const role: Role = currentUser?.role ?? 'USER';
    const roleLabel = ROLE_LABEL[role];
    const navGroups = navGroupsForRole(role);

    return (
        <aside className="relative z-10 flex h-screen w-64 flex-shrink-0 flex-col border-r border-hairline bg-canvas font-sans text-fg">
            <div className="flex h-16 items-center px-6">
                <NavLink to="/dashboard" aria-label="FedLearn">
                    <Wordmark size={26} />
                </NavLink>
            </div>

            <nav className="flex flex-1 flex-col gap-6 overflow-y-auto px-4 py-4">
                {navGroups.map((group) => (
                    <div key={group.heading} className="flex flex-col gap-1">
                        <div className="mb-1 px-3 text-[11px] font-medium uppercase tracking-[0.16em] text-fg-subtle">
                            {group.heading}
                        </div>
                        {group.items.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                end={item.end}
                                className={({ isActive }) =>
                                    cn(
                                        'group relative flex items-center gap-3 rounded-md px-3 py-2.5 text-label font-medium',
                                        'transition-colors duration-[140ms]',
                                        isActive
                                            ? 'bg-accent/10 text-fg'
                                            : 'text-fg-muted hover:bg-surface-1 hover:text-fg',
                                    )
                                }
                            >
                                {({ isActive }) => (
                                    <>
                                        {isActive && (
                                            <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-pill bg-accent" />
                                        )}
                                        <item.icon
                                            className={cn(
                                                'h-[18px] w-[18px] transition-colors',
                                                isActive
                                                    ? 'text-accent'
                                                    : 'text-fg-subtle group-hover:text-fg',
                                            )}
                                            strokeWidth={1.5}
                                        />
                                        {item.label}
                                    </>
                                )}
                            </NavLink>
                        ))}
                    </div>
                ))}
            </nav>

            <div className="p-3">
                <div className="flex items-center gap-3 rounded-card border border-hairline bg-surface-1 p-3">
                    <div className="grid h-9 w-9 flex-shrink-0 place-items-center rounded-pill bg-accent text-label font-semibold text-accent-fg">
                        {initials}
                    </div>
                    <div className="flex min-w-0 flex-1 flex-col">
                        <span className="truncate text-label font-medium tracking-tight text-fg">
                            {currentUser?.username || 'User'}
                        </span>
                        <span className="text-caption text-fg-muted">{roleLabel}</span>
                    </div>
                    <button
                        onClick={logout}
                        className="rounded-md p-1.5 text-fg-muted transition-colors hover:bg-surface-2 hover:text-danger"
                        title="Sign out"
                        aria-label="Sign out"
                    >
                        <LogOut className="h-4 w-4" strokeWidth={1.5} />
                    </button>
                </div>
            </div>
        </aside>
    );
}
