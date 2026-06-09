# RBAC Frontend — Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Commit policy:** The user has asked Claude not to run `git commit` unless explicitly requested. Each task ends with a suggested commit message and `git add` lines, but the actual `git commit` invocation MUST wait for user approval. When in doubt, prepare the staged change and ask.

**Goal:** Wire the RBAC backend (Plan 1) into the web frontend — notification bell, dashboard relationship filter, project detail tabs (Members/Clients/Requests), Discover page, My Requests page, and Admin pages.

**Architecture:** All new UI lives inside the existing `LayoutV2` shell (Sidebar + Outlet). New routes are added to App.tsx under the `ProtectedRoute` / `LayoutV2` wrapper. A `NotificationContext` manages a persistent STOMP subscription to `/user/queue/notifications` at the layout level. All new API calls are added to `apiServices.ts`. No new npm packages are needed.

**Tech stack:** React 19, TypeScript 5, Vite 6, Tailwind v4 CSS custom properties, framer-motion, lucide-react, @stomp/stompjs, axios. TypeScript compilation (`npx tsc --noEmit`) is the verification step after each task — there is no Jest/Vitest in this project.

**Source-of-truth spec:** `docs/superpowers/specs/2026-05-12-rbac-and-model-hub-design.md` §7.

**Plan series:**
- Plan 1 — Backend foundation. ✅ Done.
- **Plan 2** (this doc) — Web frontend.
- Plan 3 — Electron client discovery.
- Plan 4 — Model Hub (catalog, download, inference).

---

## House conventions (read before starting)

- **Design system:** Tailwind v4 utility classes + CSS custom properties (`var(--background-card)`, `var(--border-color)`, `var(--text-primary)`, etc.). Match DashboardV2 style exactly — `rounded-3xl`, `rounded-2xl`, `rounded-xl` for card/panel/button hierarchy. Never hardcode colors; use CSS variables.
- **Authentication:** Cookie-only. `useAuth()` gives `currentUser: { username, email, role }`. All API calls use the `api` axios instance (credentials: true). Never read the JWT from JS.
- **STOMP URL:** `const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || \`http://${window.location.hostname}:8081\`; const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');` — copy this pattern from DashboardV2.tsx.
- **TypeScript check command:** `cd frontend && npx tsc --noEmit` — no output = pass.
- **No new npm packages** — all needed libraries (lucide-react, framer-motion, @stomp/stompjs, axios) are already installed.
- **Do not touch** legacy UI under `src/components/Layout.tsx`, `src/pages/DashboardPage.tsx`, `src/pages/ClientsPage.tsx`, etc.
- **Do not modify** the `EditProjectModal` — it uses the existing PUT endpoint and its own local state.
- **Commits:** prepare `git add` lines per task; do NOT run `git commit` without user approval.

---

## File structure

### Files to **create**

```
frontend/src/
  context/
    NotificationContext.tsx     # STOMP subscription + unread state
  components/redesign/
    NotificationBell.tsx        # bell icon + dropdown, uses NotificationContext
  pages/
    DiscoverPage.tsx            # /discover — public project listing
    MyRequestsPage.tsx          # /my/requests — caller's access requests
    ProjectDetailPage.tsx       # /projects/:projectId — detail tabs
    AdminUsersPage.tsx          # /admin/users — user table with role management
    AdminProjectsPage.tsx       # /admin/projects — project table
```

### Files to **modify**

```
frontend/src/
  services/apiServices.ts            # new types + API functions
  components/redesign/DashboardV2.tsx   # filter chips + relationship badge pass-through
  components/redesign/ProjectCard.tsx   # relationship badge + Details link
  components/redesign/Sidebar.tsx       # Discover, My Requests, Admin nav items
  components/redesign/LayoutV2.tsx      # wrap with NotificationProvider; add bell to sidebar header
  main.tsx                              # add NotificationProvider inside AuthProvider
  App.tsx                               # new routes: /discover, /my/requests, /projects/:id, /admin/*
```

### Files to **leave alone**

- All files under `src/components/` not listed above.
- Legacy pages (`src/pages/DashboardPage.tsx`, `src/pages/ClientsPage.tsx`, etc.).
- `src/context/AuthContext.tsx`, `src/context/ThemeContext.tsx`.
- `src/api/axiosConfig.ts`.
- `backend/`, `framework/`, `fedlearn-desktop/` — out of scope.

---

## Task 1 — API service extensions

**Files:**
- Modify: `frontend/src/services/apiServices.ts`

Extend the service file with all types and API functions needed by the new pages. The existing functions and types must remain untouched.

- [ ] **Step 1.1 — Add new TypeScript types.**

Open `frontend/src/services/apiServices.ts`. After the existing `User` interface, add:

```typescript
// ─── RBAC types ─────────────────────────────────────────────────────────────

export interface Membership {
    projectId: string;
    userId: number;
    username: string;
    role: 'MEMBER' | 'CLIENT' | 'OWNER';
    partitionId?: number | null;
    joinedVia: string;
    addedAt: string;
}

export interface AccessRequest {
    id: number;
    projectId: string;
    projectName: string;
    userId: number;
    username: string;
    status: 'PENDING' | 'APPROVED' | 'DENIED';
    message?: string;
    requestedAt: string;
    decidedAt?: string;
    decidedByUsername?: string;
}

export interface DiscoverProject {
    id: string;
    name: string;
    visibility: 'PUBLIC' | 'PRIVATE';
    ownerUsername: string;
    modelType: string;
    myRequestStatus: 'NONE' | 'PENDING' | 'APPROVED' | 'DENIED';
    lastAccuracy?: number;
    description?: string;
}

export interface AdminUser {
    id: number;
    username: string;
    email: string;
    role: 'USER' | 'ADMIN';
    projectsOwned: number;
    memberships: number;
    createdAt: string;
}

export interface UserSearchResult {
    id: number;
    username: string;
}

export interface AppNotification {
    id: string;
    type: 'ACCESS_REQUEST_CREATED' | 'ACCESS_REQUEST_DECIDED' | 'MEMBERSHIP_ADDED' | 'MEMBERSHIP_REMOVED' | 'PROJECT_VISIBILITY_CHANGED';
    projectId: string;
    projectName: string;
    actorId: number;
    actorUsername: string;
    subjectId?: number;
    subjectUsername?: string;
    decision?: string;
    role?: string;
    timestamp: string;
}
```

- [ ] **Step 1.2 — Extend the `Project` interface.**

Find the existing `Project` interface and add two optional fields so that existing code that doesn't provide them still compiles:

```typescript
export interface Project {
    id: string;
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED' | 'FAILED';
    serverPort?: number;
    visibility?: 'PUBLIC' | 'PRIVATE';
    myRelationship?: 'OWNER' | 'MEMBER' | 'CLIENT' | null;
}
```

- [ ] **Step 1.3 — Add PATCH project + single-project fetch.**

After the existing `deleteProject` function, add:

```typescript
export const fetchProject = (projectId: string): Promise<AxiosResponse<Project>> =>
    api.get<Project>(`/projects/${projectId}`);

export const patchProject = (
    projectId: string,
    data: { name?: string; description?: string; visibility?: 'PUBLIC' | 'PRIVATE' }
): Promise<AxiosResponse<Project>> =>
    api.patch<Project>(`/projects/${projectId}`, data);
```

- [ ] **Step 1.4 — Add Discover + Membership functions.**

```typescript
// ─── Discover ───────────────────────────────────────────────────────────────
export const fetchDiscover = (): Promise<AxiosResponse<DiscoverProject[]>> =>
    api.get<DiscoverProject[]>('/projects/discover');

// ─── Memberships ────────────────────────────────────────────────────────────
export const fetchMemberships = (projectId: string): Promise<AxiosResponse<Membership[]>> =>
    api.get<Membership[]>(`/projects/${projectId}/memberships`);

export const addMembership = (
    projectId: string,
    body: { username: string; role: 'MEMBER' | 'CLIENT' }
): Promise<AxiosResponse<Membership>> =>
    api.post<Membership>(`/projects/${projectId}/memberships`, body);

export const removeMembership = (projectId: string, userId: number): Promise<AxiosResponse<void>> =>
    api.delete<void>(`/projects/${projectId}/memberships/${userId}`);
```

- [ ] **Step 1.5 — Add Access Request functions.**

```typescript
// ─── Access Requests (project-scoped) ───────────────────────────────────────
export const fetchProjectAccessRequests = (projectId: string): Promise<AxiosResponse<AccessRequest[]>> =>
    api.get<AccessRequest[]>(`/projects/${projectId}/access-requests`);

export const createAccessRequest = (projectId: string, message?: string): Promise<AxiosResponse<AccessRequest>> =>
    api.post<AccessRequest>(`/projects/${projectId}/access-requests`, message ? { message } : {});

export const decideAccessRequest = (
    projectId: string,
    reqId: number,
    decision: 'APPROVED' | 'DENIED'
): Promise<AxiosResponse<AccessRequest>> =>
    api.put<AccessRequest>(`/projects/${projectId}/access-requests/${reqId}`, { decision });

// ─── My Access Requests ──────────────────────────────────────────────────────
export const fetchMyAccessRequests = (): Promise<AxiosResponse<AccessRequest[]>> =>
    api.get<AccessRequest[]>('/my/access-requests');
```

- [ ] **Step 1.6 — Add Admin + User Search functions.**

```typescript
// ─── Admin ──────────────────────────────────────────────────────────────────
export const fetchAdminUsers = (): Promise<AxiosResponse<AdminUser[]>> =>
    api.get<AdminUser[]>('/admin/users');

export const updateUserRole = (userId: number, role: 'USER' | 'ADMIN'): Promise<AxiosResponse<AdminUser>> =>
    api.put<AdminUser>(`/admin/users/${userId}/role`, { role });

export const fetchAdminProjects = (): Promise<AxiosResponse<Project[]>> =>
    api.get<Project[]>('/admin/projects');

// ─── User Search ─────────────────────────────────────────────────────────────
export const searchUsers = (q: string): Promise<AxiosResponse<UserSearchResult[]>> =>
    api.get<UserSearchResult[]>(`/users/search?q=${encodeURIComponent(q)}`);
```

- [ ] **Step 1.7 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output (clean compile).

- [ ] **Step 1.8 — Stage.**

```bash
git add frontend/src/services/apiServices.ts
```

Draft commit (do NOT run without user approval):
```
feat(frontend): extend API service with RBAC types and endpoint functions
```

---

## Task 2 — Dashboard filter chips + relationship badge

**Files:**
- Modify: `frontend/src/components/redesign/DashboardV2.tsx`
- Modify: `frontend/src/components/redesign/ProjectCard.tsx`

Add four filter chips ("All", "Owned by me", "Member", "Client") above the project grid. Show a small relationship badge on each project card.

- [ ] **Step 2.1 — Add filter state and logic to DashboardV2.**

In `DashboardV2.tsx`, after the existing `useState` declarations, add:

```typescript
type RelationFilter = 'all' | 'owner' | 'member' | 'client';
const [filter, setFilter] = useState<RelationFilter>('all');
```

Replace the existing `filteredProjects` `useMemo` with:

```typescript
const filteredProjects = useMemo(() => {
    let list = projects;
    if (filter === 'owner') list = projects.filter((p) => p.myRelationship === 'OWNER');
    else if (filter === 'member') list = projects.filter((p) => p.myRelationship === 'MEMBER');
    else if (filter === 'client') list = projects.filter((p) => p.myRelationship === 'CLIENT');
    return list.filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase()));
}, [projects, filter, searchQuery]);
```

- [ ] **Step 2.2 — Render filter chips in DashboardV2.**

In the header section of DashboardV2, find the `<div className="flex flex-wrap items-center justify-between gap-4">` block. After the closing `</div>` of that div (but before the KPI grid), insert:

```tsx
<div className="flex items-center gap-2">
  {(['all', 'owner', 'member', 'client'] as RelationFilter[]).map((f) => (
    <button
      key={f}
      onClick={() => setFilter(f)}
      className={cn(
        'px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border',
        filter === f
          ? 'bg-(--accent-primary) text-white border-transparent'
          : 'text-(--text-secondary) border-(--border-color) hover:text-(--text-primary) hover:bg-(--background-card)'
      )}
      style={filter === f ? {} : { backgroundColor: 'var(--background-secondary)' }}
    >
      {f === 'all' ? 'All' : f === 'owner' ? 'Owned by me' : f === 'member' ? 'Member' : 'Client'}
    </button>
  ))}
</div>
```

Add the `cn` import if it is not already imported in DashboardV2: add `import { cn } from '../../lib/utils';` at the top.

- [ ] **Step 2.3 — Pass `myRelationship` badge to ProjectCard.**

In `DashboardV2.tsx`, update the `<ProjectCard .../>` invocation to include:

```tsx
<ProjectCard
  project={project}
  results={resultsMap[project.id] || []}
  onOpenLogs={() => setLogViewProjectId(project.id)}
  onOpenResults={() => handleOpenResults(project)}
  onToggleServer={() => handleToggleServer(project)}
  onEditProject={() => {
    setEditProject(project);
    setIsEditModalOpen(true);
  }}
  onDeleteProject={() => handleDeleteProject(project.id)}
/>
```

`ProjectCard` will read `project.myRelationship` directly from the project prop — no extra prop needed.

- [ ] **Step 2.4 — Add relationship badge to ProjectCard.**

In `ProjectCard.tsx`, find the block that renders `project.status` (the `<span className={cn(...)}>` with the pulsing dot). After it, add a relationship badge:

```tsx
{project.myRelationship && (
  <span className={cn(
    'ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
    project.myRelationship === 'OWNER'
      ? 'bg-blue-500/10 text-blue-500 border border-blue-500/20'
      : project.myRelationship === 'MEMBER'
        ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
        : 'bg-purple-500/10 text-purple-500 border border-purple-500/20'
  )}>
    {project.myRelationship}
  </span>
)}
```

- [ ] **Step 2.5 — Add "Details" link to ProjectCard.**

At the bottom of the `<div className="flex gap-3 mt-1">` action row in ProjectCard, add a Link that navigates to the project detail page. First add imports:

```typescript
import { Link } from 'react-router-dom';
```

Replace the existing "Edit Project Details" button with:

```tsx
<div className="flex items-center justify-between mt-1">
  <Link
    to={`/projects/${project.id}`}
    className="text-xs font-medium text-(--text-secondary) hover:text-(--accent-primary) transition-colors"
  >
    View Details →
  </Link>
  <button
    onClick={onEditProject}
    className="text-xs font-medium text-(--text-secondary) hover:text-(--accent-primary)"
  >
    Edit Project Details
  </button>
</div>
```

- [ ] **Step 2.6 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 2.7 — Stage.**

```bash
git add frontend/src/components/redesign/DashboardV2.tsx \
        frontend/src/components/redesign/ProjectCard.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add relationship filter chips and badges to dashboard
```

---

## Task 3 — Notification system

**Files:**
- Create: `frontend/src/context/NotificationContext.tsx`
- Create: `frontend/src/components/redesign/NotificationBell.tsx`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/components/redesign/Sidebar.tsx`

Establish a STOMP subscription to `/user/queue/notifications` that persists across page navigations, and surface it as a bell icon with unread count in the sidebar.

- [ ] **Step 3.1 — Create NotificationContext.**

Create `frontend/src/context/NotificationContext.tsx` with this exact content:

```typescript
import { createContext, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import { Client as StompClient } from '@stomp/stompjs';
import { useAuth } from './AuthContext';
import type { AppNotification } from '../services/apiServices';

interface NotificationContextType {
    notifications: AppNotification[];
    unreadCount: number;
    markAllRead: () => void;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

export function NotificationProvider({ children }: { children: ReactNode }) {
    const { currentUser } = useAuth();
    const [notifications, setNotifications] = useState<AppNotification[]>([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const clientRef = useRef<StompClient | null>(null);

    useEffect(() => {
        if (!currentUser) return;

        const client = new StompClient({
            brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`,
            reconnectDelay: 5000,
        });

        client.onConnect = () => {
            client.subscribe('/user/queue/notifications', (msg) => {
                try {
                    const notif: AppNotification = JSON.parse(msg.body);
                    setNotifications((prev) => [notif, ...prev].slice(0, 50));
                    setUnreadCount((c) => c + 1);
                } catch {
                    // ignore malformed payload
                }
            });
        };

        client.activate();
        clientRef.current = client;

        return () => {
            if (clientRef.current?.active) clientRef.current.deactivate();
            clientRef.current = null;
        };
    }, [currentUser]);

    const markAllRead = () => setUnreadCount(0);

    return (
        <NotificationContext.Provider value={{ notifications, unreadCount, markAllRead }}>
            {children}
        </NotificationContext.Provider>
    );
}

export function useNotifications(): NotificationContextType {
    const ctx = useContext(NotificationContext);
    if (!ctx) throw new Error('useNotifications must be used within NotificationProvider');
    return ctx;
}
```

- [ ] **Step 3.2 — Create NotificationBell component.**

Create `frontend/src/components/redesign/NotificationBell.tsx`:

```tsx
import { useEffect, useRef, useState } from 'react';
import { Bell } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useNotifications } from '../../context/NotificationContext';
import type { AppNotification } from '../../services/apiServices';

function notificationLabel(n: AppNotification): string {
    switch (n.type) {
        case 'ACCESS_REQUEST_CREATED':
            return `${n.actorUsername} requested access to ${n.projectName}`;
        case 'ACCESS_REQUEST_DECIDED':
            return `Your request for ${n.projectName} was ${n.decision?.toLowerCase()}`;
        case 'MEMBERSHIP_ADDED':
            return `${n.actorUsername} added you to ${n.projectName} as ${n.role?.toLowerCase()}`;
        case 'MEMBERSHIP_REMOVED':
            return `You were removed from ${n.projectName}`;
        case 'PROJECT_VISIBILITY_CHANGED':
            return `${n.projectName} visibility changed`;
        default:
            return `Update on ${n.projectName}`;
    }
}

export function NotificationBell() {
    const { notifications, unreadCount, markAllRead } = useNotifications();
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClick(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, []);

    const toggle = () => {
        setOpen((v) => !v);
        if (!open) markAllRead();
    };

    return (
        <div className="relative" ref={ref}>
            <button
                onClick={toggle}
                className="relative w-8 h-8 flex items-center justify-center rounded-xl text-(--text-secondary) hover:text-(--text-primary) hover:bg-(--background-card) transition-all"
                title="Notifications"
            >
                <Bell className="w-4 h-4" />
                {unreadCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 rounded-full bg-(--accent-primary) text-white text-[10px] font-bold flex items-center justify-center px-0.5">
                        {unreadCount > 99 ? '99+' : unreadCount}
                    </span>
                )}
            </button>

            {open && (
                <div
                    className="absolute left-full ml-2 top-0 z-50 w-80 rounded-2xl shadow-lg overflow-hidden"
                    style={{
                        backgroundColor: 'var(--background-card)',
                        border: '1px solid var(--border-color)',
                        boxShadow: 'var(--shadow-strong)',
                    }}
                >
                    <div className="px-4 py-3 border-b text-[13px] font-semibold text-(--text-primary)" style={{ borderColor: 'var(--border-color)' }}>
                        Notifications
                    </div>
                    <div className="max-h-80 overflow-y-auto">
                        {notifications.length === 0 ? (
                            <div className="px-4 py-6 text-center text-[13px] text-(--text-secondary)">
                                No notifications yet
                            </div>
                        ) : (
                            notifications.map((n) => (
                                <Link
                                    key={n.id}
                                    to={`/projects/${n.projectId}`}
                                    onClick={() => setOpen(false)}
                                    className="block px-4 py-3 text-[13px] text-(--text-primary) border-b last:border-b-0 hover:bg-(--background-secondary) transition-colors no-underline"
                                    style={{ borderColor: 'var(--border-color)' }}
                                >
                                    <div>{notificationLabel(n)}</div>
                                    <div className="text-[11px] text-(--text-secondary) mt-0.5">
                                        {new Date(n.timestamp).toLocaleString()}
                                    </div>
                                </Link>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
```

- [ ] **Step 3.3 — Register NotificationProvider in main.tsx.**

In `frontend/src/main.tsx`, add the import:

```typescript
import { NotificationProvider } from './context/NotificationContext';
```

Wrap `<App />` with `<NotificationProvider>` inside `<AuthProvider>`:

```tsx
<AuthProvider>
    <NotificationProvider>
        <App />
    </NotificationProvider>
</AuthProvider>
```

The full updated provider stack becomes:
```tsx
<React.StrictMode>
    <ErrorBoundary>
        <ThemeProvider>
            <BrowserRouter>
                <AuthProvider>
                    <NotificationProvider>
                        <App />
                    </NotificationProvider>
                </AuthProvider>
            </BrowserRouter>
        </ThemeProvider>
    </ErrorBoundary>
</React.StrictMode>
```

- [ ] **Step 3.4 — Add NotificationBell to Sidebar.**

In `frontend/src/components/redesign/Sidebar.tsx`, add the import:

```typescript
import { NotificationBell } from './NotificationBell';
```

In the sidebar header div (`<div className="h-20 flex items-center gap-3 px-8">`), add the bell after the `<span>FedLearn</span>` so it floats right:

```tsx
<div className="h-20 flex items-center gap-3 px-8">
  <div className="w-8 h-8 rounded-xl bg-(--background-card) border border-(--border-color) flex items-center justify-center shadow-(--shadow-soft)">
    <Brain className="w-5 h-5 text-(--accent-primary)" />
  </div>
  <span className="font-semibold text-lg tracking-tight text-(--text-primary) flex-1">FedLearn</span>
  <NotificationBell />
</div>
```

- [ ] **Step 3.5 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 3.6 — Stage.**

```bash
git add frontend/src/context/NotificationContext.tsx \
        frontend/src/components/redesign/NotificationBell.tsx \
        frontend/src/main.tsx \
        frontend/src/components/redesign/Sidebar.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add real-time notification bell via STOMP user-queue
```

---

## Task 4 — Sidebar navigation additions

**Files:**
- Modify: `frontend/src/components/redesign/Sidebar.tsx`

Add Discover and My Requests nav items for all users; show an Admin section (Users, Projects) only when `currentUser.role === 'ADMIN'`.

- [ ] **Step 4.1 — Add new nav items to Sidebar.**

In `Sidebar.tsx`, find the `navItems` array at the top. Replace it with a split structure — base nav + admin-gated nav:

```typescript
import { Brain, LayoutDashboard, Settings, Boxes, Network, Database, LogOut, ChartLine, Compass, Inbox, ShieldCheck, Users } from 'lucide-react';

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
```

- [ ] **Step 4.2 — Render base nav + conditional admin section.**

In the `<div className="flex-1 overflow-y-auto ...">` section, replace the single `{navItems.map(...)}` with:

```tsx
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
```

Also update the user role display in the sidebar bottom section to use the actual role:

```tsx
<span className="text-[13px] text-(--text-secondary)">{currentUser?.role ?? 'User'}</span>
```

- [ ] **Step 4.3 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 4.4 — Stage.**

```bash
git add frontend/src/components/redesign/Sidebar.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add Discover, My Requests, and Admin nav items to sidebar
```

---

## Task 5 — Discover page

**Files:**
- Create: `frontend/src/pages/DiscoverPage.tsx`
- Modify: `frontend/src/App.tsx`

A page at `/discover` showing all PUBLIC projects plus PRIVATE projects the user has already requested access to. Each card shows a visibility badge and a context-sensitive CTA: **Join** for PUBLIC (auto-adds as CLIENT), **Request Access** for PRIVATE (opens an inline message form), and **Pending** / **Approved** / **Denied** chips when a request already exists.

- [ ] **Step 5.1 — Create DiscoverPage.tsx.**

Create `frontend/src/pages/DiscoverPage.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Compass, Globe, Lock } from 'lucide-react';
import { cn } from '../lib/utils';
import * as api from '../services/apiServices';
import type { DiscoverProject } from '../services/apiServices';

function VisibilityBadge({ visibility }: { visibility: 'PUBLIC' | 'PRIVATE' }) {
    return (
        <span className={cn(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
            visibility === 'PUBLIC'
                ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-500 border border-amber-500/20'
        )}>
            {visibility === 'PUBLIC' ? <Globe className="w-3 h-3" /> : <Lock className="w-3 h-3" />}
            {visibility}
        </span>
    );
}

function RequestDialog({ projectId, onSuccess, onCancel }: { projectId: string; onSuccess: () => void; onCancel: () => void }) {
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const submit = async () => {
        setLoading(true);
        try {
            await api.createAccessRequest(projectId, message || undefined);
            onSuccess();
        } catch {
            // swallow; user stays on page
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mt-3 flex flex-col gap-2">
            <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Optional message to the owner..."
                rows={2}
                maxLength={1000}
                className="w-full rounded-xl px-3 py-2 text-[13px] resize-none"
                style={{ backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
            />
            <div className="flex gap-2">
                <button
                    onClick={submit}
                    disabled={loading}
                    className="flex-1 py-2 rounded-xl text-[13px] font-semibold text-white"
                    style={{ backgroundColor: 'var(--accent-primary)' }}
                >
                    {loading ? 'Sending…' : 'Send Request'}
                </button>
                <button
                    onClick={onCancel}
                    className="flex-1 py-2 rounded-xl text-[13px] font-medium text-(--text-secondary) border"
                    style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)' }}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
}

function DiscoverCard({ project, onJoined }: { project: DiscoverProject; onJoined: (id: string) => void }) {
    const [showDialog, setShowDialog] = useState(false);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(project.myRequestStatus);

    const handleJoin = async () => {
        setLoading(true);
        try {
            await api.createAccessRequest(project.id);
            onJoined(project.id);
        } catch {
            // swallow
        } finally {
            setLoading(false);
        }
    };

    const handleRequestSuccess = () => {
        setStatus('PENDING');
        setShowDialog(false);
    };

    const ctaButton = () => {
        if (status === 'APPROVED') {
            return <span className="text-[13px] font-semibold text-emerald-500">Joined</span>;
        }
        if (status === 'PENDING') {
            return <span className="text-[13px] font-semibold text-amber-500">Request Pending</span>;
        }
        if (status === 'DENIED') {
            return (
                <button
                    onClick={() => setShowDialog(true)}
                    className="text-[13px] font-semibold text-(--text-secondary) border rounded-xl px-3 py-1.5"
                    style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)' }}
                >
                    Re-request
                </button>
            );
        }
        if (project.visibility === 'PUBLIC') {
            return (
                <button
                    onClick={handleJoin}
                    disabled={loading}
                    className="text-[13px] font-semibold text-white rounded-xl px-4 py-1.5"
                    style={{ backgroundColor: 'var(--accent-primary)' }}
                >
                    {loading ? 'Joining…' : 'Join'}
                </button>
            );
        }
        return (
            <button
                onClick={() => setShowDialog((v) => !v)}
                className="text-[13px] font-semibold rounded-xl px-4 py-1.5 border"
                style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)' }}
            >
                Request Access
            </button>
        );
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-3xl p-5 flex flex-col gap-3"
            style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
        >
            <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                    <h3 className="text-[17px] font-semibold tracking-tight text-(--text-primary) truncate">{project.name}</h3>
                    <p className="text-[12px] text-(--text-secondary) mt-0.5">by {project.ownerUsername} · {project.modelType}</p>
                </div>
                <VisibilityBadge visibility={project.visibility} />
            </div>

            {project.description && (
                <p className="text-[13px] text-(--text-secondary) line-clamp-2">{project.description}</p>
            )}

            {project.lastAccuracy != null && (
                <p className="text-[12px] text-(--text-secondary)">
                    Latest accuracy: <span className="font-semibold text-(--text-primary)">{(project.lastAccuracy * 100).toFixed(1)}%</span>
                </p>
            )}

            <div className="flex items-center justify-end mt-1">{ctaButton()}</div>

            {showDialog && (
                <RequestDialog
                    projectId={project.id}
                    onSuccess={handleRequestSuccess}
                    onCancel={() => setShowDialog(false)}
                />
            )}
        </motion.div>
    );
}

export default function DiscoverPage() {
    const [projects, setProjects] = useState<DiscoverProject[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchDiscover();
            setProjects(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load discoverable projects.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleJoined = (id: string) => {
        setProjects((prev) =>
            prev.map((p) => p.id === id ? { ...p, myRequestStatus: 'APPROVED' as const } : p)
        );
    };

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Compass className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">Discover Projects</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">Browse publicly visible projects and request access to private ones.</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {error && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error}
                    </div>
                )}
                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : projects.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
                        <p className="text-[16px] font-medium text-(--text-primary)">No discoverable projects found.</p>
                        <p className="text-[14px]">Projects become visible here when their owners make them public.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1700px] mx-auto">
                        {projects.map((p) => (
                            <DiscoverCard key={p.id} project={p} onJoined={handleJoined} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
```

- [ ] **Step 5.2 — Add route in App.tsx.**

In `frontend/src/App.tsx`, add the import:

```typescript
import DiscoverPage from './pages/DiscoverPage';
```

Inside the `<Route element={<LayoutV2 />}>` block, add:

```tsx
<Route path="/discover" element={<DiscoverPage />} />
```

- [ ] **Step 5.3 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 5.4 — Stage.**

```bash
git add frontend/src/pages/DiscoverPage.tsx \
        frontend/src/App.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add Discover page with join and request-access flow
```

---

## Task 6 — My Requests page

**Files:**
- Create: `frontend/src/pages/MyRequestsPage.tsx`
- Modify: `frontend/src/App.tsx`

A page at `/my/requests` showing the caller's access requests with status chips and timestamps.

- [ ] **Step 6.1 — Create MyRequestsPage.tsx.**

Create `frontend/src/pages/MyRequestsPage.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Inbox } from 'lucide-react';
import { cn } from '../lib/utils';
import * as api from '../services/apiServices';
import type { AccessRequest } from '../services/apiServices';

function StatusChip({ status }: { status: AccessRequest['status'] }) {
    return (
        <span className={cn(
            'inline-flex items-center px-2.5 py-0.5 rounded-full text-[12px] font-semibold uppercase tracking-wider',
            status === 'PENDING' && 'bg-amber-500/10 text-amber-500 border border-amber-500/20',
            status === 'APPROVED' && 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20',
            status === 'DENIED' && 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
        )}>
            {status}
        </span>
    );
}

export default function MyRequestsPage() {
    const [requests, setRequests] = useState<AccessRequest[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchMyAccessRequests();
            setRequests(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load requests.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Inbox className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">My Requests</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">Access requests you have submitted to private projects.</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {error && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error}
                    </div>
                )}
                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : requests.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
                        <p className="text-[16px] font-medium text-(--text-primary)">No requests yet.</p>
                        <p className="text-[14px]">Use Discover to request access to private projects.</p>
                    </div>
                ) : (
                    <div className="flex flex-col gap-3 max-w-3xl">
                        {requests.map((req) => (
                            <motion.div
                                key={req.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="rounded-2xl px-5 py-4 flex items-center justify-between gap-4"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className="flex-1 min-w-0">
                                    <p className="text-[15px] font-semibold text-(--text-primary) truncate">{req.projectName}</p>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        Requested {new Date(req.requestedAt).toLocaleDateString()}
                                        {req.decidedAt && ` · Decided ${new Date(req.decidedAt).toLocaleDateString()}`}
                                        {req.decidedByUsername && ` by ${req.decidedByUsername}`}
                                    </p>
                                    {req.message && (
                                        <p className="text-[12px] text-(--text-secondary) mt-1 italic line-clamp-1">"{req.message}"</p>
                                    )}
                                </div>
                                <StatusChip status={req.status} />
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
```

- [ ] **Step 6.2 — Add route in App.tsx.**

Add the import:

```typescript
import MyRequestsPage from './pages/MyRequestsPage';
```

Inside `<Route element={<LayoutV2 />}>`, add:

```tsx
<Route path="/my/requests" element={<MyRequestsPage />} />
```

- [ ] **Step 6.3 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 6.4 — Stage.**

```bash
git add frontend/src/pages/MyRequestsPage.tsx \
        frontend/src/App.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add My Requests page showing access request history
```

---

## Task 7 — Project Detail page

**Files:**
- Create: `frontend/src/pages/ProjectDetailPage.tsx`
- Modify: `frontend/src/App.tsx`

A page at `/projects/:projectId` with a visibility toggle (owner/admin) and tabs: **Members**, **Clients**, **Requests**, **Model Card** (stub for Plan 4). The `ProjectCard` "View Details →" link from Task 2 routes here.

- [ ] **Step 7.1 — Create ProjectDetailPage.tsx.**

Create `frontend/src/pages/ProjectDetailPage.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Globe, Lock, Plus, Trash2, Check, X } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { Project, Membership, AccessRequest, UserSearchResult } from '../services/apiServices';

type Tab = 'members' | 'clients' | 'requests' | 'model';

function VisibilityToggle({
    visibility,
    onToggle,
    canEdit,
}: {
    visibility: 'PUBLIC' | 'PRIVATE';
    onToggle: (v: 'PUBLIC' | 'PRIVATE') => void;
    canEdit: boolean;
}) {
    return (
        <div className="flex items-center gap-3">
            <span className="text-[13px] text-(--text-secondary) font-medium">Visibility:</span>
            {canEdit ? (
                <button
                    onClick={() => onToggle(visibility === 'PUBLIC' ? 'PRIVATE' : 'PUBLIC')}
                    className={cn(
                        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-semibold border transition-all',
                        visibility === 'PUBLIC'
                            ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20 hover:bg-emerald-500/20'
                            : 'bg-amber-500/10 text-amber-500 border-amber-500/20 hover:bg-amber-500/20'
                    )}
                >
                    {visibility === 'PUBLIC' ? <Globe className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
                    {visibility}
                </button>
            ) : (
                <span className={cn(
                    'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-semibold border',
                    visibility === 'PUBLIC'
                        ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                        : 'bg-amber-500/10 text-amber-500 border-amber-500/20'
                )}>
                    {visibility === 'PUBLIC' ? <Globe className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
                    {visibility}
                </span>
            )}
        </div>
    );
}

function UserSearchInput({
    label,
    onAdd,
}: {
    label: string;
    onAdd: (username: string) => Promise<void>;
}) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<UserSearchResult[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (query.length < 2) { setResults([]); return; }
        const t = setTimeout(async () => {
            try {
                const res = await api.searchUsers(query);
                setResults(res.data);
            } catch {
                setResults([]);
            }
        }, 300);
        return () => clearTimeout(t);
    }, [query]);

    const handleAdd = async (username: string) => {
        setLoading(true);
        try {
            await onAdd(username);
            setQuery('');
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative">
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={`Add ${label} by username…`}
                    className="flex-1 rounded-xl px-3 py-2 text-[13px]"
                    style={{ backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                />
                <Plus className="w-4 h-4 text-(--text-secondary)" />
            </div>
            {results.length > 0 && (
                <div
                    className="absolute top-full mt-1 left-0 right-0 z-20 rounded-xl overflow-hidden shadow-lg"
                    style={{ backgroundColor: 'var(--background-card)', border: '1px solid var(--border-color)' }}
                >
                    {results.map((u) => (
                        <button
                            key={u.id}
                            onClick={() => handleAdd(u.username)}
                            disabled={loading}
                            className="w-full px-4 py-2.5 text-left text-[13px] text-(--text-primary) hover:bg-(--background-secondary) transition-colors"
                        >
                            {u.username}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

function MembershipRow({
    m,
    canRemove,
    onRemove,
}: {
    m: Membership;
    canRemove: boolean;
    onRemove: (userId: number) => void;
}) {
    const [confirm, setConfirm] = useState(false);
    return (
        <div
            className="flex items-center justify-between px-4 py-3 rounded-xl"
            style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
            <div>
                <span className="text-[14px] font-medium text-(--text-primary)">{m.username}</span>
                {m.partitionId != null && (
                    <span className="ml-2 text-[12px] text-(--text-secondary)">partition {m.partitionId}</span>
                )}
            </div>
            {canRemove && (
                <button
                    onClick={confirm ? () => onRemove(m.userId) : () => { setConfirm(true); setTimeout(() => setConfirm(false), 3000); }}
                    className={cn(
                        'p-1.5 rounded-lg transition-colors',
                        confirm ? 'text-rose-500 bg-rose-500/10' : 'text-(--text-secondary) hover:text-rose-500'
                    )}
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            )}
        </div>
    );
}

function RequestRow({
    req,
    canDecide,
    onDecide,
}: {
    req: AccessRequest;
    canDecide: boolean;
    onDecide: (id: number, decision: 'APPROVED' | 'DENIED') => void;
}) {
    return (
        <div
            className="flex items-center justify-between px-4 py-3 rounded-xl gap-3"
            style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
            <div className="flex-1 min-w-0">
                <span className="text-[14px] font-medium text-(--text-primary)">{req.username}</span>
                <span className="text-[12px] text-(--text-secondary) ml-2">{new Date(req.requestedAt).toLocaleDateString()}</span>
                {req.message && <p className="text-[12px] text-(--text-secondary) mt-0.5 italic line-clamp-1">"{req.message}"</p>}
            </div>
            <span className={cn(
                'px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
                req.status === 'PENDING' && 'bg-amber-500/10 text-amber-500',
                req.status === 'APPROVED' && 'bg-emerald-500/10 text-emerald-500',
                req.status === 'DENIED' && 'bg-rose-500/10 text-rose-500'
            )}>{req.status}</span>
            {canDecide && req.status === 'PENDING' && (
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => onDecide(req.id, 'APPROVED')}
                        className="p-1.5 rounded-lg text-emerald-500 hover:bg-emerald-500/10 transition-colors"
                        title="Approve"
                    >
                        <Check className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => onDecide(req.id, 'DENIED')}
                        className="p-1.5 rounded-lg text-rose-500 hover:bg-rose-500/10 transition-colors"
                        title="Deny"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            )}
        </div>
    );
}

export default function ProjectDetailPage() {
    const { projectId } = useParams<{ projectId: string }>();
    const navigate = useNavigate();
    const { currentUser } = useAuth();

    const [project, setProject] = useState<Project | null>(null);
    const [memberships, setMemberships] = useState<Membership[]>([]);
    const [requests, setRequests] = useState<AccessRequest[]>([]);
    const [tab, setTab] = useState<Tab>('members');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const isOwner = project?.myRelationship === 'OWNER';
    const isMember = project?.myRelationship === 'MEMBER';
    const isAdmin = currentUser?.role === 'ADMIN';
    const canManageMembers = isOwner || isAdmin;
    const canManageClients = isOwner || isMember || isAdmin;
    const canSeeManagement = isOwner || isMember || isAdmin;

    const loadProject = useCallback(async () => {
        if (!projectId) return;
        try {
            setIsLoading(true);
            const res = await api.fetchProject(projectId);
            setProject(res.data);
            setError('');
        } catch {
            setError('Project not found or access denied.');
        } finally {
            setIsLoading(false);
        }
    }, [projectId]);

    const loadMemberships = useCallback(async () => {
        if (!projectId || !canSeeManagement) return;
        try {
            const res = await api.fetchMemberships(projectId);
            setMemberships(Array.isArray(res.data) ? res.data : []);
        } catch {
            // silently ignore if not authorized for this tab
        }
    }, [projectId, canSeeManagement]);

    const loadRequests = useCallback(async () => {
        if (!projectId || !canSeeManagement) return;
        try {
            const res = await api.fetchProjectAccessRequests(projectId);
            setRequests(Array.isArray(res.data) ? res.data : []);
        } catch {
            // silently ignore
        }
    }, [projectId, canSeeManagement]);

    useEffect(() => { loadProject(); }, [loadProject]);
    useEffect(() => {
        if (project && canSeeManagement) {
            loadMemberships();
            loadRequests();
        }
    }, [project, canSeeManagement, loadMemberships, loadRequests]);

    const handleToggleVisibility = async (next: 'PUBLIC' | 'PRIVATE') => {
        if (!projectId) return;
        try {
            const res = await api.patchProject(projectId, { visibility: next });
            setProject(res.data);
        } catch {
            // swallow
        }
    };

    const handleAddMember = async (username: string) => {
        if (!projectId) return;
        await api.addMembership(projectId, { username, role: 'MEMBER' });
        loadMemberships();
    };

    const handleAddClient = async (username: string) => {
        if (!projectId) return;
        await api.addMembership(projectId, { username, role: 'CLIENT' });
        loadMemberships();
    };

    const handleRemoveMembership = async (userId: number) => {
        if (!projectId) return;
        await api.removeMembership(projectId, userId);
        loadMemberships();
    };

    const handleDecide = async (reqId: number, decision: 'APPROVED' | 'DENIED') => {
        if (!projectId) return;
        await api.decideAccessRequest(projectId, reqId, decision);
        loadRequests();
        if (decision === 'APPROVED') loadMemberships();
    };

    const members = memberships.filter((m) => m.role === 'MEMBER');
    const clients = memberships.filter((m) => m.role === 'CLIENT');

    const tabs: { key: Tab; label: string }[] = [
        { key: 'members', label: 'Members' },
        { key: 'clients', label: 'Clients' },
        { key: 'requests', label: `Requests${requests.filter((r) => r.status === 'PENDING').length > 0 ? ` (${requests.filter((r) => r.status === 'PENDING').length})` : ''}` },
        { key: 'model', label: 'Model Card' },
    ];

    if (isLoading) {
        return <div className="flex-1 flex items-center justify-center text-(--text-secondary)">Loading…</div>;
    }
    if (error || !project) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 text-(--text-secondary)">
                <p>{error || 'Project not found.'}</p>
                <button onClick={() => navigate('/dashboard')} className="text-sm text-(--accent-primary)">← Back to Dashboard</button>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            {/* Header */}
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex flex-col gap-4">
                    <button
                        onClick={() => navigate(-1)}
                        className="inline-flex items-center gap-1.5 text-sm text-(--text-secondary) hover:text-(--text-primary) transition-colors w-fit"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back
                    </button>
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">{project.name}</h1>
                            <p className="text-sm text-(--text-secondary) mt-1">{project.modelType} · {project.modelName} · {project.status}</p>
                        </div>
                        {project.visibility && (
                            <VisibilityToggle
                                visibility={project.visibility}
                                onToggle={handleToggleVisibility}
                                canEdit={canManageMembers}
                            />
                        )}
                    </div>
                </div>
            </div>

            {/* Tabs — only visible to owner/member/admin */}
            {canSeeManagement && (
                <>
                    <div className="flex gap-1 px-8 pt-4 border-b" style={{ borderColor: 'var(--border-color)' }}>
                        {tabs.map((t) => (
                            <button
                                key={t.key}
                                onClick={() => setTab(t.key)}
                                className={cn(
                                    'px-4 py-2.5 text-[14px] font-medium rounded-t-xl transition-colors border-b-2',
                                    tab === t.key
                                        ? 'text-(--accent-primary) border-(--accent-primary) bg-(--background-card)'
                                        : 'text-(--text-secondary) border-transparent hover:text-(--text-primary)'
                                )}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>

                    <div className="flex-1 overflow-y-auto px-8 py-6">
                        {tab === 'members' && (
                            <motion.div key="members" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {canManageMembers && (
                                    <UserSearchInput label="member" onAdd={handleAddMember} />
                                )}
                                {members.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No members yet.</p>
                                ) : (
                                    members.map((m) => (
                                        <MembershipRow key={m.userId} m={m} canRemove={canManageMembers} onRemove={handleRemoveMembership} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'clients' && (
                            <motion.div key="clients" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {canManageClients && (
                                    <UserSearchInput label="client" onAdd={handleAddClient} />
                                )}
                                {clients.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No clients yet.</p>
                                ) : (
                                    clients.map((m) => (
                                        <MembershipRow key={m.userId} m={m} canRemove={canManageClients} onRemove={handleRemoveMembership} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'requests' && (
                            <motion.div key="requests" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {requests.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No access requests.</p>
                                ) : (
                                    requests.map((r) => (
                                        <RequestRow key={r.id} req={r} canDecide={canManageClients} onDecide={handleDecide} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'model' && (
                            <motion.div key="model" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl">
                                <div
                                    className="rounded-2xl p-6 text-center text-(--text-secondary)"
                                    style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
                                >
                                    <p className="text-[15px] font-medium text-(--text-primary)">Model Hub</p>
                                    <p className="text-[13px] mt-1">Model publishing and inference UI coming in Plan 4.</p>
                                </div>
                            </motion.div>
                        )}
                    </div>
                </>
            )}

            {/* For users who are only clients — no management tabs */}
            {!canSeeManagement && (
                <div className="flex-1 flex flex-col items-center justify-center text-(--text-secondary) gap-2">
                    <p className="text-[15px] font-medium text-(--text-primary)">You are a client of this project.</p>
                    <p className="text-[13px]">Connect via the Electron app to participate in training.</p>
                </div>
            )}
        </div>
    );
}
```

- [ ] **Step 7.2 — Add route in App.tsx.**

Add import:

```typescript
import ProjectDetailPage from './pages/ProjectDetailPage';
```

Inside `<Route element={<LayoutV2 />}>`, add:

```tsx
<Route path="/projects/:projectId" element={<ProjectDetailPage />} />
```

- [ ] **Step 7.3 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 7.4 — Stage.**

```bash
git add frontend/src/pages/ProjectDetailPage.tsx \
        frontend/src/App.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add project detail page with membership, clients, and request tabs
```

---

## Task 8 — Admin pages

**Files:**
- Create: `frontend/src/pages/AdminUsersPage.tsx`
- Create: `frontend/src/pages/AdminProjectsPage.tsx`
- Modify: `frontend/src/App.tsx`

Admin-only pages (redirect non-admins back to /dashboard). Users table with promote/demote role buttons; projects table with deep links.

- [ ] **Step 8.1 — Create AdminUsersPage.tsx.**

Create `frontend/src/pages/AdminUsersPage.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Users } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { AdminUser } from '../services/apiServices';

function RoleBadge({ role }: { role: 'USER' | 'ADMIN' }) {
    return (
        <span className={cn(
            'inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
            role === 'ADMIN'
                ? 'bg-blue-500/10 text-blue-500 border border-blue-500/20'
                : 'bg-(--border-color) text-(--text-secondary) border border-(--border-color)'
        )}>
            {role}
        </span>
    );
}

export default function AdminUsersPage() {
    const { currentUser } = useAuth();
    const navigate = useNavigate();
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [actionError, setActionError] = useState('');

    useEffect(() => {
        if (currentUser?.role !== 'ADMIN') {
            navigate('/dashboard', { replace: true });
        }
    }, [currentUser, navigate]);

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchAdminUsers();
            setUsers(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load users.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleRoleChange = async (user: AdminUser, newRole: 'USER' | 'ADMIN') => {
        setActionError('');
        try {
            const res = await api.updateUserRole(user.id, newRole);
            setUsers((prev) => prev.map((u) => u.id === user.id ? res.data : u));
        } catch (err: any) {
            const msg = err?.response?.data?.message || err?.response?.data || 'Failed to change role.';
            if (err?.response?.status === 409) {
                setActionError('Cannot demote the only remaining admin.');
            } else {
                setActionError(String(msg));
            }
        }
    };

    const adminCount = users.filter((u) => u.role === 'ADMIN').length;

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Users className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">Manage Users</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">{users.length} total users · {adminCount} admin{adminCount !== 1 ? 's' : ''}</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {(error || actionError) && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error || actionError}
                    </div>
                )}

                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : (
                    <div className="max-w-4xl flex flex-col gap-2">
                        {users.map((user) => (
                            <motion.div
                                key={user.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-4 px-5 py-4 rounded-2xl"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className="w-10 h-10 rounded-full bg-(--background-secondary) flex items-center justify-center text-sm font-semibold text-(--text-primary) shrink-0">
                                    {user.username.slice(0, 2).toUpperCase()}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[15px] font-semibold text-(--text-primary) truncate">{user.username}</span>
                                        <RoleBadge role={user.role} />
                                    </div>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        {user.email} · {user.projectsOwned} project{user.projectsOwned !== 1 ? 's' : ''} owned · {user.memberships} membership{user.memberships !== 1 ? 's' : ''}
                                    </p>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                    {user.role === 'USER' ? (
                                        <button
                                            onClick={() => handleRoleChange(user, 'ADMIN')}
                                            className="px-3 py-1.5 rounded-xl text-[13px] font-semibold text-white"
                                            style={{ backgroundColor: 'var(--accent-primary)' }}
                                        >
                                            Promote to Admin
                                        </button>
                                    ) : (
                                        <button
                                            onClick={() => handleRoleChange(user, 'USER')}
                                            disabled={adminCount <= 1}
                                            title={adminCount <= 1 ? 'Cannot demote the only admin' : 'Demote to User'}
                                            className={cn(
                                                'px-3 py-1.5 rounded-xl text-[13px] font-semibold border transition-all',
                                                adminCount <= 1
                                                    ? 'border-(--border-color) text-(--text-secondary) opacity-40 cursor-not-allowed'
                                                    : 'border-rose-500/30 text-rose-500 hover:bg-rose-500/10'
                                            )}
                                        >
                                            Demote to User
                                        </button>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
```

- [ ] **Step 8.2 — Create AdminProjectsPage.tsx.**

Create `frontend/src/pages/AdminProjectsPage.tsx`:

```tsx
import { useCallback, useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, Globe, Lock, ExternalLink } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { Project } from '../services/apiServices';

function statusDot(status: Project['status']) {
    if (status === 'RUNNING') return 'bg-blue-500 animate-pulse';
    if (status === 'COMPLETED') return 'bg-emerald-500';
    if (status === 'FAILED') return 'bg-rose-500';
    return 'bg-(--text-secondary)';
}

export default function AdminProjectsPage() {
    const { currentUser } = useAuth();
    const navigate = useNavigate();
    const [projects, setProjects] = useState<Project[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        if (currentUser?.role !== 'ADMIN') {
            navigate('/dashboard', { replace: true });
        }
    }, [currentUser, navigate]);

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchAdminProjects();
            setProjects(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load projects.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <ShieldCheck className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">All Projects</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">{projects.length} total projects across all users</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {error && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error}
                    </div>
                )}
                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : (
                    <div className="max-w-5xl flex flex-col gap-2">
                        {projects.map((p) => (
                            <motion.div
                                key={p.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-4 px-5 py-4 rounded-2xl"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className={cn('w-2 h-2 rounded-full shrink-0', statusDot(p.status))} />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[15px] font-semibold text-(--text-primary) truncate">{p.name}</span>
                                        {p.visibility === 'PUBLIC'
                                            ? <Globe className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                                            : <Lock className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                                        }
                                    </div>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        {p.modelType} · {p.modelName} · {p.status}
                                    </p>
                                </div>
                                <Link
                                    to={`/projects/${p.id}`}
                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-medium border transition-all hover:bg-(--background-secondary) text-(--text-secondary) hover:text-(--text-primary)"
                                    style={{ borderColor: 'var(--border-color)' }}
                                >
                                    <ExternalLink className="w-3.5 h-3.5" />
                                    View
                                </Link>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
```

- [ ] **Step 8.3 — Add routes in App.tsx.**

Add imports:

```typescript
import AdminUsersPage from './pages/AdminUsersPage';
import AdminProjectsPage from './pages/AdminProjectsPage';
```

Inside `<Route element={<LayoutV2 />}>`, add:

```tsx
<Route path="/admin/users" element={<AdminUsersPage />} />
<Route path="/admin/projects" element={<AdminProjectsPage />} />
```

- [ ] **Step 8.4 — Verify TypeScript.**

Run: `cd frontend && npx tsc --noEmit`
Expected: no output.

- [ ] **Step 8.5 — Stage.**

```bash
git add frontend/src/pages/AdminUsersPage.tsx \
        frontend/src/pages/AdminProjectsPage.tsx \
        frontend/src/App.tsx
```

Draft commit (do NOT run without user approval):
```
feat(frontend): add admin pages for user role management and project oversight
```
