import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';
import { useAuth, type Role } from './context/AuthContext';
import { makeAuth } from './test/authFixtures';

vi.mock('./context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
// The two surfaces this spec can land on are stubbed so rendering them fires
// no data fetching. Everything under test stays real: App's routing table,
// ProtectedRoute, and RoleRoute.
vi.mock('./components/redesign/NodeNetwork', () => ({
  NodeNetwork: () => <div>node-network-surface</div>,
}));
vi.mock('./components/redesign/RoleDashboard', () => ({
  RoleDashboard: () => <div>role-dashboard-surface</div>,
}));
const mockedUseAuth = vi.mocked(useAuth);

function signInAs(role: Role) {
  mockedUseAuth.mockReturnValue(
    makeAuth({
      currentUser: { username: 'u', email: 'u@example.com', role },
      isAdmin: role === 'PLATFORM_ADMIN',
      isOwner: role !== 'USER',
    }),
  );
}

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

// FE-1: /nodes is platform user-account management backed by an admin-only
// endpoint. App.tsx must keep it inside the PLATFORM_ADMIN RoleRoute block —
// any other authenticated role is bounced to /dashboard, never onto the admin
// surface.
describe('App routing — /nodes is gated to PLATFORM_ADMIN (FE-1)', () => {
  it('bounces a plain USER to /dashboard', async () => {
    signInAs('USER');
    renderAt('/nodes');

    expect(await screen.findByText('role-dashboard-surface')).toBeInTheDocument();
    expect(screen.queryByText('node-network-surface')).not.toBeInTheDocument();
  });

  it('bounces a PROJECT_OWNER to /dashboard — owner is not enough', async () => {
    signInAs('PROJECT_OWNER');
    renderAt('/nodes');

    expect(await screen.findByText('role-dashboard-surface')).toBeInTheDocument();
    expect(screen.queryByText('node-network-surface')).not.toBeInTheDocument();
  });

  it('lets a PLATFORM_ADMIN through', async () => {
    signInAs('PLATFORM_ADMIN');
    renderAt('/nodes');

    expect(await screen.findByText('node-network-surface')).toBeInTheDocument();
    expect(screen.queryByText('role-dashboard-surface')).not.toBeInTheDocument();
  });
});
