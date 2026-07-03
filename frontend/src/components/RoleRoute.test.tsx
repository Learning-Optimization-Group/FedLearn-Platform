import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import RoleRoute from './RoleRoute';
import { useAuth, type Role } from '../context/AuthContext';
import { makeAuth } from '../test/authFixtures';

vi.mock('../context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
const mockedUseAuth = vi.mocked(useAuth);

function renderRoleGate(allow: Role[]) {
  return render(
    <MemoryRouter initialEntries={['/admin']}>
      <Routes>
        <Route element={<RoleRoute allow={allow} />}>
          <Route path="/admin" element={<div>admin area</div>} />
        </Route>
        <Route path="/dashboard" element={<div>dashboard landing</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('RoleRoute', () => {
  beforeEach(() => mockedUseAuth.mockReset());

  it('renders the nested route when the role is allowed', () => {
    mockedUseAuth.mockReturnValue(
      makeAuth({ currentUser: { username: 'admin', email: 'admin@x.io', role: 'PLATFORM_ADMIN' }, isAdmin: true, isOwner: true }),
    );
    renderRoleGate(['PLATFORM_ADMIN']);
    expect(screen.getByText('admin area')).toBeInTheDocument();
  });

  it('redirects to /dashboard when the role is not allowed', () => {
    mockedUseAuth.mockReturnValue(
      makeAuth({ currentUser: { username: 'u', email: 'u@x.io', role: 'USER' } }),
    );
    renderRoleGate(['PLATFORM_ADMIN']);
    expect(screen.getByText('dashboard landing')).toBeInTheDocument();
    expect(screen.queryByText('admin area')).not.toBeInTheDocument();
  });

  it('redirects when there is no user at all', () => {
    mockedUseAuth.mockReturnValue(makeAuth({ currentUser: null }));
    renderRoleGate(['USER', 'PROJECT_OWNER', 'PLATFORM_ADMIN']);
    expect(screen.getByText('dashboard landing')).toBeInTheDocument();
  });
});
