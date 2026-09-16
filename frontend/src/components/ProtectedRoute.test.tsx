import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';
import ProtectedRoute from './ProtectedRoute';
import { useAuth } from '../context/AuthContext';
import { makeAuth } from '../test/authFixtures';

vi.mock('../context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
const mockedUseAuth = vi.mocked(useAuth);

// Stands in for the login page and reports the route the user was bounced from.
function LoginProbe() {
  const location = useLocation() as { state?: { from?: { pathname?: string } } };
  return <div>login page — from: {location.state?.from?.pathname ?? 'none'}</div>;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route element={<ProtectedRoute />}>
          <Route path="/secret" element={<div>secret content</div>} />
        </Route>
        <Route path="/login" element={<LoginProbe />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ProtectedRoute', () => {
  beforeEach(() => mockedUseAuth.mockReset());

  it('shows the auth-check loader while the session is resolving', () => {
    mockedUseAuth.mockReturnValue(makeAuth({ isLoading: true }));
    renderAt('/secret');
    expect(screen.getByText('Checking authentication...')).toBeInTheDocument();
    expect(screen.queryByText('secret content')).not.toBeInTheDocument();
  });

  it('renders the nested route when authenticated', () => {
    mockedUseAuth.mockReturnValue(makeAuth({ currentUser: { username: 'a', email: 'a@x.io', role: 'USER' } }));
    renderAt('/secret');
    expect(screen.getByText('secret content')).toBeInTheDocument();
  });

  it('redirects to /login and remembers the intended route (FE-10)', () => {
    mockedUseAuth.mockReturnValue(makeAuth({ currentUser: null }));
    renderAt('/secret');
    expect(screen.getByText(/login page/)).toBeInTheDocument();
    expect(screen.getByText(/from: \/secret/)).toBeInTheDocument();
  });
});
