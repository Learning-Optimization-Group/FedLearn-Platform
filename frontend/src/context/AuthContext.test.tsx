import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import { AuthProvider, useAuth, type Role } from './AuthContext';
import * as api from '../services/apiServices';

vi.mock('../services/apiServices');
const mockedFetch = vi.mocked(api.fetchCurrentUser);

function identity(role: Role) {
  return { data: { username: 'u', email: 'u@x.io', role } } as Awaited<ReturnType<typeof api.fetchCurrentUser>>;
}

function Consumer() {
  const { currentUser, isLoading, isOwner } = useAuth();
  return (
    <div>
      <span>loading:{String(isLoading)}</span>
      <span>role:{currentUser?.role ?? 'none'}</span>
      <span>owner:{String(isOwner)}</span>
    </div>
  );
}

describe('AuthContext', () => {
  beforeEach(() => vi.clearAllMocks());

  it('bootstraps the session from /auth/me', async () => {
    mockedFetch.mockResolvedValue(identity('USER'));
    render(<AuthProvider><Consumer /></AuthProvider>);
    await waitFor(() => expect(screen.getByText('role:USER')).toBeInTheDocument());
    expect(screen.getByText('loading:false')).toBeInTheDocument();
  });

  it('stays anonymous when /auth/me rejects (no valid cookie)', async () => {
    mockedFetch.mockRejectedValue(new Error('401'));
    render(<AuthProvider><Consumer /></AuthProvider>);
    await waitFor(() => expect(screen.getByText('loading:false')).toBeInTheDocument());
    expect(screen.getByText('role:none')).toBeInTheDocument();
  });

  it('re-fetches identity on window focus so a server-side role change surfaces without reload (FE-10)', async () => {
    mockedFetch.mockResolvedValueOnce(identity('USER'));   // bootstrap
    render(<AuthProvider><Consumer /></AuthProvider>);
    await waitFor(() => expect(screen.getByText('role:USER')).toBeInTheDocument());
    expect(screen.getByText('owner:false')).toBeInTheDocument();

    // Admin promoted this user server-side; the next /auth/me carries the new role.
    mockedFetch.mockResolvedValue(identity('PROJECT_OWNER'));
    act(() => { window.dispatchEvent(new Event('focus')); });

    await waitFor(() => expect(screen.getByText('role:PROJECT_OWNER')).toBeInTheDocument());
    expect(screen.getByText('owner:true')).toBeInTheDocument();
  });
});
