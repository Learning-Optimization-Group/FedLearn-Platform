import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { NodeNetwork } from './NodeNetwork';
import * as api from '../../services/apiServices';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';

vi.mock('../../services/apiServices');
vi.mock('../../context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
const mockedUseAuth = vi.mocked(useAuth);

/** Minimal AxiosResponse wrapper — the view only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

const USERS: api.User[] = [
  { id: 1, username: 'root', email: 'root@example.com' },
  { id: 2, username: 'bob', email: 'bob@example.com' },
];

// FE-1: this is account management, not a device inventory. Removing a user
// PERMANENTLY DELETES the account, and the confirmation must say so and name
// its target — no euphemistic "Remove device" copy.
describe('NodeNetwork — user removal is honest about what it does (FE-1)', () => {
  beforeEach(() => {
    // Signed in as "root"; the list contains root plus one other account.
    mockedUseAuth.mockReturnValue(
      makeAuth({
        currentUser: { username: 'root', email: 'root@example.com', role: 'PLATFORM_ADMIN' },
        isAdmin: true,
        isOwner: true,
      }),
    );
    vi.mocked(api.fetchUsers).mockResolvedValue(resp(USERS));
  });

  it('confirmation names the target account and says it is permanently deleted', async () => {
    render(<NodeNetwork />);
    fireEvent.click(await screen.findByRole('button', { name: 'Remove user bob' }));

    const dialog = await screen.findByRole('dialog');
    expect(
      within(dialog).getByText(/PERMANENTLY DELETES the account "bob" \(bob@example\.com\)/),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/cannot be undone/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Permanently delete' })).toBeInTheDocument();
    // The old device-inventory framing must be gone everywhere on the page.
    expect(screen.queryByText(/remove device/i)).not.toBeInTheDocument();
  });

  it('deletes the named account, and only after confirmation', async () => {
    vi.mocked(api.deleteUser).mockResolvedValue(resp(undefined));
    render(<NodeNetwork />);

    fireEvent.click(await screen.findByRole('button', { name: 'Remove user bob' }));
    expect(api.deleteUser).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Permanently delete' }));
    await waitFor(() => expect(api.deleteUser).toHaveBeenCalledWith(2));
  });

  it('refuses to remove the signed-in account', async () => {
    render(<NodeNetwork />);
    const selfRemove = await screen.findByRole('button', { name: 'You cannot remove your own account' });
    expect(selfRemove).toBeDisabled();
    // The guard is self-only: the other account's control stays live.
    expect(screen.getByRole('button', { name: 'Remove user bob' })).toBeEnabled();
    expect(api.deleteUser).not.toHaveBeenCalled();
  });
});
