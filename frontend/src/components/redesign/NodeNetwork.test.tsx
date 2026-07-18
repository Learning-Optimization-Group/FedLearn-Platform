import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { AxiosResponse } from 'axios';
import { NodeNetwork } from './NodeNetwork';
import * as api from '../../services/apiServices';
import type { AdminUser, Paged } from '../../services/apiServices';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';

// Keep the REAL error helpers (errorMessage/errorStatus) so the 409-guard
// paths render the backend's message; stub only the calls the view makes.
vi.mock('../../services/apiServices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../services/apiServices')>();
  return {
    ...actual,
    searchAdminUsers: vi.fn(),
    updateUserRole: vi.fn(),
    updateUserStatus: vi.fn(),
    createUser: vi.fn(),
    deleteUser: vi.fn(),
  };
});
vi.mock('../../context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
const mockedUseAuth = vi.mocked(useAuth);

/** Minimal AxiosResponse wrapper — the view only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

/** Axios-shaped rejection that satisfies isAxiosError so errorMessage extracts `message`. */
function axiosErr(status: number, message: string) {
  return { isAxiosError: true, response: { status, data: { message } } };
}

function user(overrides: Partial<AdminUser> = {}): AdminUser {
  return {
    id: 2,
    username: 'bob',
    email: 'bob@example.com',
    role: 'USER',
    projectsOwned: 0,
    memberships: 2,
    createdAt: '2026-01-05T00:00:00Z',
    status: 'ACTIVE',
    displayName: 'Bob Builder',
    lastLoginAt: '2026-07-16T10:00:00Z',
    ...overrides,
  };
}

const ROOT = user({
  id: 1,
  username: 'root',
  email: 'root@example.com',
  role: 'PLATFORM_ADMIN',
  projectsOwned: 3,
  displayName: 'Root Admin',
});
const BOB = user();

function paged(items: AdminUser[], over: Partial<Paged<AdminUser>> = {}): Paged<AdminUser> {
  return { items, page: 0, size: 25, total: items.length, ...over };
}

/** The view reads/writes the URL query string, so it needs a router. */
function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/nodes']}>
      <NodeNetwork />
    </MemoryRouter>,
  );
}

/** Open the manage drawer for a row and return the drawer dialog element. */
async function openDrawer(username: string) {
  fireEvent.click(await screen.findByRole('button', { name: `Manage ${username}` }));
  return await screen.findByRole('dialog');
}

/** The topmost dialog — the confirm layer stacks above the drawer. */
function topDialog() {
  const dialogs = screen.getAllByRole('dialog');
  return dialogs[dialogs.length - 1];
}

beforeEach(() => {
  // Signed in as "root"; the directory contains root plus one other account.
  mockedUseAuth.mockReturnValue(
    makeAuth({
      currentUser: { username: 'root', email: 'root@example.com', role: 'PLATFORM_ADMIN' },
      isAdmin: true,
      isOwner: true,
    }),
  );
  vi.mocked(api.searchAdminUsers).mockResolvedValue(resp(paged([ROOT, BOB])));
});

describe('NodeNetwork — search-first users directory', () => {
  it('renders rows with role chip (not a live select), status pill, and the result range', async () => {
    renderPage();

    expect(await screen.findByText('bob')).toBeInTheDocument();
    await waitFor(() =>
      expect(api.searchAdminUsers).toHaveBeenCalledWith(expect.objectContaining({ page: 0, size: 25 })),
    );

    // User cell shows the displayName caption under the username.
    const bobRow = screen.getByText('bob').closest('tr')!;
    expect(within(bobRow).getByText('Bob Builder')).toBeInTheDocument();
    expect(within(bobRow).getByText('bob@example.com')).toBeInTheDocument();
    // Role is a quiet chip in the row, NOT a live select — the only comboboxes
    // on the page are the two filters.
    expect(within(bobRow).getByText('User')).toBeInTheDocument();
    expect(within(bobRow).queryByRole('combobox')).not.toBeInTheDocument();
    expect(screen.getAllByRole('combobox')).toHaveLength(2);
    // Status pill + owned count + relative last-active.
    expect(within(bobRow).getByText('Active')).toBeInTheDocument();
    expect(within(bobRow).getByText('0')).toBeInTheDocument();

    expect(screen.getByText('1–2 of 2')).toBeInTheDocument();
  });

  it('debounces the search box and resets to the first page', async () => {
    renderPage();
    await screen.findByText('bob');
    vi.mocked(api.searchAdminUsers).mockClear();

    fireEvent.change(screen.getByRole('textbox', { name: 'Search users' }), { target: { value: 'bo' } });
    // Debounced: no immediate call on each keystroke.
    expect(api.searchAdminUsers).not.toHaveBeenCalled();

    await waitFor(() =>
      expect(api.searchAdminUsers).toHaveBeenCalledWith(expect.objectContaining({ q: 'bo', page: 0 })),
    );
  });

  it('role and status filters drive the search query', async () => {
    renderPage();
    await screen.findByText('bob');

    fireEvent.change(screen.getByRole('combobox', { name: 'Filter by role' }), {
      target: { value: 'USER' },
    });
    await waitFor(() =>
      expect(api.searchAdminUsers).toHaveBeenCalledWith(expect.objectContaining({ role: 'USER', page: 0 })),
    );

    fireEvent.change(screen.getByRole('combobox', { name: 'Filter by status' }), {
      target: { value: 'SUSPENDED' },
    });
    await waitFor(() =>
      expect(api.searchAdminUsers).toHaveBeenCalledWith(
        expect.objectContaining({ status: 'SUSPENDED', page: 0 }),
      ),
    );
  });

  it('pages forward with an x–y of N range readout', async () => {
    vi.mocked(api.searchAdminUsers).mockResolvedValue(resp(paged([ROOT, BOB], { total: 60 })));
    renderPage();

    expect(await screen.findByText('1–25 of 60')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();

    vi.mocked(api.searchAdminUsers).mockResolvedValue(resp(paged([BOB], { page: 1, total: 60 })));
    fireEvent.click(screen.getByRole('button', { name: 'Next' }));

    await waitFor(() =>
      expect(api.searchAdminUsers).toHaveBeenCalledWith(expect.objectContaining({ page: 1 })),
    );
    expect(await screen.findByText('26–50 of 60')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
  });
});

describe('NodeNetwork — manage drawer', () => {
  it('shows the identity block for the selected user', async () => {
    renderPage();
    const drawer = await openDrawer('bob');

    expect(within(drawer).getByText('Bob Builder')).toBeInTheDocument();
    expect(within(drawer).getByText('bob@example.com')).toBeInTheDocument();
    expect(within(drawer).getByText('Member since')).toBeInTheDocument();
    expect(within(drawer).getByText('Last login')).toBeInTheDocument();
  });

  it('role change is confirm-gated: select + button, dialog copy, call only on confirm', async () => {
    vi.mocked(api.updateUserRole).mockResolvedValue(resp({ ...BOB, role: 'PROJECT_OWNER' }));
    renderPage();
    const drawer = await openDrawer('bob');

    const roleSelect = within(drawer).getByRole('combobox', { name: 'New role for bob' });
    // Current role is preselected and the CTA is disabled until it changes.
    expect(roleSelect).toHaveValue('USER');
    expect(within(drawer).getByRole('button', { name: 'Change role' })).toBeDisabled();

    fireEvent.change(roleSelect, { target: { value: 'PROJECT_OWNER' } });
    fireEvent.click(within(drawer).getByRole('button', { name: 'Change role' }));
    expect(api.updateUserRole).not.toHaveBeenCalled();

    const confirm = topDialog();
    expect(within(confirm).getByText(/Change bob from User to Owner\?/)).toBeInTheDocument();
    expect(
      within(confirm).getByText(/Owners can create and run projects; Admins manage the whole platform\./),
    ).toBeInTheDocument();

    fireEvent.click(within(confirm).getByRole('button', { name: 'Change role' }));
    await waitFor(() => expect(api.updateUserRole).toHaveBeenCalledWith(2, 'PROJECT_OWNER'));

    // The drawer reflects the server response: new role preselected, CTA idle.
    await waitFor(() =>
      expect(within(drawer).getByRole('combobox', { name: 'New role for bob' })).toHaveValue('PROJECT_OWNER'),
    );
    expect(within(drawer).getByRole('button', { name: 'Change role' })).toBeDisabled();
  });

  it('surfaces the 409 last-admin guard inline on role change', async () => {
    vi.mocked(api.updateUserRole).mockRejectedValue(
      axiosErr(409, 'You cannot demote the last platform admin.'),
    );
    renderPage();
    const drawer = await openDrawer('root');

    fireEvent.change(within(drawer).getByRole('combobox', { name: 'New role for root' }), {
      target: { value: 'USER' },
    });
    fireEvent.click(within(drawer).getByRole('button', { name: 'Change role' }));
    fireEvent.click(within(topDialog()).getByRole('button', { name: 'Change role' }));

    expect(
      await within(drawer).findByText('You cannot demote the last platform admin.'),
    ).toBeInTheDocument();
  });
});

describe('NodeNetwork — suspend / reactivate', () => {
  it('suspend is confirm-gated with the sign-out warning, then offers reactivate', async () => {
    vi.mocked(api.updateUserStatus).mockResolvedValue(resp({ ...BOB, status: 'SUSPENDED' }));
    renderPage();
    const drawer = await openDrawer('bob');

    fireEvent.click(within(drawer).getByRole('button', { name: 'Suspend account' }));
    expect(api.updateUserStatus).not.toHaveBeenCalled();

    const confirm = topDialog();
    expect(
      within(confirm).getByText(
        'bob will be signed out and blocked from all API access until reactivated.',
      ),
    ).toBeInTheDocument();

    fireEvent.click(within(confirm).getByRole('button', { name: 'Suspend' }));
    await waitFor(() => expect(api.updateUserStatus).toHaveBeenCalledWith(2, 'SUSPENDED'));

    // The drawer flips to the reactivate entry once the account is suspended.
    expect(await within(drawer).findByRole('button', { name: 'Reactivate account' })).toBeInTheDocument();
  });

  it('reactivate is confirm-gated and calls updateUserStatus with ACTIVE', async () => {
    vi.mocked(api.searchAdminUsers).mockResolvedValue(
      resp(paged([ROOT, user({ status: 'SUSPENDED' })])),
    );
    vi.mocked(api.updateUserStatus).mockResolvedValue(resp(user({ status: 'ACTIVE' })));
    renderPage();
    const drawer = await openDrawer('bob');

    fireEvent.click(within(drawer).getByRole('button', { name: 'Reactivate account' }));
    expect(api.updateUserStatus).not.toHaveBeenCalled();
    fireEvent.click(within(topDialog()).getByRole('button', { name: 'Reactivate' }));

    await waitFor(() => expect(api.updateUserStatus).toHaveBeenCalledWith(2, 'ACTIVE'));
  });

  it('surfaces the 409 guard inline when suspension is refused', async () => {
    vi.mocked(api.updateUserStatus).mockRejectedValue(
      axiosErr(409, 'You cannot suspend the last active platform admin.'),
    );
    renderPage();
    const drawer = await openDrawer('root');

    fireEvent.click(within(drawer).getByRole('button', { name: 'Suspend account' }));
    fireEvent.click(within(topDialog()).getByRole('button', { name: 'Suspend' }));

    expect(
      await within(drawer).findByText('You cannot suspend the last active platform admin.'),
    ).toBeInTheDocument();
  });
});

// FE-1: this is account management, not a device inventory. Removing a user
// PERMANENTLY DELETES the account, and the confirmation must say so and name
// its target — no euphemistic "Remove device" copy. (The entry point moved
// into the manage drawer; the dialog contract is unchanged.)
describe('NodeNetwork — user removal is honest about what it does (FE-1)', () => {
  it('confirmation names the target account and says it is permanently deleted', async () => {
    renderPage();
    const drawer = await openDrawer('bob');
    fireEvent.click(within(drawer).getByRole('button', { name: 'Remove user bob' }));

    const dialog = topDialog();
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
    renderPage();
    const drawer = await openDrawer('bob');

    fireEvent.click(within(drawer).getByRole('button', { name: 'Remove user bob' }));
    expect(api.deleteUser).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Permanently delete' }));
    await waitFor(() => expect(api.deleteUser).toHaveBeenCalledWith(2));
  });

  it('refuses to remove the signed-in account', async () => {
    renderPage();
    const drawer = await openDrawer('root');

    const selfRemove = within(drawer).getByRole('button', { name: 'You cannot remove your own account' });
    expect(selfRemove).toBeDisabled();
    expect(api.deleteUser).not.toHaveBeenCalled();

    // The guard is self-only: another account's remove entry stays live.
    fireEvent.click(within(drawer).getByRole('button', { name: 'Close' }));
    const bobDrawer = await openDrawer('bob');
    expect(within(bobDrawer).getByRole('button', { name: 'Remove user bob' })).toBeEnabled();
  });
});

describe('NodeNetwork — add-user flow', () => {
  it('creates the account and reloads the directory', async () => {
    vi.mocked(api.createUser).mockResolvedValue(
      resp({ id: 9, username: 'anna', email: 'anna@example.com' } as api.User),
    );
    renderPage();
    await screen.findByText('bob');

    fireEvent.click(screen.getByRole('button', { name: /add user/i }));
    const dialog = await screen.findByRole('dialog');

    fireEvent.change(within(dialog).getByLabelText('Username'), { target: { value: 'anna' } });
    fireEvent.change(within(dialog).getByLabelText('Email'), { target: { value: 'anna@example.com' } });
    fireEvent.change(within(dialog).getByLabelText('Password'), { target: { value: 'pw12345678' } });

    vi.mocked(api.searchAdminUsers).mockClear();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Add user' }));

    await waitFor(() =>
      expect(api.createUser).toHaveBeenCalledWith({
        username: 'anna',
        email: 'anna@example.com',
        password: 'pw12345678',
      }),
    );
    await waitFor(() => expect(api.searchAdminUsers).toHaveBeenCalled());
  });
});
