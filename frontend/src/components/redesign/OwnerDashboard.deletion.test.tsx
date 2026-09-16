import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { OwnerDashboard } from './OwnerDashboard';
import * as api from '../../services/apiServices';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';

// Mock only the network calls; the pure helpers stay REAL. isEmptyBody in
// particular: the "no deletion request pending yet" branch must track the
// actual empty-body semantics (a 204 reaches axios as ''), not a copy of the
// helper pasted into the test that would keep passing if the real one drifted.
vi.mock('../../services/apiServices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../services/apiServices')>();
  return {
    ...actual,
    fetchOwnedProjects: vi.fn(),
    fetchProjectResults: vi.fn(),
    fetchProjectDeletionRequest: vi.fn(),
    submitDeletionRequest: vi.fn(),
    deleteProject: vi.fn(),
  };
});
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));
vi.mock('../../context/AuthContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../context/AuthContext')>();
  return { ...actual, useAuth: vi.fn() };
});
const mockedUseAuth = vi.mocked(useAuth);

/** Minimal AxiosResponse wrapper — the dashboard only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

const PROJECT: api.OwnedProject = {
  id: 'p1',
  name: 'Fraud model',
  modelType: 'CNN',
  modelName: 'net',
  optimizer: 'Adam',
  status: 'CREATED',
  visibility: 'PRIVATE',
  myRelationship: 'OWNER',
};

async function renderAndOpenProjectMenu() {
  render(<OwnerDashboard />);
  await screen.findByText('Fraud model');
  fireEvent.click(screen.getByRole('button', { name: 'Project actions' }));
}

// FE-2: DELETE /projects/{id} is 403 for non-admins on the backend, so the card
// must route owners to the deletion-request approval flow and only admins to
// the hard delete. Rendered through OwnerDashboard — the live surface — so the
// test covers both the ProjectCard role routing and the dashboard's api wiring.
describe('OwnerDashboard/ProjectCard — deletion routes by role (FE-2)', () => {
  beforeEach(() => {
    // Inert STOMP client so the dashboard never opens a real socket.
    vi.mocked(StompClient).mockImplementation(() => ({
      onConnect: null,
      active: false,
      activate: vi.fn(),
      deactivate: vi.fn(),
      subscribe: vi.fn(),
    }) as unknown as InstanceType<typeof StompClient>);
    vi.mocked(api.fetchOwnedProjects).mockResolvedValue(resp([PROJECT]));
    vi.mocked(api.fetchProjectResults).mockResolvedValue(resp([]));
    // 204 ⇒ axios yields '' — the real isEmptyBody must read this as
    // "no deletion request pending yet" or the card hides the request flow.
    vi.mocked(api.fetchProjectDeletionRequest).mockResolvedValue(resp(''));
  });

  it('a non-admin owner files a deletion request — never DELETE /projects/{id}', async () => {
    mockedUseAuth.mockReturnValue(
      makeAuth({
        currentUser: { username: 'olive', email: 'olive@example.com', role: 'PROJECT_OWNER' },
        isOwner: true,
      }),
    );
    vi.mocked(api.submitDeletionRequest).mockResolvedValue(resp({} as api.DeletionRequest));

    await renderAndOpenProjectMenu();

    // Owners see the approval flow, not the hard delete. (FE-13: the kebab items are ARIA menuitems.)
    expect(screen.queryByRole('menuitem', { name: 'Delete project' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Request deletion' }));
    // Reason-capture modal → confirm the request (reason left blank). The modal button is a plain button.
    fireEvent.click(screen.getByRole('button', { name: 'Request deletion' }));

    await waitFor(() => expect(api.submitDeletionRequest).toHaveBeenCalledWith('p1', undefined));
    expect(api.deleteProject).not.toHaveBeenCalled();
  });

  it('a platform admin hard-deletes — never the request flow', async () => {
    mockedUseAuth.mockReturnValue(
      makeAuth({
        currentUser: { username: 'root', email: 'root@example.com', role: 'PLATFORM_ADMIN' },
        isAdmin: true,
        isOwner: true,
      }),
    );
    vi.mocked(api.deleteProject).mockResolvedValue(resp({ projectId: 'p1', message: 'deleted' }));

    await renderAndOpenProjectMenu();

    expect(screen.queryByRole('menuitem', { name: 'Request deletion' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Delete project' }));
    // ConfirmDialog → confirm (a plain button in the shared Modal).
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(api.deleteProject).toHaveBeenCalledWith('p1'));
    expect(api.submitDeletionRequest).not.toHaveBeenCalled();
  });

  it('FE-13: the actions menu has ARIA menu semantics; arrow keys roam and Escape closes + restores focus', async () => {
    mockedUseAuth.mockReturnValue(
      makeAuth({
        currentUser: { username: 'olive', email: 'olive@example.com', role: 'PROJECT_OWNER' },
        isOwner: true,
      }),
    );

    render(<OwnerDashboard />);
    await screen.findByText('Fraud model');
    const trigger = screen.getByRole('button', { name: 'Project actions' });
    expect(trigger).toHaveAttribute('aria-haspopup', 'menu');
    expect(trigger).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(trigger);
    const menu = screen.getByRole('menu', { name: 'Project actions' });
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    // Opening the menu moves focus onto the first item (WAI-ARIA menu-button pattern).
    const items = screen.getAllByRole('menuitem');
    expect(items[0]).toHaveFocus();

    fireEvent.keyDown(menu, { key: 'ArrowDown' });
    expect(items[1]).toHaveFocus();

    // Escape closes the menu and returns focus to the trigger — no lost place.
    fireEvent.keyDown(menu, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('menu')).not.toBeInTheDocument());
    expect(trigger).toHaveFocus();
  });
});
