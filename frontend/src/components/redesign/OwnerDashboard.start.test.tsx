import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { OwnerDashboard } from './OwnerDashboard';
import * as api from '../../services/apiServices';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';

// Mock only the network calls; errorMessage stays REAL so the failure path is
// exercised end to end — dashboard handler → rethrow → modal → real
// backend-detail extraction.
vi.mock('../../services/apiServices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../services/apiServices')>();
  return {
    ...actual,
    fetchOwnedProjects: vi.fn(),
    fetchProjectResults: vi.fn(),
    fetchProjectDeletionRequest: vi.fn(),
    startProjectServer: vi.fn(),
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

/** The shape axios errors take when the backend returns a message body. */
function backendError(message: string): Error {
  return Object.assign(new Error(message), {
    isAxiosError: true,
    response: { data: { message } },
  });
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

async function renderAndOpenStartModal() {
  render(<OwnerDashboard />);
  await screen.findByText('Fraud model');
  fireEvent.click(screen.getByRole('button', { name: 'Start' }));
  return screen.findByRole('dialog');
}

// FE-4: StartProjectModal.test.tsx covers the modal half with a mocked
// onSubmit. This covers the dashboard half: handleStartSubmit must let a
// rejected api.startProjectServer propagate INTO the modal — swallowing it
// (dashboard setError + no rethrow) would flash a banner behind the modal and
// leave the modal blank.
describe('OwnerDashboard — a failed start propagates into the modal (FE-4)', () => {
  beforeEach(() => {
    // Inert STOMP client so the dashboard never opens a real socket.
    vi.mocked(StompClient).mockImplementation(() => ({
      onConnect: null,
      active: false,
      activate: vi.fn(),
      deactivate: vi.fn(),
      subscribe: vi.fn(),
    }) as unknown as InstanceType<typeof StompClient>);
    mockedUseAuth.mockReturnValue(
      makeAuth({
        currentUser: { username: 'olive', email: 'olive@example.com', role: 'PROJECT_OWNER' },
        isOwner: true,
      }),
    );
    vi.mocked(api.fetchOwnedProjects).mockResolvedValue(resp([PROJECT]));
    vi.mocked(api.fetchProjectResults).mockResolvedValue(resp([]));
    vi.mocked(api.fetchProjectDeletionRequest).mockResolvedValue(resp(''));
  });

  it('keeps the modal open with the backend detail inline — not on the route behind it', async () => {
    vi.mocked(api.startProjectServer).mockRejectedValue(
      backendError('No free port available in the 50000-50010 range.'),
    );

    const dialog = await renderAndOpenStartModal();
    fireEvent.click(within(dialog).getByRole('button', { name: /start training/i }));

    // The detail must render INSIDE the dialog — a dashboard-banner rendering
    // (the swallowed-error regression) does not satisfy this query.
    expect(
      await within(dialog).findByText('No free port available in the 50000-50010 range.'),
    ).toBeInTheDocument();
    // Still open and retryable.
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: /start training/i })).toBeEnabled();
    expect(api.startProjectServer).toHaveBeenCalledWith('p1', {
      strategy: 'FedAvg',
      numRounds: 5,
      minClients: 2,
    });
  });

  it('closes the modal and flips the card to RUNNING on success', async () => {
    vi.mocked(api.startProjectServer).mockResolvedValue(
      resp({ ...PROJECT, status: 'RUNNING' as const }),
    );

    const dialog = await renderAndOpenStartModal();
    fireEvent.click(within(dialog).getByRole('button', { name: /start training/i }));

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    // The card reflects the server response: a running project offers Stop.
    expect(screen.getByRole('button', { name: 'Stop' })).toBeInTheDocument();
  });
});
