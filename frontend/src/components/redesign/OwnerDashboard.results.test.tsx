import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { OwnerDashboard } from './OwnerDashboard';
import * as api from '../../services/apiServices';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';

vi.mock('../../services/apiServices');
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));
vi.mock('../../context/AuthContext', async (importOriginal) => ({
  ...await importOriginal<typeof import('../../context/AuthContext')>(),
  useAuth: vi.fn(),
}));

function response<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as AxiosResponse<T>;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

const project: api.OwnedProject = {
  id: 'p1', name: 'Rerun results', modelType: 'CNN', modelName: 'net',
  optimizer: 'Adam', status: 'COMPLETED', visibility: 'PRIVATE', myRelationship: 'OWNER',
};
const oldRows: api.ProjectResult[] = [
  { id: 'old-1', serverRound: 1, loss: 0.8, accuracy: 0.2 },
  { id: 'old-2', serverRound: 2, loss: 0.7, accuracy: 0.3 },
  { id: 'old-3', serverRound: 3, loss: 0.6, accuracy: 0.4 },
];
const newRows: api.ProjectResult[] = [{ id: 'new-1', serverRound: 1, loss: 0.1, accuracy: 0.9 }];

type Message = { body: string; headers: { destination: string } };
let connect: () => void;
let subscriptions: Map<string, (message: Message) => void>;

function emit(topic: 'status' | 'results', body: unknown) {
  act(() => subscriptions.get(`/topic/${topic}/*`)?.({
    body: JSON.stringify(body), headers: { destination: `/topic/${topic}/p1` },
  }));
}

async function mount() {
  render(<OwnerDashboard />);
  await screen.findByText(project.name);
  act(connect);
}

beforeEach(() => {
  vi.resetAllMocks();
  subscriptions = new Map();
  vi.mocked(StompClient).mockImplementation(function () {
    const client = {
      onConnect: null as (() => void) | null,
      active: false,
      activate: vi.fn(), deactivate: vi.fn(),
      subscribe: vi.fn((topic: string, callback: (message: Message) => void) => {
        subscriptions.set(topic, callback);
        return { unsubscribe: vi.fn() };
      }),
    };
    connect = () => client.onConnect?.();
    return client as unknown as InstanceType<typeof StompClient>;
  });
  vi.mocked(useAuth).mockReturnValue(makeAuth({
    currentUser: { username: 'owner', email: 'owner@example.invalid', role: 'PROJECT_OWNER' },
    isOwner: true,
  }));
  vi.mocked(api.fetchOwnedProjects).mockResolvedValue(response([project]));
  vi.mocked(api.fetchProjectDeletionRequest).mockResolvedValue(response(''));
  vi.mocked(api.fetchProjectResults).mockResolvedValue(response(oldRows));
});

describe('OwnerDashboard results across runs', () => {
  it('replaces the earlier curve on a status change and reloads full results for duplicate frames', async () => {
    await mount();
    await screen.findByText('40.0%');
    vi.mocked(api.fetchProjectResults).mockResolvedValue(response([]));
    emit('status', { projectId: 'p1', newStatus: 'RUNNING', serverPort: 55100 });
    await waitFor(() => expect(screen.queryByText('40.0%')).not.toBeInTheDocument());

    vi.mocked(api.fetchProjectResults).mockResolvedValue(response(newRows));
    emit('results', newRows[0]);
    await screen.findByText('90.0%');
    emit('results', newRows[0]);
    await waitFor(() => expect(api.fetchProjectResults).toHaveBeenCalledTimes(4));
    expect(screen.queryByText('40.0%')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Results' }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Raw Data' }));
    expect(within(dialog).getAllByRole('row')).toHaveLength(2);
  });

  it('does not let a delayed initial load restore an earlier run after the live refresh', async () => {
    const initial = deferred<AxiosResponse<api.ProjectResult[]>>();
    vi.mocked(api.fetchProjectResults).mockReturnValueOnce(initial.promise);
    await mount();
    vi.mocked(api.fetchProjectResults).mockResolvedValue(response(newRows));
    emit('results', newRows[0]);
    await screen.findByText('90.0%');
    await act(async () => initial.resolve(response(oldRows)));
    expect(screen.getByText('90.0%')).toBeInTheDocument();
    expect(screen.queryByText('40.0%')).not.toBeInTheDocument();
  });

  it('does not let an older live refresh overwrite a later current-run response', async () => {
    await mount();
    await screen.findByText('40.0%');
    const earlier = deferred<AxiosResponse<api.ProjectResult[]>>();
    vi.mocked(api.fetchProjectResults).mockReturnValueOnce(earlier.promise)
      .mockResolvedValueOnce(response(newRows));
    emit('results', oldRows[2]);
    emit('results', newRows[0]);
    await waitFor(() => expect(api.fetchProjectResults).toHaveBeenCalledTimes(3));
    await screen.findByText('90.0%');
    await act(async () => earlier.resolve(response(oldRows)));
    expect(screen.getByText('90.0%')).toBeInTheDocument();
    expect(screen.queryByText('40.0%')).not.toBeInTheDocument();
  });

  it('refreshes an already open results dialog when a new run starts', async () => {
    await mount();
    await screen.findByText('40.0%');
    fireEvent.click(screen.getByRole('button', { name: 'Results' }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Raw Data' }));
    expect(within(dialog).getAllByRole('row')).toHaveLength(4);
    vi.mocked(api.fetchProjectResults).mockResolvedValue(response([]));
    emit('status', { projectId: 'p1', newStatus: 'RUNNING' });
    await waitFor(() => expect(within(dialog).queryAllByRole('row')).toHaveLength(0));
  });
});
