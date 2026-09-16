import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { OwnerDashboard } from './OwnerDashboard';
import * as api from '../../services/apiServices';

// FE-8: the owner dashboard's live-status socket must surface an honest
// connecting/live/reconnecting/error indicator next to "My projects" —
// never a silent stall while `/topic/status/*` is unreachable.
vi.mock('../../services/apiServices');
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));

function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

interface FakeClient {
  onConnect: ((frame: unknown) => void) | null;
  onStompError: ((frame: unknown) => void) | null;
  onWebSocketError: ((evt: unknown) => void) | null;
  onWebSocketClose: ((evt: unknown) => void) | null;
  active: boolean;
  activate: ReturnType<typeof vi.fn>;
  deactivate: ReturnType<typeof vi.fn>;
  subscribe: ReturnType<typeof vi.fn>;
}

function makeFakeClient(): FakeClient {
  const client = {
    onConnect: null,
    onStompError: null,
    onWebSocketError: null,
    onWebSocketClose: null,
    active: false,
    activate: vi.fn(),
    deactivate: vi.fn(),
    subscribe: vi.fn(() => ({ unsubscribe: vi.fn() })),
  } as FakeClient;
  client.activate.mockImplementation(() => { client.active = true; });
  client.deactivate.mockImplementation(() => { client.active = false; });
  return client;
}

describe('OwnerDashboard — honest connection indicator (FE-8)', () => {
  let fakeClient: FakeClient;

  beforeEach(() => {
    fakeClient = makeFakeClient();
    vi.mocked(StompClient).mockImplementation(
      () => fakeClient as unknown as InstanceType<typeof StompClient>,
    );
    vi.mocked(api.fetchOwnedProjects).mockResolvedValue(resp([]));
    vi.mocked(api.fetchProjectResults).mockResolvedValue(resp([]));
  });

  it('shows "Connecting…" while the status socket has not connected yet', async () => {
    render(<OwnerDashboard />);
    await screen.findByText('My projects');

    expect(await screen.findByText('Connecting…')).toBeInTheDocument();
  });

  it('flips to "Live" once the status socket connects', async () => {
    render(<OwnerDashboard />);
    await screen.findByText('My projects');

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });

    expect(await screen.findByText('Live')).toBeInTheDocument();
    expect(screen.queryByText('Connecting…')).not.toBeInTheDocument();
  });

  it('reports "Reconnecting…" when a live status socket drops', async () => {
    render(<OwnerDashboard />);
    await screen.findByText('My projects');

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    await screen.findByText('Live');

    act(() => {
      fakeClient.onWebSocketClose?.({ code: 1006 });
    });

    expect(await screen.findByText('Reconnecting…')).toBeInTheDocument();
  });

  it('subscribes to the wildcard status/results destinations', async () => {
    render(<OwnerDashboard />);
    await screen.findByText('My projects');

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });

    const subscribedTopics = fakeClient.subscribe.mock.calls.map((call) => call[0]);
    expect(subscribedTopics).toEqual(['/topic/status/*', '/topic/results/*']);
  });
});
