import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, act, fireEvent } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import { LogViewerV2 } from './LogViewer';
import * as api from '../../services/apiServices';
import { logStore } from '../../services/logStore';

// FE-8: LogViewer's status pill must reflect the REAL useStompClient state
// (connecting / live / reconnecting / error), not a fire-and-forget flag.
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

describe('LogViewerV2 — honest connection indicator (FE-8)', () => {
  let fakeClient: FakeClient;

  beforeEach(() => {
    logStore.clear('proj-1');
    vi.mocked(api.fetchProjectLogs).mockResolvedValue(resp([]));
    fakeClient = makeFakeClient();
    vi.mocked(StompClient).mockImplementation(
      () => fakeClient as unknown as InstanceType<typeof StompClient>,
    );
  });

  it('shows "Connecting…" before the socket connects, then "Live Streaming" once it does', async () => {
    render(<LogViewerV2 projectId="proj-1" onClose={() => {}} />);

    expect(await screen.findByText('Connecting…')).toBeInTheDocument();

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });

    expect(await screen.findByText('Live Streaming')).toBeInTheDocument();
    expect(screen.queryByText('Connecting…')).not.toBeInTheDocument();
  });

  it('shows "Reconnecting…" when a live socket drops', async () => {
    render(<LogViewerV2 projectId="proj-1" onClose={() => {}} />);

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    expect(await screen.findByText('Live Streaming')).toBeInTheDocument();

    act(() => {
      fakeClient.onWebSocketClose?.({ code: 1006 });
    });

    expect(await screen.findByText('Reconnecting…')).toBeInTheDocument();
  });

  it('lets an explicit Pause override the connection label', async () => {
    render(<LogViewerV2 projectId="proj-1" onClose={() => {}} />);
    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    expect(await screen.findByText('Live Streaming')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /pause/i }));

    expect(await screen.findByText('Paused')).toBeInTheDocument();
  });

  it('deactivates the socket on unmount', async () => {
    const { unmount } = render(<LogViewerV2 projectId="proj-1" onClose={() => {}} />);
    await screen.findByText('Connecting…');

    unmount();

    expect(fakeClient.deactivate).toHaveBeenCalled();
  });

  it('ignores a malformed telemetry frame instead of crashing the render', async () => {
    // A hostile/malformed backend (or a MITM on the plaintext dev ws://) sends a frame whose
    // loss/accuracy are non-numeric. Before the finiteness guard this entered the telemetry cache and
    // threw at render (latest.loss.toFixed(...)), white-screening the whole SPA. It must be ignored.
    render(<LogViewerV2 projectId="proj-1" onClose={() => {}} />);
    act(() => { fakeClient.onConnect?.({ headers: {} }); });
    await screen.findByText('Live Streaming');

    const sub = fakeClient.subscribe.mock.calls.find((c) => String(c[0]).includes('/topic/logs/'));
    const onMessage = sub?.[1] as (m: { body: string }) => void;
    expect(onMessage).toBeTypeOf('function');

    act(() => {
      onMessage({ body: JSON.stringify({ level: 'INFO', message: 'hi', serverRound: 1, loss: null, accuracy: 'x' }) });
    });

    // No crash — the component is still mounted and live.
    expect(screen.getByText('Live Streaming')).toBeInTheDocument();
  });
});
