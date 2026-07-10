import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { Client as StompClient } from '@stomp/stompjs';
import { useStompClient } from './useStompClient';

// FE-9: useStompClient is the one place that owns a STOMP client's lifecycle
// for every WS surface (PlaygroundView, LogViewer, OwnerDashboard). This spec
// asserts the honest-state contract directly against the real onConnect /
// onWebSocketClose / onStompError / onWebSocketError callbacks, not a
// fire-and-forget "connected" flag.
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));

interface FakeSubscription {
  unsubscribe: ReturnType<typeof vi.fn>;
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
    subscribe: vi.fn(() => ({ unsubscribe: vi.fn() }) as FakeSubscription),
  } as FakeClient;
  client.activate.mockImplementation(() => {
    client.active = true;
  });
  client.deactivate.mockImplementation(() => {
    client.active = false;
  });
  return client;
}

describe('useStompClient', () => {
  let fakeClient: FakeClient;

  beforeEach(() => {
    fakeClient = makeFakeClient();
    vi.mocked(StompClient).mockImplementation(
      () => fakeClient as unknown as InstanceType<typeof StompClient>,
    );
  });

  it('activates on mount and subscribes to every destination once connected', () => {
    const onMessage = vi.fn();
    const { result } = renderHook(() =>
      useStompClient({
        brokerURL: 'ws://broker/ws-logs',
        subscriptions: [{ topic: '/topic/foo', onMessage }],
      }),
    );

    expect(fakeClient.activate).toHaveBeenCalledTimes(1);
    expect(fakeClient.subscribe).not.toHaveBeenCalled();
    expect(result.current).toEqual({ isConnected: false, isReconnecting: false, lastError: null });

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });

    expect(fakeClient.subscribe).toHaveBeenCalledWith('/topic/foo', expect.any(Function));
    expect(result.current.isConnected).toBe(true);
    expect(result.current.isReconnecting).toBe(false);
    expect(result.current.lastError).toBeNull();

    // The subscribe callback forwards the message straight to onMessage.
    const forwarded = fakeClient.subscribe.mock.calls[0][1];
    const fakeMessage = { body: '{}' };
    forwarded(fakeMessage);
    expect(onMessage).toHaveBeenCalledWith(fakeMessage);
  });

  it('does not activate while disabled', () => {
    renderHook(() =>
      useStompClient({
        brokerURL: 'ws://broker/ws-logs',
        subscriptions: [],
        enabled: false,
      }),
    );

    expect(fakeClient.activate).not.toHaveBeenCalled();
  });

  it('deactivates and unsubscribes on unmount — no leaked sockets', () => {
    const sub = { unsubscribe: vi.fn() };
    fakeClient.subscribe.mockReturnValue(sub);

    const { unmount } = renderHook(() =>
      useStompClient({
        brokerURL: 'ws://broker/ws-logs',
        subscriptions: [{ topic: '/topic/foo', onMessage: vi.fn() }],
      }),
    );

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });

    unmount();

    expect(sub.unsubscribe).toHaveBeenCalledTimes(1);
    expect(fakeClient.deactivate).toHaveBeenCalledTimes(1);
  });

  it('flips isConnected false and isReconnecting true when a live socket drops', () => {
    const { result } = renderHook(() =>
      useStompClient({ brokerURL: 'ws://broker/ws-logs', subscriptions: [] }),
    );

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    expect(result.current.isConnected).toBe(true);

    act(() => {
      fakeClient.onWebSocketClose?.({ code: 1006 });
    });

    expect(result.current.isConnected).toBe(false);
    expect(result.current.isReconnecting).toBe(true);
  });

  it('does not report "reconnecting" for a close before the first successful connect', () => {
    const { result } = renderHook(() =>
      useStompClient({ brokerURL: 'ws://broker/ws-logs', subscriptions: [] }),
    );

    act(() => {
      fakeClient.onWebSocketClose?.({ code: 1006 });
    });

    expect(result.current.isConnected).toBe(false);
    expect(result.current.isReconnecting).toBe(false);
  });

  it('records the last STOMP broker error', () => {
    const { result } = renderHook(() =>
      useStompClient({ brokerURL: 'ws://broker/ws-logs', subscriptions: [] }),
    );

    act(() => {
      fakeClient.onStompError?.({ headers: { message: 'bad frame' }, body: '' });
    });

    expect(result.current.lastError).toBe('bad frame');
  });

  it('records a WebSocket-level error and clears it on the next successful connect', () => {
    const { result } = renderHook(() =>
      useStompClient({ brokerURL: 'ws://broker/ws-logs', subscriptions: [] }),
    );

    act(() => {
      fakeClient.onWebSocketError?.(new Error('connection refused'));
    });
    expect(result.current.lastError).toBe('WebSocket connection error.');
    expect(result.current.isConnected).toBe(false);

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    expect(result.current.lastError).toBeNull();
    expect(result.current.isConnected).toBe(true);
  });

  it('does not resubscribe when only the onMessage identity changes across renders', () => {
    const { rerender } = renderHook(
      ({ onMessage }: { onMessage: () => void }) =>
        useStompClient({
          brokerURL: 'ws://broker/ws-logs',
          subscriptions: [{ topic: '/topic/foo', onMessage }],
        }),
      { initialProps: { onMessage: vi.fn() } },
    );

    act(() => {
      fakeClient.onConnect?.({ headers: {} });
    });
    expect(fakeClient.subscribe).toHaveBeenCalledTimes(1);
    expect(StompClient).toHaveBeenCalledTimes(1);

    // A brand new inline closure (the norm at every real call site) must not
    // tear down and recreate the socket.
    rerender({ onMessage: vi.fn() });

    expect(StompClient).toHaveBeenCalledTimes(1);
    expect(fakeClient.deactivate).not.toHaveBeenCalled();
  });
});
