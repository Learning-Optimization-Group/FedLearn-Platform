// =============================================================================
// FedLearn Frontend — shared STOMP client lifecycle hook (FE-9)
// =============================================================================
// Every WS surface in this app (PlaygroundView, LogViewer, OwnerDashboard) used
// to hand-roll its own `new StompClient({ brokerURL, reconnectDelay: 5000 })`,
// its own activate/deactivate + subscribe/unsubscribe bookkeeping via refs, and
// its own (often approximate) notion of "connected". This hook is the one
// place that owns the client's lifecycle and reports HONEST connection state —
// derived only from the real onConnect / onWebSocketClose / onStompError /
// onWebSocketError callbacks — so a dropped socket can never present as a
// silent stall or a fake "connected".

import { useEffect, useRef, useState } from 'react';
import { Client as StompClient, type IFrame, type IMessage, type StompSubscription } from '@stomp/stompjs';

/** One subscription entry: a STOMP destination plus its message handler. */
export interface StompSubscriptionSpec {
    topic: string;
    onMessage: (message: IMessage) => void;
}

export interface UseStompClientOptions {
    /** ws(s):// broker URL. The hook stays inactive (no socket) while falsy. */
    brokerURL: string;
    /** Destinations to (re)subscribe to on every CONNECTED frame, including reconnects. */
    subscriptions: StompSubscriptionSpec[];
    /** Fired after the hook has (re)subscribed, once per fresh CONNECTED frame. */
    onConnect?: (frame: IFrame) => void;
    /** Set false to tear the socket down without unmounting the caller. Default true. */
    enabled?: boolean;
    /** Passed straight through to the underlying Client. Default 5000 (existing app convention). */
    reconnectDelay?: number;
}

export interface UseStompClientState {
    /** True only once the STOMP CONNECTED frame has actually been received. */
    isConnected: boolean;
    /** True once a previously-live connection has dropped and the client is auto-retrying. */
    isReconnecting: boolean;
    /** The most recent STOMP/WebSocket error message, or null once cleared by a fresh attempt. */
    lastError: string | null;
}

/**
 * Owns one STOMP client's full lifecycle: activates when `enabled`, (re)subscribes
 * to `subscriptions` on every CONNECTED frame (including automatic reconnects),
 * and deactivates + unsubscribes on unmount or when `enabled` flips false.
 *
 * `subscriptions`/`onConnect` are read through refs that are refreshed on every
 * render, so passing fresh inline closures each render (the norm at every call
 * site) does NOT tear down and recreate the socket — only a change to
 * `brokerURL`, `enabled`, `reconnectDelay`, or the *set of topic strings* does.
 */
export function useStompClient({
    brokerURL,
    subscriptions,
    onConnect,
    enabled = true,
    reconnectDelay = 5000,
}: UseStompClientOptions): UseStompClientState {
    const [isConnected, setIsConnected] = useState(false);
    const [isReconnecting, setIsReconnecting] = useState(false);
    const [lastError, setLastError] = useState<string | null>(null);

    const subscriptionsRef = useRef(subscriptions);
    subscriptionsRef.current = subscriptions;
    const onConnectRef = useRef(onConnect);
    onConnectRef.current = onConnect;

    // Structural stand-in for "the set of destinations changed" — deliberately
    // NOT the `subscriptions` array itself, whose identity changes on every
    // render for any call site that builds it inline (all three current ones).
    const topicsKey = subscriptions.map((s) => s.topic).join('::');

    useEffect(() => {
        if (!enabled || !brokerURL) {
            setIsConnected(false);
            setIsReconnecting(false);
            setLastError(null);
            return;
        }

        let isDisposed = false;
        let everConnected = false;
        let liveSubs: StompSubscription[] = [];
        const client = new StompClient({ brokerURL, reconnectDelay });

        client.onConnect = (frame) => {
            if (isDisposed) return;
            liveSubs = subscriptionsRef.current.map(({ topic }, index) =>
                client.subscribe(topic, (message) => {
                    if (isDisposed) return;
                    const sub = subscriptionsRef.current[index];
                    if (sub && sub.topic === topic) {
                        sub.onMessage(message);
                    } else {
                        subscriptionsRef.current
                            .filter((s) => s.topic === topic)
                            .forEach((s) => s.onMessage(message));
                    }
                }),
            );
            everConnected = true;
            setIsConnected(true);
            setIsReconnecting(false);
            setLastError(null);
            onConnectRef.current?.(frame);
        };

        client.onWebSocketClose = () => {
            if (isDisposed || !client.active) return;
            liveSubs.forEach((sub) => {
                try {
                    sub.unsubscribe();
                } catch {
                    // Ignore failures on torn-down connection
                }
            });
            liveSubs = [];
            setIsConnected(false);
            // A drop AFTER having been live auto-retries (reconnectDelay > 0) —
            // that is "reconnecting". A close before the first CONNECTED frame
            // is still the initial connect attempt, not a regression to report.
            if (everConnected && reconnectDelay > 0) setIsReconnecting(true);
        };

        client.onStompError = (frame) => {
            if (isDisposed) return;
            setLastError(frame.headers?.message || frame.body || 'STOMP broker reported an error.');
        };

        client.onWebSocketError = () => {
            if (isDisposed) return;
            setLastError('WebSocket connection error.');
        };

        client.activate();

        return () => {
            isDisposed = true;
            client.onConnect = () => {};
            client.onWebSocketClose = () => {};
            client.onStompError = () => {};
            client.onWebSocketError = () => {};
            liveSubs.forEach((sub) => {
                try {
                    sub.unsubscribe();
                } catch {
                    // Ignore failures on torn-down connection
                }
            });
            liveSubs = [];
            if (client.active) void client.deactivate();
            setIsConnected(false);
            setIsReconnecting(false);
        };
        // subscriptions/onConnect are read through refs (see above), so they're
        // deliberately not listed here — topicsKey is the real dependency for
        // "should the socket be rebuilt".
    }, [brokerURL, enabled, reconnectDelay, topicsKey]);

    return { isConnected, isReconnecting, lastError };
}
