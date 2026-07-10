// =============================================================================
// FedLearn Frontend — honest STOMP connection display (FE-8)
// =============================================================================
// Maps a useStompClient() snapshot to the one status-semantics scale
// (StatusPill's StatusKind) plus a caller-chosen label. Kept separate from
// useStompClient itself so the hook stays presentation-agnostic — every WS
// surface (LogViewer, OwnerDashboard, PlaygroundView) renders its own copy,
// with its own wording, from the same four honest phases:
//
//   connecting   — never connected yet, no error observed
//   live         — the STOMP CONNECTED frame has been received
//   reconnecting — was live, the socket dropped, auto-retry is under way
//   error        — never connected, and a STOMP/WebSocket error was observed
//
// "error" and "reconnecting" both correspond to the client actively retrying
// in the background (reconnectDelay > 0) — the distinction is purely about
// whether the surface has EVER been live, so the label can say "still trying"
// (reconnecting) vs. "hasn't connected yet" (error) instead of a silent stall
// or a fake "live".

import type { StatusKind } from '../components/ui';

export interface StompConnectionSnapshot {
    isConnected: boolean;
    isReconnecting: boolean;
    lastError: string | null;
}

export interface ConnectionDisplay {
    kind: StatusKind;
    label: string;
}

export interface ConnectionLabels {
    live: string;
    connecting: string;
    reconnecting: string;
    error: string;
}

export function describeStompConnection(
    state: StompConnectionSnapshot,
    labels: ConnectionLabels,
): ConnectionDisplay {
    if (state.isConnected) return { kind: 'running', label: labels.live };
    if (state.isReconnecting) return { kind: 'pending', label: labels.reconnecting };
    if (state.lastError) return { kind: 'error', label: labels.error };
    return { kind: 'idle', label: labels.connecting };
}
