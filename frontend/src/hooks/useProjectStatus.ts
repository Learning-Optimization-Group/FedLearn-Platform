// =============================================================================
// FedLearn Frontend — project status/results subscription hook (FE-9)
// =============================================================================
// Domain-specific convenience over useStompClient for the status/results
// contract every project-list surface shares: builds the two destinations,
// JSON-decodes each frame, and resolves the owning project id for the
// (possibly wildcarded) results destination. Callers own what happens with
// the decoded payloads (merging into their own project list, sparkline map,
// etc.) — this hook only owns the socket + the parsing.

import { WS_BROKER_URL } from '../lib/serverConfig';
import { useStompClient, type StompSubscriptionSpec, type UseStompClientState } from './useStompClient';

/** Mirrors the backend's `/topic/status/*` push payload (WebSocketService). */
export interface ProjectStatusUpdate {
    projectId: string;
    newStatus: string;
    serverPort?: number;
}

export interface UseProjectStatusOptions<TResult> {
    /**
     * Project to track, or the STOMP wildcard segment `'*'` for the owner
     * dashboard's "every owned project at once" subscription
     * (`/topic/status/*` + `/topic/results/*`).
     */
    projectId: string;
    /** Called for every parsed status push on the subscribed destination. */
    onStatusUpdate?: (update: ProjectStatusUpdate) => void;
    /**
     * Called for every parsed result push. `projectId` is read off the STOMP
     * message's destination header (not the subscribed pattern), so a `'*'`
     * subscription still reports which project each result belongs to.
     */
    onResult?: (projectId: string, result: TResult) => void;
    enabled?: boolean;
}

/**
 * Subscribes to `/topic/status/{projectId}` + `/topic/results/{projectId}`
 * and exposes the honest connection state from {@link useStompClient}. Message
 * parsing lives here (shared, low-risk JSON decode + destination lookup);
 * what the decoded status/results MEAN to a given surface stays with that
 * surface's own `onStatusUpdate`/`onResult` handlers.
 */
export function useProjectStatus<TResult = unknown>({
    projectId,
    onStatusUpdate,
    onResult,
    enabled = true,
}: UseProjectStatusOptions<TResult>): UseStompClientState {
    const subscriptions: StompSubscriptionSpec[] = [
        {
            topic: `/topic/status/${projectId}`,
            onMessage: (message) => {
                try {
                    onStatusUpdate?.(JSON.parse(message.body) as ProjectStatusUpdate);
                } catch {
                    /* ignore malformed status frames */
                }
            },
        },
        {
            topic: `/topic/results/${projectId}`,
            onMessage: (message) => {
                try {
                    const result = JSON.parse(message.body) as TResult;
                    const parts = message.headers.destination?.split('/') ?? [];
                    const resolvedProjectId = parts[parts.length - 1] || projectId;
                    onResult?.(resolvedProjectId, result);
                } catch {
                    /* ignore malformed result frames */
                }
            },
        },
    ];

    return useStompClient({ brokerURL: WS_BROKER_URL, subscriptions, enabled });
}
