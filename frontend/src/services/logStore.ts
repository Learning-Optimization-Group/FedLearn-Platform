// =============================================================================
// FedLearn Frontend — Log Store
// =============================================================================
// Module-level cache of log entries keyed by projectId. Survives unmount of
// any LogViewer modal so reopening a project's logs shows prior session output
// rather than resetting to an empty list.
//
// Consumers:
//   - logStore.get(projectId)             -> current cached entries
//   - logStore.append(projectId, entry)   -> push new entry (auto-trimmed)
//   - logStore.subscribe(id, cb)          -> observe changes for a projectId
//   - logStore.clear(projectId)           -> wipe cache (user action)
//   - logStore.hasLoadedHistorical(id)    -> avoid re-fetching /logs REST
//
// The store is intentionally framework-agnostic so it can be consumed by the
// classic LogViewer, the redesigned V2 LogViewerV2, and any future telemetry
// widget without duplicating caching logic.

export interface StoredLogEntry {
    level?: string;
    message: string;
    timestamp?: string;
    stackTrace?: string;
}

type Listener = (logs: StoredLogEntry[]) => void;

const MAX_LOGS_PER_PROJECT = 2000;

const cache = new Map<string, StoredLogEntry[]>();
const historicalLoaded = new Set<string>();
const listeners = new Map<string, Set<Listener>>();

function emit(projectId: string): void {
    const subs = listeners.get(projectId);
    if (!subs || subs.size === 0) return;
    const snapshot = cache.get(projectId) ?? [];
    subs.forEach((cb) => {
        try {
            cb(snapshot);
        } catch (err) {
            // A misbehaving listener must not break the store.
            // eslint-disable-next-line no-console
            console.error('[logStore] listener error:', err);
        }
    });
}

export const logStore = {
    get(projectId: string): StoredLogEntry[] {
        return cache.get(projectId) ?? [];
    },

    append(projectId: string, entry: StoredLogEntry): void {
        const arr = cache.get(projectId) ?? [];
        arr.push(entry);
        if (arr.length > MAX_LOGS_PER_PROJECT) {
            arr.splice(0, arr.length - MAX_LOGS_PER_PROJECT);
        }
        cache.set(projectId, arr);
        emit(projectId);
    },

    setAll(projectId: string, entries: StoredLogEntry[]): void {
        cache.set(projectId, entries.slice(-MAX_LOGS_PER_PROJECT));
        emit(projectId);
    },

    mergeHistorical(projectId: string, entries: StoredLogEntry[]): void {
        // Historical logs are considered authoritative for their time range and
        // get prepended before any live entries that arrived first. We dedupe
        // trivially by timestamp+message pairing.
        const existing = cache.get(projectId) ?? [];
        const existingKeys = new Set(existing.map((e) => `${e.timestamp ?? ''}|${e.message}`));
        const merged = [
            ...entries.filter((e) => !existingKeys.has(`${e.timestamp ?? ''}|${e.message}`)),
            ...existing,
        ].slice(-MAX_LOGS_PER_PROJECT);
        cache.set(projectId, merged);
        historicalLoaded.add(projectId);
        emit(projectId);
    },

    clear(projectId: string): void {
        cache.delete(projectId);
        historicalLoaded.delete(projectId);
        emit(projectId);
    },

    hasLoadedHistorical(projectId: string): boolean {
        return historicalLoaded.has(projectId);
    },

    markHistoricalLoaded(projectId: string): void {
        historicalLoaded.add(projectId);
    },

    subscribe(projectId: string, listener: Listener): () => void {
        let subs = listeners.get(projectId);
        if (!subs) {
            subs = new Set();
            listeners.set(projectId, subs);
        }
        subs.add(listener);
        return () => {
            const current = listeners.get(projectId);
            if (!current) return;
            current.delete(listener);
            if (current.size === 0) listeners.delete(projectId);
        };
    },
};
