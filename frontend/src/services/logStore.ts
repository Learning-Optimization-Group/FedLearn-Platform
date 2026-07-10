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
//
// Each entry receives a monotonic, never-reused {@link StoredLogEntry.id}
// at insertion time. Consumers MUST use this id as the React key when
// rendering log lists — array-index keys break under {@link mergeHistorical}
// because historical entries are *prepended*, which shifts every existing
// index and causes React to reconcile the wrong DOM node to the wrong entry
// (garbled timestamps, repeated lines, broken auto-scroll).

/** Shape callers pass into the store. The store assigns the id. */
export interface LogEntryInput {
    level?: string;
    message: string;
    timestamp?: string;
    stackTrace?: string;
}

/** Shape returned by the store. {@link id} is stable for the entry's lifetime. */
export interface StoredLogEntry extends LogEntryInput {
    id: number;
}

type Listener = (logs: StoredLogEntry[]) => void;

const MAX_LOGS_PER_PROJECT = 2000;

const cache = new Map<string, StoredLogEntry[]>();
const historicalLoaded = new Set<string>();
const listeners = new Map<string, Set<Listener>>();

// Module-scoped, monotonically increasing. Wraps after Number.MAX_SAFE_INTEGER
// in the abstract — practically unreachable for a log viewer (would require
// 9e15 entries within a single tab session).
let nextId = 1;

function stamp(entry: LogEntryInput): StoredLogEntry {
    return { ...entry, id: nextId++ };
}

function emit(projectId: string): void {
    const subs = listeners.get(projectId);
    if (!subs || subs.size === 0) return;
    const snapshot = cache.get(projectId) ?? [];
    subs.forEach((cb) => {
        try {
            cb(snapshot);
        } catch (err) {
            // A misbehaving listener must not break the store.
            console.error('[logStore] listener error:', err);
        }
    });
}

export const logStore = {
    get(projectId: string): StoredLogEntry[] {
        return cache.get(projectId) ?? [];
    },

    append(projectId: string, entry: LogEntryInput): void {
        // Build a NEW array (don't mutate the cached one in place): listeners pass this
        // reference straight into React setState, and an identical reference triggers the
        // Object.is bailout — so an in-place push would leave the live log pane frozen until
        // some other state change forced a re-render. Mirrors setAll/mergeHistorical.
        const prev = cache.get(projectId) ?? [];
        const next = [...prev, stamp(entry)];
        if (next.length > MAX_LOGS_PER_PROJECT) {
            next.splice(0, next.length - MAX_LOGS_PER_PROJECT);
        }
        cache.set(projectId, next);
        emit(projectId);
    },

    setAll(projectId: string, entries: LogEntryInput[]): void {
        cache.set(
            projectId,
            entries.slice(-MAX_LOGS_PER_PROJECT).map(stamp),
        );
        emit(projectId);
    },

    mergeHistorical(projectId: string, entries: LogEntryInput[]): void {
        // Historical logs are considered authoritative for their time range and
        // get prepended before any live entries that arrived first. We dedupe
        // trivially by timestamp+message pairing.
        const existing = cache.get(projectId) ?? [];
        const existingKeys = new Set(existing.map((e) => `${e.timestamp ?? ''}|${e.message}`));
        const survivors = entries
            .filter((e) => !existingKeys.has(`${e.timestamp ?? ''}|${e.message}`))
            .map(stamp);
        const merged = [...survivors, ...existing].slice(-MAX_LOGS_PER_PROJECT);
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
