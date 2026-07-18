// Device-local contribution ledger — the persisted "what did this phone contribute" record.
//
// One entry per COMPLETED round (appended from the training loop's onRound callback). Entries
// are client-side facts only: the round ran here and an update was SUBMITTED. Whether the
// server ultimately accepted the update (vs FR-12 robust-aggregation filtering) needs a backend
// API that does not exist yet — never label these rows "accepted".
//
// Persistence: react-native-encrypted-storage (the same at-rest store modelStore.ts uses —
// neither @react-native-async-storage/async-storage nor react-native-fs is a dependency here).
// The storage seam is injectable so tests run against an in-memory map.
import EncryptedStorage from 'react-native-encrypted-storage';

export interface ContributionEntry {
  projectId: string;
  projectName: string;
  round: number;
  /** Wall-clock compute time this round took on this device (RoundResult.computeMs). */
  wallClockMs: number;
  /** Bytes uploaded this round (RoundResult.uplinkBytes). */
  bytesUp: number;
  /** Bytes downloaded this round (RoundResult.downlinkBytes). */
  bytesDown: number;
  /** ISO-8601 timestamp of when the round completed on this device. */
  at: string;
}

export interface ContributionTotals {
  rounds: number;
  wallClockMs: number;
  bytesUp: number;
  bytesDown: number;
}

export interface ProjectContribution extends ContributionTotals {
  projectId: string;
  projectName: string;
}

/** The subset of the storage API the ledger needs (EncryptedStorage satisfies it). */
export interface LedgerStorage {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
}

const KEY = 'fedlearn.contributionLedger';

/** Newest-first cap so a long-lived install can't grow storage unbounded. */
export const MAX_LEDGER_ENTRIES = 500;

/** Rolling window the "this week" summary folds over (7 days, not calendar-aligned). */
export const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

const emptyTotals = (): ContributionTotals => ({
  rounds: 0,
  wallClockMs: 0,
  bytesUp: 0,
  bytesDown: 0,
});

function foldTotals(entries: ContributionEntry[]): ContributionTotals {
  return entries.reduce(
    (acc, e) => ({
      rounds: acc.rounds + 1,
      wallClockMs: acc.wallClockMs + e.wallClockMs,
      bytesUp: acc.bytesUp + e.bytesUp,
      bytesDown: acc.bytesDown + e.bytesDown,
    }),
    emptyTotals(),
  );
}

export class ContributionLedger {
  constructor(private readonly storage: LedgerStorage) {}

  /** All entries, newest first. A missing or corrupted store reads as empty (never throws). */
  async list(limit?: number): Promise<ContributionEntry[]> {
    let raw: string | null = null;
    try {
      raw = await this.storage.getItem(KEY);
    } catch {
      return [];
    }
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw) as unknown;
      const entries = Array.isArray(parsed) ? (parsed as ContributionEntry[]) : [];
      return limit != null ? entries.slice(0, limit) : entries;
    } catch {
      return [];
    }
  }

  /** Prepend a completed round (newest first), capped at MAX_LEDGER_ENTRIES. */
  async record(entry: ContributionEntry): Promise<void> {
    const all = await this.list();
    const next = [entry, ...all].slice(0, MAX_LEDGER_ENTRIES);
    await this.storage.setItem(KEY, JSON.stringify(next));
  }

  /** Lifetime totals across every project (within the entry cap). */
  async totals(): Promise<ContributionTotals> {
    return foldTotals(await this.list());
  }

  /** Per-project slices, ordered by most recent contribution first. */
  async totalsByProject(): Promise<ProjectContribution[]> {
    const entries = await this.list();
    const byProject = new Map<string, ProjectContribution>();
    for (const e of entries) {
      const existing = byProject.get(e.projectId);
      if (existing) {
        existing.rounds += 1;
        existing.wallClockMs += e.wallClockMs;
        existing.bytesUp += e.bytesUp;
        existing.bytesDown += e.bytesDown;
      } else {
        byProject.set(e.projectId, {
          projectId: e.projectId,
          projectName: e.projectName,
          ...emptyTotals(),
          rounds: 1,
          wallClockMs: e.wallClockMs,
          bytesUp: e.bytesUp,
          bytesDown: e.bytesDown,
        });
      }
    }
    return [...byProject.values()];
  }

  /** This project's entries, newest first. */
  async entriesForProject(projectId: string, limit?: number): Promise<ContributionEntry[]> {
    const entries = (await this.list()).filter((e) => e.projectId === projectId);
    return limit != null ? entries.slice(0, limit) : entries;
  }

  /** Totals over the rolling last-7-days window ("this week"). */
  async thisWeek(nowMs: number = Date.now()): Promise<ContributionTotals> {
    const cutoff = nowMs - WEEK_MS;
    const recent = (await this.list()).filter((e) => {
      const t = Date.parse(e.at);
      return Number.isFinite(t) && t >= cutoff;
    });
    return foldTotals(recent);
  }
}

/** App-wide ledger bound to encrypted at-rest storage (same store as the model registry). */
export const contributionLedger = new ContributionLedger(EncryptedStorage);
