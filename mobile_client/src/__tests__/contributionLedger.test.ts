// Device-local contribution ledger: per completed round {project, round, wallClockMs,
// bytesUp/Down, at}, persisted (encrypted at rest) with totals / per-project slices / a
// rolling this-week summary. Driven through the injectable storage seam; one test exercises
// the default EncryptedStorage-backed instance via the manual mock.
import EncryptedStorage from 'react-native-encrypted-storage';

import {
  ContributionLedger,
  contributionLedger,
  MAX_LEDGER_ENTRIES,
  WEEK_MS,
  type ContributionEntry,
  type LedgerStorage,
} from '../lib/contributionLedger';

function memoryStorage(): LedgerStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: async (key) => data.get(key) ?? null,
    setItem: async (key, value) => {
      data.set(key, value);
    },
  };
}

function entry(over: Partial<ContributionEntry> = {}): ContributionEntry {
  return {
    projectId: 'proj-1',
    projectName: 'Pneumonia CNN',
    round: 1,
    wallClockMs: 1500,
    bytesUp: 1024,
    bytesDown: 4096,
    at: '2026-07-17T12:00:00.000Z',
    ...over,
  };
}

describe('ContributionLedger — record + list', () => {
  it('starts empty and lists recorded rounds newest first', async () => {
    const ledger = new ContributionLedger(memoryStorage());
    expect(await ledger.list()).toEqual([]);

    await ledger.record(entry({ round: 1 }));
    await ledger.record(entry({ round: 2 }));
    const all = await ledger.list();
    expect(all.map((e) => e.round)).toEqual([2, 1]);
  });

  it('serializes overlapping record() calls so neither entry is dropped', async () => {
    // Without the internal write queue both records read the same (empty) store before
    // either writes, and the second setItem clobbers the first entry.
    const ledger = new ContributionLedger(memoryStorage());
    await Promise.all([ledger.record(entry({ round: 1 })), ledger.record(entry({ round: 2 }))]);
    expect((await ledger.list()).map((e) => e.round)).toEqual([2, 1]);
  });

  it('honors the list limit', async () => {
    const ledger = new ContributionLedger(memoryStorage());
    for (let i = 1; i <= 4; i++) await ledger.record(entry({ round: i }));
    expect((await ledger.list(2)).map((e) => e.round)).toEqual([4, 3]);
  });

  it(`caps the store at ${MAX_LEDGER_ENTRIES} entries (newest kept)`, async () => {
    const storage = memoryStorage();
    const ledger = new ContributionLedger(storage);
    const full = Array.from({ length: MAX_LEDGER_ENTRIES }, (_, i) => entry({ round: i }));
    storage.data.set('fedlearn.contributionLedger', JSON.stringify(full));

    await ledger.record(entry({ round: 9999 }));
    const all = await ledger.list();
    expect(all).toHaveLength(MAX_LEDGER_ENTRIES);
    expect(all[0]?.round).toBe(9999);
  });

  it('reads a corrupted or failing store as empty instead of throwing', async () => {
    const storage = memoryStorage();
    storage.data.set('fedlearn.contributionLedger', 'not json {');
    expect(await new ContributionLedger(storage).list()).toEqual([]);

    const failing: LedgerStorage = {
      getItem: async () => {
        throw new Error('keystore locked');
      },
      setItem: async () => {},
    };
    expect(await new ContributionLedger(failing).list()).toEqual([]);
  });
});

describe('ContributionLedger — totals + slices', () => {
  async function seeded() {
    const ledger = new ContributionLedger(memoryStorage());
    await ledger.record(entry({ projectId: 'a', projectName: 'A', round: 1, wallClockMs: 100, bytesUp: 10, bytesDown: 20 }));
    await ledger.record(entry({ projectId: 'b', projectName: 'B', round: 1, wallClockMs: 200, bytesUp: 30, bytesDown: 40 }));
    await ledger.record(entry({ projectId: 'a', projectName: 'A', round: 2, wallClockMs: 300, bytesUp: 50, bytesDown: 60 }));
    return ledger;
  }

  it('folds lifetime totals across all projects', async () => {
    const ledger = await seeded();
    expect(await ledger.totals()).toEqual({
      rounds: 3,
      wallClockMs: 600,
      bytesUp: 90,
      bytesDown: 120,
    });
  });

  it('slices totals per project, most recently contributed first', async () => {
    const ledger = await seeded();
    const slices = await ledger.totalsByProject();
    expect(slices).toEqual([
      { projectId: 'a', projectName: 'A', rounds: 2, wallClockMs: 400, bytesUp: 60, bytesDown: 80 },
      { projectId: 'b', projectName: 'B', rounds: 1, wallClockMs: 200, bytesUp: 30, bytesDown: 40 },
    ]);
  });

  it('filters entries for one project', async () => {
    const ledger = await seeded();
    const a = await ledger.entriesForProject('a');
    expect(a.map((e) => e.round)).toEqual([2, 1]);
    expect(await ledger.entriesForProject('a', 1)).toHaveLength(1);
    expect(await ledger.entriesForProject('missing')).toEqual([]);
  });
});

describe('ContributionLedger — thisWeek (rolling 7 days)', () => {
  it('sums only entries inside the window and ignores unparseable timestamps', async () => {
    const now = Date.parse('2026-07-17T12:00:00.000Z');
    const ledger = new ContributionLedger(memoryStorage());
    await ledger.record(entry({ round: 1, at: new Date(now - WEEK_MS - 1000).toISOString(), bytesUp: 999 })); // too old
    await ledger.record(entry({ round: 2, at: new Date(now - 1000).toISOString(), bytesUp: 10, bytesDown: 20, wallClockMs: 100 }));
    await ledger.record(entry({ round: 3, at: 'garbage-timestamp', bytesUp: 999 })); // dropped

    expect(await ledger.thisWeek(now)).toEqual({
      rounds: 1,
      wallClockMs: 100,
      bytesUp: 10,
      bytesDown: 20,
    });
  });
});

describe('default app-wide ledger (EncryptedStorage-backed)', () => {
  it('persists through the encrypted store', async () => {
    const store = EncryptedStorage as unknown as {
      getItem: jest.Mock;
      setItem: jest.Mock;
    };
    store.getItem.mockResolvedValue(null);
    store.setItem.mockResolvedValue(undefined);

    await contributionLedger.record(entry());
    expect(store.setItem).toHaveBeenCalledWith(
      'fedlearn.contributionLedger',
      JSON.stringify([entry()]),
    );
  });
});
