import EncryptedStorage from 'react-native-encrypted-storage';

const KEY = 'fedlearn.submittedRounds';
const MAX_RUNS = 32;

interface Storage {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
}

type Checkpoints = Record<string, number>;

// An upload checkpoint means the native round call returned after submitting, not that the
// server included the update in the aggregate. Keep only run IDs and round numbers on-device.
export class SubmittedRoundStore {
  private tail: Promise<void> = Promise.resolve();

  constructor(private readonly storage: Storage) {}

  private async read(): Promise<Checkpoints> {
    const raw = await this.storage.getItem(KEY);
    if (raw === null) return {};
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed === null || Array.isArray(parsed) || typeof parsed !== 'object' ||
          Object.values(parsed).some((round) => !Number.isSafeInteger(round) || (round as number) < 0)) {
        throw new Error('invalid checkpoint');
      }
      return parsed as Checkpoints;
    } catch {
      throw new Error('Round checkpoint is corrupt; training stopped to avoid a duplicate upload.');
    }
  }

  async load(runId: string): Promise<number | null> {
    await this.tail;
    const checkpoints = await this.read();
    return checkpoints[runId] ?? null;
  }

  save(runId: string, round: number): Promise<void> {
    if (!Number.isSafeInteger(round) || round < 0) {
      return Promise.reject(new Error('Invalid submitted round checkpoint.'));
    }
    const task = this.tail.then(async () => {
      const checkpoints = await this.read();
      const latest = Math.max(round, checkpoints[runId] ?? -1);
      // Reinsert so the bounded map evicts oldest run IDs first, never the active run.
      delete checkpoints[runId];
      checkpoints[runId] = latest;
      const entries = Object.entries(checkpoints).slice(-MAX_RUNS);
      await this.storage.setItem(KEY, JSON.stringify(Object.fromEntries(entries)));
    });
    this.tail = task.catch(() => undefined);
    return task;
  }
}

export const submittedRoundStore = new SubmittedRoundStore(EncryptedStorage);
