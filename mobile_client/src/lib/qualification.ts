import DeviceInfo from 'react-native-device-info';
import EncryptedStorage from 'react-native-encrypted-storage';

import type { QualificationReport, TrainableProbe } from './nativeCore';
import { BRIDGE_ABI_VERSION } from './nativeCompatibility';
import { readError } from './errors';

// Stage 3 D2: before a device first trains with a trainable program, it runs the exporter's qualification probe on it
// (two SGD steps on a synthetic batch, checked against the exporter's losses). The result is a property of the
// device, the app build, the native bridge and the program, so it is cached under that key: a pass is not rerun, and
// a failure quarantines the key until the app or the program changes.

const KEY = 'fedlearn.qualification.v1';
const MAX_ENTRIES = 64;

interface Storage {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
}

type Entries = Record<string, QualificationReport & { at: string }>;

/** The cache key: device model and OS, app version and build, bridge ABI, program digest, and the CPU backend. */
export function qualificationKey(programSha256: string): string {
  return [DeviceInfo.getModel(), `android ${DeviceInfo.getSystemVersion()}`,
    `${DeviceInfo.getVersion()}+${DeviceInfo.getBuildNumber()}`, `abi ${BRIDGE_ABI_VERSION}`, programSha256, 'cpu']
    .join('|');
}

export class QualificationStore {
  constructor(private readonly storage: Storage) {}

  private async read(): Promise<Entries> {
    try {
      const parsed: unknown = JSON.parse((await this.storage.getItem(KEY)) ?? '{}');
      return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Entries : {};
    } catch {
      return {};  // a corrupt cache is empty, never trusted
    }
  }

  async get(key: string): Promise<QualificationReport | undefined> {
    const entry = (await this.read())[key];
    return entry && typeof entry.passed === 'boolean' ? entry : undefined;
  }

  async put(key: string, report: QualificationReport): Promise<void> {
    const entries = await this.read();
    entries[key] = { ...report, at: new Date().toISOString() };
    const kept = Object.entries(entries).sort((a, b) => b[1].at.localeCompare(a[1].at)).slice(0, MAX_ENTRIES);
    await this.storage.setItem(KEY, JSON.stringify(Object.fromEntries(kept)));
  }
}

export const qualificationStore = new QualificationStore(EncryptedStorage);

interface Deps {
  store: QualificationStore;
  native: { qualifyTrainable(probe: TrainableProbe): Promise<QualificationReport> };
}

/**
 * Whether this device may train with the trainable program `programSha256`, from the cache or by running `probe`.
 * A program without a probe does not qualify: there is nothing to check it against.
 */
export async function qualifyTrainableProgram(
  programSha256: string, probe: TrainableProbe | undefined, deps: Deps,
): Promise<QualificationReport> {
  const empty = { lossStep1: 0, lossStep2: 0, wallMs: 0 };
  if (!probe) {
    return { passed: false, failedCheck: 'NO_PROBE', detail: 'the run\'s trainable program has no probe', ...empty };
  }
  const key = qualificationKey(programSha256);
  const cached = await deps.store.get(key);
  if (cached) return cached;
  let report: QualificationReport;
  try {
    report = await deps.native.qualifyTrainable(probe);
  } catch (e) {
    report = { passed: false, failedCheck: 'NATIVE_ERROR', detail: readError(e), ...empty };
  }
  await deps.store.put(key, report);
  return report;
}
