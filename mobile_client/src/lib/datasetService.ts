import { NativeModules } from 'react-native';
import { DType, type DataRequirement } from '../gen/fedlearn/contract/v1/execution_contract_pb';

// On-device datasets (Stage 3, wikis/mobile/11 section B). The Kotlin DatasetService imports a file the user picks
// (.csv, or a zipped dataset package) into an immutable snapshot in app-private storage; only its metadata and local
// paths cross the bridge, and nothing about the data leaves the device.

/** An imported, immutable dataset snapshot on this device. */
export interface DatasetSnapshot {
  snapshotId: string;
  recordCount: number;
  inputShape: number[];
  inputDtype: string;
  classNames: string[];
  labelSchemaId: string;
  inputsPath: string;
  targetsPath: string;
}

interface DatasetServiceModule {
  pickAndImport(classNames: string[], inputWidth: number): Promise<DatasetSnapshot>;
  list(): Promise<DatasetSnapshot[]>;
  delete(snapshotId: string, pinned: string[]): Promise<void>;
}

const DatasetService = NativeModules.DatasetService as DatasetServiceModule | undefined;

function service(): DatasetServiceModule {
  if (!DatasetService) {
    throw new Error('DatasetService is not available in this build');
  }
  return DatasetService;
}

/** Let the user pick a file and import it against the run's classes; rejects with a DATASET_* code naming why not. */
export function pickAndImportDataset(classNames: string[], inputWidth: number): Promise<DatasetSnapshot> {
  return service().pickAndImport(classNames, inputWidth);
}

export function listDatasets(): Promise<DatasetSnapshot[]> {
  return service().list();
}

/** Delete a snapshot; refused (DATASET_PINNED) while a run uses it. */
export function deleteDataset(snapshotId: string, pinned: string[]): Promise<void> {
  return service().delete(snapshotId, pinned);
}

const DTYPE_NAMES: Partial<Record<DType, string>> = { [DType.DTYPE_F32]: 'f32' };

/**
 * What stops `snapshot` binding to a run whose contract requires `requirement`; empty means it may bind. The label
 * schema id is labels-sha256 of the class list, so an equal id means the same classes in the same order.
 */
export function snapshotMismatches(snapshot: DatasetSnapshot, requirement: DataRequirement): string[] {
  const mismatches: string[] = [];
  if (snapshot.labelSchemaId !== requirement.labelSchemaId) {
    mismatches.push('labelSchemaId');
  }
  const shape = requirement.inputShape.map(Number);
  if (shape.length !== snapshot.inputShape.length || shape.some((d, i) => d !== snapshot.inputShape[i])) {
    mismatches.push('inputShape');
  }
  if (DTYPE_NAMES[requirement.inputDtype] !== snapshot.inputDtype) {
    mismatches.push('inputDtype');
  }
  if (snapshot.classNames.length !== requirement.classCount) {
    mismatches.push('classCount');
  }
  if (!(snapshot.recordCount >= 1)) {
    mismatches.push('recordCount');
  }
  return mismatches;
}
