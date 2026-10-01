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
  /** How an image snapshot's values were prepared (see transformsId); absent for a vector snapshot. */
  transforms?: string;
}

/** What one example of a run's data must be: a float vector, or an 8-bit image and its contract preparation. */
export type DataShape =
  | { kind: 'vector'; width: number }
  | { kind: 'image'; height: number; width: number; channels: number; mean?: number[]; std?: number[] };

interface DatasetServiceModule {
  pickAndImport(classNames: string[], inputWidth: number): Promise<DatasetSnapshot>;
  pickAndImportImage(classNames: string[], height: number, width: number, channels: number,
    mean: number[] | null, std: number[] | null): Promise<DatasetSnapshot>;
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
export function pickAndImportDataset(classNames: string[], shape: DataShape): Promise<DatasetSnapshot> {
  if (shape.kind === 'vector') {
    return service().pickAndImport(classNames, shape.width);
  }
  return service().pickAndImportImage(classNames, shape.height, shape.width, shape.channels, shape.mean ?? null,
    shape.std ?? null);
}

/**
 * The shape a run's contract requires of each example, from its transforms: an identity vector, or an image
 * conversion optionally followed by per-channel normalisation. Null for anything else.
 */
export function requirementShape(requirement: DataRequirement): DataShape | null {
  const [first, second, ...rest] = requirement.transforms.map((t) => t.operation);
  if (first?.case === 'identityVector' && second === undefined) {
    return { kind: 'vector', width: first.value.width };
  }
  if (first?.case !== 'imageToUnitTensor' || rest.length > 0) {
    return null;
  }
  const { height, width, channels } = first.value;
  if (second === undefined) {
    return { kind: 'image', height, width, channels };
  }
  if (second.case !== 'normalizeChannels') {
    return null;
  }
  return { kind: 'image', height, width, channels, mean: [...second.value.mean], std: [...second.value.std] };
}

function float32Hex(value: number): string {
  const bits = new Uint32Array(new Float32Array([value]).buffer)[0]!;
  return bits.toString(16).padStart(8, '0');
}

/**
 * How an image snapshot prepared for `shape` records its preparation, exactly as the device importer writes it:
 * `image:HxWxC`, then `;mean=..;std=..` as float32 bit patterns in hex. Undefined for vectors, which record none.
 */
export function transformsId(shape: DataShape): string | undefined {
  if (shape.kind === 'vector') {
    return undefined;
  }
  const base = `image:${shape.height}x${shape.width}x${shape.channels}`;
  if (!shape.mean || !shape.std) {
    return base;
  }
  return `${base};mean=${shape.mean.map(float32Hex).join(',')};std=${shape.std.map(float32Hex).join(',')}`;
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
  // An image snapshot's values depend on how they were prepared, which its shape and classes do not show.
  const shapeRequired = requirementShape(requirement);
  const preparation = shapeRequired ? transformsId(shapeRequired) : undefined;
  if (preparation !== (snapshot.transforms ?? undefined)) {
    mismatches.push('transforms');
  }
  return mismatches;
}
