import { create } from '@bufbuild/protobuf';
import {
  DataRequirementSchema,
  DataSource,
  DType,
  Task,
  type DataRequirement,
} from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import { snapshotMismatches, type DatasetSnapshot } from '@/lib/datasetService';

// Stage 3 B2: a device binds one of its own dataset snapshots to a run only when the snapshot is exactly what the
// run's contract requires. The label schema carries the class list (labels-sha256 of it), so equal ids mean the same
// classes in the same order.
const REQUIREMENT: DataRequirement = create(DataRequirementSchema, {
  task: Task.VECTOR_CLASSIFICATION,
  inputShape: [4n],
  inputDtype: DType.DTYPE_F32,
  classCount: 3,
  labelSchemaId: 'labels-sha256:4817507d9942bf24635535b19b1588bb44e00d2b51314536926f3580c7f5d896',
  source: DataSource.LOCAL_SNAPSHOT,
});

const SNAPSHOT: DatasetSnapshot = {
  snapshotId: 'a'.repeat(64),
  recordCount: 8,
  inputShape: [4],
  inputDtype: 'f32',
  classNames: ['c0', 'c1', 'c2'],
  labelSchemaId: REQUIREMENT.labelSchemaId,
  inputsPath: '/data/datasets/x/inputs.f32',
  targetsPath: '/data/datasets/x/targets.i64',
};

describe('binding a snapshot to a run', () => {
  it('accepts a snapshot that is exactly what the contract requires', () => {
    expect(snapshotMismatches(SNAPSHOT, REQUIREMENT)).toEqual([]);
  });

  const cases: [string, Partial<DatasetSnapshot>][] = [
    ['labelSchemaId', { labelSchemaId: 'labels-sha256:' + 'b'.repeat(64) }],
    ['inputShape', { inputShape: [5] }],
    ['inputDtype', { inputDtype: 'f64' }],
    ['classCount', { classNames: ['c0', 'c1'] }],
    ['recordCount', { recordCount: 0 }],
  ];
  it.each(cases)('names %s when it differs', (field, change) => {
    expect(snapshotMismatches({ ...SNAPSHOT, ...change }, REQUIREMENT)).toContain(field);
  });
});
