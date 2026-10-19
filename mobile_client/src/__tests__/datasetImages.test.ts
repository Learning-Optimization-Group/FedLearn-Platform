// Stage 4 S5: a run on images asks the device for an image package, prepared exactly as its contract states
// (ImageToUnitTensor, then NormalizeChannels). The snapshot records how its values were prepared, and a snapshot
// prepared any other way does not fit the run, even with the same shape and classes.
import { create } from '@bufbuild/protobuf';
import {
  DataRequirementSchema,
  DataSource,
  DType,
  Task,
  TransformSchema,
  type DataRequirement,
} from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import {
  pickAndImportDataset,
  requirementShape,
  snapshotMismatches,
  transformsId,
  type DatasetSnapshot,
} from '@/lib/datasetService';

jest.mock('react-native', () => ({
  NativeModules: {
    DatasetService: {
      pickAndImport: jest.fn().mockResolvedValue({}),
      pickAndImportImage: jest.fn().mockResolvedValue({}),
      list: jest.fn(),
      delete: jest.fn(),
    },
  },
}));

const mockNative = (jest.requireMock('react-native') as {
  NativeModules: { DatasetService: { pickAndImport: jest.Mock; pickAndImportImage: jest.Mock } };
}).NativeModules.DatasetService;

const HALF = [0.5, 0.5, 0.5];

function imageRequirement(normalize = true, h = 2, w = 3, c = 3): DataRequirement {
  return create(DataRequirementSchema, {
    task: Task.IMAGE_CLASSIFICATION,
    inputShape: [BigInt(c), BigInt(h), BigInt(w)],
    inputDtype: DType.DTYPE_F32,
    classCount: 2,
    labelSchemaId: 'labels-sha256:' + 'c'.repeat(64),
    source: DataSource.LOCAL_SNAPSHOT,
    transforms: [
      create(TransformSchema, { operation: { case: 'imageToUnitTensor', value: { height: h, width: w, channels: c } } }),
      ...(normalize ? [create(TransformSchema, {
        operation: { case: 'normalizeChannels', value: { mean: HALF, std: HALF } },
      })] : []),
    ],
  });
}

function imageSnapshot(requirement: DataRequirement, transforms: string | undefined): DatasetSnapshot {
  return {
    snapshotId: 'a'.repeat(64),
    recordCount: 4,
    inputShape: requirement.inputShape.map(Number),
    inputDtype: 'f32',
    classNames: ['cat', 'dog'],
    labelSchemaId: requirement.labelSchemaId,
    inputsPath: '/x/inputs.f32',
    targetsPath: '/x/targets.i64',
    transforms,
  };
}

describe('what an image run asks of the device', () => {
  it('reads the image and its normalisation from the contract', () => {
    expect(requirementShape(imageRequirement())).toEqual(
      { kind: 'image', height: 2, width: 3, channels: 3, mean: HALF, std: HALF });
    expect(requirementShape(imageRequirement(false))).toEqual(
      { kind: 'image', height: 2, width: 3, channels: 3 });
  });

  it('reads a vector run as before', () => {
    const vector = create(DataRequirementSchema, {
      task: Task.VECTOR_CLASSIFICATION, inputShape: [4n],
      transforms: [create(TransformSchema, { operation: { case: 'identityVector', value: { width: 4 } } })],
    });
    expect(requirementShape(vector)).toEqual({ kind: 'vector', width: 4 });
    expect(transformsId(requirementShape(vector)!)).toBeUndefined();
  });

  it('names the preparation exactly as the device importer records it', () => {
    // The same string ImageImportTest pins on the Kotlin side: float32 bit patterns, never decimal text.
    expect(transformsId(requirementShape(imageRequirement())!))
      .toBe('image:2x3x3;mean=3f000000,3f000000,3f000000;std=3f000000,3f000000,3f000000');
    expect(transformsId(requirementShape(imageRequirement(false))!)).toBe('image:2x3x3');
  });

  it('writes a mean as its float32 bits, so 0.485 is float32(0.485)', () => {
    const r = imageRequirement(false, 1, 1, 1);
    r.transforms.push(create(TransformSchema, {
      operation: { case: 'normalizeChannels', value: { mean: [0.485], std: [0.229] } },
    }));
    expect(transformsId(requirementShape(r)!)).toBe('image:1x1x1;mean=3ef851ec;std=3e6a7efa');
  });
});

describe('an image snapshot fits the run only if prepared as the run states', () => {
  it('fits when its preparation is the contract\'s', () => {
    const r = imageRequirement();
    expect(snapshotMismatches(imageSnapshot(r, transformsId(requirementShape(r)!)), r)).toEqual([]);
  });

  it('does not fit when normalised differently, although shape and classes agree', () => {
    const r = imageRequirement();
    expect(snapshotMismatches(imageSnapshot(r, 'image:2x3x3'), r)).toEqual(['transforms']);
  });

  it('does not fit when it records no preparation at all', () => {
    const r = imageRequirement();
    expect(snapshotMismatches(imageSnapshot(r, undefined), r)).toEqual(['transforms']);
  });
});

describe('importing for an image run', () => {
  it('asks the device for an image package with the contract\'s preparation', async () => {
    await pickAndImportDataset(['cat', 'dog'], requirementShape(imageRequirement())!);
    expect(mockNative.pickAndImportImage).toHaveBeenCalledWith(['cat', 'dog'], 2, 3, 3, HALF, HALF);
    expect(mockNative.pickAndImport).not.toHaveBeenCalled();
  });

  it('passes no normalisation when the contract states none', async () => {
    mockNative.pickAndImportImage.mockClear();
    await pickAndImportDataset(['cat', 'dog'], requirementShape(imageRequirement(false))!);
    expect(mockNative.pickAndImportImage).toHaveBeenCalledWith(['cat', 'dog'], 2, 3, 3, null, null);
  });

  it('imports a vector run as before', async () => {
    await pickAndImportDataset(['c0', 'c1'], { kind: 'vector', width: 4 });
    expect(mockNative.pickAndImport).toHaveBeenCalledWith(['c0', 'c1'], 4);
  });
});
