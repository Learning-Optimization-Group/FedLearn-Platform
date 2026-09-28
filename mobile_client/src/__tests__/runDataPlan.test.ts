// Stage 3 B2: before Start, the phone reads the run's contract to learn whether it trains on the device's own data,
// and if so what a dataset must look like; each dataset on the device is then shown as usable or not, with why.
import { toJson } from '@bufbuild/protobuf';
import { ExecutionContractSchema } from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import { parseContractBinary } from '@/lib/executionContract';
import { api } from '../lib/restClient';
import { datasetFit, importForRun, loadRunDataPlan, type RunDataPlan } from '../lib/runDataPlan';
import { pickAndImportDataset } from '../lib/datasetService';
import type { JoinedRun, RunManifest } from '../lib/runJoin';
import type { DatasetSnapshot } from '../lib/datasetService';

jest.mock('../lib/restClient', () => ({ api: { get: jest.fn() } }));
jest.mock('../lib/datasetService', () => ({
  ...jest.requireActual('../lib/datasetService'),
  pickAndImportDataset: jest.fn(),
}));
const mPick = pickAndImportDataset as unknown as jest.Mock;
const mGet = api.get as unknown as jest.Mock;

declare const __dirname: string;
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs: { readFileSync(path: string): Uint8Array } = require('fs');
const GOLDEN = `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb`;
const RUN_ID = '4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17';
const PROJECT_ID = '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64';
const TINYNET_LABELS = 'labels-sha256:4817507d9942bf24635535b19b1588bb44e00d2b51314536926f3580c7f5d896';

function contractJson(source?: string): Record<string, unknown> {
  const c = toJson(ExecutionContractSchema, parseContractBinary(new Uint8Array(fs.readFileSync(GOLDEN)))) as
    { modelTraining: { data: Record<string, unknown> } };
  if (source) c.modelTraining.data.source = source;
  return c as unknown as Record<string, unknown>;
}

function joined(over: Partial<RunManifest> = {}): JoinedRun {
  return {
    runId: RUN_ID, projectId: PROJECT_ID, partitionId: 0, assignedRound: 0, grpcEndpoint: 'localhost:50000',
    message: '',
    manifest: {
      runId: RUN_ID, projectId: PROJECT_ID, recipeKey: 'TINYNET_GOLDEN', strategy: 'FedAvg', numRounds: 3,
      clientsPerRound: 4, partitioningMode: 'SHARDED', seed: 42, torchVersion: '2.12.0', firstOrderSupported: true,
      contractState: 'READY', contractId: 'a'.repeat(64), executionContract: contractJson('DATA_SOURCE_LOCAL_SNAPSHOT'),
      ...over,
    },
  };
}

const SNAPSHOT: DatasetSnapshot = {
  snapshotId: 'd'.repeat(64), recordCount: 6, inputShape: [4], inputDtype: 'f32', classNames: ['c0', 'c1', 'c2'],
  labelSchemaId: TINYNET_LABELS, inputsPath: '/d/inputs.f32', targetsPath: '/d/targets.i64',
};

beforeEach(() => {
  mGet.mockReset().mockResolvedValue({ data: { classNames: ['c0', 'c1', 'c2'] } });
});

describe('loadRunDataPlan', () => {
  test('a run on the device\'s own data states what a dataset must be, with the run\'s classes to import against',
    async () => {
      const plan = await loadRunDataPlan(joined());

      expect(plan).toMatchObject({ source: 'LOCAL_SNAPSHOT', classNames: ['c0', 'c1', 'c2'], inputWidth: 4 });
      const own = plan as Extract<RunDataPlan, { source: 'LOCAL_SNAPSHOT' }>;
      expect(own.requirement.labelSchemaId).toBe(TINYNET_LABELS);
      expect(own.batchSize).toBeGreaterThan(0);
      expect(mGet).toHaveBeenCalledWith(`/api/runs/${RUN_ID}/model-bundle`);
    });

  test('a run on the built-in data needs nothing from the device', async () => {
    const plan = await loadRunDataPlan(joined({ executionContract: contractJson() }));

    expect(plan).toEqual({ source: 'FIXTURE' });
    expect(mGet).not.toHaveBeenCalled();
  });

  test('waits for a contract that is still being published', async () => {
    const fetchManifest = jest.fn().mockResolvedValue(joined().manifest);

    const plan = await loadRunDataPlan(joined({ contractState: 'PENDING' }),
      { fetchManifest, delay: () => Promise.resolve(), intervalMs: 1, timeoutMs: 10 });

    expect(fetchManifest).toHaveBeenCalledWith(RUN_ID);
    expect(plan?.source).toBe('LOCAL_SNAPSHOT');
  });

  test('has no plan for a run this device will not train; Start then says why', async () => {
    expect(await loadRunDataPlan(joined({ contractState: 'LEGACY_ONLY' }))).toBeNull();
    expect(await loadRunDataPlan(joined({ contractState: 'PENDING' }))).toBeNull();
  });

  test('has no plan when the run does not say which classes to import against', async () => {
    mGet.mockResolvedValue({ data: { classNames: [] } });
    expect(await loadRunDataPlan(joined())).toBeNull();
  });
});

describe('datasetFit', () => {
  let plan: Extract<RunDataPlan, { source: 'LOCAL_SNAPSHOT' }>;
  beforeEach(async () => {
    plan = (await loadRunDataPlan(joined())) as typeof plan;
  });

  test('a dataset matching the run\'s model and fitting one batch is usable', () => {
    expect(datasetFit({ ...SNAPSHOT, recordCount: plan.batchSize }, plan)).toEqual([]);
  });

  test('names each way a dataset does not fit the run', () => {
    const other = { ...SNAPSHOT, labelSchemaId: 'labels-sha256:' + '0'.repeat(64), inputShape: [3],
      classNames: ['a', 'b'] };
    expect(datasetFit(other, plan)).toEqual(['different classes', 'different input size', 'different class count']);
  });

  // The native trainer takes one whole-dataset step per epoch until minibatching lands (Stage 3 slice C).
  test('a dataset larger than one batch is not usable yet', () => {
    expect(datasetFit({ ...SNAPSHOT, recordCount: plan.batchSize + 1 }, plan))
      .toEqual([`more than ${plan.batchSize} examples`]);
  });
});

describe('importForRun', () => {
  let plan: Extract<RunDataPlan, { source: 'LOCAL_SNAPSHOT' }>;
  beforeEach(async () => {
    plan = (await loadRunDataPlan(joined())) as typeof plan;
    mPick.mockReset();
  });

  test('imports the picked file against the run\'s classes and example width', async () => {
    mPick.mockResolvedValue(SNAPSHOT);
    expect(await importForRun(plan)).toEqual({ snapshot: SNAPSHOT });
    expect(mPick).toHaveBeenCalledWith(['c0', 'c1', 'c2'], 4);
  });

  test('choosing no file is not an error', async () => {
    mPick.mockRejectedValue(Object.assign(new Error('no file was chosen'), { code: 'DATASET_PICK_CANCELLED' }));
    expect(await importForRun(plan)).toEqual({});
  });

  test('a refused import says why', async () => {
    mPick.mockRejectedValue(Object.assign(new Error('row 3 has 5 values, expected 4'), { code: 'DATASET_BAD_ROW' }));
    expect(await importForRun(plan)).toEqual({ error: 'row 3 has 5 values, expected 4' });
  });
});
