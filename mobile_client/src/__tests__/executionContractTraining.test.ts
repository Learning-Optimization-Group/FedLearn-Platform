// The training loop acts on the run's execution contract: it refuses before touching the device when the phone may
// not train the run, waits while the contract is still being published, runs the rounds the contract states, and
// refuses a staged bundle the contract did not bind.
import { toJson } from '@bufbuild/protobuf';
import { ExecutionContractSchema } from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import { parseContractBinary } from '@/lib/executionContract';
import { ExecutionContractRefusedError, runTrainingLoop } from '../lib/training';
import type { JoinedRun, RunManifest } from '../lib/runJoin';
import { provisionTrainingBundle } from '../lib/modelProvisioning';
import nativeCore from '../lib/nativeCore';

jest.mock('../lib/modelProvisioning', () => ({
  __esModule: true,
  provisionTrainingBundle: jest.fn(),
}));

jest.mock('../lib/nativeCore', () => ({
  __esModule: true,
  default: {
    loadModel: jest.fn(),
    setModelManifest: jest.fn(),
    setTrainingDataFromFiles: jest.fn(),
    getServerStatus: jest.fn(),
    runDeComFLRound: jest.fn(),
    runFedAvgRound: jest.fn(),
    getRuntimeCompatibility: jest.fn().mockResolvedValue({ bridgeAbiVersion: 1, protocolVersion: 2 }),
  },
}));

declare const __dirname: string;
type FixtureFs = { readFileSync(path: string, encoding?: 'utf8'): string & Uint8Array };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs: FixtureFs = require('fs');
const GOLDEN = `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb`;

const RUN_ID = '4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17';
const PROJECT_ID = '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64';

function contractJson(): Record<string, unknown> {
  const contract = parseContractBinary(new Uint8Array(fs.readFileSync(GOLDEN)));
  return toJson(ExecutionContractSchema, contract) as Record<string, unknown>;
}

function manifest(over: Partial<RunManifest> = {}): RunManifest {
  return {
    runId: RUN_ID,
    projectId: PROJECT_ID,
    recipeKey: 'TINYNET_GOLDEN',
    strategy: 'FedAvg',
    numRounds: 3,
    clientsPerRound: 4,
    partitioningMode: 'SHARDED',
    seed: 42,
    torchVersion: '2.12.0',
    firstOrderSupported: true,
    contractState: 'READY',
    contractId: 'a'.repeat(64),
    executionContract: contractJson(),
    ...over,
  };
}

function joined(over: Partial<RunManifest> = {}): JoinedRun {
  return {
    runId: RUN_ID,
    projectId: PROJECT_ID,
    partitionId: 0,
    assignedRound: 0,
    grpcEndpoint: 'localhost:50000',
    message: '',
    manifest: manifest(over),
  };
}

/** The bundle the golden contract binds: its programs' digests and its trainable layout. */
const STAGED_BUNDLE = {
  manifest: {
    paramLayout: [
      { name: 'fc1.weight', shape: [5, 4] },
      { name: 'fc1.bias', shape: [5] },
    ],
    totalParamCount: 43,
    inferPtePath: 'infer.pte',
    inferSha256: 'cf8744b9579d78f14bbb82e2d4ce98dcaffc8d2c6ed2253349c39342de546746',
    trainablePtePath: 'trainable.pte',
    trainableSha256: 'ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412',
    trainableParamNames: ['base.fc1.weight', 'base.fc1.bias'],
  },
  lossPtePath: 'loss.pte',
  lossSha256: '2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56',
  inputsF32Path: 'inputs.f32',
  inputShape: [8, 4],
  targetsI64Path: 'targets.i64',
};

const hooks = { onLog: jest.fn(), onRound: jest.fn(), shouldStop: () => false };
const POLICY = { maxRoundRetries: 0, maxRejoins: 0, baseBackoffMs: 1, pacingMs: 0, rejoinRecoveryRounds: 1 };

beforeEach(() => jest.clearAllMocks());

function oneRound() {
  const getServerStatus = jest.fn()
    .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
    .mockResolvedValueOnce({ serverState: 'TRAINING_COMPLETE', currentRound: 1 });
  const runFedAvgRound = jest.fn().mockResolvedValue({
    round: 1, loss: 1, accuracy: 0, scalarsTransmitted: 0,
    uplinkBytes: 100, downlinkBytes: 100, computeMs: 1, reverted: false,
  });
  return { getServerStatus, runFedAvgRound };
}

describe('runTrainingLoop — the execution contract decides', () => {
  test('runs the training the contract states', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce(STAGED_BUNDLE);
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const { getServerStatus, runFedAvgRound } = oneRound();

    await runTrainingLoop(joined(), hooks, {
      policy: POLICY,
      ops: { getServerStatus, runFedAvgRound, loadSubmittedRound: async () => null, saveSubmittedRound: async () => {} },
    });

    expect(runFedAvgRound).toHaveBeenCalledTimes(1);
    const cfg = runFedAvgRound.mock.calls[0][1];
    // The training the golden contract states, from the shared projection fixture the native C++ round test
    // also trains with — so one chain is pinned: contract -> projection -> the numbers that reach the trainer.
    const shared = JSON.parse(fs.readFileSync(
      `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/projection_tinynet_fedavg.json`,
      'utf8')) as { learningRate: number; numLocalSteps: number };
    expect(cfg).toMatchObject({
      strategy: 'FedAvg', learningRate: shared.learningRate, numLocalSteps: shared.numLocalSteps });
  });

  test('runs a FedOpt contract as FedOpt, at the rate the contract states', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce(STAGED_BUNDLE);
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const { getServerStatus, runFedAvgRound } = oneRound();
    const contract = contractJson() as { strategy: string; modelTraining: { localTraining: { sgd: Record<string, unknown> } } };
    contract.strategy = 'STRATEGY_FEDOPT';
    contract.modelTraining.localTraining.sgd.learningRate = 0.01;

    await runTrainingLoop(joined({ strategy: 'FedOpt', executionContract: contract }), hooks, {
      policy: POLICY,
      ops: { getServerStatus, runFedAvgRound, loadSubmittedRound: async () => null, saveSubmittedRound: async () => {} },
    });

    expect(runFedAvgRound.mock.calls[0][1]).toMatchObject({ strategy: 'FedOpt', learningRate: 0.01 });
  });

  test('runs a DeComFL contract through the DeComFL round, with the contract\'s own training', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce(STAGED_BUNDLE);
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const getServerStatus = jest.fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING_COMPLETE', currentRound: 1 });
    const runDeComFLRound = jest.fn().mockResolvedValue({
      round: 1, loss: 1, accuracy: 0, scalarsTransmitted: 10,
      uplinkBytes: 80, downlinkBytes: 0, computeMs: 1, reverted: false,
    });
    const runFedAvgRound = jest.fn();
    const contract = contractJson() as {
      strategy: string; modelTraining: { updateProtocol: string; localTraining: Record<string, unknown> } };
    contract.strategy = 'STRATEGY_DECOMFL';
    contract.modelTraining.updateProtocol = 'UPDATE_DECOMFL_SCALAR';
    delete contract.modelTraining.localTraining.sgd;
    delete contract.modelTraining.localTraining.localEpochs;
    contract.modelTraining.localTraining.zerothOrderSgd = {
      learningRate: 0.001, smoothing: 0.002, numLocalSteps: 1, numPerturbations: 10,
      estimator: 'ESTIMATOR_FORWARD', rng: 'RNG_TORCH_CPU_RANDN_F32',
    };

    await runTrainingLoop(joined({ strategy: 'DeComFL', executionContract: contract }), hooks, {
      policy: POLICY,
      ops: { getServerStatus, runFedAvgRound, runDeComFLRound,
        loadSubmittedRound: async () => null, saveSubmittedRound: async () => {} },
    });

    expect(runFedAvgRound).not.toHaveBeenCalled();
    expect(runDeComFLRound.mock.calls[0][1]).toMatchObject({
      strategy: 'DeComFL', learningRate: 0.001, mu: 0.002, numLocalSteps: 1, numPerturbations: 10,
      gradEstimateMethod: 'forward',
    });
  });

  test('refuses a run with no published contract before touching the device', async () => {
    await expect(runTrainingLoop(joined({ contractState: undefined, executionContract: undefined }), hooks))
      .rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
    expect(nativeCore.loadModel).not.toHaveBeenCalled();
  });

  test('refuses a run whose contract is unavailable, naming the reason', async () => {
    const run = joined({
      contractState: 'UNAVAILABLE', executionContract: undefined,
      contractUnavailableReason: 'NOT_REPRESENTABLE',
    });
    await expect(runTrainingLoop(run, hooks)).rejects.toThrow(/NOT_REPRESENTABLE/);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('waits while the contract is being published, then trains on it', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce(STAGED_BUNDLE);
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const { getServerStatus, runFedAvgRound } = oneRound();
    const fetchManifest = jest.fn()
      .mockResolvedValueOnce(manifest({ contractState: 'PENDING', executionContract: undefined }))
      .mockResolvedValueOnce(manifest());
    const delay = jest.fn().mockResolvedValue(undefined);

    await runTrainingLoop(joined({ contractState: 'PENDING', executionContract: undefined }), hooks, {
      policy: POLICY,
      contract: { fetchManifest, delay, timeoutMs: 60_000, intervalMs: 1_000 },
      ops: { getServerStatus, runFedAvgRound, loadSubmittedRound: async () => null, saveSubmittedRound: async () => {} },
    });

    expect(fetchManifest).toHaveBeenCalledTimes(2);
    expect(delay).toHaveBeenCalledWith(1_000);
    expect(runFedAvgRound).toHaveBeenCalledTimes(1);
  });

  test('gives up when the contract never arrives', async () => {
    const fetchManifest = jest.fn().mockResolvedValue(
      manifest({ contractState: 'PENDING', executionContract: undefined }));

    await expect(runTrainingLoop(joined({ contractState: 'PENDING', executionContract: undefined }), hooks, {
      contract: { fetchManifest, delay: jest.fn().mockResolvedValue(undefined), timeoutMs: 3_000, intervalMs: 1_000 },
    })).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a staged bundle the contract did not bind, before any round', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce({
      ...STAGED_BUNDLE, lossSha256: 'b'.repeat(64),
    });
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const { getServerStatus, runFedAvgRound } = oneRound();

    await expect(runTrainingLoop(joined(), hooks, {
      policy: POLICY,
      ops: { getServerStatus, runFedAvgRound, loadSubmittedRound: async () => null, saveSubmittedRound: async () => {} },
    })).rejects.toThrow(/loss\.pte/);
    expect(runFedAvgRound).not.toHaveBeenCalled();
  });
});
