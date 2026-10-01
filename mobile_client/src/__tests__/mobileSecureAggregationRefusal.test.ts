// Secure aggregation guard. The phone cannot take part in a LightSecAgg round yet: it has no native key
// exchange, share sealing or masking, and a secure server refuses plaintext gradient scalars by design. Joining
// would fail every round, so runTrainingLoop refuses up front - before any provisioning or native work - and
// tells the user where they can join instead. Mirrors the MO-4 capability guard (mobileFedAvgRefusal.test.ts).
import { runTrainingLoop, ExecutionContractRefusedError, MobileSecureAggregationUnsupportedError } from '../lib/training';
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
    getRuntimeCompatibility: jest.fn().mockResolvedValue({ bridgeAbiVersion: 4, protocolVersion: 2 }),
  },
}));

function joinedRun(overrides: Partial<RunManifest>): JoinedRun {
  return {
    runId: 'run-1',
    projectId: 'proj-1',
    partitionId: 0,
    assignedRound: 0,
    grpcEndpoint: 'localhost:50000',
    message: '',
    manifest: {
      runId: 'run-1',
      projectId: 'proj-1',
      recipeKey: 'CNN',
      strategy: 'DeComFL',
      numRounds: 5,
      clientsPerRound: 3,
      partitioningMode: 'iid',
      seed: 0,
      torchVersion: '',
      ...overrides,
    },
  };
}

const hooks = { onLog: jest.fn(), onRound: jest.fn(), shouldStop: () => false };

beforeEach(() => jest.clearAllMocks());

describe('runTrainingLoop — secure aggregation guard', () => {
  test('refuses a secure-aggregation run before any provisioning or native work', async () => {
    await expect(runTrainingLoop(joinedRun({ secureAggregation: true }), hooks)).rejects.toBeInstanceOf(
      MobileSecureAggregationUnsupportedError,
    );
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
    expect(nativeCore.loadModel).not.toHaveBeenCalled();
    expect(nativeCore.setTrainingDataFromFiles).not.toHaveBeenCalled();
  });

  test('the refusal names secure aggregation and points to the desktop app', async () => {
    await expect(runTrainingLoop(joinedRun({ secureAggregation: true }), hooks)).rejects.toThrow(/secure aggregation/i);
    await expect(runTrainingLoop(joinedRun({ secureAggregation: true }), hooks)).rejects.toThrow(/desktop/i);
  });

  test('refuses even a first-order-provisioned run: a weight upload is not masked either', async () => {
    await expect(
      runTrainingLoop(joinedRun({ secureAggregation: true, firstOrderSupported: true }), hooks),
    ).rejects.toBeInstanceOf(MobileSecureAggregationUnsupportedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('a run with secure aggregation off passes this guard (the contract then decides)', async () => {
    // The guard is specific: with masking off it does not fire, and the run is stopped one step later by the
    // execution-contract gate, because this manifest carries no contract.
    await expect(runTrainingLoop(joinedRun({ secureAggregation: false }), hooks))
      .rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('a manifest from an older backend does not trip the masking guard', async () => {
    // An absent secureAggregation field means "off", so this guard stays silent; such a manifest carries no
    // execution contract either, so the run is refused for that reason instead.
    await expect(runTrainingLoop(joinedRun({}), hooks))
      .rejects.toBeInstanceOf(ExecutionContractRefusedError);
  });
});
