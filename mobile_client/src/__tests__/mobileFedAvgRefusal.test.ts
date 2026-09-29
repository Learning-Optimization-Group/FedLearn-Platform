// The phone trains only what a run's published execution contract states. A run whose manifest carries no READY
// contract is refused before any provisioning or native work, whatever its legacy fields say — that is why every
// run here is refused, including the DeComFL and FedOpt runs an earlier build trained on the legacy fields alone.
// Execution contract v1 covers TinyNet FedAvg; runs it does not cover yet are refused rather than approximated.
// One guard still runs first because it names the obstacle better: secure aggregation.
import { runTrainingLoop, ExecutionContractRefusedError } from '../lib/training';
import type { JoinedRun } from '../lib/runJoin';
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
    getRuntimeCompatibility: jest.fn().mockResolvedValue({ bridgeAbiVersion: 2, protocolVersion: 2 }),
  },
}));

function joinedRun(strategy: string, firstOrderSupported = false): JoinedRun {
  return {
    runId: 'run-1',
    projectId: 'proj-1',
    partitionId: 0,
    assignedRound: 0,
    grpcEndpoint: 'localhost:50000',
    message: '',
    manifest: {
      firstOrderSupported,
      runId: 'run-1',
      projectId: 'proj-1',
      recipeKey: 'CNN',
      strategy,
      numRounds: 15,
      clientsPerRound: 1,
      partitioningMode: 'iid',
      seed: 0,
      torchVersion: '',
    },
  };
}

const hooks = { onLog: jest.fn(), onRound: jest.fn(), shouldStop: () => false };

beforeEach(() => jest.clearAllMocks());

describe('runTrainingLoop — execution contract gated', () => {
  test('refuses a FedAvg run without execution contract, fail-closed before provisioning', async () => {
    const p = runTrainingLoop(joinedRun('FedAvg'), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    await expect(runTrainingLoop(joinedRun('FedAvg'), hooks)).rejects.toThrow(/contract/i);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
    expect(nativeCore.loadModel).not.toHaveBeenCalled();
    expect(nativeCore.setTrainingDataFromFiles).not.toHaveBeenCalled();
  });

  test('refuses a FedAvg run with first-order support but no execution contract', async () => {
    const p = runTrainingLoop(joinedRun('FedAvg', /*firstOrderSupported=*/ true), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a DeComFL run without execution contract, fail-closed before provisioning', async () => {
    const p = runTrainingLoop(joinedRun('DeComFL'), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a FedOpt run without execution contract, fail-closed before provisioning', async () => {
    const p = runTrainingLoop(joinedRun('FedOpt', /*firstOrderSupported=*/ true), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a Robust run without execution contract, fail-closed before provisioning', async () => {
    const p = runTrainingLoop(joinedRun('Robust', /*firstOrderSupported=*/ true), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a FedProx run without execution contract, like every other strategy', async () => {
    const p = runTrainingLoop(joinedRun('FedProx', /*firstOrderSupported=*/ true), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('refuses a FutureStrategy run without execution contract, fail-closed before provisioning', async () => {
    const p = runTrainingLoop(joinedRun('FutureStrategy', /*firstOrderSupported=*/ true), hooks);
    await expect(p).rejects.toBeInstanceOf(ExecutionContractRefusedError);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

});
