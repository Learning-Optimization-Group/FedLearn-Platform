// MO-4: the on-device "FedAvg" round path (FederatedLoop::fedAvgRound) does K local ZO-SGD steps and
// uploads per-(k,p) seeds + gradient SCALARS via SubmitGradientScalars — the DeComFL wire, NOT a weight
// blob via SubmitModelUpdateStream. A server running the FedAvg *strategy* aggregates weight updates and
// cannot consume those scalars, so a phone joining a FedAvg run would "train" and submit into a void.
// Until SubmitModelUpdateStream is wired end-to-end (and the server can aggregate a mobile weight blob),
// runTrainingLoop must refuse a FedAvg run FAIL-CLOSED — before it provisions a model or touches the
// device — instead of silently no-op'ing. These tests pin that contract.
import { runTrainingLoop, MobileFedAvgUnsupportedError } from '../lib/training';
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
  },
}));

function joinedRun(strategy: string): JoinedRun {
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

describe('runTrainingLoop — mobile FedAvg refusal (MO-4)', () => {
  test('refuses a FedAvg run fail-closed, before provisioning a model or touching the device', async () => {
    const p = runTrainingLoop(joinedRun('FedAvg'), hooks);
    await expect(p).rejects.toBeInstanceOf(MobileFedAvgUnsupportedError);
    await expect(runTrainingLoop(joinedRun('FedAvg'), hooks)).rejects.toThrow(/FedAvg/i);
    // Fail-closed = no wasted work: no fetch/stage, no native load.
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
    expect(nativeCore.loadModel).not.toHaveBeenCalled();
    expect(nativeCore.setTrainingDataFromFiles).not.toHaveBeenCalled();
  });

  test('does NOT refuse a DeComFL run — it proceeds past the guard into provisioning', async () => {
    // Prove the guard is FedAvg-specific: a DeComFL run reaches provisionTrainingBundle. We make that
    // fetch reject with a sentinel so the loop unwinds there (not at the guard), which is enough to show
    // the guard let DeComFL through — without standing up the whole native round.
    (provisionTrainingBundle as jest.Mock).mockRejectedValueOnce(new Error('SENTINEL_PAST_GUARD'));
    await expect(runTrainingLoop(joinedRun('DeComFL'), hooks)).rejects.toThrow('SENTINEL_PAST_GUARD');
    expect(provisionTrainingBundle).toHaveBeenCalledWith('run-1');
  });
});
