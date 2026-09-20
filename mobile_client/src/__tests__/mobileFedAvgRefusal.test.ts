// MO-4 (capability-gated): a phone runs a FedAvg round on-device ONLY when the backend provisioned a
// first-order-capable bundle (manifest.firstOrderSupported) — then FederatedLoop::firstOrderRound does
// real backprop and uploads a WEIGHT blob a FedAvg server aggregates (SubmitModelUpdateStream). WITHOUT
// it, the only on-device path is the ZO-scalar fedAvgRound the server can't consume, so runTrainingLoop
// refuses fail-closed before provisioning. These tests pin both sides of that gate (refuse without the
// capability; proceed with it), plus the DeComFL path is unaffected.
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
    getRuntimeCompatibility: jest.fn().mockResolvedValue({ bridgeAbiVersion: 1, protocolVersion: 2 }),
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

describe('runTrainingLoop — MO-4 capability-gated FedAvg', () => {
  test('refuses a FedAvg run WITHOUT first-order support, fail-closed before any provisioning', async () => {
    // firstOrderSupported defaults false => the only on-device path is the ZO-scalar fedAvgRound a
    // FedAvg server can't aggregate => refuse before touching the device (unchanged MO-4 behavior).
    const p = runTrainingLoop(joinedRun('FedAvg'), hooks);
    await expect(p).rejects.toBeInstanceOf(MobileFedAvgUnsupportedError);
    await expect(runTrainingLoop(joinedRun('FedAvg'), hooks)).rejects.toThrow(/FedAvg/i);
    // Fail-closed = no wasted work: no fetch/stage, no native load.
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
    expect(nativeCore.loadModel).not.toHaveBeenCalled();
    expect(nativeCore.setTrainingDataFromFiles).not.toHaveBeenCalled();
  });

  test('a FedAvg run WITH first-order support proceeds past the guard into provisioning', async () => {
    // firstOrderSupported=true (backend provisioned a trainable-.pte bundle) => FedAvg is no longer
    // refused; it enters the same provision->load->round flow as DeComFL. Sentinel-reject at
    // provisioning proves the guard let it through, without standing up the whole native round.
    (provisionTrainingBundle as jest.Mock).mockRejectedValueOnce(new Error('SENTINEL_PAST_GUARD'));
    await expect(
      runTrainingLoop(joinedRun('FedAvg', /*firstOrderSupported=*/ true), hooks),
    ).rejects.toThrow('SENTINEL_PAST_GUARD');
    expect(provisionTrainingBundle).toHaveBeenCalledWith('run-1');
  });

  test('does NOT refuse a DeComFL run — it proceeds past the guard into provisioning', async () => {
    // Prove the guard is FedAvg-specific: a DeComFL run reaches provisionTrainingBundle. We make that
    // fetch reject with a sentinel so the loop unwinds there (not at the guard), which is enough to show
    // the guard let DeComFL through — without standing up the whole native round.
    (provisionTrainingBundle as jest.Mock).mockRejectedValueOnce(new Error('SENTINEL_PAST_GUARD'));
    await expect(runTrainingLoop(joinedRun('DeComFL'), hooks)).rejects.toThrow('SENTINEL_PAST_GUARD');
    expect(provisionTrainingBundle).toHaveBeenCalledWith('run-1');
  });

  test('DeComFL uses scalar training even when a trainable graph is staged', async () => {
    (provisionTrainingBundle as jest.Mock).mockResolvedValueOnce({
      manifest: {}, lossPtePath: 'loss.pte', lossSha256: 'hash',
      inputsF32Path: 'inputs.f32', inputShape: [8, 4], targetsI64Path: 'targets.i64',
    });
    (nativeCore.loadModel as jest.Mock).mockResolvedValueOnce({ trainableParamCount: 25, tier: '' });
    const getServerStatus = jest.fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING_COMPLETE', currentRound: 1 });
    const runDeComFLRound = jest.fn().mockResolvedValue({
      round: 1, loss: 1, accuracy: 0, scalarsTransmitted: 1,
      uplinkBytes: 8, downlinkBytes: 0, computeMs: 1, reverted: true,
    });
    const runFedAvgRound = jest.fn().mockRejectedValue(new Error('wrong weight-upload path'));

    await runTrainingLoop(joinedRun('DeComFL', /*firstOrderSupported=*/ true), hooks, {
      policy: { maxRoundRetries: 0, maxRejoins: 0, baseBackoffMs: 1, pacingMs: 0, rejoinRecoveryRounds: 1 },
      ops: {
        getServerStatus, runDeComFLRound, runFedAvgRound,
        loadSubmittedRound: async () => null,
        saveSubmittedRound: async () => {},
      },
    });

    expect(runDeComFLRound).toHaveBeenCalledTimes(1);
    expect(runFedAvgRound).not.toHaveBeenCalled();
  });

  test('FedProx refuses before provisioning because native training has no proximal term', async () => {
    (provisionTrainingBundle as jest.Mock).mockRejectedValueOnce(new Error('SENTINEL_PAST_GUARD'));
    await expect(
      runTrainingLoop(joinedRun('FedProx', /*firstOrderSupported=*/ true), hooks),
    ).rejects.toThrow(/proximal term/i);
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test.each(['FedOpt', 'Robust'])(
    'a %s run WITH first-order support proceeds into first-order (no DeComFL fallback, no refusal)',
    async (strategy) => {
      // FedOpt and Robust use the same first-order upload as FedAvg; FedProx cannot join until its
      // client-side proximal term is implemented. Sentinel-reject at
      // provisioning proves it got past the guard on the first-order path.
      (provisionTrainingBundle as jest.Mock).mockRejectedValueOnce(new Error('SENTINEL_PAST_GUARD'));
      await expect(
        runTrainingLoop(joinedRun(strategy, /*firstOrderSupported=*/ true), hooks),
      ).rejects.toThrow('SENTINEL_PAST_GUARD');
      expect(provisionTrainingBundle).toHaveBeenCalledWith('run-1');
    },
  );

  test.each(['FedProx', 'FedOpt', 'Robust'])(
    'a %s run WITHOUT first-order support is still refused (no on-device path against a non-DeComFL server)',
    async (strategy) => {
      await expect(runTrainingLoop(joinedRun(strategy), hooks)).rejects.toBeInstanceOf(
        MobileFedAvgUnsupportedError,
      );
      expect(provisionTrainingBundle).not.toHaveBeenCalled();
    },
  );
});
