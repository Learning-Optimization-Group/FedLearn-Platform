// On-device federated training loop. After the client has joined + registered (runJoin.ts), this stages
// the model + local data and runs rounds against the server run until it ends. All training compute
// (forward passes, DeComFL zeroth-order perturbations) happens natively ON THE DEVICE; per round only
// perturbation seeds + gradient scalars are uploaded (DLG-resistant) — raw features/labels never leave.
import nativeCore, { type RoundConfig, type RoundResult } from './nativeCore';
import type { JoinedRun } from './runJoin';
import { provisionTrainingBundle } from './modelProvisioning';

// Server run states that mean "stop looping" (mirrors GetServerStatusResponse.ServerState names).
const TERMINAL_STATES = new Set(['COMPLETED', 'FINISHED', 'FAILED', 'STOPPED', 'ABORTED']);
const ROUND_PACING_MS = 1500; // brief pause between rounds so we don't hot-poll the server

export interface TrainingHooks {
  onLog: (line: string) => void;
  onRound: (r: RoundResult) => void;
  shouldStop: () => boolean;
}

// The client proposes a config; the server is authoritative on K/P (applied inside the native round).
function roundConfigFor(joined: JoinedRun): RoundConfig {
  const m = joined.manifest;
  return {
    strategy: m.strategy === 'FedAvg' ? 'FedAvg' : 'DeComFL',
    learningRate: 0.001,
    mu: 0.001,
    numPerturbations: 1,
    numLocalSteps: 1,
    gradEstimateMethod: 'forward',
    seed: typeof m.seed === 'number' ? m.seed : 0,
    torchVersion: m.torchVersion ?? '',
  };
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Run the on-device training loop to completion (or until `shouldStop`). Throws
 * ModelDeliveryUnavailableError if the model/data bundle can't be staged yet (see modelProvisioning.ts).
 */
export async function runTrainingLoop(joined: JoinedRun, hooks: TrainingHooks): Promise<void> {
  const isFedAvg = joined.manifest.strategy === 'FedAvg';

  hooks.onLog('Provisioning model + on-device data…');
  const bundle = await provisionTrainingBundle(joined.runId);

  await nativeCore.setModelManifest(bundle.manifest);
  const info = await nativeCore.loadModel(bundle.lossPtePath, bundle.lossSha256);
  hooks.onLog(`Model loaded — ${info.trainableParamCount} trainable params (tier ${info.tier}).`);

  await nativeCore.setTrainingDataFromFiles(
    bundle.inputsF32Path,
    bundle.inputShape,
    bundle.targetsI64Path,
  );
  hooks.onLog('On-device data staged. Training starts — your data never leaves this device.');

  const cfg = roundConfigFor(joined);
  for (;;) {
    if (hooks.shouldStop()) {
      hooks.onLog('Training stopped.');
      return;
    }
    const status = await nativeCore.getServerStatus(joined.runId);
    if (TERMINAL_STATES.has(status.serverState)) {
      hooks.onLog(`Run ${status.serverState.toLowerCase()}.`);
      return;
    }
    try {
      const r = isFedAvg
        ? await nativeCore.runFedAvgRound(joined.runId, cfg)
        : await nativeCore.runDeComFLRound(joined.runId, cfg);
      hooks.onRound(r);
      hooks.onLog(
        `Round ${r.round}: loss ${r.loss.toFixed(4)} · ${r.scalarsTransmitted} scalars up · ${r.computeMs}ms`,
      );
    } catch (e) {
      // The native layer rejects a clean stop with a "STOP:"-prefixed message (abort / server ended).
      if (String(e).includes('STOP:')) {
        hooks.onLog('Server ended this client’s participation.');
        return;
      }
      throw e;
    }
    await delay(ROUND_PACING_MS);
  }
}
