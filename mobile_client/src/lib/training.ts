// On-device federated training loop. After the client has joined + registered (runJoin.ts), this stages
// the model + local data and runs rounds against the server run until it ends. All training compute
// (forward passes, DeComFL zeroth-order perturbations) happens natively ON THE DEVICE; per round only
// perturbation seeds + gradient scalars are uploaded (DLG-resistant) — raw features/labels never leave.
import nativeCore, { type RoundConfig, type RoundResult } from './nativeCore';
import { joinRun, type JoinedRun } from './runJoin';
import { provisionTrainingBundle } from './modelProvisioning';

// Server run states that mean "stop looping" (mirrors GetServerStatusResponse.ServerState names).
const TERMINAL_STATES = new Set(['COMPLETED', 'FINISHED', 'FAILED', 'STOPPED', 'ABORTED']);
const ROUND_PACING_MS = 1500; // brief pause between rounds so we don't hot-poll the server

export interface TrainingHooks {
  onLog: (line: string) => void;
  onRound: (r: RoundResult) => void;
  shouldStop: () => boolean;
}

/**
 * MO-4: raised when a phone joins a FedAvg run that has NOT been provisioned for first-order on-device
 * training (manifest.firstOrderSupported is absent/false). Without first-order support the only
 * on-device path is FederatedLoop::fedAvgRound — local ZO-SGD uploading seeds + gradient SCALARS via
 * SubmitGradientScalars (the DeComFL wire), which a FedAvg *strategy* server cannot aggregate (it
 * expects a weight blob via SubmitModelUpdateStream), so the "training" would submit into a void. We
 * refuse fail-closed rather than no-op. Once the backend provisions a trainable-.pte bundle AND the
 * native firstOrderRound (real backprop -> weight-blob upload) is wired, the run's manifest sets
 * firstOrderSupported and the phone runs the first-order path instead of raising this.
 * Caught by the training UI to show a clear "not provisioned for on-device training yet" message.
 */
export class MobileFedAvgUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MobileFedAvgUnsupportedError';
  }
}

// The client proposes a config; the server is authoritative on K/P (applied inside the native round).
function roundConfigFor(joined: JoinedRun): RoundConfig {
  const m = joined.manifest;
  return {
    // First-order runs upload a WEIGHT blob regardless of the server strategy (the server aggregates
    // per its own strategy — FedAvg/FedProx/FedOpt/Robust — all consume SubmitModelUpdateStream). Only
    // fall back to the zeroth-order DeComFL wire when no first-order bundle was provisioned.
    strategy: m.strategy === 'FedAvg' || m.firstOrderSupported ? 'FedAvg' : 'DeComFL',
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

// ---------------------------------------------------------------------------
// MO-8: bounded retry / backoff / rejoin so one network blip doesn't end participation.
// ---------------------------------------------------------------------------

/** The per-round operations the resilient loop drives — injectable so the state machine is unit-testable
 *  without the native module. `getServerStatus`'s ServerStatus is narrowed to the field the loop reads. */
export interface RoundOps {
  getServerStatus: (runId: string) => Promise<{ serverState: string }>;
  runFedAvgRound: (runId: string, cfg: RoundConfig) => Promise<RoundResult>;
  runDeComFLRound: (runId: string, cfg: RoundConfig) => Promise<RoundResult>;
  /** Re-establish the run connection (re-enroll + re-register). Returns the (possibly new) run id. */
  rejoin: () => Promise<{ runId: string }>;
  delay: (ms: number) => Promise<void>;
}

export interface ResiliencePolicy {
  /** Consecutive transient failures tolerated (with exponential backoff) before escalating to a rejoin. */
  maxRoundRetries: number;
  /** Total rejoins allowed for the whole run before the loop finally gives up. */
  maxRejoins: number;
  /** Exponential backoff base: attempt N waits baseBackoffMs * 2^(N-1). */
  baseBackoffMs: number;
  /** Pause between successful rounds (avoids hot-polling the server). */
  pacingMs: number;
  /** After this many CONSECUTIVE good rounds following a rejoin, the connection is deemed stable again
   *  and the rejoin budget is restored — so several INDEPENDENT blips over a long run don't cumulatively
   *  end participation, while a persistently-flaky link (never this many successes in a row) stays
   *  bounded and eventually gives up. */
  rejoinRecoveryRounds: number;
}

export const DEFAULT_RESILIENCE: ResiliencePolicy = {
  maxRoundRetries: 3,
  maxRejoins: 2,
  baseBackoffMs: 1000,
  pacingMs: ROUND_PACING_MS,
  rejoinRecoveryRounds: 5,
};

const isStopSignal = (e: unknown): boolean => String(e).includes('STOP:');

/**
 * Run rounds against the server until it ends (or `shouldStop`), surviving transient failures.
 *
 * Per iteration: check the server status (terminal → done), then run one native round. A rejected
 * getServerStatus/round that is NOT a clean STOP is treated as a blip: retry the iteration with
 * exponential backoff up to `maxRoundRetries` CONSECUTIVE failures. A good round resets that streak, so
 * isolated blips never accumulate. When the retry budget is exhausted, escalate to a bounded `rejoin`
 * (re-enroll + re-register) up to `maxRejoins` times, continuing on the new run id. Only once BOTH
 * budgets are spent does the loop give up and rethrow the last error. STOP / terminal state /
 * cooperative stop always end cleanly and are never retried.
 *
 * NOTE: this bounds the common blip, which surfaces as a fast Promise REJECTION. A call that HANGS
 * (never settles) is out of scope here — that needs per-RPC deadlines on the native gRPC path (MO-2).
 */
export async function runResilientRoundLoop(
  init: { runId: string; isFedAvg: boolean; cfg: RoundConfig },
  ops: RoundOps,
  policy: ResiliencePolicy,
  hooks: TrainingHooks,
): Promise<void> {
  let runId = init.runId;
  let consecutiveFailures = 0;
  let consecutiveSuccesses = 0;
  let rejoinsUsed = 0;

  for (;;) {
    if (hooks.shouldStop()) {
      hooks.onLog('Training stopped.');
      return;
    }

    try {
      const status = await ops.getServerStatus(runId);
      if (TERMINAL_STATES.has(status.serverState)) {
        hooks.onLog(`Run ${status.serverState.toLowerCase()}.`);
        return;
      }

      const r = init.isFedAvg
        ? await ops.runFedAvgRound(runId, init.cfg)
        : await ops.runDeComFLRound(runId, init.cfg);
      hooks.onRound(r);
      hooks.onLog(
        `Round ${r.round}: loss ${r.loss.toFixed(4)} · ${r.scalarsTransmitted} scalars up · ${r.computeMs}ms`,
      );

      consecutiveFailures = 0; // a good round clears the failure streak
      consecutiveSuccesses += 1;
      // Once the connection has re-proven itself stable, restore the rejoin budget so later independent
      // blips over a long run are each survivable (bounded: a flaky link never reaches this streak).
      if (rejoinsUsed > 0 && consecutiveSuccesses >= policy.rejoinRecoveryRounds) {
        rejoinsUsed = 0;
        consecutiveSuccesses = 0;
        hooks.onLog('Connection stable — reconnect budget restored.');
      }
      if (policy.pacingMs > 0) await ops.delay(policy.pacingMs);
    } catch (e) {
      // The native layer rejects a clean stop with a "STOP:"-prefixed message (abort / server ended).
      if (isStopSignal(e)) {
        hooks.onLog('Server ended this client’s participation.');
        return;
      }

      consecutiveSuccesses = 0; // a failure breaks the stability streak
      consecutiveFailures += 1;
      if (consecutiveFailures <= policy.maxRoundRetries) {
        const backoffMs = policy.baseBackoffMs * 2 ** (consecutiveFailures - 1);
        hooks.onLog(
          `Transient error (attempt ${consecutiveFailures}/${policy.maxRoundRetries}); retrying in ${backoffMs}ms…`,
        );
        await ops.delay(backoffMs);
        continue;
      }

      // Retry budget exhausted for this streak — escalate to a bounded rejoin.
      if (rejoinsUsed < policy.maxRejoins) {
        rejoinsUsed += 1;
        hooks.onLog(`Reconnecting to the run (rejoin ${rejoinsUsed}/${policy.maxRejoins})…`);
        try {
          const rejoined = await ops.rejoin();
          runId = rejoined.runId;
          consecutiveFailures = 0;
          continue;
        } catch (rejoinErr) {
          // A failed rejoin still counts toward the budget; loop retries a rejoin or gives up below.
          hooks.onLog(`Rejoin failed: ${String(rejoinErr)}`);
          continue;
        }
      }

      // Out of both retries and rejoins — give up, surfacing the last error to the caller.
      throw e;
    }
  }
}

/**
 * Run the on-device training loop to completion (or until `shouldStop`). Throws
 * ModelDeliveryUnavailableError if the model/data bundle can't be staged yet (see modelProvisioning.ts).
 *
 * `overrides` exists only so tests / advanced callers can inject the resilience policy or ops; production
 * callers pass just (joined, hooks) and get the native ops + DEFAULT_RESILIENCE.
 */
export async function runTrainingLoop(
  joined: JoinedRun,
  hooks: TrainingHooks,
  overrides?: { policy?: ResiliencePolicy; ops?: Partial<RoundOps> },
): Promise<void> {
  // A run is FIRST-ORDER trainable on-device whenever the backend provisioned a trainable bundle
  // (manifest.firstOrderSupported) — the native firstOrderRound does real backprop and uploads a WEIGHT
  // blob that ANY gradient-aggregation server consumes (FedAvg/FedProx/FedOpt/Robust all take
  // SubmitModelUpdateStream). The server applies its own strategy to the weight update; only FedProx's
  // client-side proximal term is not yet applied on-device (see note), so it runs as FedAvg-equivalent
  // local training under a FedProx server. Non-FedAvg servers therefore no longer force the DeComFL wire.
  const isFirstOrder = joined.manifest.firstOrderSupported === true;

  // MO-4 (capability-gated, generalized): without a first-order bundle the only on-device path is the
  // ZO-scalar DeComFL round, which a NON-DeComFL server can't consume — refuse fail-closed before any
  // provisioning/native work rather than submit into a void. (A DeComFL server + no bundle is the
  // supported zeroth-order path and is allowed.)
  if (!isFirstOrder && joined.manifest.strategy !== 'DeComFL') {
    throw new MobileFedAvgUnsupportedError(
      `This run uses the ${joined.manifest.strategy} strategy but is not provisioned for on-device ` +
        'training yet: first-order (weight-update) support is not enabled. Join a first-order-provisioned ' +
        'or DeComFL project to train on this device.',
    );
  }

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

  // The model + on-device data stay loaded natively across a rejoin, so rejoin only re-establishes the
  // run connection (re-enroll + re-register) — it does NOT re-provision.
  const ops: RoundOps = {
    getServerStatus: (runId) => nativeCore.getServerStatus(runId),
    runFedAvgRound: (runId, cfg) => nativeCore.runFedAvgRound(runId, cfg),
    runDeComFLRound: (runId, cfg) => nativeCore.runDeComFLRound(runId, cfg),
    rejoin: async () => {
      const re = await joinRun({ projectId: joined.projectId });
      return { runId: re.runId };
    },
    delay,
    ...overrides?.ops,
  };

  await runResilientRoundLoop(
    { runId: joined.runId, isFedAvg: isFirstOrder, cfg: roundConfigFor(joined) },
    ops,
    overrides?.policy ?? DEFAULT_RESILIENCE,
    hooks,
  );
}
