// On-device federated training loop. After the client has joined + registered (runJoin.ts), this stages
// the model + local data and runs rounds against the server run until it ends. All training compute
// (forward passes, DeComFL perturbations, first-order updates) happens natively ON THE DEVICE;
// DeComFL uploads scalars and first-order strategies upload trainable weights; training batches are not uploaded.
import nativeCore, { type RoundConfig, type RoundResult, type Strategy } from './nativeCore';
import { joinRun, type JoinedRun, type RunManifest } from './runJoin';
import { provisionTrainingBundle } from './modelProvisioning';
import { snapshotMismatches, type DatasetSnapshot } from './datasetService';
import { assertNativeCompatibility } from './nativeCompatibility';
import type { ExecutionContract } from '../gen/fedlearn/contract/v1/execution_contract_pb';
import { contractPrograms,
  checkBundleAgainstContract,
  decideOnContract,
  type ContractProjection,
  type ContractRefusalCode,
} from './executionContractGate';
import { submittedRoundStore } from './submittedRoundStore';
import { readError } from './errors';
import { qualificationStore, qualifyTrainableProgram, type QualificationStore } from './qualification';

// Server run states that mean "stop looping" (mirrors GetServerStatusResponse.ServerState names).
const TERMINAL_STATES = new Set(['TRAINING_COMPLETE', 'COMPLETED', 'FINISHED', 'FAILED', 'STOPPED', 'ABORTED']);
const PENDING_STATES = new Set(['INITIALIZING', 'WAITING_FOR_CLIENTS', 'AGGREGATING']);
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

function supportedStrategy(value: string): Strategy {
  if (value === 'DeComFL' || value === 'FedAvg' || value === 'FedOpt' || value === 'Robust' || value === 'FedProx') {
    return value;
  }
  throw new MobileFedAvgUnsupportedError(`Unsupported strategy on this device: ${value}.`);
}

/**
 * Raised when a phone joins a run that uses secure aggregation (manifest.secureAggregation). Such a server
 * accepts only masked gradient scalars and refuses unmasked ones, and the phone has no key agreement, share
 * sealing or masking, so every round it trained would be thrown away. Refused before any provisioning or native
 * work. Caught by the training UI and shown as information, like the MO-4 refusal above.
 */
export class MobileSecureAggregationUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MobileSecureAggregationUnsupportedError';
  }
}

// DeComFL gets K/P from the server. First-order runs use per-round server settings when supplied;
// the TinyNet FedAvg/Robust fallback remains until execution contract v1 replaces it.
/**
 * Raised when this device may not train a run under its published execution contract: no contract, one that is not
 * ready, one this app does not accept, or one it would not execute exactly. The phone never falls back to training
 * on the legacy run fields, so a refusal here means the run is not for this device. Caught by the training UI and
 * shown as information, like the refusals above.
 */
export class ExecutionContractRefusedError extends Error {
  constructor(
    readonly code: ContractRefusalCode | 'CONTRACT_TIMEOUT' | 'BUNDLE_MISMATCH' | 'DATASET_REQUIRED' | 'DATASET_INCOMPATIBLE'
      | 'QUALIFICATION_FAILED',
    message: string,
  ) {
    super(message);
    this.name = 'ExecutionContractRefusedError';
  }
}

/** How long to wait for a run's contract to be published, and how often to look. */
export interface ContractWaitOps {
  fetchManifest: (runId: string) => Promise<RunManifest>;
  delay?: (ms: number) => Promise<void>;
  timeoutMs?: number;
  intervalMs?: number;
}

const CONTRACT_WAIT_TIMEOUT_MS = 60_000;
const CONTRACT_WAIT_INTERVAL_MS = 3_000;

/**
 * The contract this device may train under, waiting while the server is still publishing it. Staging a run's
 * artifacts takes a moment, so a PENDING contract is polled rather than refused; anything else is decided at once.
 */
export async function resolveContract(
  joined: JoinedRun,
  ops?: ContractWaitOps,
): Promise<{ contract: ExecutionContract; projection: ContractProjection }> {
  const expected = { runId: joined.runId, projectId: joined.projectId };
  const timeoutMs = ops?.timeoutMs ?? CONTRACT_WAIT_TIMEOUT_MS;
  const intervalMs = ops?.intervalMs ?? CONTRACT_WAIT_INTERVAL_MS;
  const wait = ops?.delay ?? delay;
  let manifest = joined.manifest;
  for (let waited = 0; ; waited += intervalMs) {
    const decision = decideOnContract(manifest, expected);
    if (decision.kind === 'refuse') {
      throw new ExecutionContractRefusedError(decision.code, decision.message);
    }
    if (decision.kind === 'train') {
      return { contract: decision.contract, projection: decision.projection };
    }
    if (!ops?.fetchManifest || waited >= timeoutMs) {
      throw new ExecutionContractRefusedError('CONTRACT_TIMEOUT',
        'This run is still preparing its execution contract. Try joining again in a moment.');
    }
    await wait(intervalMs);
    manifest = await ops.fetchManifest(joined.runId);
  }
}

/** The native round's settings: the contract's training, and the run identity the native core needs. */
function roundConfigFor(joined: JoinedRun, strategy: Strategy, projection: ContractProjection): RoundConfig {
  const m = joined.manifest;
  return {
    // The contract states the strategy and the first-order training; nothing here is a default any more.
    strategy,
    learningRate: projection.learningRate,
    // DeComFL's smoothing and perturbations, which the native round holds the server's round config to. A
    // first-order round never reads them.
    mu: projection.zerothOrder?.smoothing ?? 0.001,
    numPerturbations: projection.zerothOrder?.numPerturbations ?? 1,
    numLocalSteps: projection.numLocalSteps,
    gradEstimateMethod: 'forward',
    initialStateSha256: projection.initialStateSha256,
    proximalMu: projection.proximalMu,
    // Seeded minibatching when the contract states the reproducible batch order; otherwise one whole-dataset step.
    batchSize: projection.minibatch ? projection.batchSize : 0,
    batchSeed: projection.minibatch?.seed ?? '',
    seed: typeof m.seed === 'number' ? m.seed : 0,
    torchVersion: m.torchVersion ?? '',
  };
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// MO-8: bounded retry / backoff / rejoin so one network blip doesn't end participation.
// ---------------------------------------------------------------------------

/** The per-round operations the resilient loop drives — injectable so the state machine is unit-testable
 *  without the native module. */
export interface RoundOps {
  getServerStatus: (runId: string) => Promise<{ serverState: string; currentRound: number }>;
  runFedAvgRound: (runId: string, cfg: RoundConfig) => Promise<RoundResult>;
  runDeComFLRound: (runId: string, cfg: RoundConfig) => Promise<RoundResult>;
  /** Re-establish the run connection (re-enroll + re-register). Returns the (possibly new) run id. */
  rejoin: () => Promise<{ runId: string }>;
  delay: (ms: number) => Promise<void>;
  loadSubmittedRound: (runId: string) => Promise<number | null>;
  saveSubmittedRound: (runId: string, round: number) => Promise<void>;
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

// The native bridge prefixes a model-execution failure (fedlearn::ModelExecutionError: ExecuTorch refused to load or
// run a program on this device's data) so the loop can tell it from a network blip.
const MODEL_EXECUTION_PREFIX = 'MODEL_EXECUTION: ';

/**
 * ExecuTorch could not run the run's model on this device's data. The failure is deterministic for the program and
 * the data, so the loop ends training at once instead of retrying the round and rejoining the run.
 */
export class ModelExecutionFailedError extends Error {
  constructor(readonly detail: string) {
    super(`This device could not run the model on this data, so training stopped: ${detail}`);
    this.name = 'ModelExecutionFailedError';
  }
}

function modelExecutionDetail(e: unknown): string | undefined {
  const message = e instanceof Error ? e.message : String(e);
  const at = message.indexOf(MODEL_EXECUTION_PREFIX);
  return at < 0 ? undefined : message.slice(at + MODEL_EXECUTION_PREFIX.length);
}

class RoundCheckpointError extends Error {
  constructor(cause: unknown) {
    super(`Could not save the round checkpoint; training stopped to avoid a duplicate upload: ${String(cause)}`);
  }
}

class InvalidServerStatusError extends Error {}

/**
 * Run rounds against the server until it ends (or `shouldStop`), surviving transient failures.
 *
 * Per iteration: check the server status (terminal → done), then run one native round. A rejected
 * getServerStatus/round that is NOT a clean STOP is treated as a blip: retry the iteration with
 * exponential backoff up to `maxRoundRetries` CONSECUTIVE failures. A good round resets that streak, so
 * isolated blips never accumulate. When the retry budget is exhausted, escalate to a bounded `rejoin`
 * (re-enroll + re-register) up to `maxRejoins` times, continuing on the new run id. Only once BOTH
 * budgets are spent does the loop give up and rethrow the last error. STOP / terminal state /
 * cooperative stop always end cleanly and are never retried, and a model-execution failure (deterministic for the
 * program and this device's data) ends training at once with ModelExecutionFailedError.
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
  let submittedRound = await ops.loadSubmittedRound(runId);

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
      const currentRound = status.currentRound;
      if (typeof currentRound !== 'number' || !Number.isSafeInteger(currentRound) || currentRound < 0) {
        throw new InvalidServerStatusError('Server status is missing a valid round number.');
      }
      if (PENDING_STATES.has(status.serverState)) {
        await ops.delay(policy.pacingMs || ROUND_PACING_MS);
        continue;
      }
      if (status.serverState !== 'TRAINING') {
        throw new InvalidServerStatusError(`Unknown server training state: ${status.serverState}`);
      }

      // A successful native return means the update was sent, not that the server advanced.
      // Wait through AGGREGATING and reconnects without downloading/training the same round again.
      if (submittedRound !== null && currentRound <= submittedRound) {
        await ops.delay(policy.pacingMs || ROUND_PACING_MS);
        continue;
      }

      const r = init.isFedAvg
        ? await ops.runFedAvgRound(runId, init.cfg)
        : await ops.runDeComFLRound(runId, init.cfg);
      submittedRound = r.round;
      try {
        await ops.saveSubmittedRound(runId, r.round);
      } catch (cause) {
        throw new RoundCheckpointError(cause);
      }
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
      if (e instanceof RoundCheckpointError || e instanceof InvalidServerStatusError) throw e;
      // The native layer rejects a clean stop with a "STOP:"-prefixed message (abort / server ended).
      if (isStopSignal(e)) {
        hooks.onLog('Server ended this client’s participation.');
        return;
      }
      const modelFailure = modelExecutionDetail(e);
      if (modelFailure !== undefined) throw new ModelExecutionFailedError(modelFailure);

      consecutiveSuccesses = 0; // a failure breaks the stability streak
      consecutiveFailures += 1;
      if (consecutiveFailures <= policy.maxRoundRetries) {
        const backoffMs = policy.baseBackoffMs * 2 ** (consecutiveFailures - 1);
        hooks.onLog(
          `Transient error (attempt ${consecutiveFailures}/${policy.maxRoundRetries}): ${readError(e)}; `
          + `retrying in ${backoffMs}ms…`,
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
          const resumedRound = await ops.loadSubmittedRound(rejoined.runId);
          runId = rejoined.runId;
          submittedRound = resumedRound;
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
  // policy / ops / contract exist so tests can inject them; dataset is the snapshot the user bound for a run on the
  // device's own data (a LOCAL_SNAPSHOT contract).
  overrides?: {
    policy?: ResiliencePolicy; ops?: Partial<RoundOps>; contract?: ContractWaitOps; dataset?: DatasetSnapshot;
    qualificationStore?: QualificationStore;
  },
): Promise<void> {
  // One legacy refusal stays ahead of the contract, because it names the real obstacle better than a missing
  // contract would: a secure-aggregation run the phone cannot mask for.
  if (joined.manifest.secureAggregation === true) {
    throw new MobileSecureAggregationUnsupportedError(
      'This run uses secure aggregation, which this device cannot take part in yet: the phone cannot mask ' +
        'its update, and the server refuses unmasked ones. Join this run from the desktop app instead.',
    );
  }

  // The run's published execution contract decides everything about this round: whether this device may train it
  // at all, and with what. It is resolved before any provisioning or native work, so a run this device cannot
  // execute costs nothing. There is no fallback to the legacy run fields.
  const { contract, projection } = await resolveContract(joined, overrides?.contract);
  // A run on the device's own data needs a snapshot the user bound, exactly as the contract requires, before anything
  // is downloaded.
  const dataset = overrides?.dataset;
  if (projection.dataSource === 'LOCAL_SNAPSHOT') {
    if (!dataset) {
      throw new ExecutionContractRefusedError('DATASET_REQUIRED',
        'This run trains on your own data. Choose a dataset on this device first.');
    }
    const requirement = contract.workload.case === 'modelTraining' ? contract.workload.value.data : undefined;
    const mismatches = requirement ? snapshotMismatches(dataset, requirement) : ['data requirement'];
    if (mismatches.length > 0) {
      throw new ExecutionContractRefusedError('DATASET_INCOMPATIBLE',
        `The chosen dataset is not what this run's model takes (${mismatches.join(', ')}).`);
    }
  }
  const strategy = supportedStrategy(projection.strategy);
  // DeComFL uploads gradient scalars; every other v1 strategy uploads the float32 trainable state.
  const isFirstOrder = projection.strategy !== 'DeComFL';

  await assertNativeCompatibility(nativeCore);
  hooks.onLog(`Execution contract ${projection.contractId.slice(0, 12)}… accepted.`);
  hooks.onLog('Provisioning model + on-device data…');
  const ownData = projection.dataSource === 'LOCAL_SNAPSHOT' ? dataset : undefined;
  const bundle = await provisionTrainingBundle(joined.runId, contractPrograms(contract), { fixtureData: !ownData });
  const staged = ownData
    ? { inputsPath: ownData.inputsPath, shape: [ownData.recordCount, ...ownData.inputShape], targetsPath: ownData.targetsPath }
    : { inputsPath: bundle.inputsF32Path, shape: bundle.inputShape, targetsPath: bundle.targetsI64Path };
  if (!staged.inputsPath || !staged.shape || !staged.targetsPath) {
    throw new ExecutionContractRefusedError('BUNDLE_MISMATCH', 'This run delivered no training data.');
  }

  // The staged bundle must be the one the contract binds: same programs, same trainable layout.
  const unbound = checkBundleAgainstContract(contract, {
    lossSha256: bundle.lossSha256,
    inferSha256: bundle.manifest.inferSha256,
    trainableSha256: bundle.manifest.trainableSha256,
    paramLayout: bundle.manifest.paramLayout,
  });
  if (unbound.length > 0) {
    throw new ExecutionContractRefusedError('BUNDLE_MISMATCH',
      `This run's staged model is not the one its execution contract binds (${unbound.join(', ')}).`);
  }

  // Without the reproducible batch order the native trainer takes one whole-dataset step per epoch, so it reproduces
  // the contract only when the staged data fits in one batch; a larger dataset would train steps the contract does
  // not state. With it (projection.minibatch), any number of examples trains in the contract's minibatches.
  const records = staged.shape[0] ?? 0;
  if (!(records >= 1 && (projection.minibatch !== undefined || records <= projection.batchSize))) {
    throw new ExecutionContractRefusedError('UNSUPPORTED_BATCHING',
      `This run trains batches of ${projection.batchSize}, but this device has ${records} examples; it can only `
      + 'train data that fits in one batch.');
  }

  await nativeCore.setModelManifest(bundle.manifest);
  const info = await nativeCore.loadModel(bundle.lossPtePath, bundle.lossSha256);
  hooks.onLog(`Model loaded — ${info.trainableParamCount} trainable params (tier ${info.tier}).`);

  // Stage 3 D2: a first-order round trains with the trainable program, so the device first proves it runs that
  // program correctly (cached per device, app build and program). DeComFL trains with the loss program only.
  if (isFirstOrder && bundle.manifest.trainablePtePath && bundle.manifest.trainableSha256) {
    const q = await qualifyTrainableProgram(bundle.manifest.trainableSha256, bundle.trainableProbe, {
      store: overrides?.qualificationStore ?? qualificationStore, native: nativeCore });
    if (!q.passed) {
      throw new ExecutionContractRefusedError('QUALIFICATION_FAILED',
        `This device did not qualify to run this run's model (${q.failedCheck}: ${q.detail}).`);
    }
    hooks.onLog(`Model qualified on this device (probe ${q.wallMs} ms).`);
  }

  await nativeCore.setTrainingDataFromFiles(staged.inputsPath, staged.shape, staged.targetsPath);
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
    loadSubmittedRound: (runId) => submittedRoundStore.load(runId),
    saveSubmittedRound: (runId, round) => submittedRoundStore.save(runId, round),
    ...overrides?.ops,
  };

  await runResilientRoundLoop(
    { runId: joined.runId, isFedAvg: isFirstOrder, cfg: roundConfigFor(joined, strategy, projection) },
    ops,
    overrides?.policy ?? DEFAULT_RESILIENCE,
    hooks,
  );
}
