// Execution contract v1 on Android. Before the phone downloads a model or opens local data, it decides from the
// run's published contract whether it may train at all, and projects the contract into the settings the native
// trainer runs with. Android is v1-dependent: a run without a READY contract it can execute exactly is refused
// with a precise reason, never approximated.
import { ArtifactBackend, ContractIssueCodeSchema, GradientEstimator, PerturbationRng, SecureAggregation, Strategy,
  UpdateProtocol, type ArtifactVariant, type ExecutionContract, type LocalTraining, type ModelTraining,
} from '../gen/fedlearn/contract/v1/execution_contract_pb';
import { MalformedContractError, parseContractJson, validateContract } from './executionContract';
import { SERVER_PROTOCOL_VERSION } from './nativeCompatibility';

/** The contract fields of a run manifest or connection payload, as the backend serves them. */
export interface ContractCarrier {
  contractState?: string;
  contractId?: string;
  executionContract?: Record<string, unknown>;
  contractUnavailableReason?: string;
}

export type ContractRefusalCode =
  | 'CONTRACT_MISSING'
  | 'CONTRACT_LEGACY_ONLY'
  | 'CONTRACT_UNAVAILABLE'
  | 'CONTRACT_INVALID'
  | 'UNSUPPORTED_STRATEGY'
  | 'UNSUPPORTED_UPDATE_PROTOCOL'
  | 'UNSUPPORTED_SECURITY'
  | 'UNSUPPORTED_OPTIMIZER'
  | 'UNSUPPORTED_BATCHING'
  | 'MISSING_CPU_ARTIFACT';

/**
 * The first-order strategies v1 approves: ordinary client training, with the strategy's own work on the server.
 * FedProx also adds the proximal term its contract states.
 */
export type FirstOrderStrategy = 'FedAvg' | 'FedOpt' | 'Robust' | 'FedProx';

const FIRST_ORDER_STRATEGIES: ReadonlyMap<Strategy, FirstOrderStrategy> = new Map([
  [Strategy.FEDAVG, 'FedAvg'],
  [Strategy.FEDOPT, 'FedOpt'],
  [Strategy.ROBUST, 'Robust'],
  [Strategy.FEDPROX, 'FedProx'],
]);

/** What the native trainer needs from the contract for one round. */
export interface ContractProjection {
  contractId: string;
  strategy: FirstOrderStrategy | 'DeComFL';
  learningRate: number;
  /** First-order: local epochs of one whole-batch step each. DeComFL: K, the zeroth-order local steps. */
  numLocalSteps: number;
  batchSize: number;
  /** The run's initial trainable state; a DeComFL round starts only from the server's model with this digest. */
  initialStateSha256: string;
  /** FedProx's proximal coefficient; 0 for every other strategy. */
  proximalMu: number;
  /** DeComFL only: the zeroth-order settings the native round holds the server's round config to. */
  zerothOrder?: { smoothing: number; numPerturbations: number };
}

export type ContractDecision =
  | { kind: 'train'; contract: ExecutionContract; projection: ContractProjection }
  | { kind: 'wait' }
  | { kind: 'refuse'; code: ContractRefusalCode; message: string };

/** The device's ABI: the only one the app ships native code for. */
const DEVICE_ABI = 'arm64-v8a';

function refuse(code: ContractRefusalCode, message: string): ContractDecision {
  return { kind: 'refuse', code, message };
}

/**
 * Whether the phone may train this run, and with what. Every refusal names what the phone cannot do, so the user
 * is told the real reason rather than seeing a round fail later.
 */
export function decideOnContract(
  carrier: ContractCarrier,
  expected: { runId?: string; projectId?: string },
): ContractDecision {
  switch (carrier.contractState) {
    case undefined:
    case null:
      return refuse('CONTRACT_MISSING',
        'This server does not publish execution contracts. Update the server, or use a client that trains on '
        + 'the legacy run fields.');
    case 'PENDING':
      return { kind: 'wait' };
    case 'LEGACY_ONLY':
      return refuse('CONTRACT_LEGACY_ONLY',
        'This run started before execution contracts and cannot be joined by this app. Start a new run.');
    case 'UNAVAILABLE':
      return refuse('CONTRACT_UNAVAILABLE',
        'This run has no execution contract'
        + (carrier.contractUnavailableReason ? ` (${carrier.contractUnavailableReason})` : '')
        + ', so this app cannot train it.');
    case 'READY':
      break;
    default:
      return refuse('CONTRACT_INVALID',
        `This run reports an execution contract state this app does not understand (${carrier.contractState}).`);
  }
  if (!carrier.executionContract || !carrier.contractId) {
    return refuse('CONTRACT_INVALID', 'This run reports a ready execution contract but did not send it.');
  }
  let contract: ExecutionContract;
  try {
    contract = parseContractJson(JSON.stringify(carrier.executionContract));
  } catch (e) {
    if (e instanceof MalformedContractError) {
      return refuse('CONTRACT_INVALID', 'This run\'s execution contract could not be read.');
    }
    throw e;
  }
  const issues = validateContract(contract, {
    readerProtocolVersion: SERVER_PROTOCOL_VERSION,
    expectedRunId: expected.runId,
    expectedProjectId: expected.projectId,
  });
  if (issues.length > 0) {
    const named = issues
      .map(i => `${ContractIssueCodeSchema.value[i.code]?.name ?? i.code} at ${i.path || 'the contract'}`)
      .sort();
    return refuse('CONTRACT_INVALID', `This run's execution contract is not one this app accepts: ${named.join(', ')}.`);
  }
  return projectContract(contract, carrier.contractId);
}

/**
 * The capability layer: given a valid contract, whether this app's trainer executes exactly what it states, and the
 * settings it runs with. Exported so each refusal is testable on its own, including for runs the approved matrix
 * does not publish yet.
 */
export function projectContract(contract: ExecutionContract, contractId: string): ContractDecision {
  if (contract.workload.case !== 'modelTraining') {
    return refuse('CONTRACT_INVALID', 'This run\'s execution contract describes no model training.');
  }
  const training = contract.workload.value;
  if (contract.strategy === Strategy.DECOMFL) {
    return projectDeComFL(contract, contractId, training);
  }
  const strategy = FIRST_ORDER_STRATEGIES.get(contract.strategy);
  if (!strategy) {
    return refuse('UNSUPPORTED_STRATEGY',
      'This app trains only the first-order weight path published for FedAvg, FedOpt, Robust and FedProx runs; '
      + 'this run uses another strategy.');
  }
  if (strategy === 'FedProx' && training.fedproxMu === undefined) {
    return refuse('CONTRACT_INVALID', 'This FedProx run\'s execution contract states no proximal coefficient.');
  }
  if (training.updateProtocol !== UpdateProtocol.UPDATE_TRAINABLE_STATE_F32) {
    return refuse('UNSUPPORTED_UPDATE_PROTOCOL', 'This run expects an update this app does not produce.');
  }
  if (contract.security?.secureAggregation !== SecureAggregation.SECAGG_NONE) {
    return refuse('UNSUPPORTED_SECURITY', 'This run requires secure aggregation, which this app cannot perform.');
  }
  const local = training.localTraining;
  if (!local) {
    return refuse('CONTRACT_INVALID', 'This run\'s execution contract states no local training.');
  }
  const optimizer = unsupportedOptimizer(local);
  if (optimizer) {
    return refuse('UNSUPPORTED_OPTIMIZER', optimizer);
  }
  // The native trainer takes one step per local epoch over the whole local dataset, so it can reproduce a contract
  // whose every epoch is a single batch, keeping the final batch. The dataset's size is checked against batchSize
  // when the data is staged.
  if (local.dropLast !== false || local.maxLocalSteps !== undefined) {
    return refuse('UNSUPPORTED_BATCHING',
      'This run batches its local training in a way this app\'s trainer cannot reproduce.');
  }
  if (!portableCpuVariant(training.artifacts)) {
    return refuse('MISSING_CPU_ARTIFACT', `This run has no portable CPU model for ${DEVICE_ABI} devices.`);
  }
  const sgd = local.optimizer.case === 'sgd' ? local.optimizer.value : undefined;
  return {
    kind: 'train',
    contract,
    projection: {
      contractId,
      strategy,
      learningRate: sgd!.learningRate,
      numLocalSteps: local.localEpochs,
      batchSize: local.batchSize,
      initialStateSha256: training.initialStateSha256,
      proximalMu: training.fedproxMu ?? 0,
    },
  };
}

/**
 * DeComFL: zeroth-order training whose update is gradient scalars. The native round computes the forward
 * difference only, draws perturbations with the generator that byte-matches torch.randn, and trains the staged
 * batch whole; anything else the contract states is refused.
 */
function projectDeComFL(contract: ExecutionContract, contractId: string, training: ModelTraining): ContractDecision {
  if (training.updateProtocol !== UpdateProtocol.UPDATE_DECOMFL_SCALAR) {
    return refuse('UNSUPPORTED_UPDATE_PROTOCOL', 'This DeComFL run expects an update this app does not produce.');
  }
  if (contract.security?.secureAggregation !== SecureAggregation.SECAGG_NONE) {
    return refuse('UNSUPPORTED_SECURITY', 'This run requires secure aggregation, which this app cannot perform.');
  }
  const local = training.localTraining;
  if (!local) {
    return refuse('CONTRACT_INVALID', 'This run\'s execution contract states no local training.');
  }
  if (local.optimizer.case !== 'zerothOrderSgd') {
    return refuse('UNSUPPORTED_OPTIMIZER', 'This DeComFL run does not state zeroth-order training.');
  }
  const zo = local.optimizer.value;
  if (zo.estimator !== GradientEstimator.ESTIMATOR_FORWARD) {
    return refuse('UNSUPPORTED_OPTIMIZER', 'This run uses a gradient estimator this app does not implement; it '
      + 'computes the forward difference only.');
  }
  if (zo.rng !== PerturbationRng.RNG_TORCH_CPU_RANDN_F32) {
    return refuse('UNSUPPORTED_OPTIMIZER', 'This run draws perturbations with a generator this app does not have.');
  }
  if (local.dropLast !== false) {
    return refuse('UNSUPPORTED_BATCHING',
      'This run batches its local training in a way this app\'s trainer cannot reproduce.');
  }
  if (!portableCpuVariant(training.artifacts)) {
    return refuse('MISSING_CPU_ARTIFACT', `This run has no portable CPU model for ${DEVICE_ABI} devices.`);
  }
  return {
    kind: 'train',
    contract,
    projection: {
      contractId,
      strategy: 'DeComFL',
      learningRate: zo.learningRate,
      numLocalSteps: zo.numLocalSteps,
      batchSize: local.batchSize,
      initialStateSha256: training.initialStateSha256,
      proximalMu: 0,
      zerothOrder: { smoothing: zo.smoothing, numPerturbations: zo.numPerturbations },
    },
  };
}

/** Why this app's trainer cannot run the contract's optimizer, or undefined when it can. */
function unsupportedOptimizer(local: LocalTraining): string | undefined {
  if (local.optimizer.case !== 'sgd') {
    return `This run trains with ${local.optimizer.case ?? 'no'} optimizer; this app implements plain SGD.`;
  }
  const sgd = local.optimizer.value;
  if (sgd.momentum !== 0 || sgd.dampening !== 0 || sgd.weightDecay !== 0 || sgd.nesterov !== false) {
    return 'This run trains with SGD settings this app implements no state for (momentum, dampening, weight '
      + 'decay or Nesterov).';
  }
  if (local.gradientClipNorm !== undefined) {
    return 'This run clips gradients, which this app does not implement.';
  }
  if (local.resetOptimizerEachRound !== true) {
    return 'This run carries optimizer state between rounds, which this app does not implement.';
  }
  return undefined;
}

function portableCpuVariant(variants: readonly ArtifactVariant[]): ArtifactVariant | undefined {
  return variants.find(v => v.backend === ArtifactBackend.BACKEND_EXECUTORCH_CPU && v.abi === DEVICE_ABI);
}

/** What a staged bundle carries, as far as the contract binds it. */
export interface StagedBundleFacts {
  lossSha256: string;
  inferSha256: string;
  trainableSha256?: string;
  paramLayout: { name: string; shape: number[] }[];
}

/**
 * What the staged bundle has that the contract did not bind: each program whose digest differs from the contract's,
 * and the layout when it is not the contract's trainable layout. Empty means the bundle is the contract's.
 */
export function checkBundleAgainstContract(contract: ExecutionContract, bundle: StagedBundleFacts): string[] {
  if (contract.workload.case !== 'modelTraining') {
    return ['model training'];
  }
  const training = contract.workload.value;
  const variant = portableCpuVariant(training.artifacts);
  const digests = new Map((variant?.files ?? []).map(f => [f.relativePath, f.sha256]));
  const mismatched: string[] = [];
  for (const [name, staged] of [['loss.pte', bundle.lossSha256], ['infer.pte', bundle.inferSha256],
    ['trainable.pte', bundle.trainableSha256]] as const) {
    if (digests.get(name) !== staged) {
      mismatched.push(name);
    }
  }
  const stated = training.trainable.map(t => `${t.name}:${t.shape.join(',')}`).join('|');
  const staged = bundle.paramLayout.map(p => `${p.name}:${p.shape.join(',')}`).join('|');
  if (stated !== staged) {
    mismatched.push('trainable parameter layout');
  }
  return mismatched;
}
