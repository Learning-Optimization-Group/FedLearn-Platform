// Execution contract v1 on Android: the phone decides from the run's published contract, before it downloads a
// model or opens local data, whether it may train at all — and refuses with a precise reason when it may not.
// Android is v1-dependent: a run without a READY contract it can execute is refused, never approximated.
import { create, toJson } from '@bufbuild/protobuf';
import {
  ArtifactBackend,
  ExecutionContractSchema,
  GradientEstimator,
  PerturbationRng,
  SecureAggregation,
  Strategy,
  UpdateProtocol,
  ZerothOrderSgdSchema,
  type ExecutionContract,
} from '@/gen/fedlearn/contract/v1/execution_contract_pb';
import {
  checkBundleAgainstContract,
  contractPrograms,
  decideOnContract,
  projectContract,
  type ContractCarrier,
} from '@/lib/executionContractGate';
import { parseContractBinary } from '@/lib/executionContract';

declare const __dirname: string;
type FixtureFs = { readFileSync(path: string, encoding?: 'utf8'): string & Uint8Array };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs: FixtureFs = require('fs');
const GOLDEN = `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb`;

const RUN_ID = '4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17';
const PROJECT_ID = '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64';

function golden(): ExecutionContract {
  return parseContractBinary(new Uint8Array(fs.readFileSync(GOLDEN)));
}

/** A manifest carrying `contract` as the backend serves it: ProtoJSON beside the legacy fields. */
function manifestFor(contract: ExecutionContract, over: Partial<ContractCarrier> = {}): ContractCarrier {
  return {
    contractState: 'READY',
    contractId: 'a'.repeat(64),
    executionContract: toJson(ExecutionContractSchema, contract) as Record<string, unknown>,
    ...over,
  };
}

const EXPECTED = { runId: RUN_ID, projectId: PROJECT_ID };

function decide(contract: ExecutionContract, over: Partial<ContractCarrier> = {}) {
  return decideOnContract(manifestFor(contract, over), EXPECTED);
}

describe('the phone decides from the contract state', () => {
  it('trains on a READY contract it can execute', () => {
    const decision = decide(golden());
    expect(decision.kind).toBe('train');
    if (decision.kind !== 'train') throw new Error('expected to train');
    // The projection is the shared fixture the native C++ round test trains with, so one chain is checked:
    // contract -> projection -> the numbers the native golden replays.
    const shared = JSON.parse(fs.readFileSync(
      `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/projection_tinynet_fedavg.json`,
      'utf8')) as { learningRate: number; numLocalSteps: number; batchSize: number };
    expect(decision.projection).toEqual({
      contractId: 'a'.repeat(64),
      strategy: 'FedAvg',
      learningRate: shared.learningRate,
      numLocalSteps: shared.numLocalSteps,
      batchSize: shared.batchSize,
      initialStateSha256: '1122ba73e49f6df981861bb76d3dcff46666abb5f41e3a6a4d510db9fddd965c',
      proximalMu: 0,
    });
  });

  it('waits while the contract is still being published', () => {
    expect(decide(golden(), { contractState: 'PENDING', executionContract: undefined }))
      .toEqual({ kind: 'wait' });
  });

  it.each([
    ['UNAVAILABLE', 'CONTRACT_UNAVAILABLE'],
    ['LEGACY_ONLY', 'CONTRACT_LEGACY_ONLY'],
  ])('refuses a run whose contract state is %s', (state, code) => {
    const decision = decide(golden(), { contractState: state, executionContract: undefined });
    expect(decision).toMatchObject({ kind: 'refuse', code });
  });

  it('refuses a backend that publishes no contract at all', () => {
    expect(decideOnContract({}, EXPECTED)).toMatchObject({ kind: 'refuse', code: 'CONTRACT_MISSING' });
  });

  it('refuses a READY state with no contract', () => {
    expect(decide(golden(), { executionContract: undefined }))
      .toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
  });

  it('refuses a contract that does not parse', () => {
    expect(decide(golden(), { executionContract: { contractVersion: 'one' } as Record<string, unknown> }))
      .toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
  });

  it('names the issues of an invalid contract', () => {
    const contract = golden();
    contract.numRounds = 0;
    const decision = decide(contract);
    expect(decision).toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
    if (decision.kind !== 'refuse') throw new Error('expected a refusal');
    expect(decision.message).toContain('ISSUE_OUT_OF_RANGE');
    expect(decision.message).toContain('numRounds');
  });

  it('refuses a contract published for another run', () => {
    const contract = golden();
    contract.runId = '00000000-0000-4000-8000-000000000001';
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
  });
});

describe('the phone trains the first-order strategies v1 approves', () => {
  // FedOpt and Robust are first-order client training; the server adapts or aggregates robustly. The phone runs
  // each with the contract's own training and tells the native round which strategy it is, because a FedOpt round
  // also requires the server to confirm the contract's rate.
  it.each([
    ['FedOpt', Strategy.FEDOPT, 0.01],
    ['Robust', Strategy.ROBUST, 0.001],
  ] as const)('trains a %s contract with its own strategy and rate', (name, strategy, rate) => {
    const contract = golden();
    contract.strategy = strategy;
    local(contract).optimizer = { case: 'sgd', value: { ...sgd(contract), learningRate: rate } };
    const decision = decide(contract);
    expect(decision).toMatchObject({
      kind: 'train', projection: { strategy: name, learningRate: rate, proximalMu: 0 } });
  });

  // FedProx is first-order training plus the proximal term: the native round needs the contract's coefficient.
  it('trains a FedProx contract with its proximal coefficient', () => {
    const contract = golden();
    contract.strategy = Strategy.FEDPROX;
    local(contract).optimizer = { case: 'sgd', value: { ...sgd(contract), learningRate: 0.01 } };
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.fedproxMu = 0.1;
    }
    expect(decide(contract)).toMatchObject({
      kind: 'train', projection: { strategy: 'FedProx', learningRate: 0.01, proximalMu: 0.1 } });
  });

  it('refuses a FedProx contract that states no proximal coefficient', () => {
    const contract = golden();
    contract.strategy = Strategy.FEDPROX;
    expect(projectContract(contract, 'a'.repeat(64))).toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
  });
});

function sgd(contract: ExecutionContract) {
  const optimizer = local(contract).optimizer;
  if (optimizer.case !== 'sgd') {
    throw new Error('the golden contract uses SGD');
  }
  return optimizer.value;
}

/** The golden contract made a valid TinyNet DeComFL contract: zeroth-order training, scalar updates. */
function decomfl(): ExecutionContract {
  const contract = golden();
  contract.strategy = Strategy.DECOMFL;
  if (contract.workload.case !== 'modelTraining' || !contract.workload.value.localTraining) {
    throw new Error('the golden contract trains a model');
  }
  contract.workload.value.updateProtocol = UpdateProtocol.UPDATE_DECOMFL_SCALAR;
  const lt = contract.workload.value.localTraining;
  lt.localEpochs = 0;
  lt.optimizer = {
    case: 'zerothOrderSgd',
    value: create(ZerothOrderSgdSchema, {
      learningRate: 0.001, smoothing: 0.002, numLocalSteps: 1, numPerturbations: 10,
      estimator: GradientEstimator.ESTIMATOR_FORWARD, rng: PerturbationRng.RNG_TORCH_CPU_RANDN_F32,
    }),
  };
  return contract;
}

function zeroth(contract: ExecutionContract) {
  const optimizer = local(contract).optimizer;
  if (optimizer.case !== 'zerothOrderSgd') {
    throw new Error('a DeComFL contract trains zeroth-order');
  }
  return optimizer.value;
}

describe('the phone trains DeComFL from its contract', () => {
  it('projects the zeroth-order training the contract states', () => {
    expect(decide(decomfl())).toMatchObject({
      kind: 'train',
      projection: {
        strategy: 'DeComFL', learningRate: 0.001, numLocalSteps: 1, batchSize: 8,
        zerothOrder: { smoothing: 0.002, numPerturbations: 10 },
      },
    });
  });

  it('refuses the central estimator, which it does not implement', () => {
    const contract = decomfl();
    zeroth(contract).estimator = GradientEstimator.ESTIMATOR_CENTRAL;
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_OPTIMIZER' });
  });

  it('refuses secure aggregation it cannot perform', () => {
    const contract = decomfl();
    if (contract.security) {
      contract.security.secureAggregation = SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR;
      contract.security.secureAggThreshold = 2;
    }
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_SECURITY' });
  });

  it('refuses batching its whole-batch trainer cannot reproduce', () => {
    const contract = decomfl();
    local(contract).dropLast = true;
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_BATCHING' });
  });

  it('refuses a DeComFL contract whose update is not gradient scalars', () => {
    const contract = decomfl();
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.updateProtocol = UpdateProtocol.UPDATE_TRAINABLE_STATE_F32;
    }
    expect(projectContract(contract, 'a'.repeat(64)))
      .toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_UPDATE_PROTOCOL' });
  });
});

describe('the phone refuses what it cannot execute', () => {
  // A DeComFL or secure-aggregation run is refused as soon as it is published outside the approved v1 matrix, so
  // these drive the capability layer directly: it is what refuses them once the matrix widens to those runs.
  it('refuses a strategy it has no contract path for', () => {
    const contract = golden();
    contract.strategy = Strategy.UNSPECIFIED;
    expect(projectContract(contract, 'a'.repeat(64)))
      .toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_STRATEGY' });
  });

  it('refuses an update protocol it does not produce', () => {
    const contract = golden();
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.updateProtocol = UpdateProtocol.UPDATE_DECOMFL_SCALAR;
    }
    expect(projectContract(contract, 'a'.repeat(64)))
      .toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_UPDATE_PROTOCOL' });
  });

  it('refuses secure aggregation it cannot perform', () => {
    const contract = golden();
    if (contract.security) {
      contract.security.secureAggregation = SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR;
      contract.security.secureAggThreshold = 2;
    }
    expect(projectContract(contract, 'a'.repeat(64)))
      .toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_SECURITY' });
  });

  it.each([
    ['momentum', (c: ExecutionContract) => set(c, (sgd) => (sgd.momentum = 0.9))],
    ['weight decay', (c: ExecutionContract) => set(c, (sgd) => (sgd.weightDecay = 0.01))],
    ['dampening', (c: ExecutionContract) => set(c, (sgd) => (sgd.dampening = 0.1))],
    ['Nesterov', (c: ExecutionContract) => set(c, (sgd) => {
      sgd.momentum = 0.9;   // Nesterov without momentum is not a valid contract at all
      sgd.nesterov = true;
    })],
  ])('refuses an optimizer with %s, which its trainer does not implement', (_name, mutate) => {
    const contract = golden();
    mutate(contract);
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_OPTIMIZER' });
  });

  it('refuses gradient clipping it does not implement', () => {
    const contract = golden();
    local(contract).gradientClipNorm = 1.0;
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_OPTIMIZER' });
  });

  it('refuses optimizer state it would have to carry between rounds', () => {
    const contract = golden();
    local(contract).resetOptimizerEachRound = false;
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_OPTIMIZER' });
  });

  it('refuses batching its full-batch trainer cannot reproduce', () => {
    const dropping = golden();
    local(dropping).dropLast = true;
    expect(decide(dropping)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_BATCHING' });

    const capped = golden();
    local(capped).maxLocalSteps = 2;
    expect(decide(capped)).toMatchObject({ kind: 'refuse', code: 'UNSUPPORTED_BATCHING' });
  });

  it('refuses a contract whose portable CPU artifact is for another device', () => {
    const contract = golden();
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.artifacts[0]!.abi = 'x86_64';
    }
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'MISSING_CPU_ARTIFACT' });
  });

  it('refuses a contract with no portable CPU artifact at all', () => {
    const contract = golden();
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.artifacts[0]!.backend = ArtifactBackend.BACKEND_EXECUTORCH_GPU;
    }
    expect(projectContract(contract, 'a'.repeat(64)))
      .toMatchObject({ kind: 'refuse', code: 'MISSING_CPU_ARTIFACT' });
  });
});

function local(contract: ExecutionContract) {
  if (contract.workload.case !== 'modelTraining' || !contract.workload.value.localTraining) {
    throw new Error('the golden contract has local training');
  }
  return contract.workload.value.localTraining;
}

function set(contract: ExecutionContract, mutate: (sgd: { momentum?: number; weightDecay?: number;
  dampening?: number; nesterov?: boolean }) => void) {
  const optimizer = local(contract).optimizer;
  if (optimizer.case !== 'sgd') {
    throw new Error('the golden contract uses SGD');
  }
  mutate(optimizer.value);
}

describe('the staged bundle must be the one the contract binds', () => {
  const bundle = {
    lossSha256: '2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56',
    inferSha256: 'cf8744b9579d78f14bbb82e2d4ce98dcaffc8d2c6ed2253349c39342de546746',
    trainableSha256: 'ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412',
    paramLayout: [
      { name: 'fc1.weight', shape: [5, 4] },
      { name: 'fc1.bias', shape: [5] },
    ],
  };

  it('accepts the bundle whose programs and layout the contract names', () => {
    expect(checkBundleAgainstContract(golden(), bundle)).toEqual([]);
  });

  it('reports a program the contract did not bind', () => {
    expect(checkBundleAgainstContract(golden(), { ...bundle, trainableSha256: 'b'.repeat(64) }))
      .toEqual(['trainable.pte']);
  });

  it('reports a layout the contract did not state', () => {
    expect(checkBundleAgainstContract(golden(), {
      ...bundle, paramLayout: [{ name: 'fc1.bias', shape: [5] }, { name: 'fc1.weight', shape: [5, 4] }],
    })).toEqual(['trainable parameter layout']);
  });

  it('reports a missing trainable program', () => {
    expect(checkBundleAgainstContract(golden(), { ...bundle, trainableSha256: undefined }))
      .toEqual(['trainable.pte']);
  });
});

describe('a contract carrier without a workload', () => {
  it('is refused rather than treated as trainable', () => {
    const contract = create(ExecutionContractSchema, { contractVersion: 1 });
    expect(decide(contract)).toMatchObject({ kind: 'refuse', code: 'CONTRACT_INVALID' });
  });
});

describe('the programs the phone downloads come from the contract', () => {
  it('lists the portable CPU variant\'s files with their digests and sizes', () => {
    const contract = golden();
    const variant = contract.workload.case === 'modelTraining'
      ? contract.workload.value.artifacts.find(v => v.abi === 'arm64-v8a') : undefined;
    const programs = contractPrograms(contract);
    expect(programs.map(p => p.relativePath)).toEqual(variant!.files.map(f => f.relativePath));
    const first = variant!.files[0]!;
    expect(programs[0]).toEqual({
      relativePath: first.relativePath, sha256: first.sha256, byteSize: Number(first.byteSize),
    });
  });

  it('lists nothing when the contract has no portable CPU variant', () => {
    const contract = golden();
    if (contract.workload.case === 'modelTraining') {
      contract.workload.value.artifacts = [];
    }
    expect(contractPrograms(contract)).toEqual([]);
  });
});

