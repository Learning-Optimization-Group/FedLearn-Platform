import { runTrainingLoop, ExecutionContractRefusedError, MobileSecureAggregationUnsupportedError } from '../lib/training';
import type { JoinedRun, RunManifest } from '../lib/runJoin';
import { provisionTrainingBundle } from '../lib/modelProvisioning';
import { create } from '@bufbuild/protobuf';
import * as nativeCoreModule from '../lib/nativeCore';
import {
  GradientEstimator,
  PerturbationRng,
  SecureAggregation,
  Strategy,
  UpdateProtocol,
  ZerothOrderSgdSchema,
  type ExecutionContract,
} from '../gen/fedlearn/contract/v1/execution_contract_pb';
import { projectContract } from '../lib/executionContractGate';
import { parseContractBinary } from '../lib/executionContract';

declare const __dirname: string;
type FixtureFs = { readFileSync(path: string, encoding?: 'utf8'): string & Uint8Array };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs: FixtureFs = require('fs');
const GOLDEN = `${__dirname}/../../../framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb`;

function golden(): ExecutionContract {
  return parseContractBinary(new Uint8Array(fs.readFileSync(GOLDEN)));
}

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
      learningRate: 0.001,
      smoothing: 0.002,
      numLocalSteps: 1,
      numPerturbations: 10,
      estimator: GradientEstimator.ESTIMATOR_FORWARD,
      rng: PerturbationRng.RNG_TORCH_CPU_RANDN_F32,
    }),
  };
  return contract;
}

jest.mock('../lib/modelProvisioning', () => ({
  __esModule: true,
  provisionTrainingBundle: jest.fn(),
}));

jest.mock('../lib/nativeCore', () => {
  const actual = jest.requireActual('../lib/nativeCore');
  return {
    __esModule: true,
    ...actual,
    isNativeCoreAvailable: jest.fn(),
    default: {
      loadModel: jest.fn(),
      setModelManifest: jest.fn(),
      setTrainingDataFromFiles: jest.fn(),
      getServerStatus: jest.fn(),
      runDeComFLRound: jest.fn(),
      runFedAvgRound: jest.fn(),
      getRuntimeCompatibility: jest.fn().mockResolvedValue({ bridgeAbiVersion: 5, protocolVersion: 2 }),
    },
  };
});

function joinedRun(overrides: Partial<RunManifest>): JoinedRun {
  return {
    runId: 'run-secagg-1',
    projectId: 'proj-secagg-1',
    partitionId: 0,
    assignedRound: 0,
    grpcEndpoint: 'localhost:50000',
    message: '',
    manifest: {
      runId: 'run-secagg-1',
      projectId: 'proj-secagg-1',
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

beforeEach(() => {
  jest.clearAllMocks();
});

describe('Mobile LightSecAgg — capability gating and execution path', () => {
  test('supported DeComFL configuration does not trip refusal when native core is available', async () => {
    jest.mocked(nativeCoreModule.isNativeCoreAvailable).mockReturnValue(true);

    // With native core available, DeComFL with secureAggregation=true passes the legacy refusal guard
    // and proceeds to contract resolution (which throws ExecutionContractRefusedError because no contract is attached)
    await expect(runTrainingLoop(joinedRun({ secureAggregation: true }), hooks))
      .rejects.toBeInstanceOf(ExecutionContractRefusedError);

    // It did NOT reject with MobileSecureAggregationUnsupportedError!
    expect(provisionTrainingBundle).not.toHaveBeenCalled();
  });

  test('unsupported dense weight first-order configuration still refuses explicitly', async () => {
    jest.mocked(nativeCoreModule.isNativeCoreAvailable).mockReturnValue(true);

    // First-order weight updates cannot use scalar LightSecAgg, so it must refuse
    await expect(
      runTrainingLoop(joinedRun({ secureAggregation: true, firstOrderSupported: true }), hooks),
    ).rejects.toBeInstanceOf(MobileSecureAggregationUnsupportedError);
  });

  test('when native core is unavailable, even DeComFL secure aggregation refuses safely', async () => {
    jest.mocked(nativeCoreModule.isNativeCoreAvailable).mockReturnValue(false);

    await expect(
      runTrainingLoop(joinedRun({ secureAggregation: true }), hooks),
    ).rejects.toBeInstanceOf(MobileSecureAggregationUnsupportedError);
  });

  test('execution contract gate accepts SECAGG_LIGHTSECAGG_SCALAR for DeComFL when native core is present', () => {
    jest.mocked(nativeCoreModule.isNativeCoreAvailable).mockReturnValue(true);

    const contract = decomfl();
    if (contract.security) {
      contract.security.secureAggregation = SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR;
      contract.security.secureAggThreshold = 2;
    }

    const decision = projectContract(contract, 'valid-hash-1234');
    expect(decision.kind).toBe('train');
  });

  test('execution contract gate refuses SECAGG_LIGHTSECAGG_SCALAR for first-order full-weight strategies', () => {
    jest.mocked(nativeCoreModule.isNativeCoreAvailable).mockReturnValue(true);

    const contract = golden();
    if (contract.security) {
      contract.security.secureAggregation = SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR;
      contract.security.secureAggThreshold = 2;
    }

    const decision = projectContract(contract, 'valid-hash-5678');
    expect(decision.kind).toBe('refuse');
    if (decision.kind === 'refuse') {
      expect(decision.code).toBe('UNSUPPORTED_SECURITY');
    }
  });
});
