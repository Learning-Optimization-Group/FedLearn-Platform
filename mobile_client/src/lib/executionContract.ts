// Execution contract v1 reader: parsing and the validation rules shared with the Python and Java
// readers. The rules, limits and issue paths are specified in
// framework/tests/fixtures/execution_contract_v1/README.md and pinned by the conformance corpus
// beside it. A successful parse is never acceptance: validate before downloading a model or opening
// local data. uint64 fields arrive as bigint and are compared as bigint.
import { fromBinary, fromJsonString } from '@bufbuild/protobuf';
import type { DescEnum } from '@bufbuild/protobuf';
import {
  type ArtifactRef,
  type ArtifactVariant,
  type DataRequirement,
  type ExecutionContract,
  type LocalTraining,
  type ModelTraining,
  type RoundPolicy,
  type SecurityPolicy,
  type TensorSpec,
  ArmSchema,
  ArtifactBackend,
  ArtifactBackendSchema,
  BatchOrderSchema,
  ClientAuthSchema,
  ContractIssueCode,
  DTypeSchema,
  ExecutionContractSchema,
  ObjectiveSchema,
  PartitioningSchema,
  RecipeSchema,
  SecureAggregation,
  SecureAggregationSchema,
  Strategy,
  StrategySchema,
  Task,
  TaskSchema,
  TransportSchema,
  UpdateProtocolSchema,
  Arm,
  Objective,
  Recipe,
  UpdateProtocol,
} from '../gen/fedlearn/contract/v1/execution_contract_pb';

export const CONTRACT_VERSION = 1;

const MAX_ROUNDS = 10_000;
const MAX_CLIENTS_PER_ROUND = 10_000;
const MAX_TIMEOUT_MS = 86_400_000n;
const MAX_DECLARED_MS = 86_400_000n;
const MAX_TRANSIENT_RETRIES = 10;
const MAX_RETRY_BACKOFF_MS = 3_600_000n;
const MAX_TENSORS = 4096;
const MAX_RANK = 8;
const MAX_ELEMENTS = 2_147_483_647n;
const MAX_LOCAL_EPOCHS = 1000;
const MAX_LOCAL_STEPS = 1_000_000;
const MAX_BATCH_SIZE = 65_536;
const MAX_CLASSES = 1_000_000;
const MAX_TRANSFORMS = 16;
const MAX_VARIANTS = 16;
const MAX_FILES = 64;
const MAX_OPERATORS = 4096;
const MAX_FILE_BYTES = 1n << 36n;
const MAX_DECLARED_BYTES = 1n << 40n;
const MAX_TENSOR_NAME_LENGTH = 256;
const MAX_OPERATOR_LENGTH = 128;
const MAX_PATH_LENGTH = 255;
const MAX_PATH_SEGMENTS = 8;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const REVISION = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const VARIANT_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const ABI = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const TENSOR_NAME = /^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$/;
const SHA256 = /^[0-9a-f]{64}$/;
const PATH_SEGMENT = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;
const OPERATOR = /^[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$/;

/** The approved v1 matrix: recipe, strategy, arm, task, objective, update protocol. */
const APPROVED_MATRIX: ReadonlySet<string> = new Set([
  [
    Recipe.TINYNET_GOLDEN,
    Strategy.FEDAVG,
    Arm.FULL,
    Task.VECTOR_CLASSIFICATION,
    Objective.CROSS_ENTROPY,
    UpdateProtocol.UPDATE_TRAINABLE_STATE_F32,
  ].join(','),
]);

const CLASSIFICATION_TASKS: ReadonlySet<number> = new Set([
  Task.VECTOR_CLASSIFICATION,
  Task.IMAGE_CLASSIFICATION,
  Task.SEQUENCE_CLASSIFICATION,
]);
const TEXT_TASKS: ReadonlySet<number> = new Set([Task.SEQUENCE_CLASSIFICATION, Task.CAUSAL_LM]);

/** One reason a reader refuses a contract: an issue code and a ProtoJSON field path. */
export type ContractIssue = { code: ContractIssueCode; path: string };

export type ValidationOptions = {
  readerProtocolVersion: number;
  /** The run the contract was delivered for, when known. */
  expectedRunId?: string;
  /** The project the contract was delivered for, when known. */
  expectedProjectId?: string;
};

/** The input does not parse as an ExecutionContract. */
export class MalformedContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MalformedContractError';
  }
}

export function parseContractBinary(bytes: Uint8Array): ExecutionContract {
  try {
    return fromBinary(ExecutionContractSchema, bytes);
  } catch {
    throw new MalformedContractError('contract bytes do not parse');
  }
}

export function parseContractJson(json: string): ExecutionContract {
  let document: unknown;
  try {
    document = JSON.parse(json);
  } catch {
    throw new MalformedContractError('contract ProtoJSON is not JSON');
  }
  if (document === null || typeof document !== 'object' || Array.isArray(document)) {
    throw new MalformedContractError('contract ProtoJSON is not an object');
  }
  try {
    // Unknown fields are ignored; an unknown enum name then reads as 0, which validation refuses.
    return fromJsonString(ExecutionContractSchema, json, { ignoreUnknownFields: true });
  } catch {
    throw new MalformedContractError('contract ProtoJSON does not parse');
  }
}

/** Every v1 issue in `contract`; an empty list means the contract is accepted. */
export function validateContract(contract: ExecutionContract, options: ValidationOptions): ContractIssue[] {
  if (contract.contractVersion !== CONTRACT_VERSION) {
    return [{ code: ContractIssueCode.ISSUE_UNSUPPORTED_CONTRACT_VERSION, path: 'contractVersion' }];
  }
  return new Validator(contract, options).run();
}

function known(schema: DescEnum, value: number): boolean {
  return value !== 0 && schema.value[value] !== undefined;
}

function positiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function nonnegativeFinite(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

function openUnit(value: number): boolean {
  return Number.isFinite(value) && value > 0 && value < 1;
}

/** The element count of a valid shape, or undefined when the shape is not valid. */
function elementCount(shape: readonly bigint[]): bigint | undefined {
  if (shape.length < 1 || shape.length > MAX_RANK) {
    return undefined;
  }
  let count = 1n;
  for (const extent of shape) {
    if (extent < 1n || extent > MAX_ELEMENTS) {
      return undefined;
    }
    count *= extent;
    if (count > MAX_ELEMENTS) {
      return undefined;
    }
  }
  return count;
}

function validPath(path: string): boolean {
  if (path.length < 1 || path.length > MAX_PATH_LENGTH) {
    return false;
  }
  const segments = path.split('/');
  return segments.length <= MAX_PATH_SEGMENTS && segments.every(s => PATH_SEGMENT.test(s));
}

class Validator {
  private readonly issues: ContractIssue[] = [];

  constructor(
    private readonly c: ExecutionContract,
    private readonly options: ValidationOptions,
  ) {}

  private add(code: ContractIssueCode, path: string): void {
    this.issues.push({ code, path });
  }

  private check(ok: boolean, code: ContractIssueCode, path: string): void {
    if (!ok) {
      this.add(code, path);
    }
  }

  private bounded(value: number | bigint, low: number | bigint, high: number | bigint, path: string): void {
    this.check(value >= low && value <= high, ContractIssueCode.ISSUE_OUT_OF_RANGE, path);
  }

  private enumValue(schema: DescEnum, value: number, path: string): void {
    this.check(known(schema, value), ContractIssueCode.ISSUE_UNKNOWN_ENUM, path);
  }

  private present(has: boolean, path: string): boolean {
    this.check(has, ContractIssueCode.ISSUE_MISSING_FIELD, path);
    return has;
  }

  /** Checks a required explicit-presence value; true when present and valid. */
  private nonnegative(value: number | undefined, path: string): boolean {
    if (!this.present(value !== undefined, path)) {
      return false;
    }
    const ok = nonnegativeFinite(value as number);
    this.check(ok, ContractIssueCode.ISSUE_OUT_OF_RANGE, path);
    return ok;
  }

  run(): ContractIssue[] {
    const c = this.c;
    if (c.minClientProtocolVersion === 0 || c.minClientProtocolVersion > this.options.readerProtocolVersion) {
      this.add(ContractIssueCode.ISSUE_UNSUPPORTED_CLIENT_PROTOCOL, 'minClientProtocolVersion');
    }
    this.identity(c.runId, this.options.expectedRunId, 'runId');
    this.identity(c.projectId, this.options.expectedProjectId, 'projectId');
    this.enumValue(RecipeSchema, c.recipe, 'recipe');
    this.enumValue(StrategySchema, c.strategy, 'strategy');
    this.enumValue(PartitioningSchema, c.partitioning, 'partitioning');
    this.bounded(c.numRounds, 1, MAX_ROUNDS, 'numRounds');
    this.bounded(c.clientsPerRound, 1, MAX_CLIENTS_PER_ROUND, 'clientsPerRound');
    if (c.round !== undefined) {
      this.roundPolicy(c.round);
    } else {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, 'round');
    }
    if (c.security !== undefined) {
      this.security(c.security);
    } else {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, 'security');
    }
    if (c.workload.case === 'modelTraining') {
      this.modelTraining(c.workload.value);
      this.matrix(c.workload.value);
    } else {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, 'modelTraining');
    }
    return this.issues;
  }

  private identity(value: string, expected: string | undefined, path: string): void {
    if (!UUID.test(value)) {
      this.add(ContractIssueCode.ISSUE_INVALID_IDENTIFIER, path);
    } else if (expected !== undefined && value !== expected) {
      this.add(ContractIssueCode.ISSUE_IDENTITY_MISMATCH, path);
    }
  }

  private matrix(mt: ModelTraining): void {
    const fields: [DescEnum, number][] = [
      [RecipeSchema, this.c.recipe],
      [StrategySchema, this.c.strategy],
      [ArmSchema, mt.arm],
      [TaskSchema, mt.task],
      [ObjectiveSchema, mt.objective],
      [UpdateProtocolSchema, mt.updateProtocol],
    ];
    if (
      fields.every(([schema, value]) => known(schema, value)) &&
      !APPROVED_MATRIX.has(fields.map(([, value]) => value).join(','))
    ) {
      this.add(ContractIssueCode.ISSUE_UNSUPPORTED_COMBINATION, '');
    }
  }

  private roundPolicy(r: RoundPolicy): void {
    this.bounded(r.timeoutMs, 1n, MAX_TIMEOUT_MS, 'round.timeoutMs');
    this.check(r.oneAcceptedUpdatePerRound, ContractIssueCode.ISSUE_OUT_OF_RANGE, 'round.oneAcceptedUpdatePerRound');
    if (r.maxTransientRetries === undefined) {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, 'round.maxTransientRetries');
    } else {
      this.bounded(r.maxTransientRetries, 0, MAX_TRANSIENT_RETRIES, 'round.maxTransientRetries');
    }
    this.bounded(r.retryBackoffMs, 1n, MAX_RETRY_BACKOFF_MS, 'round.retryBackoffMs');
  }

  private security(s: SecurityPolicy): void {
    this.enumValue(TransportSchema, s.transport, 'security.transport');
    this.enumValue(ClientAuthSchema, s.clientAuth, 'security.clientAuth');
    this.enumValue(SecureAggregationSchema, s.secureAggregation, 'security.secureAggregation');
    const strategy = this.c.strategy;
    if (s.secureAggregation === SecureAggregation.SECAGG_LIGHTSECAGG_SCALAR) {
      if (s.secureAggThreshold === undefined) {
        this.add(ContractIssueCode.ISSUE_MISSING_FIELD, 'security.secureAggThreshold');
      } else {
        this.bounded(s.secureAggThreshold, 2, this.c.clientsPerRound, 'security.secureAggThreshold');
      }
      if (known(StrategySchema, strategy) && strategy !== Strategy.DECOMFL) {
        this.add(ContractIssueCode.ISSUE_INVALID_SECURITY, 'security.secureAggregation');
      }
    } else if (s.secureAggregation === SecureAggregation.SECAGG_NONE && s.secureAggThreshold !== undefined) {
      this.add(ContractIssueCode.ISSUE_INVALID_SECURITY, 'security.secureAggThreshold');
    }
    if (s.centralDp !== undefined) {
      const dp = s.centralDp;
      this.check(positiveFinite(dp.targetEpsilon), ContractIssueCode.ISSUE_OUT_OF_RANGE, 'security.centralDp.targetEpsilon');
      this.check(openUnit(dp.delta), ContractIssueCode.ISSUE_OUT_OF_RANGE, 'security.centralDp.delta');
      this.check(positiveFinite(dp.clipNorm), ContractIssueCode.ISSUE_OUT_OF_RANGE, 'security.centralDp.clipNorm');
    }
  }

  private modelTraining(mt: ModelTraining): void {
    const p = 'modelTraining';
    this.check(MODEL_ID.test(mt.modelId), ContractIssueCode.ISSUE_INVALID_IDENTIFIER, `${p}.modelId`);
    this.check(REVISION.test(mt.modelRevision), ContractIssueCode.ISSUE_INVALID_IDENTIFIER, `${p}.modelRevision`);
    this.enumValue(ArmSchema, mt.arm, `${p}.arm`);
    this.enumValue(TaskSchema, mt.task, `${p}.task`);
    this.enumValue(ObjectiveSchema, mt.objective, `${p}.objective`);
    this.enumValue(UpdateProtocolSchema, mt.updateProtocol, `${p}.updateProtocol`);
    this.trainable(mt.trainable);
    this.check(SHA256.test(mt.frozenStateSha256), ContractIssueCode.ISSUE_INVALID_HASH, `${p}.frozenStateSha256`);
    this.check(SHA256.test(mt.initialStateSha256), ContractIssueCode.ISSUE_INVALID_HASH, `${p}.initialStateSha256`);
    if (mt.localTraining !== undefined) {
      this.localTraining(mt.localTraining);
    } else {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${p}.localTraining`);
    }
    if (mt.data !== undefined) {
      this.data(mt.data, mt.task);
    } else {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${p}.data`);
    }
    this.artifacts(mt.artifacts);
    const strategy = this.c.strategy;
    if (strategy === Strategy.FEDPROX) {
      if (mt.fedproxMu === undefined) {
        this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${p}.fedproxMu`);
      } else if (!nonnegativeFinite(mt.fedproxMu)) {
        this.add(ContractIssueCode.ISSUE_OUT_OF_RANGE, `${p}.fedproxMu`);
      }
    } else if (known(StrategySchema, strategy) && mt.fedproxMu !== undefined) {
      this.add(ContractIssueCode.ISSUE_INVALID_STRATEGY_SETTINGS, `${p}.fedproxMu`);
    }
  }

  private trainable(tensors: readonly TensorSpec[]): void {
    const p = 'modelTraining.trainable';
    if (tensors.length < 1 || tensors.length > MAX_TENSORS) {
      this.add(ContractIssueCode.ISSUE_MALFORMED_LAYOUT, p);
      return;
    }
    const seen = new Set<string>();
    let total = 0n;
    tensors.forEach((tensor, i) => {
      const at = `${p}[${i}]`;
      const nameOk = tensor.name.length <= MAX_TENSOR_NAME_LENGTH && TENSOR_NAME.test(tensor.name);
      this.check(nameOk && !seen.has(tensor.name), ContractIssueCode.ISSUE_MALFORMED_LAYOUT, `${at}.name`);
      seen.add(tensor.name);
      const count = elementCount(tensor.shape);
      if (count === undefined) {
        this.add(ContractIssueCode.ISSUE_MALFORMED_LAYOUT, `${at}.shape`);
      } else {
        total += count;
      }
      this.enumValue(DTypeSchema, tensor.dtype, `${at}.dtype`);
    });
    if (total > MAX_ELEMENTS) {
      this.add(ContractIssueCode.ISSUE_MALFORMED_LAYOUT, p);
    }
  }

  private localTraining(lt: LocalTraining): void {
    const p = 'modelTraining.localTraining';
    this.bounded(lt.localEpochs, 1, MAX_LOCAL_EPOCHS, `${p}.localEpochs`);
    if (lt.maxLocalSteps !== undefined) {
      this.bounded(lt.maxLocalSteps, 1, MAX_LOCAL_STEPS, `${p}.maxLocalSteps`);
    }
    if (lt.gradientClipNorm !== undefined) {
      this.check(positiveFinite(lt.gradientClipNorm), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${p}.gradientClipNorm`);
    }
    const optimizer = lt.optimizer;
    switch (optimizer.case) {
      case 'sgd': {
        const sgd = optimizer.value;
        const at = `${p}.sgd`;
        this.check(positiveFinite(sgd.learningRate), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.learningRate`);
        const momentumOk = this.nonnegative(sgd.momentum, `${at}.momentum`);
        const dampeningOk = this.nonnegative(sgd.dampening, `${at}.dampening`);
        this.nonnegative(sgd.weightDecay, `${at}.weightDecay`);
        if (
          this.present(sgd.nesterov !== undefined, `${at}.nesterov`) &&
          sgd.nesterov === true &&
          momentumOk &&
          dampeningOk &&
          !((sgd.momentum as number) > 0 && sgd.dampening === 0)
        ) {
          this.add(ContractIssueCode.ISSUE_INVALID_OPTIMIZER, `${at}.nesterov`);
        }
        break;
      }
      case 'adam':
      case 'adamw': {
        const adam = optimizer.value;
        const at = `${p}.${optimizer.case}`;
        this.check(positiveFinite(adam.learningRate), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.learningRate`);
        this.check(openUnit(adam.beta1), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.beta1`);
        this.check(openUnit(adam.beta2), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.beta2`);
        this.check(positiveFinite(adam.epsilon), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.epsilon`);
        this.nonnegative(adam.weightDecay, `${at}.weightDecay`);
        this.present(adam.amsgrad !== undefined, `${at}.amsgrad`);
        break;
      }
      case 'rmsprop': {
        const rms = optimizer.value;
        const at = `${p}.rmsprop`;
        this.check(positiveFinite(rms.learningRate), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.learningRate`);
        this.check(openUnit(rms.alpha), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.alpha`);
        this.check(positiveFinite(rms.epsilon), ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.epsilon`);
        this.nonnegative(rms.weightDecay, `${at}.weightDecay`);
        this.nonnegative(rms.momentum, `${at}.momentum`);
        this.present(rms.centered !== undefined, `${at}.centered`);
        break;
      }
      default:
        this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${p}.optimizer`);
    }
    this.present(lt.resetOptimizerEachRound !== undefined, `${p}.resetOptimizerEachRound`);
    this.bounded(lt.batchSize, 1, MAX_BATCH_SIZE, `${p}.batchSize`);
    this.present(lt.dropLast !== undefined, `${p}.dropLast`);
    this.enumValue(BatchOrderSchema, lt.batchOrder, `${p}.batchOrder`);
  }

  private data(data: DataRequirement, trainingTask: number): void {
    const p = 'modelTraining.data';
    const task = data.task;
    const taskKnown = known(TaskSchema, task);
    if (!taskKnown) {
      this.add(ContractIssueCode.ISSUE_UNKNOWN_ENUM, `${p}.task`);
    } else if (known(TaskSchema, trainingTask) && trainingTask !== task) {
      this.add(ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, `${p}.task`);
    }
    const shapeOk = elementCount(data.inputShape) !== undefined;
    this.check(shapeOk, ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, `${p}.inputShape`);
    this.enumValue(DTypeSchema, data.inputDtype, `${p}.inputDtype`);
    if (CLASSIFICATION_TASKS.has(task)) {
      this.bounded(data.classCount, 2, MAX_CLASSES, `${p}.classCount`);
    } else if (task === Task.CAUSAL_LM && data.classCount !== 0) {
      this.add(ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, `${p}.classCount`);
    }
    this.check(REVISION.test(data.labelSchemaId), ContractIssueCode.ISSUE_INVALID_IDENTIFIER, `${p}.labelSchemaId`);
    if (data.transforms.length < 1 || data.transforms.length > MAX_TRANSFORMS) {
      this.add(ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, `${p}.transforms`);
    } else {
      data.transforms.forEach((transform, i) => {
        const at = `${p}.transforms[${i}]`;
        if (transform.operation.case !== 'identityVector') {
          this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${at}.operation`);
          return;
        }
        const width = transform.operation.value.width;
        const widthPath = `${at}.identityVector.width`;
        if (width < 1 || BigInt(width) > MAX_ELEMENTS) {
          this.add(ContractIssueCode.ISSUE_OUT_OF_RANGE, widthPath);
        } else if (
          taskKnown &&
          (task !== Task.VECTOR_CLASSIFICATION ||
            (shapeOk && !(data.inputShape.length === 1 && data.inputShape[0] === BigInt(width))))
        ) {
          this.add(ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, widthPath);
        }
      });
    }
    const hasTokenizer = data.tokenizer !== undefined;
    if (TEXT_TASKS.has(task) && !hasTokenizer) {
      this.add(ContractIssueCode.ISSUE_MISSING_FIELD, `${p}.tokenizer`);
    } else if (taskKnown && !TEXT_TASKS.has(task) && hasTokenizer) {
      this.add(ContractIssueCode.ISSUE_INVALID_DATA_REQUIREMENT, `${p}.tokenizer`);
    }
    if (data.tokenizer !== undefined) {
      this.artifactRef(data.tokenizer, `${p}.tokenizer`);
    }
  }

  /** Checks one reference; true when its byte size is in range. */
  private artifactRef(ref: ArtifactRef, p: string): boolean {
    this.check(validPath(ref.relativePath), ContractIssueCode.ISSUE_INVALID_PATH, `${p}.relativePath`);
    this.check(SHA256.test(ref.sha256), ContractIssueCode.ISSUE_INVALID_HASH, `${p}.sha256`);
    const sizeOk = ref.byteSize >= 1n && ref.byteSize <= MAX_FILE_BYTES;
    this.check(sizeOk, ContractIssueCode.ISSUE_OUT_OF_RANGE, `${p}.byteSize`);
    return sizeOk;
  }

  private artifacts(variants: readonly ArtifactVariant[]): void {
    const p = 'modelTraining.artifacts';
    if (variants.length === 0) {
      this.add(ContractIssueCode.ISSUE_MISSING_ARTIFACT, p);
      return;
    }
    if (variants.length > MAX_VARIANTS) {
      this.add(ContractIssueCode.ISSUE_INVALID_ARTIFACT, p);
      return;
    }
    const seenIds = new Set<string>();
    variants.forEach((variant, i) => {
      const at = `${p}[${i}]`;
      if (!VARIANT_ID.test(variant.variantId)) {
        this.add(ContractIssueCode.ISSUE_INVALID_IDENTIFIER, `${at}.variantId`);
      } else if (seenIds.has(variant.variantId)) {
        this.add(ContractIssueCode.ISSUE_INVALID_ARTIFACT, `${at}.variantId`);
      }
      seenIds.add(variant.variantId);
      this.enumValue(ArtifactBackendSchema, variant.backend, `${at}.backend`);
      this.check(ABI.test(variant.abi), ContractIssueCode.ISSUE_INVALID_IDENTIFIER, `${at}.abi`);
      const sizesOk = this.files(variant.files, `${at}.files`);
      this.operators(variant.requiredOperators, `${at}.requiredOperators`);
      this.bounded(variant.declaredPeakMemoryBytes, 1n, MAX_DECLARED_BYTES, `${at}.declaredPeakMemoryBytes`);
      const storage = variant.declaredStorageBytes;
      const storageOk = storage >= 1n && storage <= MAX_DECLARED_BYTES;
      this.check(storageOk, ContractIssueCode.ISSUE_OUT_OF_RANGE, `${at}.declaredStorageBytes`);
      this.bounded(variant.declaredProbeMs, 1n, MAX_DECLARED_MS, `${at}.declaredProbeMs`);
      this.bounded(variant.declaredTrainMs, 1n, MAX_DECLARED_MS, `${at}.declaredTrainMs`);
      if (storageOk && sizesOk) {
        const total = variant.files.reduce((sum, ref) => sum + ref.byteSize, 0n);
        this.check(total <= storage, ContractIssueCode.ISSUE_INVALID_ARTIFACT, `${at}.declaredStorageBytes`);
      }
    });
    if (!variants.some(v => v.backend === ArtifactBackend.BACKEND_EXECUTORCH_CPU)) {
      this.add(ContractIssueCode.ISSUE_MISSING_ARTIFACT, p);
    }
  }

  /** Checks a variant's files; true when there are 1..MAX_FILES, all with sizes in range. */
  private files(files: readonly ArtifactRef[], p: string): boolean {
    if (files.length === 0) {
      this.add(ContractIssueCode.ISSUE_MISSING_ARTIFACT, p);
      return false;
    }
    if (files.length > MAX_FILES) {
      this.add(ContractIssueCode.ISSUE_INVALID_ARTIFACT, p);
      return false;
    }
    const seenPaths = new Set<string>();
    let allSizesOk = true;
    files.forEach((ref, j) => {
      const at = `${p}[${j}]`;
      allSizesOk = this.artifactRef(ref, at) && allSizesOk;
      if (validPath(ref.relativePath)) {
        const folded = ref.relativePath.toLowerCase();
        this.check(!seenPaths.has(folded), ContractIssueCode.ISSUE_INVALID_PATH, `${at}.relativePath`);
        seenPaths.add(folded);
      }
    });
    return allSizesOk;
  }

  private operators(operators: readonly string[], p: string): void {
    if (operators.length < 1 || operators.length > MAX_OPERATORS) {
      this.add(ContractIssueCode.ISSUE_INVALID_ARTIFACT, p);
      return;
    }
    const seen = new Set<string>();
    operators.forEach((operator, k) => {
      const valid = operator.length <= MAX_OPERATOR_LENGTH && OPERATOR.test(operator);
      this.check(valid && !seen.has(operator), ContractIssueCode.ISSUE_INVALID_ARTIFACT, `${p}[${k}]`);
      seen.add(operator);
    });
  }
}
