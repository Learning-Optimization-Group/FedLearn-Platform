import type { DataRequirement } from '../gen/fedlearn/contract/v1/execution_contract_pb';
import {
  pickAndImportDataset, requirementShape, snapshotMismatches, type DataShape, type DatasetSnapshot,
} from './datasetService';
import { readError } from './errors';
import { api } from './restClient';
import type { JoinedRun } from './runJoin';
import { ExecutionContractRefusedError, resolveContract, type ContractWaitOps } from './training';

// What a joined run needs from this device's data, read from its execution contract before Start (Stage 3 slice B2).
// A run on the built-in data needs nothing; a run on the device's own data needs a dataset snapshot that matches the
// contract's data requirement, imported against the run's classes.

export type RunDataPlan =
  | { source: 'FIXTURE' }
  | {
      source: 'LOCAL_SNAPSHOT';
      requirement: DataRequirement;
      /** The run's classes in label order, which an imported file's labels are checked against. */
      classNames: string[];
      /** What one example must be: a vector of some width, or an image and how the contract prepares it. */
      shape: DataShape;
      /** The run's batch size. */
      batchSize: number;
      /**
       * The most examples a dataset may have: one batch when the contract trains the whole dataset as one step per
       * epoch, or null when it trains in seeded minibatches (BATCH_ORDER_SEEDED_PERMUTATION_V1).
       */
      maxExamples: number | null;
    };

/**
 * The run's data plan, or null when this device will not train the run (no contract it accepts, or one still being
 * published after the wait) or the run does not say which classes to import against. Start then reports why.
 */
export async function loadRunDataPlan(joined: JoinedRun, ops?: ContractWaitOps): Promise<RunDataPlan | null> {
  let resolved;
  try {
    resolved = await resolveContract(joined, ops);
  } catch (e) {
    if (e instanceof ExecutionContractRefusedError) return null;
    throw e;
  }
  const { contract, projection } = resolved;
  if (projection.dataSource === 'FIXTURE') return { source: 'FIXTURE' };
  const requirement = contract.workload.case === 'modelTraining' ? contract.workload.value.data : undefined;
  const shape = requirement ? requirementShape(requirement) : null;
  if (!requirement || !shape) return null;
  const res = await api.get<{ classNames?: string[] | null }>(`/api/runs/${joined.runId}/model-bundle`);
  const classNames = res.data?.classNames ?? [];
  if (classNames.length !== requirement.classCount) return null;
  return {
    source: 'LOCAL_SNAPSHOT',
    requirement,
    classNames,
    shape,
    batchSize: projection.batchSize,
    maxExamples: projection.minibatch ? null : projection.batchSize,
  };
}

const MISMATCH_LABELS: Record<string, string> = {
  labelSchemaId: 'different classes',
  inputShape: 'different input size',
  inputDtype: 'different input type',
  classCount: 'different class count',
  recordCount: 'no examples',
  transforms: 'different image preparation',
};

/** Why `snapshot` cannot train this run, in words for the user; empty when it can. */
export function datasetFit(snapshot: DatasetSnapshot, plan: RunDataPlan): string[] {
  if (plan.source !== 'LOCAL_SNAPSHOT') return [];
  const reasons = snapshotMismatches(snapshot, plan.requirement).map((m) => MISMATCH_LABELS[m] ?? m);
  if (plan.maxExamples !== null && snapshot.recordCount > plan.maxExamples) {
    reasons.push(`more than ${plan.maxExamples} examples`);
  }
  return reasons;
}

/**
 * Let the user pick a file and import it against the run's classes and example shape. Choosing no file is not an
 * error; a refused import carries the importer's reason.
 */
export async function importForRun(
  plan: Extract<RunDataPlan, { source: 'LOCAL_SNAPSHOT' }>,
): Promise<{ snapshot?: DatasetSnapshot; error?: string }> {
  try {
    return { snapshot: await pickAndImportDataset(plan.classNames, plan.shape) };
  } catch (e) {
    if ((e as { code?: string } | null)?.code === 'DATASET_PICK_CANCELLED') return {};
    return { error: readError(e) };
  }
}
