import * as fs from 'fs';
import * as path from 'path';

// Execution contract v1 on the desktop launch paths. The backend's connection payload carries the active run's
// contract state and, when READY, the contract as ProtoJSON. The launcher writes a READY contract to a file and hands
// it to fl-runtime/client.py (--execution-contract, --run-id), which refuses to train unless it would execute exactly
// what the contract states. Any other state keeps the legacy launch: during the compatibility window a laptop client
// may train on the legacy fields when no contract is ready.

/** Where the contract is mounted inside the client container. */
export const CONTAINER_EXECUTION_CONTRACT_PATH = '/run/fedlearn/execution-contract.json';

const CONTRACT_STATES = new Set(['PENDING', 'READY', 'UNAVAILABLE', 'LEGACY_ONLY']);
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const MAX_CONTRACT_BYTES = 256 * 1024;

export interface ExecutionContractLaunch {
  executionContractPath?: string;
  runId?: string;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Resolves the connection payload's execution contract for a launch. A READY contract is written under `dir`, named
 * by its run, and its path returned; any other state returns nothing. A malformed payload refuses the start with a
 * clear message rather than launching a client that would check against the wrong contract, or none.
 */
export function resolveExecutionContract(
  payload: { runId?: unknown; contractState?: unknown; executionContract?: unknown },
  dir: string,
): ExecutionContractLaunch {
  const { runId, contractState, executionContract } = payload;
  if (contractState === undefined || contractState === null) {
    return {};
  }
  if (typeof contractState !== 'string' || !CONTRACT_STATES.has(contractState)) {
    throw new Error(`Unrecognised execution contract state "${String(contractState)}" in the connection payload.`);
  }
  if (contractState !== 'READY') {
    return {};
  }
  if (typeof runId !== 'string' || !UUID.test(runId)) {
    throw new Error('The connection payload has a READY execution contract without a valid run ID.');
  }
  if (!isPlainObject(executionContract)) {
    throw new Error('The connection payload says the execution contract is READY but carries no contract.');
  }
  if (executionContract.runId !== runId) {
    throw new Error('The execution contract in the connection payload belongs to another run; refusing to start.');
  }
  const text = JSON.stringify(executionContract);
  if (Buffer.byteLength(text, 'utf8') > MAX_CONTRACT_BYTES) {
    throw new Error('The execution contract in the connection payload is too large; refusing to start.');
  }
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${runId}.json`);
  fs.writeFileSync(file, text, { mode: 0o644 });
  return { executionContractPath: file, runId };
}
