/**
 * Execution contract v1 on the desktop launch paths. When the run published a READY contract, the launcher
 * writes it to a file and hands it to fl-runtime/client.py (--execution-contract, --run-id), which refuses to train
 * unless it would execute that contract exactly. Other states keep the legacy launch: during the compatibility
 * window a laptop client may train on the legacy fields when no contract is ready.
 */
import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  CONTAINER_EXECUTION_CONTRACT_PATH,
  resolveExecutionContract,
} from '../main/executionContract';
import { buildContainerBinds, buildContainerEnv, type TrainingConfig } from '../main/docker.service';

const RUN_ID = '4f2c8a1e-7b3d-4c59-9e21-6a0d5b8f3c17';
const CONTRACT = { contractVersion: 1, runId: RUN_ID, projectId: '9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64' };

let dir: string;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'contract-'));
});
afterEach(() => fs.rmSync(dir, { recursive: true, force: true }));

describe('resolveExecutionContract', () => {
  it('writes a READY contract for the run and returns where it is', () => {
    const launch = resolveExecutionContract(
      { runId: RUN_ID, contractState: 'READY', executionContract: CONTRACT }, dir);
    expect(launch.runId).toBe(RUN_ID);
    expect(launch.executionContractPath).toBe(path.join(dir, `${RUN_ID}.json`));
    expect(JSON.parse(fs.readFileSync(launch.executionContractPath as string, 'utf8'))).toEqual(CONTRACT);
  });

  it.each(['PENDING', 'UNAVAILABLE', 'LEGACY_ONLY', undefined])(
    'keeps the legacy launch when the contract state is %s', (state) => {
      expect(resolveExecutionContract({ runId: RUN_ID, contractState: state }, dir)).toEqual({});
      expect(fs.readdirSync(dir)).toEqual([]);
    });

  it('refuses a READY state without a contract', () => {
    expect(() => resolveExecutionContract({ runId: RUN_ID, contractState: 'READY' }, dir)).toThrow(/contract/);
  });

  it('refuses a contract for another run', () => {
    expect(() => resolveExecutionContract({
      runId: RUN_ID, contractState: 'READY',
      executionContract: { ...CONTRACT, runId: '00000000-0000-4000-8000-000000000001' },
    }, dir)).toThrow(/run/);
  });

  it('refuses a run ID that is not a canonical UUID', () => {
    expect(() => resolveExecutionContract(
      { runId: '../escape', contractState: 'READY', executionContract: CONTRACT }, dir)).toThrow(/run ID/);
  });

  it('refuses an unknown contract state', () => {
    expect(() => resolveExecutionContract({ runId: RUN_ID, contractState: 'DONE' }, dir)).toThrow(/state/);
  });

  it('refuses a contract that is not a JSON object', () => {
    expect(() => resolveExecutionContract(
      { runId: RUN_ID, contractState: 'READY', executionContract: [CONTRACT] }, dir)).toThrow(/contract/);
  });

  it('refuses an oversized contract', () => {
    const huge = { ...CONTRACT, padding: 'x'.repeat(300 * 1024) };
    expect(() => resolveExecutionContract(
      { runId: RUN_ID, contractState: 'READY', executionContract: huge }, dir)).toThrow(/large/);
  });
});

const BASE: TrainingConfig = {
  projectId: 'p-1',
  serverAddress: 'host:50000',
  partitionId: '0',
  modelType: 'TINYNET_GOLDEN',
  datasetPath: '/data/in',
} as TrainingConfig;

describe('execution contract — Docker path', () => {
  it('mounts the contract read-only and names it for the entrypoint', () => {
    const config = { ...BASE, executionContractPath: '/host/contract.json', runId: RUN_ID } as TrainingConfig;
    expect(buildContainerEnv(config)).toEqual(expect.arrayContaining([
      `EXECUTION_CONTRACT=${CONTAINER_EXECUTION_CONTRACT_PATH}`, `RUN_ID=${RUN_ID}`,
    ]));
    expect(buildContainerBinds(config)).toContain(`/host/contract.json:${CONTAINER_EXECUTION_CONTRACT_PATH}:ro`);
  });

  it('adds nothing on a legacy launch', () => {
    expect(buildContainerEnv(BASE).some((e) => e.startsWith('EXECUTION_CONTRACT=') || e.startsWith('RUN_ID=')))
      .toBe(false);
    expect(buildContainerBinds(BASE)).toEqual(['/data/in:/data']);
  });
});

describe('client-docker entrypoint', () => {
  const entrypoint = path.resolve(__dirname, '../../../client-docker/entrypoint.sh');

  function launch(env: Record<string, string>): string {
    const bin = fs.mkdtempSync(path.join(os.tmpdir(), 'bin-'));
    fs.writeFileSync(path.join(bin, 'python3'), '#!/bin/bash\nprintf "%s\\n" "$@"\n', { mode: 0o755 });
    try {
      return execFileSync('bash', [entrypoint], {
        env: { PATH: `${bin}:${process.env.PATH}`, PROJECT_ID: 'p-1', SERVER_ADDRESS: 'h:1', PARTITION_ID: '0',
          ...env },
        encoding: 'utf8',
      });
    } finally {
      fs.rmSync(bin, { recursive: true, force: true });
    }
  }

  it('forwards the contract and run to the client', () => {
    const argv = launch({ EXECUTION_CONTRACT: CONTAINER_EXECUTION_CONTRACT_PATH, RUN_ID: RUN_ID }).split('\n');
    expect(argv[argv.indexOf('--execution-contract') + 1]).toBe(CONTAINER_EXECUTION_CONTRACT_PATH);
    expect(argv[argv.indexOf('--run-id') + 1]).toBe(RUN_ID);
  });

  it('passes neither on a legacy launch', () => {
    const argv = launch({}).split('\n');
    expect(argv).not.toContain('--execution-contract');
    expect(argv).not.toContain('--run-id');
  });
});
