import { api } from './restClient';
import nativeCore from './nativeCore';
import { getOrCreateClientId } from './clientId';

/** The run manifest the backend serves inline in the enroll response. (Determinism/model-hash
 *  fields arrive with the model-delivery slice — not present today.) */
export interface RunManifest {
  runId: string;
  projectId: string;
  recipeKey: string;
  strategy: string;
  numRounds: number;
  clientsPerRound: number;
  partitioningMode: string;
  seed: number;
  torchVersion: string;
}

export interface JoinParams {
  projectId: string;
  /** Phase-1 gRPC is plaintext → defaults false. TLS + CA-pin land in the transport-security phase. */
  useTls?: boolean;
}

export interface JoinedRun {
  runId: string;
  projectId: string;
  partitionId: number;
  assignedRound: number;
  grpcEndpoint: string;
  manifest: RunManifest;
  message: string;
}

const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 120_000;

interface ActiveRunResponse { activeRun: { runId: string; status: string } | null }
interface RunStatusResponse { status: string; grpcEndpoint: string | null; caFingerprint: string | null }
interface EnrollResponse {
  runId: string; projectId: string; grpcEndpoint: string; partitionId: number;
  clientKind: string; caFingerprint: string | null; connectionToken: string;
  expiresAt: string; manifest: RunManifest;
}

/** The project's active run id, or throw if the owner hasn't started one. */
async function resolveRunId(projectId: string): Promise<string> {
  const res = await api.get<ActiveRunResponse>(`/api/client/projects/${projectId}`);
  const runId = res.data?.activeRun?.runId;
  if (!runId) throw new Error('No active run for this project yet — the owner needs to start one.');
  return runId;
}

/** Poll the run status until RUNNING with a grpc endpoint, or time out. */
async function pollUntilRunning(runId: string): Promise<void> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const res = await api.get<RunStatusResponse>(`/api/runs/${runId}/status`);
    const { status, grpcEndpoint } = res.data;
    if (status === 'RUNNING' && grpcEndpoint) return;
    if (status === 'FAILED' || status === 'COMPLETED' || status === 'STOPPED') {
      // MO-16: actionable, not just a status code — the run the owner had running has ended; a new one
      // is a fresh /start away. (resolveRunId gives the analogous message when there is no active run.)
      throw new Error(`The active run has ended (${status}). Ask the owner to start a new one.`);
    }
    if (Date.now() > deadline) throw new Error('Timed out waiting for the run to become ready.');
    await new Promise<void>((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

/**
 * Slice 1b — connect the device to the project's active run and register the native FL client.
 * Stops at registerClient (the run-onboarding DoD). Model download + on-device training data
 * (native "task 14") + round execution are the model-delivery / execution slices.
 */
export async function joinRun(p: JoinParams): Promise<JoinedRun> {
  const runId = await resolveRunId(p.projectId);
  await pollUntilRunning(runId);

  const { data: enroll } = await api.post<EnrollResponse>(`/api/runs/${runId}/enroll`);

  const clientId = await getOrCreateClientId();
  const reg = await nativeCore.registerClient(
    enroll.grpcEndpoint, runId, clientId, enroll.connectionToken, p.useTls ?? false,
  );
  if (!reg.accepted) {
    throw new Error(reg.message || 'Server rejected the client registration.');
  }

  return {
    runId,
    projectId: enroll.projectId,
    partitionId: enroll.partitionId,
    assignedRound: reg.assignedRound,
    grpcEndpoint: enroll.grpcEndpoint,
    manifest: enroll.manifest,
    message: reg.message,
  };
}
