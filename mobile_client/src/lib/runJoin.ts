// Join-run flow (15-LLD §6.1): REST start/poll -> manifest version gate -> enrollment token ->
// stable client id -> native registerClient -> integrity-checked loadModel. Returns everything
// the round loop (TrainingScreen) needs.
import { api } from './restClient';
import { getOrCreateClientId } from './clientId';
import nativeCore, { type ModelInfo } from './nativeCore';

// DeterminismManifestDto (04 §4.4 / 15-LLD §5.4) — the run's reproducibility contract.
export interface DeterminismManifest {
  runId: string;
  seed: number;
  strategy: 'DeComFL' | 'FedAvg';
  torchVersion: string; // mobile MUST match this for RNG parity (else warn/refuse)
  numpyVersion: string;
  frameworkGitSha: string;
  datasetVersionId: string;
  datasetSha256: string;
  partitionRecipeId: string;
  modelInitSha256: string;
  goldenVectorSha256: string;
  createdAt: string;
}

interface RunStatus {
  status: string; // INITIALIZING | WAITING_FOR_CLIENTS | RUNNING | ...
  grpcEndpoint: string | null;
  enrollmentToken?: string;
}

export interface JoinParams {
  projectId: string;
  strategy: 'DeComFL' | 'FedAvg';
  numRounds: number;
  minClients: number;
  datasetVersionId: string;
  modelPath: string; // app-private path of the .pt the manifest hash is checked against
  useTls?: boolean; // default true (release); a dev build may pass false
}

export interface JoinedRun {
  runId: string;
  grpcEndpoint: string;
  manifest: DeterminismManifest;
  clientId: string;
  modelInfo: ModelInfo;
  assignedRound: number;
}

const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 120000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

async function pollUntilRunning(runId: string): Promise<RunStatus> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const { data } = await api.get<RunStatus>(`/api/runs/${runId}/status`);
    if (data.status === 'RUNNING' && data.grpcEndpoint) return data;
    if (data.status === 'FAILED' || data.status === 'TRAINING_COMPLETE') {
      throw new Error(`run ${runId} is ${data.status}; cannot join`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
  throw new Error(`run ${runId} did not reach RUNNING within ${POLL_TIMEOUT_MS} ms`);
}

export async function joinRun(p: JoinParams): Promise<JoinedRun> {
  // 1. Start (or join) a run on the project.
  const { data: run } = await api.post<{ id: string; enrollmentToken?: string }>(
    `/api/projects/${p.projectId}/runs`,
    {
      strategy: p.strategy,
      numRounds: p.numRounds,
      minClients: p.minClients,
      datasetVersionId: p.datasetVersionId,
    },
  );
  const runId = run.id;

  // 2. Poll until the FL server is RUNNING and has published its gRPC endpoint.
  const status = await pollUntilRunning(runId);

  // 3. Reproducibility / version gate (the mobile half of the C3 version-compatibility gate).
  const { data: manifest } = await api.get<DeterminismManifest>(`/api/runs/${runId}/manifest`);

  // 4. Enrollment token (backend-minted at launch; surfaced via the join or status response).
  const enrollmentToken = run.enrollmentToken ?? status.enrollmentToken;
  if (!enrollmentToken) throw new Error('no enrollment_token returned by the control plane');

  // 5. Stable client id (encrypted, persisted).
  const clientId = await getOrCreateClientId();

  // 6. Register the native client over gRPC (TLS+mTLS by default).
  const reg = await nativeCore.registerClient(
    status.grpcEndpoint as string,
    runId,
    clientId,
    enrollmentToken,
    p.useTls ?? true,
  );
  if (!reg.accepted) {
    throw new Error(`registration rejected (server protocol v${reg.serverProtocolVersion}): ${reg.message}`);
  }

  // 7. Load the model, integrity-checked against the manifest hash, before any round.
  const modelInfo = await nativeCore.loadModel(p.modelPath, manifest.modelInitSha256);

  return {
    runId,
    grpcEndpoint: status.grpcEndpoint as string,
    manifest,
    clientId,
    modelInfo,
    assignedRound: reg.assignedRound,
  };
}
