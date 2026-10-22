import api from '../api/axiosConfig';

export type SimulationStatus = 'ACCEPTED' | 'STARTING' | 'RUNNING' | 'STOPPING' | 'COMPLETED'
  | 'STOPPED' | 'FAILED' | 'TIMED_OUT' | 'INTERRUPTED';
export type SimulationChoices = {
  workloadId: string; recipe: string; head: string; numClients: number; rounds: number; localEpochs: number;
  aggregation: string; clipNorm: number; privacy: string; epsilon: number; behavior: string; seed: number;
};
export type SimulationRecipe = Partial<SimulationChoices> & {
  evaluationSplit?: string; featureReportSha256?: string; learningRate?: number;
  batchSize?: number; parameters?: number; delta?: number; trimRatio?: number | null;
  maliciousFraction?: number; researchRecordDpSafe?: boolean;
};
export type SimulationScalar = number | null | { __nonfinite_float__: 'nan' | 'inf' | '-inf' };
export type SimulationRound = {
  round: number; auc: SimulationScalar; accuracy: SimulationScalar; loss: SimulationScalar;
  balanced_accuracy: SimulationScalar; sensitivity: SimulationScalar; specificity: SimulationScalar; ece: SimulationScalar;
  comm: { up_bytes: SimulationScalar; down_bytes: SimulationScalar };
  privacy: { epsilon_spent: SimulationScalar; clip_rate: SimulationScalar; noise_std: SimulationScalar; sensitivity: SimulationScalar };
  participants: Record<string, SimulationScalar>;
  cpuSeconds: SimulationScalar; simulatedRoundSeconds: SimulationScalar;
};
export type SimulationLimits = {
  deadlineSeconds: number; cpuThreads: number; observedRssQualificationBytes: number;
  activeWorkers: number; scope: string;
};
export type SimulationWorkload = {
  id: string; ready: boolean; reason: string | null; label: string; representation: string;
  reportSha256: string | null; splitSummary: Record<string, { images?: number; patients?: number }> | null;
  supportedChoices: Record<string, unknown>; limits: SimulationLimits;
};
export type SimulationResolution = {
  workloadId: string; configHash: string; recipe: SimulationRecipe; limits: SimulationLimits; scope: string;
};
export type SimulationAccepted = {
  id: string; status: SimulationStatus; configHash: string; recipe: SimulationRecipe; createdAt: string; scope: string;
};
export type SimulationRun = SimulationAccepted & {
  startedAt: string | null; endedAt: string | null; completedRounds: number | null; requestedRounds: number | null;
  rounds: SimulationRound[]; finalSummary: SimulationRound | null; qualification: Record<string, unknown> | null;
  recordAvailable: boolean; recordEncoding: string | null; errorCode: string | null; errorMessage: string | null;
};
export type SimulationPage = { content: SimulationRun[]; totalElements: number; page: number; size: number; scope: string };
export type SimulationRecord = {
  runId: string; configHash: string; researchRecordDpSafe: boolean;
  execution: { wallSeconds: SimulationScalar; peakRssBytes: SimulationScalar; peakRssScope: string | null; scope: string };
  source: { runtime?: { git_commit?: string; python?: string; torch?: string; numpy?: string } }; scope: string;
};

const runPath = (id: string) => `/simulations/runs/${encodeURIComponent(id)}`;
export const simulationTerminal = (status: SimulationStatus) =>
  ['COMPLETED', 'STOPPED', 'FAILED', 'TIMED_OUT', 'INTERRUPTED'].includes(status);

export async function getSimulationWorkloads(signal?: AbortSignal) {
  return (await api.get<{ workloads: SimulationWorkload[]; scope: string }>('/simulations/workloads', { signal, timeout: 30000 })).data;
}
export async function resolveSimulation(choices: Partial<SimulationChoices>, signal?: AbortSignal) {
  return (await api.post<SimulationResolution>('/simulations/resolve', { choices }, { signal, timeout: 30000 })).data;
}
export async function startSimulation(choices: Partial<SimulationChoices>, signal?: AbortSignal) {
  return (await api.post<SimulationAccepted>('/simulations/runs', { choices }, { signal, timeout: 30000 })).data;
}
export async function listSimulations(page = 0, signal?: AbortSignal) {
  return (await api.get<SimulationPage>('/simulations/runs', { params: { page, size: 20, scope: 'mine' }, signal, timeout: 30000 })).data;
}
export async function getSimulation(id: string, signal?: AbortSignal) {
  return (await api.get<SimulationRun>(runPath(id), { signal, timeout: 30000 })).data;
}
export async function stopSimulation(id: string, signal?: AbortSignal) {
  return (await api.post<SimulationRun>(`${runPath(id)}/stop`, undefined, { signal, timeout: 30000 })).data;
}
export async function getSimulationRecord(id: string, signal?: AbortSignal) {
  return (await api.get<SimulationRecord>(`${runPath(id)}/record`, { signal, timeout: 30000 })).data;
}
export async function downloadSimulationRecord(id: string, signal?: AbortSignal) {
  return (await api.get<Blob>(`${runPath(id)}/record`, { responseType: 'blob', signal, timeout: 30000 })).data;
}
