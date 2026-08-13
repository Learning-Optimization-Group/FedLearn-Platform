// =============================================================================
// FedLearn Desktop — Train flow logic (pure, UI-free)
// =============================================================================
// State derivation for TrainSection's two-state flow: which layout phase the
// section is in, whether the guided setup is ready to start, and small display
// formatters. Kept free of React/DOM so the node-env jest suite can exercise
// every branch directly.
// =============================================================================

/**
 * Mirror of App's ContainerStatus union. Structurally identical on purpose —
 * TrainSection must not import from App.tsx (the shell owns that file), so the
 * contract is duplicated here and stays assignable in both directions.
 */
export type TrainRunStatus =
  | 'idle'
  | 'pulling'
  | 'running'
  | 'completed'
  | 'error'
  | 'restarting'
  | 'paused'
  | 'stopped';

/** The config shape App's handleStartTraining accepts (unchanged contract). */
export interface StartTrainingConfig {
  hardwareProfile: string;
  projectId: string;
  serverAddress: string;
  partitionId: string;
  modelType: string;
  datasetPath: string;
  connectionToken?: string;
  // Active run's aggregation strategy; forwarded to the client as --strategy (DeComFL vs default FedAvg).
  strategy?: string;
  // Forwarded to the client as --training-arm so a FROZEN_HEAD project federates head-only.
  trainingArm?: string;
}

/** Layout phase for the Train section. */
export type TrainPhase = 'setup' | 'running' | 'completed' | 'error';

/** Statuses during which a run is in flight (logs are the dominant surface). */
export const ACTIVE_STATUSES: ReadonlySet<TrainRunStatus> = new Set([
  'pulling',
  'running',
  'restarting',
  'paused',
]);

export function derivePhase(status: TrainRunStatus): TrainPhase {
  if (ACTIVE_STATUSES.has(status)) return 'running';
  if (status === 'completed') return 'completed';
  if (status === 'error') return 'error';
  return 'setup'; // idle | stopped
}

// ── Hardware detection summary ───────────────────────────────────────────────

/** Shape of window.fedLearnAPI.detectHardware()'s `detection` payload. */
export interface HardwareDetection {
  platform: string;
  arch: string;
  recommendedProfile: string;
  nativeBundleAvailable: boolean;
  cudaAvailable: boolean;
  cudaInfo?: string;
}

/** Compact one-line chip label for the detected hardware. */
export function describeDetection(d: HardwareDetection): string {
  const parts: string[] = [];
  if (d.platform === 'darwin' && d.arch === 'arm64') parts.push('Apple Silicon');
  else if (d.platform === 'win32') parts.push('Windows x64');
  else parts.push(`${d.platform}/${d.arch}`);

  if (d.cudaAvailable) parts.push(`CUDA — ${d.cudaInfo || 'NVIDIA GPU'}`);
  // Non-jetson profiles train via the bundled native client (no Docker
  // fallback); a missing bundle means training can't run until reinstall.
  if (!d.nativeBundleAvailable) parts.push('native client bundle missing — reinstall to enable training');
  return parts.join(' · ');
}

// ── Readiness checklist ──────────────────────────────────────────────────────

export type ReadinessState = 'ok' | 'warn' | 'pending' | 'blocked';

export interface ReadinessItem {
  id: 'server' | 'project' | 'hardware' | 'dataset';
  label: string;
  state: ReadinessState;
  detail?: string;
}

export interface ReadinessInput {
  projectsLoading: boolean;
  projectsError: string;
  hasProjects: boolean;
  selectedProject: { name: string; status: string } | null;
  /** Advisory self-gate result for the selected project (never blocks start). */
  eligibility: { eligible: boolean; lines: string[] } | null;
  detection: {
    done: boolean;
    failed: boolean;
    nativeBundleMissing: boolean;
    summary: string;
  };
  datasetPath: string;
  datasetSkipped: boolean;
}

/**
 * Derive the four readiness rows entirely from state the renderer already has —
 * no new IPC. Blocking rules intentionally match the pre-redesign gating
 * (project selected + accepting clients); eligibility and hardware issues stay
 * advisory warnings, exactly as before. The only new gate is the explicit
 * dataset choice: pick a folder or explicitly skip.
 */
export function deriveReadiness(input: ReadinessInput): ReadinessItem[] {
  const items: ReadinessItem[] = [];

  // 1. Server reachable — proxied by the trainable-projects fetch.
  let server: ReadinessItem;
  if (input.projectsLoading) {
    server = { id: 'server', label: 'Server reachable', state: 'pending', detail: 'Contacting server…' };
  } else if (input.projectsError) {
    server = { id: 'server', label: 'Server reachable', state: 'blocked', detail: input.projectsError };
  } else {
    server = { id: 'server', label: 'Server reachable', state: 'ok' };
  }
  items.push(server);

  // 2. Model selected and accepting clients.
  let project: ReadinessItem;
  if (server.state !== 'ok') {
    project = { id: 'project', label: 'Model selected', state: 'pending', detail: 'Waiting for the model list' };
  } else if (!input.hasProjects) {
    project = {
      id: 'project',
      label: 'Model selected',
      state: 'blocked',
      detail: 'No models available — ask a project owner for access',
    };
  } else if (!input.selectedProject) {
    project = { id: 'project', label: 'Model selected', state: 'blocked', detail: 'Choose a model to train' };
  } else if (input.selectedProject.status !== 'RUNNING') {
    project = {
      id: 'project',
      label: 'Model selected',
      state: 'blocked',
      detail: `${input.selectedProject.name} is not accepting clients yet — ask the owner to start it, then refresh`,
    };
  } else if (input.eligibility && (!input.eligibility.eligible || input.eligibility.lines.length > 0)) {
    project = {
      id: 'project',
      label: 'Model selected',
      state: 'warn',
      detail: `${input.selectedProject.name} — ${input.eligibility.lines.join(' · ')}`,
    };
  } else {
    project = { id: 'project', label: 'Model selected', state: 'ok', detail: input.selectedProject.name };
  }
  items.push(project);

  // 3. Hardware detected — advisory only; the user can always pick a profile.
  let hardware: ReadinessItem;
  if (!input.detection.done) {
    hardware = { id: 'hardware', label: 'Hardware detected', state: 'pending', detail: 'Detecting hardware…' };
  } else if (input.detection.failed) {
    hardware = {
      id: 'hardware',
      label: 'Hardware detected',
      state: 'warn',
      detail: 'Detection unavailable — pick a profile under Advanced',
    };
  } else if (input.detection.nativeBundleMissing) {
    hardware = {
      id: 'hardware',
      label: 'Hardware detected',
      state: 'warn',
      detail: 'Native client bundle missing — reinstall to enable training',
    };
  } else {
    hardware = { id: 'hardware', label: 'Hardware detected', state: 'ok', detail: input.detection.summary };
  }
  items.push(hardware);

  // 4. Dataset chosen or explicitly skipped.
  let dataset: ReadinessItem;
  if (input.datasetPath.trim() !== '') {
    dataset = { id: 'dataset', label: 'Training data', state: 'ok', detail: input.datasetPath };
  } else if (input.datasetSkipped) {
    dataset = {
      id: 'dataset',
      label: 'Training data',
      state: 'ok',
      detail: 'Skipped — using the model’s built-in dataset',
    };
  } else {
    dataset = { id: 'dataset', label: 'Training data', state: 'pending', detail: 'Choose a folder, or skip' };
  }
  items.push(dataset);

  return items;
}

/** Start is enabled when nothing is pending or blocked (warnings don't gate). */
export function isReadyToStart(items: ReadinessItem[]): boolean {
  return items.every((i) => i.state === 'ok' || i.state === 'warn');
}

// ── Display formatters ───────────────────────────────────────────────────────

/** "m:ss" under an hour, "h:mm:ss" above. Never negative. */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, '0');
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${ss}`;
  return `${m}:${ss}`;
}
