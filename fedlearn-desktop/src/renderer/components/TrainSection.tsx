// =============================================================================
// FedLearn Desktop — TrainSection Component
// =============================================================================
// The Train view as a two-state flow keyed on the run status App already owns:
//
//   SETUP   (idle/stopped/completed/error) — one guided card: model picker,
//           dataset folder (choose or explicitly skip), detected-hardware chip
//           with a details disclosure, an Advanced disclosure hiding the
//           profile override, then a readiness checklist and a single primary
//           Start button that enables when nothing is pending or blocked.
//           Completed/error additionally show an outcome banner ("Run again"
//           becomes the primary) and keep the previous run's log alongside.
//
//   RUNNING (pulling/running/restarting/paused) — logs are the dominant
//           surface: a compact run header (project, elapsed, hardware, Stop)
//           over a full-height LogPanel.
//
// Everything derives from existing renderer state + the existing preload IPC
// (listTrainableProjects, getProjectConnection, detectHardware,
// getDeviceCapabilities, selectDatasetPath) — no new channels, no new App
// state. On completion/failure transitions an HTML5 Notification fires
// (permission-guarded, renderer-only).
// =============================================================================

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  Circle,
  Cpu,
  Play,
  RefreshCw,
  Square,
  XCircle,
} from 'lucide-react';
import LogPanel from './LogPanel';
import { HardwareProfilePicker, HARDWARE_PROFILES } from './HardwareSelector';
import { evaluateEligibility, eligibilitySummary } from '../../shared/evaluateEligibility';
import type { DeviceCapabilities } from '../../shared/deviceCapabilities.types';
import type { ClientProject } from '../client.types';
import {
  ACTIVE_STATUSES,
  derivePhase,
  deriveReadiness,
  describeDetection,
  formatElapsed,
  isReadyToStart,
  type HardwareDetection,
  type ReadinessItem,
  type StartTrainingConfig,
  type TrainRunStatus,
} from './trainFlow';
import { classifyRunTransition, notifyRunOutcome } from './runNotifications';
import './sections.css';

export type { StartTrainingConfig, TrainRunStatus } from './trainFlow';

export interface TrainSectionProps {
  /**
   * App's containerStatus — drives the setup/running/outcome layout.
   * Preferred: with the full status this section can render the
   * completed/error outcome banners and fire the finish notifications.
   */
  status?: TrainRunStatus;
  /**
   * Compatibility mirror of the old HardwareSelector prop. Consulted only
   * when `status` is absent and collapses to 'running'/'idle' — outcome
   * banners and notifications need `status` instead.
   */
  isRunning?: boolean;
  /** App's log buffer (unchanged data flow). */
  logs: string[];
  /** App's handleStartTraining. */
  onStart: (config: StartTrainingConfig) => void;
  /** App's handleStopTraining. */
  onStop: () => void;
}

const READINESS_ICONS: Record<ReadinessItem['state'], React.ReactNode> = {
  ok: <CheckCircle2 strokeWidth={1.5} size={14} />,
  warn: <AlertTriangle strokeWidth={1.5} size={14} />,
  blocked: <XCircle strokeWidth={1.5} size={14} />,
  pending: <Circle strokeWidth={1.5} size={14} />,
};

export const TrainSection: React.FC<TrainSectionProps> = ({
  status: statusProp,
  isRunning,
  logs,
  onStart,
  onStop,
}) => {
  // Full status wins; the boolean fallback can only express running/idle.
  const status: TrainRunStatus = statusProp ?? (isRunning ? 'running' : 'idle');

  // ── Hardware detection (one-shot, mirrors the pre-redesign behavior) ──────
  const [selectedProfile, setSelectedProfile] = useState('cpu');
  const [detection, setDetection] = useState<HardwareDetection | null>(null);
  const [detectionDone, setDetectionDone] = useState(false);
  const [capabilities, setCapabilities] = useState<DeviceCapabilities | null>(null);

  // ── Trainable projects ────────────────────────────────────────────────────
  const [projects, setProjects] = useState<ClientProject[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [projectsError, setProjectsError] = useState('');

  // ── Dataset + start flow ──────────────────────────────────────────────────
  const [datasetPath, setDatasetPath] = useState('');
  const [datasetSkipped, setDatasetSkipped] = useState(false);
  const [validationError, setValidationError] = useState('');
  const [starting, setStarting] = useState(false);

  // ── Run header bookkeeping ────────────────────────────────────────────────
  const [elapsedMs, setElapsedMs] = useState(0);
  const runStartRef = useRef<number | null>(null);
  const lastStartedNameRef = useRef<string | null>(null);
  const prevStatusRef = useRef<TrainRunStatus>(status);

  const selectedProject = projects.find((p) => p.projectId === selectedProjectId) ?? null;

  const eligibilityByProject = useMemo(() => {
    const map: Record<string, ReturnType<typeof evaluateEligibility>> = {};
    if (!capabilities) return map;
    for (const p of projects) map[p.projectId] = evaluateEligibility(capabilities, p.requirements);
    return map;
  }, [capabilities, projects]);

  // One-shot hardware detection — pre-select the profile matching this machine.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        window.fedLearnAPI.getDeviceCapabilities().then((res) => {
          if (!cancelled && res.success && res.capabilities) setCapabilities(res.capabilities);
        });

        const result = await window.fedLearnAPI.detectHardware();
        if (cancelled) return;
        if (result.success && result.detection) {
          setDetection(result.detection);
          setSelectedProfile(result.detection.recommendedProfile);
        }
      } catch {
        // Detection is best-effort; the user can still pick a profile manually.
      } finally {
        if (!cancelled) setDetectionDone(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadProjects = useCallback(async () => {
    setLoadingProjects(true);
    setProjectsError('');
    try {
      const res = await window.fedLearnAPI.listTrainableProjects();
      if (res.success && res.projects) {
        setProjects(res.projects);
        // Keep the current selection if it still exists, else default to the
        // first project that is RUNNING (ready to join), else the first one.
        setSelectedProjectId((prev) => {
          if (prev && res.projects!.some((p) => p.projectId === prev)) return prev;
          const running = res.projects!.find((p) => p.status === 'RUNNING');
          return (running ?? res.projects![0])?.projectId ?? '';
        });
      } else {
        setProjectsError(res.error || 'Could not load your projects.');
      }
    } catch (err: unknown) {
      setProjectsError(err instanceof Error ? err.message : 'Could not load your projects.');
    } finally {
      setLoadingProjects(false);
    }
  }, []);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  // Elapsed-time clock: anchor when a run becomes active, tick each second,
  // freeze on completed/error, reset on the next run.
  useEffect(() => {
    if (!ACTIVE_STATUSES.has(status)) {
      runStartRef.current = null; // elapsedMs keeps its last (frozen) value
      return;
    }
    if (runStartRef.current === null) {
      runStartRef.current = Date.now();
      setElapsedMs(0);
    }
    const timer = setInterval(() => {
      if (runStartRef.current !== null) setElapsedMs(Date.now() - runStartRef.current);
    }, 1000);
    return () => clearInterval(timer);
  }, [status]);

  // Desktop notification on completion/failure transitions (renderer-only,
  // guarded by Notification.permission inside notifyRunOutcome).
  useEffect(() => {
    const prev = prevStatusRef.current;
    prevStatusRef.current = status;
    const outcome = classifyRunTransition(prev, status);
    if (outcome) {
      notifyRunOutcome(outcome, lastStartedNameRef.current ?? 'FedLearn project');
    }
  }, [status]);

  const handleSelectDataset = useCallback(async () => {
    try {
      const result = await window.fedLearnAPI.selectDatasetPath();
      if (result.success && result.path) {
        setDatasetPath(result.path);
        setDatasetSkipped(false);
        setValidationError('');
      } else if (result.error) {
        setValidationError(`Dataset selection failed: ${result.error}`);
      }
    } catch (err: unknown) {
      setValidationError(`Error opening dialog: ${err instanceof Error ? err.message : 'unknown'}`);
    }
  }, []);

  const handleStart = useCallback(async () => {
    setValidationError('');

    if (!selectedProject) {
      setValidationError('Select a model to train.');
      return;
    }
    if (selectedProject.status !== 'RUNNING') {
      setValidationError(
        'This model is not accepting clients yet — its owner has not started the training server. '
        + 'Ask them to start it, then refresh.',
      );
      return;
    }

    setStarting(true);
    try {
      // Resolve the live connection (gRPC address + server-assigned partition
      // id + model type) from the backend — no manual entry.
      const res = await window.fedLearnAPI.getProjectConnection(selectedProject.projectId);
      if (!res.success || !res.connection) {
        setValidationError(res.error || 'Could not get connection details for this model.');
        return;
      }
      const c = res.connection;
      lastStartedNameRef.current = selectedProject.name;
      onStart({
        hardwareProfile: selectedProfile,
        projectId: c.projectId,
        serverAddress: c.serverAddress,
        partitionId: String(c.partitionId),
        modelType: c.modelType,
        datasetPath: datasetPath.trim(),
        connectionToken: c.connectionToken,
        strategy: c.strategy,
      });
    } catch (err: unknown) {
      setValidationError(err instanceof Error ? err.message : 'Failed to start training.');
    } finally {
      setStarting(false);
    }
  }, [selectedProject, selectedProfile, datasetPath, onStart]);

  // ── Derived view state ────────────────────────────────────────────────────
  const phase = derivePhase(status);

  const readiness = useMemo(() => {
    let eligibility: { eligible: boolean; lines: string[] } | null = null;
    if (selectedProject) {
      const elig = eligibilityByProject[selectedProject.projectId];
      if (elig) {
        eligibility = { eligible: elig.eligible, lines: eligibilitySummary(elig).lines };
      }
    }
    return deriveReadiness({
      projectsLoading: loadingProjects,
      projectsError,
      hasProjects: projects.length > 0,
      selectedProject: selectedProject
        ? { name: selectedProject.name, status: selectedProject.status }
        : null,
      eligibility,
      detection: {
        done: detectionDone,
        failed: detectionDone && detection === null,
        nativeBundleMissing: detection ? !detection.nativeBundleAvailable : false,
        summary: detection ? describeDetection(detection) : '',
      },
      datasetPath,
      datasetSkipped,
    });
  }, [
    loadingProjects,
    projectsError,
    projects.length,
    selectedProject,
    eligibilityByProject,
    detectionDone,
    detection,
    datasetPath,
    datasetSkipped,
  ]);

  const ready = isReadyToStart(readiness);
  const profileLabel = HARDWARE_PROFILES.find((p) => p.id === selectedProfile)?.label ?? selectedProfile;

  // ── RUNNING: logs dominant, compact run header ────────────────────────────
  if (phase === 'running') {
    const runName = lastStartedNameRef.current ?? selectedProject?.name ?? 'Training run';
    return (
      <div className="train-section train-running">
        <div className="run-header">
          <span className="run-status-dot" aria-hidden="true" />
          <div className="run-titles">
            <span className="run-title">{runName}</span>
            <span className="run-meta">
              {status === 'pulling' ? 'Starting' : 'Training'} · {profileLabel} · {formatElapsed(elapsedMs)}
            </span>
          </div>
          <button
            id="stop-training-button"
            className="btn btn-danger"
            onClick={onStop}
            type="button"
          >
            <span className="btn-icon"><Square strokeWidth={1.5} size={16} /></span>
            Stop training
          </button>
        </div>
        <div className="run-log-area">
          <section className="panel log-panel-container">
            <div className="panel-header">
              <h2 className="panel-title">Activity log</h2>
              <span className="log-count">{logs.length} lines</span>
            </div>
            <LogPanel logs={logs} />
          </section>
        </div>
      </div>
    );
  }

  // ── SETUP (plus outcome banner after completed/error) ─────────────────────
  const showLogs = logs.length > 0;
  const startLabel = starting
    ? 'Connecting…'
    : phase === 'completed' || phase === 'error'
      ? 'Run again'
      : 'Start training';

  return (
    <div className="train-section train-setup">
      {phase === 'completed' && (
        <div className="run-banner run-banner-success" role="status">
          <span className="run-banner-icon"><CheckCircle2 strokeWidth={1.5} size={18} /></span>
          <div className="run-banner-text">
            <strong>Training run completed.</strong>{' '}
            <span>The log from this run is kept below.</span>
          </div>
        </div>
      )}
      {phase === 'error' && (
        <div className="run-banner run-banner-danger" role="alert">
          <span className="run-banner-icon"><AlertTriangle strokeWidth={1.5} size={18} /></span>
          <div className="run-banner-text">
            <strong>Training run failed.</strong>{' '}
            <span>Check the log below, then adjust and run again.</span>
          </div>
        </div>
      )}

      <div className={showLogs ? 'train-setup-grid train-setup-grid-with-logs' : 'train-setup-grid'}>
        <section className="panel setup-card">
          <div className="panel-header">
            <h2 className="panel-title">Set up training</h2>
            <span className="panel-badge">This device</span>
          </div>

          <div className="setup-card-body">
            {/* 1 — Model to train */}
            <div className="form-group">
              <div className="form-label-row">
                <span className="form-label" id="train-model-label">Model to train</span>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm btn-icon-only"
                  onClick={() => { void loadProjects(); }}
                  disabled={starting || loadingProjects}
                  aria-label="Refresh model list"
                  title="Refresh model list"
                >
                  <RefreshCw strokeWidth={1.5} size={14} />
                </button>
              </div>

              {loadingProjects ? (
                <div className="form-help">Loading your models…</div>
              ) : projectsError ? (
                <div className="validation-error" role="alert">
                  <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
                  {projectsError}
                </div>
              ) : projects.length === 0 ? (
                <div className="form-help">
                  You don&apos;t have any models to train yet. Ask a project owner to grant you
                  access, or browse available projects in the web dashboard.
                </div>
              ) : (
                <div className="project-list" role="group" aria-labelledby="train-model-label">
                  {projects.map((p) => {
                    const elig = eligibilityByProject[p.projectId];
                    const marker = elig ? eligibilitySummary(elig).marker : '';
                    const active = p.projectId === selectedProjectId;
                    return (
                      <button
                        key={p.projectId}
                        type="button"
                        className={active ? 'project-option project-option-active' : 'project-option'}
                        aria-pressed={active}
                        disabled={starting}
                        onClick={() => {
                          setSelectedProjectId(p.projectId);
                          setValidationError('');
                        }}
                      >
                        <span className="project-option-name">{p.name}</span>
                        <span className="project-option-meta">
                          {p.modelType}
                          {' · '}
                          {p.status === 'RUNNING' ? 'Accepting clients' : 'Waiting for owner to start'}
                          {marker}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* 2 — Training data */}
            <div className="form-group">
              <label className="form-label" htmlFor="config-dataset-path">
                Dataset folder
              </label>
              <div className="form-input-row">
                <input
                  id="config-dataset-path"
                  className="form-input"
                  type="text"
                  value={datasetPath}
                  readOnly
                  placeholder="No folder selected"
                  aria-describedby="config-dataset-path-help"
                  disabled={starting}
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => { void handleSelectDataset(); }}
                  disabled={starting}
                >
                  Browse…
                </button>
                {datasetPath !== '' && (
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setDatasetPath('')}
                    disabled={starting}
                  >
                    Clear
                  </button>
                )}
              </div>
              <p className="form-help" id="config-dataset-path-help">
                Choose the local folder containing your training data.
              </p>
              <label className="dataset-skip">
                <input
                  type="checkbox"
                  checked={datasetSkipped}
                  disabled={starting || datasetPath !== ''}
                  onChange={(e) => setDatasetSkipped(e.target.checked)}
                />
                <span>Skip — train with the model&apos;s built-in dataset</span>
              </label>
            </div>

            {/* 3 — Hardware */}
            <div className="form-group">
              <span className="form-label">Hardware</span>
              <details className="hw-details">
                <summary>
                  <span className="hw-chip-icon"><Cpu strokeWidth={1.5} size={14} /></span>
                  <span className="hw-chip-text" role="status">
                    {detection
                      ? describeDetection(detection)
                      : detectionDone
                        ? 'Detection unavailable — pick a profile under Advanced'
                        : 'Detecting hardware…'}
                  </span>
                  <span className="disclosure-chevron"><ChevronDown strokeWidth={1.5} size={14} /></span>
                </summary>
                <div className="hw-details-body">
                  <div className="hw-fact-row">
                    <span className="hw-fact-label">Platform</span>
                    <span className="hw-fact-value">
                      {detection ? `${detection.platform}/${detection.arch}` : 'Unknown'}
                    </span>
                  </div>
                  <div className="hw-fact-row">
                    <span className="hw-fact-label">Recommended profile</span>
                    <span className="hw-fact-value">
                      {detection
                        ? HARDWARE_PROFILES.find((p) => p.id === detection.recommendedProfile)?.label
                          ?? detection.recommendedProfile
                        : 'Unknown'}
                    </span>
                  </div>
                  <div className="hw-fact-row">
                    <span className="hw-fact-label">CUDA</span>
                    <span className="hw-fact-value">
                      {detection?.cudaAvailable ? detection.cudaInfo || 'Available' : 'Not available'}
                    </span>
                  </div>
                  <div className="hw-fact-row">
                    <span className="hw-fact-label">Native client bundle</span>
                    <span className="hw-fact-value">
                      {detection ? (detection.nativeBundleAvailable ? 'Present' : 'Missing — reinstall') : 'Unknown'}
                    </span>
                  </div>
                </div>
              </details>

              {/* Advanced — the profile override (the only manual override that
                  exists today; connection details are server-resolved). */}
              <details className="adv-details">
                <summary>
                  <span>Advanced</span>
                  <span className="adv-summary-hint">Profile: {profileLabel}</span>
                  <span className="disclosure-chevron"><ChevronDown strokeWidth={1.5} size={14} /></span>
                </summary>
                <div className="adv-details-body">
                  <p className="form-help">
                    Override how training executes on this device. The detected profile is
                    preselected.
                  </p>
                  <HardwareProfilePicker
                    value={selectedProfile}
                    onChange={setSelectedProfile}
                    disabled={starting}
                  />
                </div>
              </details>
            </div>

            {/* 4 — Readiness */}
            <div className="readiness" role="list" aria-label="Readiness checks">
              <span className="readiness-heading">Readiness</span>
              {readiness.map((item) => (
                <div key={item.id} role="listitem" className={`readiness-item readiness-${item.state}`}>
                  <span className="readiness-icon">{READINESS_ICONS[item.state]}</span>
                  <span className="readiness-label">{item.label}</span>
                  {item.detail && <span className="readiness-detail">{item.detail}</span>}
                </div>
              ))}
            </div>

            {validationError && (
              <div className="validation-error" role="alert">
                <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
                {validationError}
              </div>
            )}

            <div className="action-buttons">
              <button
                id="start-training-button"
                className="btn btn-primary btn-full"
                onClick={() => { void handleStart(); }}
                type="button"
                disabled={starting || loadingProjects || !ready}
              >
                <span className="btn-icon">
                  {phase === 'completed' || phase === 'error'
                    ? <ArrowRight strokeWidth={1.5} size={16} />
                    : <Play strokeWidth={1.5} size={16} />}
                </span>
                {startLabel}
              </button>
            </div>
          </div>
        </section>

        {showLogs && (
          <section className="panel log-panel-container setup-log-panel">
            <div className="panel-header">
              <h2 className="panel-title">Last run log</h2>
              <span className="log-count">{logs.length} lines</span>
            </div>
            <LogPanel logs={logs} />
          </section>
        )}
      </div>
    </div>
  );
};

export default TrainSection;
