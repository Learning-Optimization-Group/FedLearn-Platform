// =============================================================================
// FedLearn Desktop — HardwareSelector Component
// =============================================================================
// Card-based hardware profile selector + the "models you can train" picker.
//
// The user no longer types a project id / server address / partition id. After
// login the app lists the projects the user may train (GET /api/client/projects);
// when the user picks one and clicks Start, the live gRPC address + the
// server-assigned partition id + the model type are fetched from the backend
// (GET /api/client/projects/{id}/connection) and used to launch training.
// =============================================================================

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  MonitorCog,
  CircuitBoard,
  Command,
  Cpu,
  AlertTriangle,
  Play,
  Square,
  RefreshCw,
} from 'lucide-react';
import type { ClientProject } from '../client.types';
import { evaluateEligibility, eligibilitySummary } from '../../shared/evaluateEligibility';
import type { DeviceCapabilities } from '../../shared/deviceCapabilities.types';

interface HardwareSelectorProps {
  onStart: (config: {
    hardwareProfile: string;
    projectId: string;
    serverAddress: string;
    partitionId: string;
    modelType: string;
    datasetPath: string;
  }) => void;
  onStop: () => void;
  isRunning: boolean;
}

interface HardwareProfileOption {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ strokeWidth?: number | string; size?: number | string }>;
  dockerConfig: string;
}

const HARDWARE_PROFILES: HardwareProfileOption[] = [
  {
    id: 'discrete',
    label: 'Discrete GPU',
    description: 'NVIDIA workstation with dedicated PCIe GPU. Uses --gpus all via DeviceRequests.',
    icon: MonitorCog,
    dockerConfig: 'DeviceRequests: --gpus all',
  },
  {
    id: 'jetson',
    label: 'Jetson SoC',
    description: 'NVIDIA Jetson edge device with integrated Tegra GPU. Uses direct /dev/nvhost-* mounts.',
    icon: CircuitBoard,
    dockerConfig: 'Devices: /dev/nvhost-ctrl, nvhost-ctrl-gpu, ...',
  },
  {
    id: 'mps',
    label: 'Apple Silicon',
    description: 'Mac M1/M2/M3/M4 with Metal GPU. Runs natively (no Docker) for MPS acceleration.',
    icon: Command,
    dockerConfig: 'Native process (no Docker)',
  },
  {
    id: 'cpu',
    label: 'CPU Only',
    description: 'Standard CPU training without GPU acceleration. Compatible with any hardware.',
    icon: Cpu,
    dockerConfig: 'No GPU configuration',
  },
];

const HardwareSelector: React.FC<HardwareSelectorProps> = ({ onStart, onStop, isRunning }) => {
  const [selectedProfile, setSelectedProfile] = useState('cpu');
  const [detectionLabel, setDetectionLabel] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<DeviceCapabilities | null>(null);

  const [projects, setProjects] = useState<ClientProject[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [projectsError, setProjectsError] = useState('');

  const [datasetPath, setDatasetPath] = useState('');
  const [validationError, setValidationError] = useState('');
  const [starting, setStarting] = useState(false);

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
        if (cancelled || !result.success || !result.detection) return;

        const d = result.detection;
        setSelectedProfile(d.recommendedProfile);

        const parts: string[] = [];
        if (d.platform === 'darwin' && d.arch === 'arm64') parts.push('Apple Silicon');
        else if (d.platform === 'win32') parts.push('Windows x64');
        else parts.push(`${d.platform}/${d.arch}`);

        if (d.cudaAvailable) parts.push(`CUDA — ${d.cudaInfo || 'NVIDIA GPU'}`);
        if (!d.nativeBundleAvailable) parts.push('native bundle missing — falling back to Docker');

        setDetectionLabel(parts.join(' · '));
      } catch {
        // Detection is best-effort; the user can still pick a profile manually.
      }
    })();
    return () => { cancelled = true; };
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

  // Load the trainable-project list on mount.
  useEffect(() => { void loadProjects(); }, [loadProjects]);

  const handleSelectDataset = async () => {
    try {
      const result = await window.fedLearnAPI.selectDatasetPath();
      if (result.success && result.path) {
        setDatasetPath(result.path);
        setValidationError('');
      } else if (result.error) {
        setValidationError(`Dataset selection failed: ${result.error}`);
      }
    } catch (err: unknown) {
      setValidationError(`Error opening dialog: ${err instanceof Error ? err.message : 'unknown'}`);
    }
  };

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
      // Resolve the live connection (gRPC address + server-assigned partition id
      // + model type) from the backend — no manual entry.
      const res = await window.fedLearnAPI.getProjectConnection(selectedProject.projectId);
      if (!res.success || !res.connection) {
        setValidationError(res.error || 'Could not get connection details for this model.');
        return;
      }
      const c = res.connection;
      onStart({
        hardwareProfile: selectedProfile,
        projectId: c.projectId,
        serverAddress: c.serverAddress,
        partitionId: String(c.partitionId),
        modelType: c.modelType,
        datasetPath: datasetPath.trim(),
      });
    } catch (err: unknown) {
      setValidationError(err instanceof Error ? err.message : 'Failed to start training.');
    } finally {
      setStarting(false);
    }
  }, [selectedProject, selectedProfile, datasetPath, onStart]);

  return (
    <div className="hardware-selector">
      {detectionLabel && (
        <div className="detection-label" role="status" style={{ fontSize: '0.75rem', color: 'var(--fg-muted)', marginBottom: 'var(--space-2)' }}>
          Detected: {detectionLabel}
        </div>
      )}

      {/* Hardware Profile Cards */}
      <div className="profile-cards">
        {HARDWARE_PROFILES.map((profile) => (
          <button
            key={profile.id}
            id={`profile-${profile.id}`}
            className={`profile-card ${selectedProfile === profile.id ? 'profile-card-active' : ''}`}
            onClick={() => setSelectedProfile(profile.id)}
            disabled={isRunning}
            type="button"
          >
            <span className="profile-icon"><profile.icon strokeWidth={1.5} size={20} /></span>
            <span className="profile-label">{profile.label}</span>
            <span className="profile-desc">{profile.description}</span>
            <span className="profile-docker">{profile.dockerConfig}</span>
          </button>
        ))}
      </div>

      {/* Models you can train */}
      <div className="config-inputs">
        <div className="form-group">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label className="form-label" htmlFor="config-project">
              Model to train
            </label>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => { void loadProjects(); }}
              disabled={isRunning || loadingProjects}
              style={{ whiteSpace: 'nowrap', fontSize: '0.75rem' }}
            >
              <span className="btn-icon"><RefreshCw strokeWidth={1.5} size={14} /></span>
              Refresh
            </button>
          </div>

          {loadingProjects ? (
            <div style={{ fontSize: '0.85rem', color: 'var(--fg-muted)' }}>Loading your models…</div>
          ) : projectsError ? (
            <div className="validation-error" role="alert">
              <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
              {projectsError}
            </div>
          ) : projects.length === 0 ? (
            <div style={{ fontSize: '0.85rem', color: 'var(--fg-muted)' }}>
              You don&apos;t have any models to train yet. Ask a project owner to grant you access,
              or browse available projects in the web dashboard.
            </div>
          ) : (
            <>
              <select
                id="config-project"
                className="form-input"
                value={selectedProjectId}
                onChange={(e) => { setSelectedProjectId(e.target.value); setValidationError(''); }}
                disabled={isRunning}
              >
                {projects.map((p) => {
                  const elig = eligibilityByProject[p.projectId];
                  const marker = elig ? eligibilitySummary(elig).marker + ' ' : '';
                  return (
                    <option key={p.projectId} value={p.projectId}>
                      {marker}{p.name} — {p.modelType} ({p.status})
                    </option>
                  );
                })}
              </select>
              {selectedProject && selectedProject.status !== 'RUNNING' && (
                <div style={{ fontSize: '0.75rem', color: 'var(--warning, var(--fg-muted))', marginTop: 'var(--space-1)' }}>
                  Waiting for the owner to start this model&apos;s training server.
                </div>
              )}
              {selectedProject && (() => {
                const elig = eligibilityByProject[selectedProject.projectId];
                if (!elig) return null;
                const s = eligibilitySummary(elig);
                if (s.marker === '✅') return null;
                return (
                  <div
                    className={s.marker === '⚠️' ? 'eligibility-warn' : 'eligibility-info'}
                    style={{ fontSize: '0.75rem', marginTop: 'var(--space-1)' }}
                  >
                    {s.marker} {s.lines.join(' · ')}
                  </div>
                );
              })()}
            </>
          )}
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="config-dataset-path">
            Local Dataset Path (Optional)
          </label>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            <input
              id="config-dataset-path"
              className="form-input"
              type="text"
              value={datasetPath}
              readOnly
              placeholder="Select the local folder containing your training data. E.g., C:/Datasets/CIFAR10"
              disabled={isRunning}
            />
            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleSelectDataset}
              disabled={isRunning}
              style={{ whiteSpace: 'nowrap' }}
            >
              Browse...
            </button>
          </div>
        </div>
      </div>

      {/* Validation Error */}
      {validationError && (
        <div className="validation-error" role="alert">
          <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
          {validationError}
        </div>
      )}

      {/* Action Buttons */}
      <div className="action-buttons">
        {!isRunning ? (
          <button
            id="start-training-button"
            className="btn btn-primary btn-full"
            onClick={() => { void handleStart(); }}
            type="button"
            disabled={starting || loadingProjects || !selectedProject || selectedProject.status !== 'RUNNING'}
          >
            <span className="btn-icon"><Play strokeWidth={1.5} size={16} /></span>
            {starting ? 'Connecting…' : 'Start Training'}
          </button>
        ) : (
          <button
            id="stop-training-button"
            className="btn btn-danger btn-full"
            onClick={onStop}
            type="button"
          >
            <span className="btn-icon"><Square strokeWidth={1.5} size={16} /></span>
            Stop Training
          </button>
        )}
      </div>
    </div>
  );
};

export default HardwareSelector;
