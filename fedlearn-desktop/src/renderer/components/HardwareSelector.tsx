// =============================================================================
// FedLearn Desktop — HardwareSelector Component
// =============================================================================
// Card-based hardware profile selector with input fields for training config.
// The three profiles (discrete, jetson, cpu) map directly to the Docker
// runtime configurations specified in Section 4.2 of the deployment guide.
// =============================================================================

import React, { useState, useCallback } from 'react';

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
  icon: string;
  dockerConfig: string;
}

const HARDWARE_PROFILES: HardwareProfileOption[] = [
  {
    id: 'discrete',
    label: 'Discrete GPU',
    description: 'NVIDIA workstation with dedicated PCIe GPU. Uses --gpus all via DeviceRequests.',
    icon: '🖥️',
    dockerConfig: 'DeviceRequests: --gpus all',
  },
  {
    id: 'jetson',
    label: 'Jetson SoC',
    description: 'NVIDIA Jetson edge device with integrated Tegra GPU. Uses direct /dev/nvhost-* mounts.',
    icon: '🔧',
    dockerConfig: 'Devices: /dev/nvhost-ctrl, nvhost-ctrl-gpu, ...',
  },
  {
    id: 'mps',
    label: 'Apple Silicon',
    description: 'Mac M1/M2/M3/M4 with Metal GPU. Runs natively (no Docker) for MPS acceleration.',
    icon: '🍎',
    dockerConfig: 'Native process (no Docker)',
  },
  {
    id: 'cpu',
    label: 'CPU Only',
    description: 'Standard CPU training without GPU acceleration. Compatible with any hardware.',
    icon: '💻',
    dockerConfig: 'No GPU configuration',
  },
];

const HardwareSelector: React.FC<HardwareSelectorProps> = ({ onStart, onStop, isRunning }) => {
  const [selectedProfile, setSelectedProfile] = useState('discrete');
  const [projectId, setProjectId] = useState('');
  const [serverAddress, setServerAddress] = useState('');
  const [partitionId, setPartitionId] = useState('0');
  const [modelType, setModelType] = useState('CNN');
  const [datasetPath, setDatasetPath] = useState('');
  const [validationError, setValidationError] = useState('');

  const handleSelectDataset = async () => {
    try {
      const result = await window.fedLearnAPI.selectDatasetPath();
      if (result.success && result.path) {
        setDatasetPath(result.path);
        setValidationError('');
      } else if (result.error) {
        setValidationError(`Dataset selection failed: ${result.error}`);
      }
    } catch (err: any) {
      setValidationError(`Error opening dialog: ${err.message}`);
    }
  };

  const handleStart = useCallback(() => {
    setValidationError('');

    if (!projectId.trim()) {
      setValidationError('Project ID is required.');
      return;
    }

    if (!serverAddress.trim()) {
      setValidationError('Server address is required.');
      return;
    }

    if (!partitionId.trim()) {
      setValidationError('Partition ID is required.');
      return;
    }

    // Pattern validation (matches preload allowlists)
    if (!/^[a-zA-Z0-9_-]{1,128}$/.test(projectId)) {
      setValidationError('Project ID must be alphanumeric (max 128 chars).');
      return;
    }

    if (!/^[a-zA-Z0-9._:/-]{1,256}$/.test(serverAddress)) {
      setValidationError('Invalid server address format.');
      return;
    }

    if (!/^[0-9]{1,10}$/.test(partitionId)) {
      setValidationError('Partition ID must be a number.');
      return;
    }

    if (!modelType.trim()) {
      setValidationError('Model Architecture is required.');
      return;
    }

    if (!datasetPath.trim()) {
      setValidationError('Local Dataset Path is required.');
      return;
    }

    onStart({
      hardwareProfile: selectedProfile,
      projectId: projectId.trim(),
      serverAddress: serverAddress.trim(),
      partitionId: partitionId.trim(),
      modelType: modelType.trim(),
      datasetPath: datasetPath.trim(),
    });
  }, [selectedProfile, projectId, serverAddress, partitionId, modelType, datasetPath, onStart]);

  return (
    <div className="hardware-selector">
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
            <span className="profile-icon">{profile.icon}</span>
            <span className="profile-label">{profile.label}</span>
            <span className="profile-desc">{profile.description}</span>
            <span className="profile-docker">{profile.dockerConfig}</span>
          </button>
        ))}
      </div>

      {/* Configuration Inputs */}
      <div className="config-inputs">
        <div className="form-group">
          <label className="form-label" htmlFor="config-project-id">
            Project ID
          </label>
          <input
            id="config-project-id"
            className="form-input"
            type="text"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="e.g., cardiac-ecg-001"
            disabled={isRunning}
            maxLength={128}
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="config-server-address">
            Server Address
          </label>
          <input
            id="config-server-address"
            className="form-input"
            type="text"
            value={serverAddress}
            onChange={(e) => setServerAddress(e.target.value)}
            placeholder="e.g., 192.168.1.100:8080"
            disabled={isRunning}
            maxLength={256}
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="config-partition-id">
            Partition ID
          </label>
          <input
            id="config-partition-id"
            className="form-input"
            type="text"
            value={partitionId}
            onChange={(e) => setPartitionId(e.target.value)}
            placeholder="e.g., 0"
            disabled={isRunning}
            maxLength={10}
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="config-model-type">
            Model Architecture
          </label>
          <select
            id="config-model-type"
            className="form-input"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={isRunning}
          >
            <option value="CNN">CNN</option>
            <option value="OPT-125M">OPT-125M</option>
            <option value="Transformer">Transformer</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="config-dataset-path">
            Local Dataset Path
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
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
          <span className="error-icon">⚠</span>
          {validationError}
        </div>
      )}

      {/* Action Buttons */}
      <div className="action-buttons">
        {!isRunning ? (
          <button
            id="start-training-button"
            className="btn btn-primary btn-full"
            onClick={handleStart}
            type="button"
          >
            <span className="btn-icon">▶</span>
            Start Training
          </button>
        ) : (
          <button
            id="stop-training-button"
            className="btn btn-danger btn-full"
            onClick={onStop}
            type="button"
          >
            <span className="btn-icon">■</span>
            Stop Training
          </button>
        )}
      </div>
    </div>
  );
};

export default HardwareSelector;
