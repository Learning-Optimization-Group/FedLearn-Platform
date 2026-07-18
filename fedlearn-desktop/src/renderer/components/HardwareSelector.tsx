// =============================================================================
// FedLearn Desktop — Hardware profile picker
// =============================================================================
// The hardware-profile card grid, as a controlled component. The old composite
// "HardwareSelector" (profile cards + project picker + dataset row + start/stop
// buttons in one form) was superseded by TrainSection's guided setup flow;
// TrainSection embeds this picker under its Advanced disclosure and owns
// detection/preselection, the project list, and the start flow itself.
// =============================================================================

import React from 'react';
import { MonitorCog, CircuitBoard, Command, Cpu } from 'lucide-react';

export interface HardwareProfileOption {
  id: string;
  label: string;
  description: string;
  icon: React.ComponentType<{ strokeWidth?: number | string; size?: number | string }>;
  dockerConfig: string;
}

// dockerConfig states how the profile actually executes. Only Jetson runs under
// Docker (direct /dev device mounts); every other profile runs the bundled
// native client — see DockerService.startTraining, where hardwareProfile is the
// sole dispatcher.
export const HARDWARE_PROFILES: HardwareProfileOption[] = [
  {
    id: 'discrete',
    label: 'Discrete GPU',
    description: 'NVIDIA workstation with a dedicated PCIe GPU (CUDA). Runs the bundled native client.',
    icon: MonitorCog,
    dockerConfig: 'Native process (bundled client)',
  },
  {
    id: 'jetson',
    label: 'Jetson SoC',
    description: 'NVIDIA Jetson edge device with an integrated Tegra GPU. Runs in a Docker container with direct /dev device mounts.',
    icon: CircuitBoard,
    dockerConfig: 'Docker container (direct /dev device mounts)',
  },
  {
    id: 'mps',
    label: 'Apple Silicon',
    description: 'Mac M1/M2/M3/M4 with Metal (MPS) acceleration. Runs the bundled native client.',
    icon: Command,
    dockerConfig: 'Native process (bundled client)',
  },
  {
    id: 'cpu',
    label: 'CPU Only',
    description: 'Standard CPU training without GPU acceleration. Runs the bundled native client.',
    icon: Cpu,
    dockerConfig: 'Native process (bundled client)',
  },
];

export interface HardwareProfilePickerProps {
  /** Currently selected profile id (one of HARDWARE_PROFILES[].id). */
  value: string;
  onChange: (profileId: string) => void;
  disabled?: boolean;
}

/**
 * Controlled hardware-profile card grid. Pure presentation — detection and
 * preselection live in the consumer (TrainSection detects once on mount and
 * preselects the recommended profile).
 */
export const HardwareProfilePicker: React.FC<HardwareProfilePickerProps> = ({
  value,
  onChange,
  disabled = false,
}) => (
  <div className="profile-cards">
    {HARDWARE_PROFILES.map((profile) => (
      <button
        key={profile.id}
        id={`profile-${profile.id}`}
        className={`profile-card ${value === profile.id ? 'profile-card-active' : ''}`}
        onClick={() => onChange(profile.id)}
        disabled={disabled}
        type="button"
        aria-pressed={value === profile.id}
      >
        <span className="profile-icon"><profile.icon strokeWidth={1.5} size={20} /></span>
        <span className="profile-label">{profile.label}</span>
        <span className="profile-desc">{profile.description}</span>
        <span className="profile-docker">{profile.dockerConfig}</span>
      </button>
    ))}
  </div>
);

export default HardwareProfilePicker;
