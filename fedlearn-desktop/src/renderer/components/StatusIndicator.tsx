// =============================================================================
// FedLearn Desktop — StatusIndicator Component
// =============================================================================
// Visual status badge showing the current Docker container state.
// Animated pulse for active states (running, pulling).
// =============================================================================

import React from 'react';
import { ContainerStatus } from '../App';

interface StatusIndicatorProps {
  status: ContainerStatus;
}

const STATUS_CONFIG: Record<
  ContainerStatus,
  { label: string; colorClass: string; animate: boolean }
> = {
  idle: { label: 'Idle', colorClass: 'status-idle', animate: false },
  pulling: { label: 'Pulling image', colorClass: 'status-pulling', animate: true },
  running: { label: 'Training', colorClass: 'status-running', animate: true },
  completed: { label: 'Completed', colorClass: 'status-completed', animate: false },
  error: { label: 'Error', colorClass: 'status-error', animate: false },
  restarting: { label: 'Restarting', colorClass: 'status-pulling', animate: true },
  paused: { label: 'Paused', colorClass: 'status-idle', animate: false },
  stopped: { label: 'Stopped', colorClass: 'status-idle', animate: false },
};

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status }) => {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.idle;

  return (
    <div className={`status-indicator ${config.colorClass}`} id="status-indicator">
      <span className={`status-dot ${config.animate ? 'status-dot-pulse' : ''}`} />
      <span className="status-label">{config.label}</span>
    </div>
  );
};

export default StatusIndicator;
