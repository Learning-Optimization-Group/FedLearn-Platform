// =============================================================================
// FedLearn Desktop — StatusBar Component
// =============================================================================
// Persistent full-width bottom strip rendered by the app shell OUTSIDE the
// section outlet, so it survives Train / Models / Settings switches. Shows:
//   - backend connection (dot + host, parsed from the configured server URL)
//   - the detected hardware profile chip (same detection the Train flow uses)
//   - the run state: StatusIndicator badge (unchanged states + #status-indicator
//     hook) plus "<project> · 12:34 elapsed" with a live 1s timer while a run
//     is active
//   - the app version, right-aligned
// All data is App-owned and arrives via props; the only local state is the
// elapsed-timer tick. The run start time is anchored by App when the user
// presses Start — never derived from the 3s status poll.
// Ledger design system: token-driven classes from styles.css only.
// =============================================================================

import React, { useState, useEffect } from 'react';
import StatusIndicator from './StatusIndicator';
import { formatElapsed } from './trainFlow';
import type { ContainerStatus } from '../App';

export interface ActiveRun {
  /** Human label for the running project (name when resolvable, else model type). */
  projectLabel: string;
  /** Epoch ms captured when the user pressed Start. */
  startedAt: number;
}

interface StatusBarProps {
  containerStatus: ContainerStatus;
  /** Backend host (e.g. "localhost:8081"); empty while unresolved. */
  serverHost: string;
  /** Detected hardware profile label (e.g. "Apple Silicon"); empty while detecting. */
  hardwareLabel: string;
  /** The run whose elapsed time is shown while containerStatus is active. */
  activeRun: ActiveRun | null;
  appVersion: string;
}

const StatusBar: React.FC<StatusBarProps> = ({
  containerStatus,
  serverHost,
  hardwareLabel,
  activeRun,
  appVersion,
}) => {
  const isActive = containerStatus === 'running' || containerStatus === 'pulling';
  const [now, setNow] = useState(() => Date.now());

  // 1s tick, running only while a training run is active.
  useEffect(() => {
    if (!isActive || !activeRun) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [isActive, activeRun]);

  return (
    <footer className="statusbar" id="status-bar">
      <div className="statusbar-left">
        <span
          className="statusbar-item"
          title={serverHost ? `Connected to ${serverHost}` : 'Server not resolved yet'}
        >
          <span
            className={`statusbar-dot ${serverHost ? 'statusbar-dot-ok' : ''}`}
            aria-hidden="true"
          />
          <span className="statusbar-host">{serverHost || 'Not connected'}</span>
        </span>
        {hardwareLabel && <span className="statusbar-chip">{hardwareLabel}</span>}
        <span className="statusbar-item">
          <StatusIndicator status={containerStatus} />
          {isActive && activeRun && (
            <span className="statusbar-run-detail">
              {activeRun.projectLabel} · {formatElapsed(now - activeRun.startedAt)} elapsed
            </span>
          )}
        </span>
      </div>
      <span className="statusbar-version">v{appVersion}</span>
    </footer>
  );
};

export default StatusBar;
