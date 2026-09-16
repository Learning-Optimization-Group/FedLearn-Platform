// =============================================================================
// FedLearn Desktop — UpdateBanner Component
// =============================================================================
// Shows a persistent "Check for Updates" button in idle state.
// When an update is found/downloading/ready, the banner expands with status.
//
// Ledger design system: all color/radius/space/motion comes from token vars
// via shared classes in styles.css. No inline hex/rgb, no gradients, no glow.
// State semantics: available/checking/downloading→accent, ready/upToDate→
// success, error→danger.
// =============================================================================

import React, { useState, useEffect, useRef } from 'react';
import { CheckCircle2, AlertTriangle, RefreshCw, Download, Bell, X } from 'lucide-react';

type UpdateState = 'idle' | 'checking' | 'upToDate' | 'available' | 'downloading' | 'ready' | 'error';

interface UpdateInfo {
  version: string;
}

interface DownloadProgress {
  percent: number;
  bytesPerSecond: number;
  transferred: number;
  total: number;
}

const UpdateBanner: React.FC = () => {
  const [updateState, setUpdateState] = useState<UpdateState>('idle');
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [dismissed, setDismissed] = useState(false);
  const checkingRef = useRef(false);

  useEffect(() => {
    // Passive listeners — fired by the main process when autoUpdater emits events
    window.fedLearnAPI.onUpdateAvailable((info: UpdateInfo) => {
      setUpdateInfo(info);
      setUpdateState('available');
      setDismissed(false);
      checkingRef.current = false;
    });

    window.fedLearnAPI.onUpdateProgress((prog) => {
      setProgress(prog);
      setUpdateState('downloading');
    });

    window.fedLearnAPI.onUpdateDownloaded((info: UpdateInfo) => {
      setUpdateInfo(info);
      setUpdateState('ready');
      setProgress(null);
    });

    window.fedLearnAPI.onUpdateNotAvailable(() => {
      checkingRef.current = false;
      setUpdateState('upToDate');
      // Auto-dismiss the "up to date" notice after 4 s
      setTimeout(() => setUpdateState('idle'), 4000);
    });

    window.fedLearnAPI.onUpdateError((msg: string) => {
      checkingRef.current = false;
      setErrorMsg(msg);
      setUpdateState('error');
    });
  }, []);

  const handleCheck = async () => {
    if (checkingRef.current) return;
    checkingRef.current = true;
    setUpdateState('checking');
    setErrorMsg('');
    const result = await window.fedLearnAPI.checkForUpdates();
    if (!result.success) {
      checkingRef.current = false;
      setErrorMsg(result.error ?? 'Unknown error');
      setUpdateState('error');
    }
    // On success we wait for the updater events above to drive the state
  };

  const handleInstall = async () => {
    await window.fedLearnAPI.installUpdate();
  };

  // ── Idle state: just a small inline "Check for updates" button ──────────────
  if (updateState === 'idle') {
    return (
      <div className="update-check-row">
        <button
          id="check-for-updates-button"
          className="btn btn-ghost btn-banner"
          onClick={handleCheck}
        >
          <RefreshCw strokeWidth={1.5} size={16} /> Check for Updates
        </button>
      </div>
    );
  }

  // ── Dismissed active banners: nothing rendered ───────────────────────────────
  if (dismissed && (updateState === 'available' || updateState === 'error')) {
    return null;
  }

  // ── Active banner ────────────────────────────────────────────────────────────
  const isReady = updateState === 'ready';
  const isError = updateState === 'error';
  const isUpToDate = updateState === 'upToDate';
  const bannerClass = isReady || isUpToDate
    ? 'update-banner update-banner-success'
    : isError
    ? 'update-banner update-banner-error'
    : 'update-banner';

  return (
    <div className={bannerClass} role="status">
      {/* Icon */}
      <span className="update-banner-icon">
        {updateState === 'ready'
          ? <CheckCircle2 strokeWidth={1.5} size={16} />
          : updateState === 'downloading'
          ? <Download strokeWidth={1.5} size={16} />
          : updateState === 'upToDate'
          ? <CheckCircle2 strokeWidth={1.5} size={16} />
          : updateState === 'error'
          ? <AlertTriangle strokeWidth={1.5} size={16} />
          : updateState === 'checking'
          ? <RefreshCw strokeWidth={1.5} size={16} />
          : <Bell strokeWidth={1.5} size={16} />}
      </span>

      {/* Message */}
      <div className="update-banner-message">
        {updateState === 'checking' && (
          <span className="update-banner-muted">Checking for updates…</span>
        )}
        {updateState === 'upToDate' && (
          <span className="update-banner-success-text">You're up to date!</span>
        )}
        {updateState === 'available' && (
          <span>
            <strong>Update available:</strong> v{updateInfo?.version} — downloading in background…
          </span>
        )}
        {updateState === 'downloading' && progress && (
          <div>
            <span>
              Downloading update v{updateInfo?.version}…{' '}
              <strong className="update-banner-percent">{progress.percent.toFixed(1)}%</strong>
              {'  '}
              <span className="update-banner-rate">
                ({(progress.bytesPerSecond / 1024).toFixed(0)} KB/s)
              </span>
            </span>
            <div className="update-progress-track">
              <div
                className="update-progress-fill"
                style={{ width: `${progress.percent}%` }}
              />
            </div>
          </div>
        )}
        {updateState === 'ready' && (
          <span>
            <strong>Update v{updateInfo?.version} ready</strong> — restart the app to install.
          </span>
        )}
        {updateState === 'error' && (
          <span className="update-banner-error-text">
            <strong>Update check failed:</strong>{' '}
            {errorMsg.length > 80 ? errorMsg.slice(0, 80) + '…' : errorMsg}
          </span>
        )}
      </div>

      {/* Actions */}
      <div className="update-banner-actions">
        {updateState === 'ready' && (
          <button
            id="update-restart-button"
            className="btn btn-primary btn-banner"
            onClick={handleInstall}
          >
            Restart &amp; Install
          </button>
        )}

        {updateState === 'error' && (
          <button
            id="update-retry-button"
            className="btn btn-secondary btn-banner"
            onClick={handleCheck}
          >
            Retry
          </button>
        )}

        {/* Dismiss — only rendered when dismissing is actually possible.
            The "ready" state is not dismissible (a restart is pending), so the
            control is hidden there rather than shown disabled. */}
        {(updateState === 'available' || updateState === 'error') && (
          <button
            id="update-dismiss-button"
            className="btn-dismiss"
            aria-label="Dismiss notification"
            title="Dismiss"
            onClick={() => {
              setDismissed(true);
              setUpdateState('idle');
            }}
          >
            <X strokeWidth={1.5} size={16} />
          </button>
        )}
      </div>
    </div>
  );
};

export default UpdateBanner;
