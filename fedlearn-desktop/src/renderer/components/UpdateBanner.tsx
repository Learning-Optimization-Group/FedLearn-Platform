// =============================================================================
// FedLearn Desktop — UpdateBanner Component
// =============================================================================
// Shows a persistent "Check for Updates" button in idle state.
// When an update is found/downloading/ready, the banner expands with status.
// =============================================================================

import React, { useState, useEffect, useRef } from 'react';

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

const BTN_BASE: React.CSSProperties = {
  padding: '4px 12px',
  border: 'none',
  borderRadius: '6px',
  fontSize: '12px',
  fontWeight: 600,
  cursor: 'pointer',
  transition: 'opacity 0.15s ease',
};

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
      <div
        style={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          padding: '6px 20px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          minHeight: '36px',
        }}
      >
        <button
          id="check-for-updates-button"
          onClick={handleCheck}
          style={{
            ...BTN_BASE,
            background: 'rgba(99, 102, 241, 0.15)',
            color: '#a5b4fc',
            border: '1px solid rgba(99, 102, 241, 0.3)',
          }}
        >
          🔄 Check for Updates
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
  const bannerBg = isReady
    ? 'linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05))'
    : isError
    ? 'linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05))'
    : 'linear-gradient(90deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05))';
  const bannerBorder = isReady
    ? '1px solid rgba(16, 185, 129, 0.3)'
    : isError
    ? '1px solid rgba(239, 68, 68, 0.3)'
    : '1px solid rgba(99, 102, 241, 0.3)';

  return (
    <div
      className="update-banner"
      role="status"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '10px 20px',
        background: bannerBg,
        borderBottom: bannerBorder,
        fontSize: '13px',
        color: 'var(--color-text)',
        minHeight: '44px',
      }}
    >
      {/* Icon */}
      <span style={{ fontSize: '16px', flexShrink: 0 }}>
        {updateState === 'ready'
          ? '✅'
          : updateState === 'downloading'
          ? '⬇️'
          : updateState === 'upToDate'
          ? '✅'
          : updateState === 'error'
          ? '⚠️'
          : updateState === 'checking'
          ? '🔄'
          : '🔔'}
      </span>

      {/* Message */}
      <div style={{ flex: 1 }}>
        {updateState === 'checking' && (
          <span style={{ color: 'var(--color-text-muted)' }}>Checking for updates…</span>
        )}
        {updateState === 'upToDate' && (
          <span style={{ color: '#10b981', fontWeight: 600 }}>You're up to date!</span>
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
              <strong>{progress.percent.toFixed(1)}%</strong>
              {'  '}
              <span style={{ color: 'var(--color-text-muted)', fontSize: '11px' }}>
                ({(progress.bytesPerSecond / 1024).toFixed(0)} KB/s)
              </span>
            </span>
            <div
              style={{
                marginTop: '5px',
                height: '3px',
                borderRadius: '2px',
                background: 'rgba(255,255,255,0.1)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${progress.percent}%`,
                  background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
                  transition: 'width 0.3s ease',
                  borderRadius: '2px',
                }}
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
          <span style={{ color: '#fca5a5' }}>
            <strong>Update check failed:</strong>{' '}
            {errorMsg.length > 80 ? errorMsg.slice(0, 80) + '…' : errorMsg}
          </span>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
        {updateState === 'ready' && (
          <button
            id="update-restart-button"
            onClick={handleInstall}
            style={{
              ...BTN_BASE,
              background: 'linear-gradient(135deg, #10b981, #059669)',
              color: '#fff',
            }}
          >
            Restart &amp; Install
          </button>
        )}

        {updateState === 'error' && (
          <button
            id="update-retry-button"
            onClick={handleCheck}
            style={{
              ...BTN_BASE,
              background: 'rgba(239, 68, 68, 0.2)',
              color: '#fca5a5',
              border: '1px solid rgba(239, 68, 68, 0.3)',
            }}
          >
            Retry
          </button>
        )}

        {/* Dismiss — shown for available / error / ready */}
        {(updateState === 'available' || updateState === 'error' || updateState === 'ready') && (
          <button
            id="update-dismiss-button"
            onClick={() => {
              if (updateState === 'ready') return; // don't allow dismissing "ready"
              setDismissed(true);
              setUpdateState('idle');
            }}
            title={updateState === 'ready' ? 'Please restart to apply the update' : 'Dismiss'}
            style={{
              ...BTN_BASE,
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.15)',
              color: 'var(--color-text-muted)',
              opacity: updateState === 'ready' ? 0.4 : 1,
              cursor: updateState === 'ready' ? 'not-allowed' : 'pointer',
            }}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
};

export default UpdateBanner;
