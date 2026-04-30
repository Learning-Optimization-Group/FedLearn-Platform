// =============================================================================
// FedLearn Desktop — UpdateBanner Component
// =============================================================================
// Shows a non-intrusive banner at the top of the dashboard when an update
// is available, downloading, or ready to install.
// =============================================================================

import React, { useState, useEffect } from 'react';

type UpdateState = 'idle' | 'available' | 'downloading' | 'ready';

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
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    window.fedLearnAPI.onUpdateAvailable((info: UpdateInfo) => {
      setUpdateInfo(info);
      setUpdateState('available');
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
  }, []);

  const handleInstall = async () => {
    await window.fedLearnAPI.installUpdate();
  };

  if (updateState === 'idle' || dismissed) {
    return null;
  }

  return (
    <div
      className="update-banner"
      role="status"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        padding: '10px 20px',
        background: updateState === 'ready'
          ? 'linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05))'
          : 'linear-gradient(90deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05))',
        borderBottom: updateState === 'ready'
          ? '1px solid rgba(16, 185, 129, 0.3)'
          : '1px solid rgba(99, 102, 241, 0.3)',
        fontSize: '13px',
        color: 'var(--color-text)',
      }}
    >
      {/* Icon */}
      <span style={{ fontSize: '16px' }}>
        {updateState === 'ready' ? '✅' : updateState === 'downloading' ? '⬇️' : '🔔'}
      </span>

      {/* Message */}
      <div style={{ flex: 1 }}>
        {updateState === 'available' && (
          <span>
            <strong>Update available:</strong> v{updateInfo?.version} — downloading in background...
          </span>
        )}
        {updateState === 'downloading' && progress && (
          <div>
            <span>Downloading update v{updateInfo?.version}... {progress.percent.toFixed(1)}%</span>
            <div
              style={{
                marginTop: '4px',
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
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '8px' }}>
        {updateState === 'ready' && (
          <button
            id="update-restart-button"
            onClick={handleInstall}
            style={{
              padding: '5px 14px',
              background: 'linear-gradient(135deg, #10b981, #059669)',
              border: 'none',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Restart & Install
          </button>
        )}
        <button
          id="update-dismiss-button"
          onClick={() => setDismissed(true)}
          style={{
            padding: '5px 10px',
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.15)',
            borderRadius: '6px',
            color: 'var(--color-text-muted)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          ✕
        </button>
      </div>
    </div>
  );
};

export default UpdateBanner;
