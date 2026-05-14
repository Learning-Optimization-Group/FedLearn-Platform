import React, { useEffect, useState } from 'react';
import HardwareSelector from '../components/HardwareSelector';
import type { ContainerStatus } from '../App';

interface SettingsViewProps {
  containerStatus: ContainerStatus;
  onManualStartTraining: (config: {
    hardwareProfile: string; projectId: string; serverAddress: string;
    partitionId: string; modelType: string; datasetPath: string;
  }) => void;
  onManualStopTraining: () => void;
}

const SettingsView: React.FC<SettingsViewProps> = ({ containerStatus, onManualStartTraining, onManualStopTraining }) => {
  const [serverUrl, setServerUrl] = useState<string>('');
  const [serverStatus, setServerStatus] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      const r = await window.fedLearnAPI.getServerUrl();
      if (r.success && r.url) setServerUrl(r.url);
    })();
  }, []);

  const handleSaveServerUrl = async () => {
    setServerStatus('Saving…');
    const r = await window.fedLearnAPI.setServerUrl(serverUrl);
    setServerStatus(r.success ? `Saved: ${r.url}` : (r.error || 'Failed to save'));
  };

  return (
    <>
      <div className="view-header">
        <div>
          <div className="view-header__title">Settings</div>
          <div className="view-header__subtitle">Connection, hardware, and advanced options</div>
        </div>
      </div>

      <div className="settings-panel">
        <div className="settings-panel__title">Backend Server</div>
        <div className="settings-panel__desc">
          URL of the FedLearn platform API. `/api` is appended automatically.
        </div>
        <div className="settings-panel__row">
          <input
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            placeholder="http://localhost:8081"
          />
          <button className="btn-primary" onClick={handleSaveServerUrl}>Save</button>
        </div>
        {serverStatus && <div className="settings-panel__status">{serverStatus}</div>}
      </div>

      <span className="settings-advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
        {showAdvanced ? '▾' : '▸'} Advanced — Manual Connection
      </span>
      {showAdvanced && (
        <div className="settings-panel">
          <div className="settings-panel__title">Manual Connection (legacy)</div>
          <div className="settings-panel__desc">
            Enter a project UUID, gRPC server address, and partition ID directly. Use this only when
            the project picker is unavailable; this option will be removed in a future release.
          </div>
          <HardwareSelector
            onStart={onManualStartTraining}
            onStop={onManualStopTraining}
            isRunning={containerStatus === 'running' || containerStatus === 'pulling'}
          />
        </div>
      )}
    </>
  );
};

export default SettingsView;
