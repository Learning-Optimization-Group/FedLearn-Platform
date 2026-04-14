// =============================================================================
// FedLearn Desktop — Main Application Component
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import AuthModal from './components/AuthModal';
import HardwareSelector from './components/HardwareSelector';
import LogPanel from './components/LogPanel';
import StatusIndicator from './components/StatusIndicator';
import './styles.css';

// Type declaration for the secure preload API
declare global {
  interface Window {
    fedLearnAPI: {
      startTraining: (config: {
        hardwareProfile: string;
        projectId: string;
        serverAddress: string;
        partitionId: string;
      }) => Promise<{ success: boolean; error?: string }>;
      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (username: string, password: string) => Promise<{ success: boolean }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean }>;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
    };
  }
}

export type ContainerStatus = 'idle' | 'pulling' | 'running' | 'completed' | 'error' | 'restarting' | 'paused' | 'stopped';

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
  const [logs, setLogs] = useState<string[]>([]);

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      try {
        const result = await window.fedLearnAPI.checkAuth();
        setIsAuthenticated(result.authenticated === true);
      } catch {
        setIsAuthenticated(false);
      } finally {
        setIsAuthChecking(false);
      }
    };
    checkAuth();
  }, []);

  // Subscribe to container logs
  useEffect(() => {
    if (!isAuthenticated) return;

    window.fedLearnAPI.onTrainingLog((logLine: string) => {
      setLogs((prev) => [...prev, logLine]);
    });

    return () => {
      window.fedLearnAPI.removeTrainingLogListener();
    };
  }, [isAuthenticated]);

  // Poll container status
  useEffect(() => {
    if (!isAuthenticated) return;

    const pollStatus = async () => {
      try {
        const result = await window.fedLearnAPI.getDockerStatus();
        if (result.success && result.status) {
          setContainerStatus(result.status as ContainerStatus);
        }
      } catch {
        // Silently handle polling failures
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 3000);
    return () => clearInterval(interval);
  }, [isAuthenticated]);

  const handleLogin = useCallback(() => {
    setIsAuthenticated(true);
  }, []);

  const handleLogout = useCallback(async () => {
    await window.fedLearnAPI.logout();
    setIsAuthenticated(false);
    setLogs([]);
    setContainerStatus('idle');
  }, []);

  const handleStartTraining = useCallback(
    async (config: {
      hardwareProfile: string;
      projectId: string;
      serverAddress: string;
      partitionId: string;
    }) => {
      setLogs([]);
      setContainerStatus('pulling');
      setLogs((prev) => [...prev, '[System] Starting training container...\n']);

      const result = await window.fedLearnAPI.startTraining(config);

      if (!result.success) {
        setContainerStatus('error');
        setLogs((prev) => [...prev, `[System] Failed to start: ${result.error || 'Unknown error'}\n`]);
      }
    },
    [],
  );

  const handleStopTraining = useCallback(async () => {
    setLogs((prev) => [...prev, '[System] Stopping training container...\n']);
    const result = await window.fedLearnAPI.stopTraining();

    if (result.success) {
      setContainerStatus('idle');
      setLogs((prev) => [...prev, '[System] Container stopped.\n']);
    } else {
      setLogs((prev) => [...prev, `[System] Failed to stop: ${result.error || 'Unknown error'}\n`]);
    }
  }, []);

  // Loading state
  if (isAuthChecking) {
    return (
      <div className="app-container">
        <div className="loading-screen">
          <div className="loading-spinner" />
          <p className="loading-text">Initializing FedLearn Desktop...</p>
        </div>
      </div>
    );
  }

  // Auth gate
  if (!isAuthenticated) {
    return (
      <div className="app-container">
        <AuthModal onLoginSuccess={handleLogin} />
      </div>
    );
  }

  // Main dashboard
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="app-logo">
            <span className="logo-icon">◆</span>
            <h1 className="app-title">FedLearn Desktop</h1>
          </div>
          <StatusIndicator status={containerStatus} />
        </div>
        <div className="header-right">
          <button className="btn btn-ghost" onClick={handleLogout} id="logout-button">
            Sign Out
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="main-grid">
          {/* Left Panel: Configuration */}
          <section className="panel config-panel">
            <div className="panel-header">
              <h2 className="panel-title">Training Configuration</h2>
              <span className="panel-badge">Docker Orchestration</span>
            </div>
            <HardwareSelector
              onStart={handleStartTraining}
              onStop={handleStopTraining}
              isRunning={containerStatus === 'running' || containerStatus === 'pulling'}
            />
          </section>

          {/* Right Panel: Logs */}
          <section className="panel log-panel-container">
            <div className="panel-header">
              <h2 className="panel-title">Container Output</h2>
              <span className="log-count">{logs.length} lines</span>
            </div>
            <LogPanel logs={logs} />
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span className="footer-text">FedLearn Platform — Privacy-Preserving Federated Learning</span>
        <span className="footer-version">v1.0.0</span>
      </footer>
    </div>
  );
};

export default App;
