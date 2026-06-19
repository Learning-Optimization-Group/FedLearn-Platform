// =============================================================================
// FedLearn Desktop — Main Application Component
// =============================================================================

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Network, Settings } from 'lucide-react';
import AuthModal from './components/AuthModal';
import HardwareSelector from './components/HardwareSelector';
import LogPanel from './components/LogPanel';
import StatusIndicator from './components/StatusIndicator';
import SettingsModal from './components/SettingsModal';
import UpdateBanner from './components/UpdateBanner';
import ModelPlayground from './components/ModelPlayground';
import type { InferableModel, InferenceResult } from './inference.types';
import type { ClientProject, ProjectConnection } from './client.types';
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
        modelType: string;
        datasetPath: string;
      }) => Promise<{ success: boolean; error?: string }>;
      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (username: string, password: string) => Promise<{ success: boolean }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean }>;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
      listTrainableProjects: () => Promise<{ success: boolean; projects?: ClientProject[]; error?: string }>;
      getProjectConnection: (
        projectId: string,
      ) => Promise<{ success: boolean; connection?: ProjectConnection; error?: string }>;
      setServerUrl: (url: string) => Promise<{ success: boolean; url?: string; error?: string }>;
      getServerUrl: () => Promise<{ success: boolean; url?: string }>;
      selectDatasetPath: () => Promise<{ success: boolean; path?: string; error?: string }>;
      listModels: () => Promise<{ success: boolean; models?: InferableModel[]; error?: string }>;
      runInference: (
        projectId: string,
        payload: { imageBase64?: string; values?: number[] },
      ) => Promise<{ success: boolean; result?: InferenceResult; error?: string }>;
      detectHardware: () => Promise<{
        success: boolean;
        detection?: {
          platform: string;
          arch: string;
          recommendedProfile: string;
          nativeBundleAvailable: boolean;
          cudaAvailable: boolean;
          cudaInfo?: string;
        };
        error?: string;
      }>;
      // Auto-updater
      onUpdateAvailable: (callback: (info: any) => void) => void;
      onUpdateProgress: (callback: (progress: { percent: number; bytesPerSecond: number; transferred: number; total: number }) => void) => void;
      onUpdateDownloaded: (callback: (info: any) => void) => void;
      onUpdateNotAvailable: (callback: () => void) => void;
      onUpdateError: (callback: (message: string) => void) => void;
      installUpdate: () => Promise<{ success: boolean; error?: string }>;
      checkForUpdates: () => Promise<{ success: boolean; error?: string }>;
    };
  }
}

export type ContainerStatus = 'idle' | 'pulling' | 'running' | 'completed' | 'error' | 'restarting' | 'paused' | 'stopped';

// Cap log buffer at 10K lines to prevent unbounded memory growth during long training runs.
// When exceeded, the oldest lines are dropped.
const MAX_LOG_LINES = 10_000;

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [view, setView] = useState<'train' | 'use'>('train');

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

  // Subscribe to container logs — batch rapid IPC events into single state updates
  // to prevent per-line re-renders that thrash layout during fast Docker output.
  const logBufferRef = useRef<string[]>([]);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isAuthenticated) return;

    window.fedLearnAPI.onTrainingLog((logLine: string) => {
      logBufferRef.current.push(logLine);

      if (rafIdRef.current === null) {
        rafIdRef.current = requestAnimationFrame(() => {
          const batch = logBufferRef.current;
          logBufferRef.current = [];
          rafIdRef.current = null;
          setLogs((prev) => {
            const merged = [...prev, ...batch];
            if (merged.length > MAX_LOG_LINES) {
              return merged.slice(merged.length - MAX_LOG_LINES);
            }
            return merged;
          });
        });
      }
    });

    return () => {
      window.fedLearnAPI.removeTrainingLogListener();
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
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
      modelType: string;
      datasetPath: string;
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
            <span className="logo-icon"><Network strokeWidth={1.5} size={20} /></span>
            <h1 className="app-title">FedLearn Desktop</h1>
          </div>
          <StatusIndicator status={containerStatus} />
          <nav className="view-tabs" role="tablist" aria-label="View">
            <button
              role="tab"
              aria-selected={view === 'train'}
              className={`view-tab ${view === 'train' ? 'view-tab-active' : ''}`}
              onClick={() => setView('train')}
            >
              Train
            </button>
            <button
              role="tab"
              aria-selected={view === 'use'}
              className={`view-tab ${view === 'use' ? 'view-tab-active' : ''}`}
              onClick={() => setView('use')}
            >
              Use a model
            </button>
          </nav>
        </div>
        <div className="header-right header-actions">
          <button className="btn btn-ghost" onClick={() => setShowSettings(true)} id="settings-button">
            <span><Settings strokeWidth={1.5} size={16} /> Settings</span>
          </button>
          <button className="btn btn-ghost" onClick={handleLogout} id="logout-button">
            Sign out
          </button>
        </div>
      </header>

      {/* Auto-Update Banner */}
      <UpdateBanner />

      {/* Main Content */}
      <main className="app-main">
        {view === 'train' ? (
          <div className="main-grid">
            {/* Left Panel: Configuration */}
            <section className="panel config-panel">
              <div className="panel-header">
                <h2 className="panel-title">Set up training</h2>
                <span className="panel-badge">This device</span>
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
                <h2 className="panel-title">Activity log</h2>
                <span className="log-count">{logs.length} lines</span>
              </div>
              <LogPanel logs={logs} />
            </section>
          </div>
        ) : (
          <ModelPlayground />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <span className="footer-text">FedLearn — Train AI together. Share nothing.</span>
        <span className="footer-version">v1.0.0</span>
      </footer>

      {/* Settings Modal Layer */}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  );
};

export default App;
