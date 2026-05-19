import React, { useCallback, useEffect, useRef, useState } from 'react';
import AppShell from './components/AppShell';
import AuthModal from './components/AuthModal';
import DatasetConfirmDialog from './components/DatasetConfirmDialog';
import LogDrawer from './components/LogDrawer';
import type { RouteKey } from './components/Sidebar';
import UpdateBanner from './components/UpdateBanner';
import { trainProject } from './lib/api';
import type { ClientProject } from './lib/types';
import './styles.css';
import DiscoverView from './views/DiscoverView';
import ModelsView from './views/ModelsView';
import MyProjectsView from './views/MyProjectsView';
import MyRequestsView from './views/MyRequestsView';
import SettingsView from './views/SettingsView';

declare global {
  interface Window {
    fedLearnAPI: {
      startTraining: (config: any) => Promise<{ success: boolean; error?: string }>;
      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (
        username: string,
        password: string
      ) => Promise<{ success: boolean; username?: string }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean; username?: string }>;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
      onDockerUnavailable: (callback: (message: string) => void) => void;
      setServerUrl: (url: string) => Promise<{ success: boolean; url?: string; error?: string }>;
      getServerUrl: () => Promise<{ success: boolean; url?: string }>;
      selectDatasetPath: () => Promise<{ success: boolean; path?: string; error?: string }>;
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
      listClientProjects: () => Promise<{ success: boolean; projects?: any[] }>;
      listDiscover: () => Promise<{ success: boolean; projects?: any[] }>;
      listMyRequests: () => Promise<{ success: boolean; requests?: any[] }>;
      requestAccess: (
        projectId: string,
        message: string
      ) => Promise<{ success: boolean; status?: string; error?: string }>;
      trainProject: (
        projectId: string,
        datasetPath: string
      ) => Promise<{ success: boolean; error?: string }>;
      getLastDatasetPath: (projectId: string) => Promise<{ success: boolean; path?: string }>;
      setLastDatasetPath: (projectId: string, path: string) => Promise<{ success: boolean }>;
      onUpdateAvailable: (callback: (info: any) => void) => void;
      onUpdateProgress: (callback: (progress: any) => void) => void;
      onUpdateDownloaded: (callback: (info: any) => void) => void;
      onUpdateNotAvailable: (callback: () => void) => void;
      onUpdateError: (callback: (message: string) => void) => void;
      installUpdate: () => Promise<{ success: boolean; error?: string }>;
      checkForUpdates: () => Promise<{ success: boolean; error?: string }>;
    };
  }
}

export type ContainerStatus =
  | 'idle'
  | 'pulling'
  | 'running'
  | 'completed'
  | 'error'
  | 'restarting'
  | 'paused'
  | 'stopped';

const MAX_LOG_LINES = 10_000;

function ensureFedLearnBridge(): Window['fedLearnAPI'] {
  if (window.fedLearnAPI) {
    return window.fedLearnAPI;
  }

  // Browser preview mode (webpack dev server) does not have Electron preload.
  // Provide a no-op bridge so the redesigned UI can still be tested safely.
  const fallback: Window['fedLearnAPI'] = {
    startTraining: async () => ({
      success: false,
      error: 'Electron bridge unavailable in browser preview mode.',
    }),
    stopTraining: async () => ({
      success: false,
      error: 'Electron bridge unavailable in browser preview mode.',
    }),
    getDockerStatus: async () => ({ success: true, status: 'idle' }),
    login: async (requestedUsername: string) => ({
      success: true,
      username: requestedUsername || 'preview-user',
    }),
    logout: async () => ({ success: true }),
    checkAuth: async () => ({ success: true, authenticated: true, username: 'preview-user' }),
    onTrainingLog: () => {},
    removeTrainingLogListener: () => {},
    onDockerUnavailable: (callback: (message: string) => void) => {
      callback('Desktop backend bridge is unavailable in browser preview mode.');
    },
    setServerUrl: async (url: string) => ({ success: true, url }),
    getServerUrl: async () => ({ success: true, url: 'http://localhost:8081/api' }),
    selectDatasetPath: async () => ({
      success: false,
      error: 'Directory picker only works in Electron.',
    }),
    detectHardware: async () => ({
      success: true,
      detection: {
        platform: 'browser-preview',
        arch: 'web',
        recommendedProfile: 'cpu',
        nativeBundleAvailable: false,
        cudaAvailable: false,
      },
    }),
    listClientProjects: async () => ({ success: true, projects: [] }),
    listDiscover: async () => ({ success: true, projects: [] }),
    listMyRequests: async () => ({ success: true, requests: [] }),
    requestAccess: async () => ({ success: true, status: 'PENDING' }),
    trainProject: async () => ({
      success: false,
      error: 'Training can only be launched from Electron.',
    }),
    getLastDatasetPath: async () => ({ success: true, path: '' }),
    setLastDatasetPath: async () => ({ success: true }),
    onUpdateAvailable: () => {},
    onUpdateProgress: () => {},
    onUpdateDownloaded: () => {},
    onUpdateNotAvailable: () => {},
    onUpdateError: () => {},
    installUpdate: async () => ({
      success: false,
      error: 'Updater is only available in Electron.',
    }),
    checkForUpdates: async () => ({
      success: false,
      error: 'Updater is only available in Electron.',
    }),
  };

  window.fedLearnAPI = fallback;
  return fallback;
}

const App: React.FC = () => {
  const fedLearnAPI = ensureFedLearnBridge();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [dockerWarning, setDockerWarning] = useState<string | null>(null);
  const [trainingProjectId, setTrainingProjectId] = useState<string | null>(null);
  const [pendingTrain, setPendingTrain] = useState<ClientProject | null>(null);
  const [autoOpenLogs, setAutoOpenLogs] = useState<boolean>(false);
  const [hardwareLabel, setHardwareLabel] = useState<string>('Detecting…');
  const [username, setUsername] = useState<string>('');

  useEffect(() => {
    fedLearnAPI.onDockerUnavailable((msg: string) => {
      setDockerWarning(`Docker is not running: ${msg}`);
    });
  }, [fedLearnAPI]);

  useEffect(() => {
    (async () => {
      try {
        const r = await fedLearnAPI.checkAuth();
        setIsAuthenticated(r.authenticated === true);
        setUsername(r.authenticated ? r.username || '' : '');
      } catch {
        setIsAuthenticated(false);
        setUsername('');
      } finally {
        setIsAuthChecking(false);
      }
    })();
  }, [fedLearnAPI]);

  useEffect(() => {
    if (!isAuthenticated) return;
    (async () => {
      const r = await fedLearnAPI.detectHardware();
      if (r.success && r.detection) {
        const { recommendedProfile, platform } = r.detection;
        setHardwareLabel(`${recommendedProfile.toUpperCase()} · ${platform}`);
      } else {
        setHardwareLabel('Unknown');
      }
    })();
  }, [fedLearnAPI, isAuthenticated]);

  const logBufferRef = useRef<string[]>([]);
  const rafIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isAuthenticated) return;
    fedLearnAPI.onTrainingLog((logLine: string) => {
      logBufferRef.current.push(logLine);
      if (rafIdRef.current === null) {
        rafIdRef.current = requestAnimationFrame(() => {
          const batch = logBufferRef.current;
          logBufferRef.current = [];
          rafIdRef.current = null;
          setLogs((prev) => {
            const merged = [...prev, ...batch];
            return merged.length > MAX_LOG_LINES ? merged.slice(-MAX_LOG_LINES) : merged;
          });
        });
      }
    });
    return () => {
      fedLearnAPI.removeTrainingLogListener();
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [fedLearnAPI, isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated) return;
    const poll = async () => {
      try {
        const r = await fedLearnAPI.getDockerStatus();
        if (r.success && r.status) {
          setContainerStatus(r.status as ContainerStatus);
          if (r.status !== 'running' && r.status !== 'pulling') {
            setTrainingProjectId(null);
          }
        }
      } catch {
        /* silent */
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [fedLearnAPI, isAuthenticated]);

  const handleLogin = useCallback((nextUsername: string) => {
    setIsAuthenticated(true);
    setUsername(nextUsername);
  }, []);

  const handleLogout = useCallback(async () => {
    await fedLearnAPI.logout();
    setIsAuthenticated(false);
    setLogs([]);
    setContainerStatus('idle');
    setTrainingProjectId(null);
    setUsername('');
  }, [fedLearnAPI]);

  const handleTrainClick = (project: ClientProject) => {
    setPendingTrain(project);
  };

  const handleConfirmTrain = async (datasetPath: string) => {
    if (!pendingTrain) return;
    const project = pendingTrain;
    setPendingTrain(null);
    setLogs([]);
    setAutoOpenLogs(true);
    setTrainingProjectId(project.projectId);
    setContainerStatus('pulling');
    setLogs((prev) => [...prev, `[System] Connecting to ${project.name}...`]);
    const r = await trainProject(project.projectId, datasetPath);
    if (!r.success) {
      setContainerStatus('error');
      setTrainingProjectId(null);
      setLogs((prev) => [...prev, `[System] Failed to start: ${r.error || 'unknown error'}`]);
    }
  };

  const handleManualStart = useCallback(
    async (config: {
      hardwareProfile: string;
      projectId: string;
      serverAddress: string;
      partitionId: string;
      modelType: string;
      datasetPath: string;
    }) => {
      setLogs([]);
      setAutoOpenLogs(true);
      setContainerStatus('pulling');
      setLogs((prev) => [...prev, '[System] Starting manual training container...']);
      const r = await fedLearnAPI.startTraining(config);
      if (!r.success) {
        setContainerStatus('error');
        setLogs((prev) => [...prev, `[System] Failed to start: ${r.error || 'unknown error'}`]);
      }
    },
    []
  );

  const handleManualStop = useCallback(async () => {
    setLogs((prev) => [...prev, '[System] Stopping training container...']);
    const r = await fedLearnAPI.stopTraining();
    if (r.success) {
      setContainerStatus('idle');
      setLogs((prev) => [...prev, '[System] Container stopped.']);
    } else {
      setLogs((prev) => [...prev, `[System] Failed to stop: ${r.error || 'unknown error'}`]);
    }
  }, []);

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

  if (!isAuthenticated) {
    return (
      <div className="app-container">
        <AuthModal onLoginSuccess={handleLogin} />
      </div>
    );
  }

  const renderView = (route: RouteKey) => {
    switch (route) {
      case 'projects':
        return <MyProjectsView onTrain={handleTrainClick} trainingProjectId={trainingProjectId} />;
      case 'discover':
        return <DiscoverView />;
      case 'requests':
        return <MyRequestsView />;
      case 'models':
        return <ModelsView />;
      case 'settings':
        return (
          <SettingsView
            containerStatus={containerStatus}
            onManualStartTraining={handleManualStart}
            onManualStopTraining={handleManualStop}
          />
        );
    }
  };

  return (
    <>
      <UpdateBanner />
      {dockerWarning && (
        <div className="docker-warning" role="alert">
          <span className="error-icon">⚠</span>
          <span>{dockerWarning}</span>
          <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Start Docker Desktop and restart the app.
          </span>
        </div>
      )}
      <AppShell
        username={username}
        status={containerStatus}
        hardwareLabel={hardwareLabel}
        onLogout={handleLogout}
        renderView={renderView}
        drawer={<LogDrawer logs={logs} autoOpen={autoOpenLogs} />}
      />
      {pendingTrain && (
        <DatasetConfirmDialog
          project={pendingTrain}
          onCancel={() => setPendingTrain(null)}
          onConfirm={handleConfirmTrain}
        />
      )}
    </>
  );
};

export default App;
