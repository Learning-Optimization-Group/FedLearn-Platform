// =============================================================================
// FedLearn Desktop — Main Application Component
// =============================================================================
// Shell layout (authenticated): a top drag strip (hiddenInset traffic-light
// spacing), a 64px left icon rail (Train / Models / Settings + account item),
// a section outlet, and a persistent bottom StatusBar. Sections mount once and
// toggle visibility with CSS so ModelPlayground's inference stream/chat state
// and the training log buffer survive section switches. UpdateBanner stays
// mounted at shell level forever (its preload listeners have no removal API)
// and overlays the outlet as a top layer regardless of section.
// =============================================================================

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, Boxes, LogOut, Network, Settings, User } from 'lucide-react';
import AuthModal from './components/AuthModal';
import UpdateBanner from './components/UpdateBanner';
import ModelPlayground from './components/ModelPlayground';
import StatusBar from './components/StatusBar';
import type { ActiveRun } from './components/StatusBar';
import { TrainSection } from './components/TrainSection';
import { SettingsSection } from './components/SettingsSection';
import type { InferableModel, InferenceResult } from './inference.types';
import type { ClientProject, ProjectConnection } from './client.types';
import type { UpdateInfo, ProgressInfo } from 'electron-updater';
import './styles.css';

// Injected at build time by webpack DefinePlugin (reads `version` from package.json).
// See webpack.renderer.config.js and webpack.prod.config.js.
declare const __APP_VERSION__: string;

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
        connectionToken?: string;
        strategy?: string;
        trainingArm?: string;
      }) => Promise<{ success: boolean; error?: string }>;
      stopTraining: () => Promise<{ success: boolean; error?: string }>;
      getDockerStatus: () => Promise<{ success: boolean; status?: string }>;
      login: (username: string, password: string) => Promise<{ success: boolean }>;
      logout: () => Promise<{ success: boolean }>;
      checkAuth: () => Promise<{ success: boolean; authenticated?: boolean }>;
      onSessionExpired: (callback: () => void) => void;
      removeSessionExpiredListener: () => void;
      onTrainingLog: (callback: (logLine: string) => void) => void;
      removeTrainingLogListener: () => void;
      listTrainableProjects: () => Promise<{ success: boolean; projects?: ClientProject[]; error?: string }>;
      getProjectConnection: (
        projectId: string,
      ) => Promise<{ success: boolean; connection?: ProjectConnection; error?: string }>;
      setServerUrl: (
        url: string,
        opts?: { allowInsecureHttp?: boolean },
      ) => Promise<{ success: boolean; url?: string; error?: string; code?: string; warning?: string }>;
      getServerUrl: () => Promise<{ success: boolean; url?: string }>;
      saveCredentials: (username: string, password: string) => Promise<{ success: boolean }>;
      getSavedCredentials: () => Promise<{ success: boolean; username?: string; password?: string }>;
      clearSavedCredentials: () => Promise<{ success: boolean }>;
      selectDatasetPath: () => Promise<{ success: boolean; path?: string; error?: string }>;
      listModels: () => Promise<{ success: boolean; models?: InferableModel[]; error?: string }>;
      runInference: (
        projectId: string,
        payload: { imageBase64?: string; values?: number[]; text?: string },
      ) => Promise<{ success: boolean; result?: InferenceResult; error?: string }>;
      runGeneration: (
        projectId: string,
        payload: { prompt: string; maxNewTokens: number; temperature: number; history?: { role: 'user' | 'assistant'; content: string }[] },
      ) => Promise<{ success: boolean; result?: unknown; error?: string }>;
      stopGeneration: (projectId: string) => Promise<{ success: boolean; stopped?: boolean; error?: string }>;
      onInferenceToken: (callback: (token: string) => void) => void;
      removeInferenceTokenListener: () => void;
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
      getDeviceCapabilities: () => Promise<{
        success: boolean;
        capabilities?: import('../shared/deviceCapabilities.types').DeviceCapabilities;
        error?: string;
      }>;
      // Auto-updater
      onUpdateAvailable: (callback: (info: UpdateInfo) => void) => void;
      onUpdateProgress: (callback: (progress: ProgressInfo) => void) => void;
      onUpdateDownloaded: (callback: (info: UpdateInfo) => void) => void;
      onUpdateNotAvailable: (callback: () => void) => void;
      onUpdateError: (callback: (message: string) => void) => void;
      installUpdate: () => Promise<{ success: boolean; error?: string }>;
      checkForUpdates: () => Promise<{ success: boolean; error?: string }>;
    };
  }
}

export type ContainerStatus = 'idle' | 'pulling' | 'running' | 'completed' | 'error' | 'restarting' | 'paused' | 'stopped';

export type Section = 'train' | 'models' | 'settings';

// Cap log buffer at 10K lines to prevent unbounded memory growth during long training runs.
// When exceeded, the oldest lines are dropped.
const MAX_LOG_LINES = 10_000;

// Labels for the hardware profile chip in the StatusBar — same ids the Train
// flow's profile cards use (HardwareSelector HARDWARE_PROFILES).
const PROFILE_LABELS: Record<string, string> = {
  discrete: 'Discrete GPU',
  jetson: 'Jetson SoC',
  mps: 'Apple Silicon',
  cpu: 'CPU Only',
};

// hiddenInset traffic lights only exist on macOS; the drag strip inset and the
// shortcut hints are keyed off this.
const IS_MAC = navigator.userAgent.includes('Mac');

const shortcutHint = (n: number): string => (IS_MAC ? `⌘${n}` : `Ctrl+${n}`);

const App: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const [containerStatus, setContainerStatus] = useState<ContainerStatus>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [section, setSection] = useState<Section>('train');
  const [serverHost, setServerHost] = useState('');
  const [hardwareLabel, setHardwareLabel] = useState('');
  const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);

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

  // DE-8: Main pushes this when a 401 (or a locally-detected expired token)
  // invalidates the session mid-use. Registered unconditionally on mount —
  // unlike the log/status subscriptions below, this is exactly what detects
  // the authenticated -> expired transition, so it can't be gated on
  // isAuthenticated. React back to the login screen instead of leaving the
  // dashboard up with opaque per-call "Not authenticated" errors.
  useEffect(() => {
    window.fedLearnAPI.onSessionExpired(() => {
      setIsAuthenticated(false);
      setLogs([]);
      setContainerStatus('idle');
      setSection('train');
      setActiveRun(null);
      setServerHost('');
      setHardwareLabel('');
    });
    return () => {
      window.fedLearnAPI.removeSessionExpiredListener();
    };
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

  // Hardware detection for the StatusBar chip — same preload call the Train
  // flow uses for its profile preselection; advisory only, so failures just
  // leave the chip hidden.
  useEffect(() => {
    if (!isAuthenticated) return;
    let cancelled = false;
    window.fedLearnAPI
      .detectHardware()
      .then((result) => {
        if (cancelled || !result.success || !result.detection) return;
        const profile = result.detection.recommendedProfile;
        setHardwareLabel(PROFILE_LABELS[profile] ?? profile);
      })
      .catch(() => {
        // Detection is advisory; the chip stays hidden.
      });
    return () => {
      cancelled = true;
    };
  }, [isAuthenticated]);

  // Server host for the StatusBar connection readout. Re-fetched on section
  // change so a URL saved in Settings is reflected when navigating away.
  useEffect(() => {
    if (!isAuthenticated) return;
    let cancelled = false;
    window.fedLearnAPI
      .getServerUrl()
      .then((result) => {
        if (cancelled || !result.success || !result.url) return;
        try {
          setServerHost(new URL(result.url).host);
        } catch {
          setServerHost(result.url);
        }
      })
      .catch(() => {
        // Strip shows "Not connected" until the URL resolves.
      });
    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, section]);

  // Keyboard section switching: Cmd/Ctrl+1..3. Pure renderer listener — no
  // menu accelerators, no IPC.
  useEffect(() => {
    if (!isAuthenticated) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return;
      const target: Section | null =
        e.key === '1' ? 'train' : e.key === '2' ? 'models' : e.key === '3' ? 'settings' : null;
      if (target) {
        e.preventDefault();
        setSection(target);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isAuthenticated]);

  const handleLogin = useCallback(() => {
    setIsAuthenticated(true);
  }, []);

  const handleLogout = useCallback(async () => {
    await window.fedLearnAPI.logout();
    setIsAuthenticated(false);
    setLogs([]);
    setContainerStatus('idle');
    setSection('train');
    setActiveRun(null);
    setServerHost('');
    setHardwareLabel('');
  }, []);

  const handleStartTraining = useCallback(
    async (config: {
      hardwareProfile: string;
      projectId: string;
      serverAddress: string;
      partitionId: string;
      modelType: string;
      datasetPath: string;
      connectionToken?: string;
      strategy?: string;
      trainingArm?: string;
    }) => {
      // Anchor the elapsed timer at the moment the user pressed Start — never
      // derived from the 3s status poll. Label falls back to the model type
      // until the project name resolves.
      setActiveRun({
        projectLabel: config.modelType || config.projectId.slice(0, 8),
        startedAt: Date.now(),
      });
      window.fedLearnAPI
        .listTrainableProjects()
        .then((result) => {
          const project = result.projects?.find((p) => p.projectId === config.projectId);
          if (project) {
            setActiveRun((prev) => (prev ? { ...prev, projectLabel: project.name } : prev));
          }
        })
        .catch(() => {
          // Keep the model-type fallback label.
        });

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

  // Main shell
  return (
    <div className="app-container">
      {/* Top drag strip — the only draggable window chrome. */}
      <header className={`shell-titlebar ${IS_MAC ? 'shell-titlebar-mac' : ''}`}>
        <span className="shell-titlebar-logo">
          <Network strokeWidth={1.5} size={18} />
        </span>
        <h1 className="shell-titlebar-title">FedLearn Desktop</h1>
      </header>

      {/* Auto-Update layer — mounted exactly once at shell level (its preload
          listeners have no removal API) and overlaying every section. */}
      <div className="shell-update-layer">
        <UpdateBanner />
      </div>

      <div className="shell-body">
        {/* Left icon rail */}
        <nav className="rail" aria-label="Primary">
          <div className="rail-items">
            <button
              type="button"
              className={`rail-item ${section === 'train' ? 'rail-item-active' : ''}`}
              aria-current={section === 'train' ? 'page' : undefined}
              title={`Train (${shortcutHint(1)})`}
              onClick={() => setSection('train')}
            >
              <Activity strokeWidth={1.5} size={20} />
              <span className="rail-item-label">Train</span>
            </button>
            <button
              type="button"
              className={`rail-item ${section === 'models' ? 'rail-item-active' : ''}`}
              aria-current={section === 'models' ? 'page' : undefined}
              title={`Models (${shortcutHint(2)})`}
              onClick={() => setSection('models')}
            >
              <Boxes strokeWidth={1.5} size={20} />
              <span className="rail-item-label">Models</span>
            </button>
            <button
              type="button"
              id="settings-button"
              className={`rail-item ${section === 'settings' ? 'rail-item-active' : ''}`}
              aria-current={section === 'settings' ? 'page' : undefined}
              title={`Settings (${shortcutHint(3)})`}
              onClick={() => setSection('settings')}
            >
              <Settings strokeWidth={1.5} size={20} />
              <span className="rail-item-label">Settings</span>
            </button>
          </div>
          <div className="rail-foot">
            {/* The frozen preload API exposes no username readback, so the
                account item shows a generic signed-in identity. */}
            <div className="rail-user" title="Signed in">
              <User strokeWidth={1.5} size={20} />
              <span className="rail-item-label">Signed in</span>
            </div>
            <button
              type="button"
              id="logout-button"
              className="rail-item"
              title="Sign out"
              onClick={handleLogout}
            >
              <LogOut strokeWidth={1.5} size={20} />
              <span className="rail-item-label">Sign out</span>
            </button>
          </div>
        </nav>

        {/* Section outlet — every section stays mounted; visibility toggles
            via CSS so training/playground state survives switches. */}
        <main className="shell-outlet">
          <section
            className={`shell-section ${section === 'train' ? '' : 'shell-section-hidden'}`}
            aria-label="Train"
          >
            {/* Pass the FULL container status — TrainSection derives its
                setup/running/outcome phases from it. A collapsed boolean
                would make the completed/error banners and the run-finished
                notifications unreachable, and would show the setup card
                mid-run during 'restarting'/'paused'. */}
            <TrainSection
              onStart={handleStartTraining}
              onStop={handleStopTraining}
              status={containerStatus}
              logs={logs}
            />
          </section>
          <section
            className={`shell-section ${section === 'models' ? '' : 'shell-section-hidden'}`}
            aria-label="Models"
          >
            <ModelPlayground />
          </section>
          <section
            className={`shell-section ${section === 'settings' ? '' : 'shell-section-hidden'}`}
            aria-label="Settings"
          >
            <SettingsSection />
          </section>
        </main>
      </div>

      {/* Persistent bottom status strip */}
      <StatusBar
        containerStatus={containerStatus}
        serverHost={serverHost}
        hardwareLabel={hardwareLabel}
        activeRun={activeRun}
        appVersion={__APP_VERSION__}
      />
    </div>
  );
};

export default App;
