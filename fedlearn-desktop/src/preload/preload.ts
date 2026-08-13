// =============================================================================
// FedLearn Desktop — Preload Script (contextBridge)
// =============================================================================
// Per Section 5.3: All IPC inputs validated against explicit allowlists
// before forwarding to Main. Rejections are logged via electron-log.
//
// This script is the ONLY bridge between the sandboxed Renderer and the
// Main Process. It exposes a strictly typed, parameter-validated API via
// contextBridge.exposeInMainWorld.
//
// SECURITY CONTRACTS:
//   1. No raw ipcRenderer exposure to the Renderer
//   2. Every input is validated before ipcRenderer.invoke
//   3. Rejections logged with electron-log
//   4. No Node.js globals leak into window scope
// =============================================================================

import { contextBridge, ipcRenderer } from 'electron';
import type { UpdateInfo, ProgressInfo } from 'electron-updater';
// NOTE: electron-log cannot be used in sandboxed preload scripts.
// console.error is forwarded to the main process console automatically.
// electron-updater is type-only here (erased at compile time) — the preload bundle never pulls
// in its runtime code, only the UpdateInfo/ProgressInfo shapes forwarded from Main via IPC.

// ========== Validation Constants ==========

const ALLOWED_HARDWARE_PROFILES = ['discrete', 'jetson', 'cpu', 'mps'] as const;
const PROJECT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;
const MAX_IMAGE_BASE64_LEN = 14 * 1024 * 1024; // ~10 MB decoded
const MAX_VECTOR_LEN = 100_000;
const PARTITION_ID_PATTERN = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
const MAX_STRING_LENGTH = 256;

// ========== Validation Helpers ==========

function isValidHardwareProfile(profile: unknown): boolean {
  if (typeof profile !== 'string') {
    console.error(`[Preload:Validation] Hardware profile is not a string: ${typeof profile}`);
    return false;
  }
  const valid = (ALLOWED_HARDWARE_PROFILES as readonly string[]).includes(profile);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected hardware profile not in allowlist: "${profile}"`);
  }
  return valid;
}

function isValidProjectId(id: unknown): boolean {
  if (typeof id !== 'string') {
    console.error(`[Preload:Validation] Project ID is not a string: ${typeof id}`);
    return false;
  }
  const valid = PROJECT_ID_PATTERN.test(id);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected project ID failing pattern: "${id}"`);
  }
  return valid;
}

function isValidPartitionId(id: unknown): boolean {
  if (typeof id !== 'string') {
    console.error(`[Preload:Validation] Partition ID is not a string: ${typeof id}`);
    return false;
  }
  const valid = PARTITION_ID_PATTERN.test(id);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected partition ID failing pattern: "${id}"`);
  }
  return valid;
}

function isValidModelType(val: unknown): boolean {
  if (typeof val !== 'string') {
    console.error(`[Preload:Validation] Model type is not a string: ${typeof val}`);
    return false;
  }
  const valid = /^[a-zA-Z0-9_\-.]{1,128}$/.test(val);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected model type failing pattern: "${val}"`);
  }
  return valid;
}

function isValidDatasetPath(val: unknown): boolean {
  // Dataset path is optional — empty string is allowed (means 'use default data')
  if (typeof val !== 'string') {
    console.error(`[Preload:Validation] Dataset path is not a string: ${typeof val}`);
    return false;
  }
  if (val.length > 2048) {
    console.error(`[Preload:Validation] Dataset path length out of bounds: ${val.length}`);
    return false;
  }
  return true;
}

function isValidServerAddress(addr: unknown): boolean {
  if (typeof addr !== 'string') {
    console.error(`[Preload:Validation] Server address is not a string: ${typeof addr}`);
    return false;
  }
  const valid = SERVER_ADDRESS_PATTERN.test(addr);
  if (!valid) {
    console.error(`[Preload:Validation] Rejected server address failing pattern: "${addr}"`);
  }
  return valid;
}

function isValidStringInput(val: unknown, fieldName: string): boolean {
  if (typeof val !== 'string') {
    console.error(`[Preload:Validation] ${fieldName} is not a string: ${typeof val}`);
    return false;
  }
  if (val.length === 0 || val.length > MAX_STRING_LENGTH) {
    console.error(`[Preload:Validation] ${fieldName} length out of bounds: ${val.length}`);
    return false;
  }
  return true;
}

// The FL connection token is an HMAC-JWT (three base64url segments joined by dots).
// It's optional here — absent means the legacy no-auth flow — but when present it must
// be a bounded token-charset string before we forward it to Main.
function isValidConnectionToken(val: unknown): boolean {
  if (val === undefined || val === null) {
    return true;
  }
  if (typeof val !== 'string' || val.length === 0 || val.length > 8192) {
    console.error('[Preload:Validation] Connection token missing or out of bounds');
    return false;
  }
  return /^[A-Za-z0-9._-]+$/.test(val);
}

// ========== Secure API exposed to Renderer ==========

export interface TrainingConfigInput {
  hardwareProfile: string;
  projectId: string;
  serverAddress: string;
  partitionId: string;
  modelType: string;
  datasetPath: string;
  connectionToken?: string;
  strategy?: string;
  trainingArm?: string;
}

interface InferencePayloadInput {
  imageBase64?: string;
  values?: number[];
  text?: string;
}

const MAX_TEXT_LEN = 10_000; // matches backend @Size(max = 10_000)

function isValidInferencePayload(payload: unknown): payload is InferencePayloadInput {
  if (!payload || typeof payload !== 'object') {
    console.error('[Preload:Validation] Inference payload is not an object');
    return false;
  }
  const p = payload as Record<string, unknown>;
  if (typeof p.imageBase64 === 'string') {
    return p.imageBase64.length > 0 && p.imageBase64.length <= MAX_IMAGE_BASE64_LEN;
  }
  if (Array.isArray(p.values)) {
    return (
      p.values.length > 0 &&
      p.values.length <= MAX_VECTOR_LEN &&
      p.values.every((v) => typeof v === 'number' && Number.isFinite(v))
    );
  }
  if (typeof p.text === 'string') {
    return p.text.trim().length > 0 && p.text.length <= MAX_TEXT_LEN;
  }
  console.error('[Preload:Validation] Inference payload has neither imageBase64, values, nor text');
  return false;
}

contextBridge.exposeInMainWorld('fedLearnAPI', {
  /**
   * Start a federated learning training container.
   * All parameters are validated against allowlists before IPC transmission.
   */
  startTraining: async (config: TrainingConfigInput): Promise<{ success: boolean; error?: string }> => {
    // Validate every field before forwarding to Main
    if (!isValidHardwareProfile(config.hardwareProfile)) {
      return { success: false, error: 'Invalid hardware profile' };
    }
    if (!isValidProjectId(config.projectId)) {
      return { success: false, error: 'Invalid project ID' };
    }
    if (!isValidServerAddress(config.serverAddress)) {
      return { success: false, error: 'Invalid server address' };
    }
    if (!isValidPartitionId(config.partitionId)) {
      return { success: false, error: 'Invalid partition ID' };
    }
    if (!isValidModelType(config.modelType)) {
      return { success: false, error: 'Invalid model type' };
    }
    if (!isValidDatasetPath(config.datasetPath)) {
      return { success: false, error: 'Invalid dataset path' };
    }
    if (!isValidConnectionToken(config.connectionToken)) {
      return { success: false, error: 'Invalid connection token' };
    }
    // strategy is optional (a bounded token from the backend connection payload); reject a malformed
    // one rather than forwarding garbage. Main re-validates with the same pattern (defense in depth).
    if (config.strategy !== undefined && !/^[a-zA-Z0-9_\-.]{1,64}$/.test(config.strategy)) {
      return { success: false, error: 'Invalid strategy' };
    }

    return ipcRenderer.invoke('docker:start-training', {
      hardwareProfile: config.hardwareProfile,
      projectId: config.projectId,
      serverAddress: config.serverAddress,
      partitionId: config.partitionId,
      modelType: config.modelType,
      datasetPath: config.datasetPath,
      connectionToken: config.connectionToken,
      strategy: config.strategy,
    });
  },

  /**
   * Stop the currently running training container.
   */
  stopTraining: async (): Promise<{ success: boolean; error?: string }> => {
    return ipcRenderer.invoke('docker:stop-training');
  },

  /**
   * Get the current Docker container status.
   */
  getDockerStatus: async (): Promise<{ success: boolean; status?: string }> => {
    return ipcRenderer.invoke('docker:get-status');
  },

  /**
   * Authenticate with the FedLearn backend.
   * Returns { success: boolean } ONLY — JWT never leaves Main Process.
   */
  login: async (username: string, password: string): Promise<{ success: boolean }> => {
    if (!isValidStringInput(username, 'username')) {
      return { success: false };
    }
    if (!isValidStringInput(password, 'password')) {
      return { success: false };
    }

    return ipcRenderer.invoke('auth:login', { username, password });
  },

  /**
   * Clear stored authentication credentials.
   */
  logout: async (): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('auth:logout');
  },

  /**
   * Check if the user is currently authenticated.
   */
  checkAuth: async (): Promise<{ success: boolean; authenticated?: boolean }> => {
    return ipcRenderer.invoke('auth:check');
  },

  /**
   * Register a callback fired when Main invalidates the current session
   * (a 401 from the backend, or a locally-detected expired token) mid-use.
   * The renderer reacts by clearing its own auth state and showing the login
   * screen again — see App.tsx.
   */
  onSessionExpired: (callback: () => void): void => {
    ipcRenderer.on('auth:session-expired', () => callback());
  },

  /**
   * Remove all session-expired listeners (cleanup on unmount).
   */
  removeSessionExpiredListener: (): void => {
    ipcRenderer.removeAllListeners('auth:session-expired');
  },

  /**
   * Register a callback for real-time training log events.
   * LogPanel renders these as plain text only — no HTML.
   */
  onTrainingLog: (callback: (logLine: string) => void): void => {
    ipcRenderer.on('docker:training-log', (_event, value: string) => {
      // Ensure we only pass string data to the callback
      if (typeof value === 'string') {
        callback(value);
      }
    });
  },

  /**
   * Remove all training log listeners (cleanup on unmount).
   */
  removeTrainingLogListener: (): void => {
    ipcRenderer.removeAllListeners('docker:training-log');
  },

  /**
   * Set the backend server URL. Persisted across app restarts; /api is appended
   * automatically. Plaintext http:// to a non-loopback host is refused with
   * code 'INSECURE_HTTP' unless opts.allowInsecureHttp is set — and even then
   * the response carries a warning the caller must surface (credentials and the
   * session token traverse the network unencrypted).
   */
  setServerUrl: async (
    url: string,
    opts?: { allowInsecureHttp?: boolean },
  ): Promise<{ success: boolean; url?: string; error?: string; code?: string; warning?: string }> => {
    if (!isValidStringInput(url, 'serverUrl')) {
      return { success: false, error: 'Invalid server URL' };
    }
    // Forward only the known flag — never an arbitrary renderer object.
    const forwarded = opts?.allowInsecureHttp === true ? { allowInsecureHttp: true } : undefined;
    return ipcRenderer.invoke('auth:set-server-url', url, forwarded);
  },

  /**
   * Get the currently configured backend server URL.
   */
  getServerUrl: async (): Promise<{ success: boolean; url?: string }> => {
    return ipcRenderer.invoke('auth:get-server-url');
  },

  /**
   * "Save password" opt-in: persist the login credentials encrypted (OS keychain) in Main.
   * The renderer only ever passes/reads plaintext; the encrypted blob never leaves Main's store.
   */
  saveCredentials: async (username: string, password: string): Promise<{ success: boolean }> => {
    if (!isValidStringInput(username, 'username') || !isValidStringInput(password, 'password')) {
      return { success: false };
    }
    return ipcRenderer.invoke('auth:save-credentials', { username, password });
  },

  /**
   * Load the saved credentials to pre-fill the login form. { success: false } when none stored.
   */
  getSavedCredentials: async (): Promise<{ success: boolean; username?: string; password?: string }> => {
    return ipcRenderer.invoke('auth:get-credentials');
  },

  /**
   * Forget any saved credentials (unchecked "Save password").
   */
  clearSavedCredentials: async (): Promise<{ success: boolean }> => {
    return ipcRenderer.invoke('auth:clear-credentials');
  },

  /**
   * Trigger native system file dialog to select a dataset path securely.
   */
  selectDatasetPath: async (): Promise<{ success: boolean; path?: string; error?: string }> => {
    return ipcRenderer.invoke('dialog:open-directory');
  },

  // ===================== Client Projects ("models I can train") =====================

  /**
   * List the projects the authenticated user may train (owner or approved
   * CLIENT). Replaces manual project-id / server / partition entry.
   */
  listTrainableProjects: async (): Promise<{ success: boolean; projects?: unknown[]; error?: string }> => {
    return ipcRenderer.invoke('client:list-projects');
  },

  /**
   * Resolve a project's live gRPC connection (address + server-assigned
   * partition id + model type) so training can start without manual entry.
   */
  getProjectConnection: async (
    projectId: string,
  ): Promise<{ success: boolean; connection?: unknown; error?: string }> => {
    if (!isValidProjectId(projectId)) {
      return { success: false, error: 'Invalid project ID' };
    }
    return ipcRenderer.invoke('client:get-connection', projectId);
  },

  // ===================== Inference ("Use a model") =====================

  /**
   * List the authenticated user's trained models that can be run interactively.
   */
  listModels: async (): Promise<{ success: boolean; models?: unknown[]; error?: string }> => {
    return ipcRenderer.invoke('inference:list-models');
  },

  /**
   * Run inference against a project's trained model. The payload carries either
   * a base64 image (image models) or a numeric vector (tabular models).
   */
  runInference: async (
    projectId: string,
    payload: InferencePayloadInput,
  ): Promise<{ success: boolean; result?: unknown; error?: string }> => {
    if (!isValidProjectId(projectId)) {
      return { success: false, error: 'Invalid project ID' };
    }
    if (!isValidInferencePayload(payload)) {
      return { success: false, error: 'Invalid input payload' };
    }
    return ipcRenderer.invoke('inference:run', { projectId, payload });
  },

  /**
   * Run streaming text generation against a project's trained generative model.
   * Tokens arrive via the onInferenceToken listener; this call resolves with
   * the full result once generation is complete.
   */
  runGeneration: async (
    projectId: string,
    payload: { prompt: string; maxNewTokens: number; temperature: number; history?: { role: 'user' | 'assistant'; content: string }[] },
  ): Promise<{ success: boolean; result?: unknown; error?: string }> => {
    if (!isValidProjectId(projectId)) return { success: false, error: 'Invalid project ID' };
    if (
      typeof payload?.prompt !== 'string' ||
      !payload.prompt.trim() ||
      payload.prompt.length > 10_000
    ) {
      return { success: false, error: 'Invalid prompt' };
    }
    return ipcRenderer.invoke('inference:run-generation', { projectId, payload });
  },

  /**
   * Cancel an in-flight generation. Best-effort: the streamed partial is kept
   * by the renderer regardless of the server response.
   */
  stopGeneration: async (projectId: string): Promise<{ success: boolean; stopped?: boolean; error?: string }> => {
    if (!isValidProjectId(projectId)) return { success: false, error: 'Invalid project ID' };
    return ipcRenderer.invoke('inference:stop-generation', { projectId });
  },

  /**
   * Register a callback for streaming generation token events pushed by Main.
   * Call removeInferenceTokenListener() on component unmount.
   */
  onInferenceToken: (callback: (token: string) => void): void => {
    ipcRenderer.on('inference:token', (_event, value: string) => {
      if (typeof value === 'string') callback(value);
    });
  },

  /**
   * Remove all inference:token listeners (cleanup on unmount).
   */
  removeInferenceTokenListener: (): void => {
    ipcRenderer.removeAllListeners('inference:token');
  },

  /**
   * One-shot hardware detection. Returns the platform/arch, whether a CUDA
   * GPU is visible, whether the bundled native client is shipped with this
   * install, and a recommended hardware profile to pre-select in the UI.
   */
  detectHardware: async (): Promise<{
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
  }> => {
    return ipcRenderer.invoke('hardware:detect');
  },

  getDeviceCapabilities: (): Promise<{
    success: boolean;
    capabilities?: import('../shared/deviceCapabilities.types').DeviceCapabilities;
    error?: string;
  }> =>
    ipcRenderer.invoke('device:capabilities'),

  // ===================== Auto Updater =====================

  onUpdateAvailable: (callback: (info: UpdateInfo) => void): void => {
    ipcRenderer.on('updater:update-available', (_event, info) => callback(info));
  },

  onUpdateProgress: (callback: (progress: ProgressInfo) => void): void => {
    ipcRenderer.on('updater:download-progress', (_event, progress) => callback(progress));
  },

  onUpdateDownloaded: (callback: (info: UpdateInfo) => void): void => {
    ipcRenderer.on('updater:update-downloaded', (_event, info) => callback(info));
  },

  onUpdateNotAvailable: (callback: () => void): void => {
    ipcRenderer.on('updater:not-available', () => callback());
  },

  onUpdateError: (callback: (message: string) => void): void => {
    ipcRenderer.on('updater:error', (_event, message) => callback(message));
  },

  installUpdate: async (): Promise<{ success: boolean; error?: string }> => {
    return ipcRenderer.invoke('updater:install');
  },

  checkForUpdates: async (): Promise<{ success: boolean; error?: string }> => {
    return ipcRenderer.invoke('updater:check');
  },
});
