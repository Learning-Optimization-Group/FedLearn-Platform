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
// NOTE: electron-log cannot be used in sandboxed preload scripts.
// console.error is forwarded to the main process console automatically.

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
  const valid = /^[a-zA-Z0-9_\-\.]{1,128}$/.test(val);
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

// ========== Secure API exposed to Renderer ==========

export interface TrainingConfigInput {
  hardwareProfile: string;
  projectId: string;
  serverAddress: string;
  partitionId: string;
  modelType: string;
  datasetPath: string;
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

    return ipcRenderer.invoke('docker:start-training', {
      hardwareProfile: config.hardwareProfile,
      projectId: config.projectId,
      serverAddress: config.serverAddress,
      partitionId: config.partitionId,
      modelType: config.modelType,
      datasetPath: config.datasetPath,
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
   * Set the backend server URL. Persisted across app restarts.
   * Users enter the URL (e.g. http://192.168.1.100:8081) and /api is appended automatically.
   */
  setServerUrl: async (url: string): Promise<{ success: boolean; url?: string; error?: string }> => {
    if (!isValidStringInput(url, 'serverUrl')) {
      return { success: false, error: 'Invalid server URL' };
    }
    return ipcRenderer.invoke('auth:set-server-url', url);
  },

  /**
   * Get the currently configured backend server URL.
   */
  getServerUrl: async (): Promise<{ success: boolean; url?: string }> => {
    return ipcRenderer.invoke('auth:get-server-url');
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

  onUpdateAvailable: (callback: (info: any) => void): void => {
    ipcRenderer.on('updater:update-available', (_event, info) => callback(info));
  },

  onUpdateProgress: (callback: (progress: any) => void): void => {
    ipcRenderer.on('updater:download-progress', (_event, progress) => callback(progress));
  },

  onUpdateDownloaded: (callback: (info: any) => void): void => {
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
