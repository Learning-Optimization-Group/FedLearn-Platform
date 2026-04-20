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
  if (typeof val !== 'string') {
    console.error(`[Preload:Validation] Dataset path is not a string: ${typeof val}`);
    return false;
  }
  if (val.length === 0 || val.length > 2048) {
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
   * Register a callback for Docker daemon unavailability events.
   * Fired once on startup if the Docker socket is unreachable.
   */
  onDockerUnavailable: (callback: (message: string) => void): void => {
    ipcRenderer.on('docker:daemon-unavailable', (_event, value: string) => {
      if (typeof value === 'string') {
        callback(value);
      }
    });
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
});
