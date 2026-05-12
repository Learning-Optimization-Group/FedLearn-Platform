// =============================================================================
// FedLearn Desktop — IPC Handler Registrations
// =============================================================================
// All ipcMain.handle registrations are centralized here.
// Defense-in-depth: inputs are re-validated in Main even though
// preload.ts performs primary allowlist validation.
// =============================================================================

import { ipcMain, BrowserWindow, dialog } from 'electron';
import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import * as fs from 'fs';
import * as path from 'path';
import { DockerService, TrainingConfig, HardwareProfile } from './docker.service';
import { AuthService } from './auth.service';
import { detectHardware } from './hardware.probe';

const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> = new Set(['discrete', 'jetson', 'cpu', 'mps']);
const PROJECT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
const MAX_DATASET_PATH_LEN = 2048;

/**
 * Normalizes and validates a dataset path before it's bind-mounted into a
 * training container. The renderer normally selects the path through the
 * native dialog (which is safe), but a compromised renderer or future text
 * input could craft a path that, once interpolated into the Docker bind
 * string `${path}:/data`, escapes to a sensitive host directory.
 *
 * Rules:
 *   - String of bounded length, no NUL bytes (no directory-traversal via
 *     embedded null terminator).
 *   - Resolves to an absolute path with no remaining `..` segments.
 *   - Path must currently exist and be a directory (catches typos and
 *     prevents bind-mounting non-existent paths which Docker would create
 *     as empty directories owned by root).
 *
 * Returns the canonical absolute path on success, or null on rejection.
 */
function sanitizeDatasetPath(raw: unknown): string | null {
  // Dataset path is optional — empty string means "use default dataset inside container".
  if (typeof raw === 'string' && raw.trim() === '') {
    return '';
  }
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > MAX_DATASET_PATH_LEN) {
    return null;
  }
  if (raw.includes('\0')) {
    return null;
  }
  let resolved: string;
  try {
    resolved = path.resolve(raw);
  } catch {
    return null;
  }
  // After resolve(), `..` segments should already be collapsed. If any
  // remain (only possible on platforms with unusual semantics), bail out.
  if (resolved.split(path.sep).some((seg) => seg === '..')) {
    return null;
  }
  if (!path.isAbsolute(resolved)) {
    return null;
  }
  let stat: fs.Stats;
  try {
    stat = fs.statSync(resolved);
  } catch {
    return null;
  }
  if (!stat.isDirectory()) {
    return null;
  }
  return resolved;
}

let dockerService: DockerService;
let authService: AuthService;

function validateHardwareProfile(profile: unknown): profile is HardwareProfile {
  return typeof profile === 'string' && ALLOWED_HARDWARE_PROFILES.has(profile);
}

function validateProjectId(id: unknown): id is string {
  return typeof id === 'string' && PROJECT_ID_PATTERN.test(id);
}

function validatePartitionId(id: unknown): id is string {
  return typeof id === 'string' && PARTITION_ID_PATTERN.test(id);
}

function validateServerAddress(addr: unknown): addr is string {
  return typeof addr === 'string' && SERVER_ADDRESS_PATTERN.test(addr);
}

function validateStringInput(val: unknown, maxLength: number): val is string {
  return typeof val === 'string' && val.length > 0 && val.length <= maxLength;
}

export function registerIpcHandlers(mainWindow: BrowserWindow): void {
  dockerService = new DockerService(mainWindow);
  authService = new AuthService();

  // ===================== File Dialogs =====================
  ipcMain.handle('dialog:open-directory', async () => {
    try {
      const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory', 'createDirectory'],
        title: 'Select Dataset Directory'
      });
      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, error: 'User canceled.' };
      }
      return { success: true, path: result.filePaths[0] };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:dialog:open-directory] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  // ===================== Docker Channels =====================

  ipcMain.handle('docker:start-training', async (_event, config: unknown) => {
    try {
      // Defense-in-depth validation (preload also validates)
      if (!config || typeof config !== 'object') {
        log.error('[IPC:docker:start-training] Invalid config object received');
        return { success: false, error: 'Invalid configuration' };
      }

      const cfg = config as Record<string, unknown>;

      if (!validateHardwareProfile(cfg.hardwareProfile)) {
        log.error(`[IPC:docker:start-training] Rejected invalid hardware profile: ${String(cfg.hardwareProfile)}`);
        return { success: false, error: 'Invalid hardware profile' };
      }

      if (!validateProjectId(cfg.projectId)) {
        log.error(`[IPC:docker:start-training] Rejected invalid project ID: ${String(cfg.projectId)}`);
        return { success: false, error: 'Invalid project ID' };
      }

      if (!validateServerAddress(cfg.serverAddress)) {
        log.error(`[IPC:docker:start-training] Rejected invalid server address: ${String(cfg.serverAddress)}`);
        return { success: false, error: 'Invalid server address' };
      }

      if (!validatePartitionId(cfg.partitionId)) {
        log.error(`[IPC:docker:start-training] Rejected invalid partition ID: ${String(cfg.partitionId)}`);
        return { success: false, error: 'Invalid partition ID' };
      }

      if (typeof cfg.modelType !== 'string' || !/^[a-zA-Z0-9_\-\.]{1,128}$/.test(cfg.modelType)) {
        log.error(`[IPC:docker:start-training] Rejected invalid model type: ${String(cfg.modelType)}`);
        return { success: false, error: 'Invalid model type' };
      }

      const safeDatasetPath = sanitizeDatasetPath(cfg.datasetPath);
      if (safeDatasetPath === null) {
        log.error('[IPC:docker:start-training] Rejected invalid dataset path');
        return {
          success: false,
          error: 'Invalid dataset path: must be an existing absolute directory',
        };
      }

      const validConfig: TrainingConfig = {
        hardwareProfile: cfg.hardwareProfile as HardwareProfile,
        projectId: cfg.projectId as string,
        serverAddress: cfg.serverAddress as string,
        partitionId: cfg.partitionId as string,
        modelType: cfg.modelType as string,
        // Use the canonical resolved path, not the raw string from the renderer.
        datasetPath: safeDatasetPath,
      };

      log.info(`[IPC:docker:start-training] Starting training with profile=${validConfig.hardwareProfile}, project=${validConfig.projectId}, model=${validConfig.modelType}`);
      await dockerService.startTraining(validConfig);
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:docker:start-training] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('docker:stop-training', async () => {
    try {
      log.info('[IPC:docker:stop-training] Stopping training container');
      await dockerService.stopTraining();
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:docker:stop-training] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('hardware:detect', async () => {
    try {
      const detection = await detectHardware();
      return { success: true, detection };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:hardware:detect] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('docker:get-status', async () => {
    try {
      const status = await dockerService.getStatus();
      return { success: true, status };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:docker:get-status] Failed: ${message}`);
      return { success: false, status: 'error', error: message };
    }
  });

  // ===================== Auth Channels =====================

  ipcMain.handle('auth:login', async (_event, credentials: unknown) => {
    try {
      if (!credentials || typeof credentials !== 'object') {
        log.error('[IPC:auth:login] Invalid credentials object');
        return { success: false };
      }

      const creds = credentials as Record<string, unknown>;

      if (!validateStringInput(creds.username, 256)) {
        log.error('[IPC:auth:login] Rejected invalid username input');
        return { success: false };
      }

      if (!validateStringInput(creds.password, 256)) {
        log.error('[IPC:auth:login] Rejected invalid password input');
        return { success: false };
      }

      // JWT is confined to Main process — Renderer receives only { success }
      const result = await authService.login(
        creds.username as string,
        creds.password as string,
      );

      return { success: result };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:auth:login] Failed: ${message}`);
      return { success: false };
    }
  });

  ipcMain.handle('auth:logout', async () => {
    try {
      authService.logout();
      log.info('[IPC:auth:logout] User logged out, JWT cleared');
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:auth:logout] Failed: ${message}`);
      return { success: false };
    }
  });

  ipcMain.handle('auth:check', async () => {
    try {
      const authenticated = authService.isAuthenticated();
      return { success: true, authenticated };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:auth:check] Failed: ${message}`);
      return { success: false, authenticated: false };
    }
  });

  // ===================== Server URL Channel =====================

  ipcMain.handle('auth:set-server-url', async (_event, url: unknown) => {
    try {
      if (typeof url !== 'string' || url.length === 0 || url.length > 512) {
        log.error('[IPC:auth:set-server-url] Invalid URL input');
        return { success: false, error: 'Invalid server URL' };
      }

      // Require http:// or https:// protocol
      if (!/^https?:\/\//i.test(url.trim())) {
        log.error('[IPC:auth:set-server-url] Rejected URL missing http(s):// protocol');
        return { success: false, error: 'URL must start with http:// or https://' };
      }

      // Normalize: ensure it ends with /api
      let normalized = url.trim().replace(/\/+$/, '');
      if (!normalized.endsWith('/api')) {
        normalized += '/api';
      }

      authService.setApiUrl(normalized);
      return { success: true, url: normalized };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:auth:set-server-url] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('auth:get-server-url', async () => {
    try {
      const url = authService.getApiUrl();
      return { success: true, url };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:auth:get-server-url] Failed: ${message}`);
      return { success: false, url: '' };
    }
  });

  // ===================== Auto Updater =====================
  ipcMain.handle('updater:install', async () => {
    try {
      log.info('[IPC:updater:install] User requested restart to install update');
      autoUpdater.quitAndInstall();
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:updater:install] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('updater:check', async () => {
    try {
      log.info('[IPC:updater:check] Manual update check triggered');
      // Relay "not available" and "error" events back to the renderer
      autoUpdater.once('update-not-available', () => {
        if (!mainWindow.isDestroyed()) {
          mainWindow.webContents.send('updater:not-available');
        }
      });
      autoUpdater.once('error', (err: Error) => {
        if (!mainWindow.isDestroyed()) {
          mainWindow.webContents.send('updater:error', err.message);
        }
      });
      await autoUpdater.checkForUpdates();
      return { success: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:updater:check] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  log.info('[IPC] All handlers registered: docker:start-training, docker:stop-training, docker:get-status, hardware:detect, auth:login, auth:logout, auth:check, auth:set-server-url, auth:get-server-url, dialog:open-directory, updater:install');
}
