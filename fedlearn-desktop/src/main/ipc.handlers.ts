// =============================================================================
// FedLearn Desktop — IPC Handler Registrations
// =============================================================================
// All ipcMain.handle registrations are centralized here.
// Defense-in-depth: inputs are re-validated in Main even though
// preload.ts performs primary allowlist validation.
// =============================================================================

import { ipcMain, BrowserWindow } from 'electron';
import log from 'electron-log';
import { DockerService, TrainingConfig, HardwareProfile } from './docker.service';
import { AuthService } from './auth.service';

const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> = new Set(['discrete', 'jetson', 'cpu']);
const PROJECT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;
const PARTITION_ID_PATTERN = /^[0-9]{1,10}$/;
const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;

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

      const validConfig: TrainingConfig = {
        hardwareProfile: cfg.hardwareProfile as HardwareProfile,
        projectId: cfg.projectId as string,
        serverAddress: cfg.serverAddress as string,
        partitionId: cfg.partitionId as string,
      };

      log.info(`[IPC:docker:start-training] Starting training with profile=${validConfig.hardwareProfile}, project=${validConfig.projectId}`);
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

  ipcMain.handle('docker:get-status', async () => {
    try {
      const status = await dockerService.getStatus();
      return { success: true, status };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:docker:get-status] Failed: ${message}`);
      return { success: true, status: 'error' };
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

  log.info('[IPC] All handlers registered: docker:start-training, docker:stop-training, docker:get-status, auth:login, auth:logout, auth:check, auth:set-server-url, auth:get-server-url');
}
