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
import { DockerService, TrainingConfig, HardwareProfile } from './docker.service';
import {
  sanitizeDatasetPath,
  validateHardwareProfile,
  validateProjectId,
  validatePartitionId,
  validateServerAddress,
  validateStringInput,
} from './validators';
import { AuthService } from './auth.service';
import { InferenceService, InferencePayload } from './inference.service';
import { ClientProjectService } from './client-projects.service';
import { InferenceStreamService } from './inference-stream.service';
import { detectHardware } from './hardware.probe';
import { collectDeviceCapabilities } from './deviceCapabilities.collector';

const MAX_IMAGE_BASE64_LEN = 14 * 1024 * 1024; // ~10 MB decoded
const MAX_VECTOR_LEN = 100_000;

// Input validators (sanitizeDatasetPath, validateHardwareProfile, validateProjectId,
// validatePartitionId, validateServerAddress, validateStringInput) live in ./validators
// so they can be unit-tested directly. This file used to carry a diverged inline copy.

let dockerService: DockerService;
let authService: AuthService;
let inferenceService: InferenceService;
let clientProjectService: ClientProjectService;

/**
 * Validates a renderer-supplied inference payload (defense-in-depth — preload
 * validates too). Exactly one of imageBase64 / values must be present and within
 * bounds. Returns a clean payload or null on rejection.
 */
const MAX_TEXT_LEN = 10_000; // matches backend @Size(max = 10_000)

function sanitizeInferencePayload(raw: unknown): InferencePayload | null {
  if (!raw || typeof raw !== 'object') return null;
  const p = raw as Record<string, unknown>;

  if (typeof p.imageBase64 === 'string') {
    const b64 = p.imageBase64;
    if (b64.length === 0 || b64.length > MAX_IMAGE_BASE64_LEN) return null;
    return { imageBase64: b64 };
  }
  if (Array.isArray(p.values)) {
    if (p.values.length === 0 || p.values.length > MAX_VECTOR_LEN) return null;
    if (!p.values.every((v) => typeof v === 'number' && Number.isFinite(v))) return null;
    return { values: p.values as number[] };
  }
  if (typeof p.text === 'string') {
    const txt = p.text;
    if (txt.trim().length === 0 || txt.length > MAX_TEXT_LEN) return null;
    return { text: txt };
  }
  return null;
}

/**
 * Accessor for the singleton DockerService created in registerIpcHandlers.
 * Used by the main process's before-quit handler to drain a running training
 * container/native process on app exit. Returns undefined if IPC handlers were
 * never registered (e.g. registration threw).
 */
export function getDockerService(): DockerService | undefined {
  return dockerService;
}

export function registerIpcHandlers(mainWindow: BrowserWindow): void {
  dockerService = new DockerService(mainWindow);
  authService = new AuthService();
  inferenceService = new InferenceService(authService);
  clientProjectService = new ClientProjectService(authService);
  const inferenceStreamService = new InferenceStreamService(authService, mainWindow);

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

  // ===================== Inference ("Use a model") =====================

  ipcMain.handle('inference:list-models', async () => {
    try {
      return await inferenceService.listModels();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:inference:list-models] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('inference:run', async (_event, args: unknown) => {
    try {
      if (!args || typeof args !== 'object') {
        return { success: false, error: 'Invalid request' };
      }
      const a = args as Record<string, unknown>;
      if (!validateProjectId(a.projectId)) {
        log.error(`[IPC:inference:run] Rejected invalid project ID: ${String(a.projectId)}`);
        return { success: false, error: 'Invalid project ID' };
      }
      const payload = sanitizeInferencePayload(a.payload);
      if (payload === null) {
        log.error('[IPC:inference:run] Rejected invalid payload');
        return { success: false, error: 'Invalid input payload' };
      }
      return await inferenceService.runInference(a.projectId as string, payload);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:inference:run] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  // ===================== Inference — Generation (streaming) =====================

  ipcMain.handle('inference:run-generation', async (_event, args: unknown) => {
    try {
      const a = (args ?? {}) as Record<string, unknown>;
      if (!validateProjectId(a.projectId)) return { success: false, error: 'Invalid project ID' };
      const p = (a.payload ?? {}) as Record<string, unknown>;
      const prompt = typeof p.prompt === 'string' ? p.prompt : '';
      if (!prompt.trim() || prompt.length > 10_000) return { success: false, error: 'Invalid prompt' };
      const mnt = Number(p.maxNewTokens);
      const maxNewTokens = Math.max(1, Math.min(2048, Number.isFinite(mnt) ? mnt : 256));
      const t = Number(p.temperature);
      const temperature = Math.max(0, Math.min(2, Number.isFinite(t) ? t : 0.7));
      const history = Array.isArray(p.history)
        ? (p.history as unknown[])
            .filter(
              (turn): turn is { role: 'user' | 'assistant'; content: string } =>
                !!turn &&
                typeof turn === 'object' &&
                ((turn as Record<string, unknown>).role === 'user' ||
                  (turn as Record<string, unknown>).role === 'assistant') &&
                typeof (turn as Record<string, unknown>).content === 'string',
            )
            .slice(0, 100)
        : undefined;
      return await inferenceStreamService.runGeneration(a.projectId as string, {
        prompt,
        maxNewTokens,
        temperature,
        history,
      });
    } catch (err: unknown) {
      return { success: false, error: err instanceof Error ? err.message : 'Unknown error' };
    }
  });

  ipcMain.handle('inference:stop-generation', async (_event, args: unknown) => {
    const a = (args ?? {}) as Record<string, unknown>;
    if (!validateProjectId(a.projectId)) return { success: false, error: 'Invalid project ID' };
    return await inferenceStreamService.stopGeneration(a.projectId as string);
  });

  // ===================== Client Projects ("models I can train") =====================

  ipcMain.handle('client:list-projects', async () => {
    try {
      return await clientProjectService.listProjects();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:client:list-projects] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('client:get-connection', async (_event, projectId: unknown) => {
    try {
      if (!validateProjectId(projectId)) {
        log.error(`[IPC:client:get-connection] Rejected invalid project ID: ${String(projectId)}`);
        return { success: false, error: 'Invalid project ID' };
      }
      return await clientProjectService.getConnection(projectId as string);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[IPC:client:get-connection] Failed: ${message}`);
      return { success: false, error: message };
    }
  });

  ipcMain.handle('device:capabilities', async () => {
    try {
      return { success: true, capabilities: collectDeviceCapabilities() };
    } catch (e) {
      return { success: false, error: e instanceof Error ? e.message : 'capability probe failed' };
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

  log.info('[IPC] All handlers registered: docker:start-training, docker:stop-training, docker:get-status, hardware:detect, auth:login, auth:logout, auth:check, auth:set-server-url, auth:get-server-url, dialog:open-directory, inference:list-models, inference:run, updater:install');
}
