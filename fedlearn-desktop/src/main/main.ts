// =============================================================================
// FedLearn Desktop — Main Process Entry Point
// =============================================================================
// Initializes the BrowserWindow with strict security settings per Section 5.1:
//   - nodeIntegration: false
//   - contextIsolation: true
//   - sandbox: true
// Sets Content Security Policy headers on every window.
// No use of the deprecated 'remote' module.
// =============================================================================

import { app, BrowserWindow, session, crashReporter } from 'electron';
import * as path from 'path';
import log from 'electron-log';
import { registerIpcHandlers, stopAllTrainingForShutdown } from './ipc.handlers';
import { initializeUpdater } from './updater';

// Crash reports written to disk — visible via app.getPath('crashDumps').
// No remote submission configured; dumps stay local for debugging.
crashReporter.start({ uploadToServer: false });

// Configure electron-log as the sole logging mechanism
log.transports.file.level = 'info';
log.transports.console.level = 'debug';
log.initialize();

// Replace default console with electron-log in main process
Object.assign(console, log.functions);

let mainWindow: BrowserWindow | null = null;
let quittingCleanly = false;

const isDev = process.env.NODE_ENV !== 'production' && !app.isPackaged;

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    title: 'FedLearn Desktop',
    backgroundColor: '#0a0a0f',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      // ========== SECURITY: Non-Negotiable ==========
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      // ===============================================
      preload: path.join(__dirname, '..', 'preload', 'preload.js'),
      devTools: isDev,
      webSecurity: true,
      allowRunningInsecureContent: false,
      experimentalFeatures: false,
    },
  });

  // ========== Content Security Policy ==========
  // Applied to every response via session-level header injection.
  // This is a defense-in-depth layer on top of contextIsolation.
  //
  // connect-src must explicitly list backend API origins. Under a packaged file:// origin,
  // 'self' does NOT resolve to the API host, so without these entries every XHR/fetch/WebSocket
  // to the backend is blocked. FEDLEARN_API_ORIGINS can override at runtime (comma-separated).
  const apiOriginsFromEnv = (process.env.FEDLEARN_API_ORIGINS || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const defaultApiOrigins = isDev
    ? ['http://localhost:8081', 'ws://localhost:8081', 'http://localhost:9000', 'ws://localhost:9000']
    : [];
  const apiConnectSrc = [...defaultApiOrigins, ...apiOriginsFromEnv].join(' ');

  // CSP is injected via a <meta> tag in index.html for packaged (file://) builds,
  // because Chromium's interpretation of 'self' under file:// origins is inconsistent
  // and can block legitimate scripts bundled in the asar. For dev builds served over
  // HTTP, the response-header approach works correctly.
  if (isDev) {
    session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
      callback({
        responseHeaders: {
          ...details.responseHeaders,
          'Content-Security-Policy': [
            [
              "default-src 'self'",
              "script-src 'self' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
              "font-src 'self' https://fonts.gstatic.com",
              "img-src 'self' data:",
              `connect-src 'self' ${apiConnectSrc}`.trim(),
              "frame-src 'none'",
              "object-src 'none'",
              "base-uri 'self'",
            ].join('; '),
          ],
        },
      });
    });
  }

  // Register all IPC handlers (docker, auth). An exception here must not
  // prevent the renderer from loading — otherwise the window stays black
  // and the user has no way to recover (e.g. log out, switch server).
  try {
    registerIpcHandlers(mainWindow);
  } catch (err) {
    log.error('[Main] registerIpcHandlers failed; renderer will still load', err);
  }

  // Load the renderer
  if (isDev) {
    mainWindow.loadURL('http://localhost:9000');
    log.info('[Main] Loaded renderer from dev server at http://localhost:9000');
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
    log.info('[Main] Loaded renderer from packaged file');
  }

  // Initialize auto-updater
  initializeUpdater(mainWindow);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  log.info('[Main] BrowserWindow created with strict security settings');
  log.info('[Main] nodeIntegration=false, contextIsolation=true, sandbox=true');
}

// ========== App Lifecycle ==========

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // macOS: re-create window when dock icon clicked and no other windows open
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('before-quit', async (event) => {
  if (quittingCleanly) {
    return;
  }

  event.preventDefault();
  quittingCleanly = true;

  try {
    await stopAllTrainingForShutdown();
  } catch (err) {
    log.error('[Main] cleanup on quit failed', err);
  }

  app.quit();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// ========== Security Hardening ==========

// Prevent new window creation from renderer
app.on('web-contents-created', (_event, contents) => {
  contents.setWindowOpenHandler(() => {
    log.warn('[Security] Blocked attempt to open new window from renderer');
    return { action: 'deny' };
  });

  // Prevent navigation away from the app.
  // In production, only allow file:// URLs rooted in the app's own directory.
  // In dev, also allow the webpack dev server origin.
  const appDir = isDev ? '' : path.join(__dirname, '..');
  contents.on('will-navigate', (event, url) => {
    let allowed = false;
    if (isDev && url.startsWith('http://localhost:9000')) {
      allowed = true;
    } else if (url.startsWith('file://')) {
      // Restrict file:// navigation to the packaged app directory
      const filePath = decodeURIComponent(new URL(url).pathname);
      allowed = appDir ? filePath.startsWith(appDir) : true;
    }

    if (!allowed) {
      event.preventDefault();
      log.warn(`[Security] Blocked navigation to: ${url}`);
    }
  });
});

// Disable GPU acceleration if not needed for the UI itself
// (GPU compute happens inside Docker containers, not in Electron)
app.disableHardwareAcceleration();
