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

import { app, BrowserWindow, session } from 'electron';
import * as path from 'path';
import log from 'electron-log';
import { registerIpcHandlers } from './ipc.handlers';

// Configure electron-log as the sole logging mechanism
log.transports.file.level = 'info';
log.transports.console.level = 'debug';
log.initialize();

// Replace default console with electron-log in main process
Object.assign(console, log.functions);

let mainWindow: BrowserWindow | null = null;

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
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data:",
            "connect-src 'self'",
            "frame-src 'none'",
            "object-src 'none'",
            "base-uri 'self'",
          ].join('; '),
        ],
      },
    });
  });

  // Register all IPC handlers (docker, auth)
  registerIpcHandlers(mainWindow);

  // Load the renderer
  if (isDev) {
    mainWindow.loadURL('http://localhost:9000');
    log.info('[Main] Loaded renderer from dev server at http://localhost:9000');
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
    log.info('[Main] Loaded renderer from packaged file');
  }

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

  // Prevent navigation away from the app
  contents.on('will-navigate', (event, url) => {
    const parsedUrl = new URL(url);
    const allowedOrigins = ['http://localhost:9000', `file://`];
    const isAllowed = allowedOrigins.some((origin) => url.startsWith(origin));

    if (!isAllowed) {
      event.preventDefault();
      log.warn(`[Security] Blocked navigation to: ${parsedUrl.origin}`);
    }
  });
});

// Disable GPU acceleration if not needed for the UI itself
// (GPU compute happens inside Docker containers, not in Electron)
app.disableHardwareAcceleration();
