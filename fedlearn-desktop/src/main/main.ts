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

import { app, BrowserWindow, Menu, session, crashReporter } from 'electron';
import type { MenuItemConstructorOptions } from 'electron';
import * as path from 'path';
import log from 'electron-log';
import { registerIpcHandlers, getDockerService } from './ipc.handlers';
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

const isDev = process.env.NODE_ENV !== 'production' && !app.isPackaged;

// Application menu with STANDARD ROLES ONLY: restores the system copy/paste,
// zoom, and window shortcuts without custom items or IPC. Section-switching
// shortcuts (Cmd/Ctrl+1..3) live in a renderer keydown listener, deliberately
// NOT here — a menu item reaching the renderer would require a new IPC channel.
function setApplicationMenu(): void {
  const template: MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin'
      ? ([{ role: 'appMenu' }] as MenuItemConstructorOptions[])
      : []),
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    // Shell layout budget: 64px rail + ~380px setup column + usable log pane
    // needs >= 1024 wide; drag strip + checklist + logs + status bar needs
    // >= 700 tall.
    width: 1360,
    height: 860,
    minWidth: 1024,
    minHeight: 700,
    title: 'FedLearn Desktop',
    // Mirrors the light canvas token from design/tokens.json — the main
    // process cannot read CSS vars, so this literal must be kept in sync
    // manually on any palette swap.
    backgroundColor: '#F6F3EE',
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

  // CSP is injected via a <meta> tag in index.html for packaged (file://) builds
  // (baked in at build time by HtmlWebpackPlugin — see webpack.csp.js and
  // webpack.prod.config.js), because Chromium's interpretation of 'self' under
  // file:// origins is inconsistent and can block legitimate scripts bundled in
  // the asar. For dev builds served over HTTP, the response-header approach
  // works correctly, and 'unsafe-eval' is required here because webpack's
  // development build uses the `eval` devtool — the packaged production build
  // carries neither this header nor 'unsafe-eval' in its meta CSP. Fonts are
  // bundled locally (src/renderer/fonts.css), so no remote font host is needed
  // in either environment.
  if (isDev) {
    session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
      callback({
        responseHeaders: {
          ...details.responseHeaders,
          'Content-Security-Policy': [
            [
              "default-src 'self'",
              "script-src 'self' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "font-src 'self'",
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
  setApplicationMenu();
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

// Drain any running training before exit. Containers are created with AutoRemove:false and the
// native client is spawned non-detached, so quitting mid-run would otherwise orphan a Jetson
// Docker container until it is lazily cleaned up on the next run. This covers Cmd+Q and — on
// non-macOS — closing the last window (window-all-closed -> app.quit() above, which fires
// before-quit). On macOS, closing the window does NOT quit the app, so training keeps running
// under the still-live app and nothing is orphaned. We intercept the first quit and best-effort
// drain, with a hard timeout so a wedged Docker daemon can't make the app unquittable.
let isDraining = false;
app.on('before-quit', (event) => {
  if (isDraining) {
    return; // drain already in progress; let the eventual app.exit(0) proceed
  }
  // getDockerService() is non-undefined whenever IPC handlers registered (the normal case after
  // startup); it's undefined only if registration threw. stopTraining() is a cheap no-op when
  // nothing is running, so we don't gate on training state here.
  const docker = getDockerService();
  if (!docker) {
    return;
  }
  isDraining = true;
  event.preventDefault();
  log.info('[Main] before-quit: draining any active training before exit');

  let exited = false;
  const exit = () => { if (!exited) { exited = true; app.exit(0); } };
  // Hard cap: a hung/unresponsive Docker daemon must never make quit hang forever. Must exceed
  // the Jetson drain's worst responsive case — stopDockerContainer does container.stop({t:10})
  // (up to ~10s for a SIGTERM-ignoring container) THEN container.remove — so 8s would force-exit
  // before remove() and orphan the container. 15s covers the slow-but-responsive path; only a
  // genuinely wedged daemon (where cleanup is impossible anyway) hits this backstop.
  const hardTimeout = setTimeout(() => {
    log.warn('[Main] before-quit: drain exceeded timeout; forcing exit');
    exit();
  }, 15000);

  Promise.resolve(docker.stopTraining())
    .catch((err) => log.error('[Main] before-quit: stopTraining failed', err))
    .finally(() => { clearTimeout(hardTimeout); exit(); });
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
