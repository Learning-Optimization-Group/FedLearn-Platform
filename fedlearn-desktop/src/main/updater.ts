import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import { BrowserWindow } from 'electron';

// autoUpdater is a process-wide singleton. createWindow() can run more than once
// per process (e.g. macOS 'activate' re-creates the window after all windows are
// closed), so guard registration to happen EXACTLY once — otherwise every
// updater event listener stacks up and each IPC message fires N times.
let updaterInitialized = false;

export function initializeUpdater(mainWindow: BrowserWindow) {
  if (updaterInitialized) {
    return;
  }
  updaterInitialized = true;

  // Configure logging for the auto-updater
  autoUpdater.logger = log;
  log.transports.file.level = 'info';

  log.info('App starting... Initializing autoUpdater');

  // Auto-download an update as soon as one is found. The UpdateBanner surfaces
  // the "downloading in background… → progress → restart to install" flow, so
  // there is no separate download prompt. autoInstallOnAppQuit then applies the
  // already-downloaded update on the next quit.
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;

  // Let's force dev update config if not in production to test
  // NOTE: If testing locally with "npm run dev", you need to set forceDevUpdateConfig = true.
  autoUpdater.forceDevUpdateConfig = process.env.NODE_ENV === 'development';

  autoUpdater.on('checking-for-update', () => {
    log.info('Checking for update...');
  });

  autoUpdater.on('update-available', (info) => {
    log.info('Update available.');
    mainWindow.webContents.send('updater:update-available', info);
  });

  autoUpdater.on('update-not-available', () => {
    log.info('Update not available.');
  });

  autoUpdater.on('error', (err) => {
    log.error('Error in auto-updater. ' + err);
  });

  autoUpdater.on('download-progress', (progressObj) => {
    let log_message = "Download speed: " + progressObj.bytesPerSecond;
    log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
    log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
    log.info(log_message);
    mainWindow.webContents.send('updater:download-progress', progressObj);
  });

  autoUpdater.on('update-downloaded', (info) => {
    log.info('Update downloaded');
    mainWindow.webContents.send('updater:update-downloaded', info);
  });

  // Check for updates immediately
  autoUpdater.checkForUpdatesAndNotify();
}
