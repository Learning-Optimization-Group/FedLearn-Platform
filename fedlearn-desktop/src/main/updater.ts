import { autoUpdater } from 'electron-updater';
import log from 'electron-log';
import { BrowserWindow } from 'electron';

export function initializeUpdater(mainWindow: BrowserWindow) {
  // Configure logging for the auto-updater
  autoUpdater.logger = log;
  (autoUpdater.logger as any).transports.file.level = 'info';

  log.info('App starting... Initializing autoUpdater');

  // Disable auto-download so we can prompt the user first (or show progress)
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

  autoUpdater.on('update-not-available', (info) => {
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
