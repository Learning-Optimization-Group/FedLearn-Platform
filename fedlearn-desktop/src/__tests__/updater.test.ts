// DE-4: initializeUpdater must register autoUpdater listeners EXACTLY once.
//
// autoUpdater is a process-wide singleton; createWindow() can run more than once
// per process (macOS 'activate' re-creates the window after all windows close),
// so a second initializeUpdater() call must NOT stack duplicate listeners — that
// would fire every updater IPC event N times against the renderer.

// electron-updater is not stubbed by the shared __mocks__, so mock it inline.
// The factory captures no out-of-scope vars, so jest's hoisting is happy.
jest.mock('electron-updater', () => ({
  autoUpdater: {
    on: jest.fn(),
    logger: undefined,
    autoDownload: false,
    autoInstallOnAppQuit: false,
    forceDevUpdateConfig: false,
    checkForUpdatesAndNotify: jest.fn().mockResolvedValue(undefined),
  },
}));

import { autoUpdater } from 'electron-updater';
import { initializeUpdater } from '../main/updater';

describe('initializeUpdater (DE-4)', () => {
  it('registers autoUpdater listeners exactly once across repeated calls', () => {
    const on = autoUpdater.on as unknown as jest.Mock;
    const check = autoUpdater.checkForUpdatesAndNotify as unknown as jest.Mock;

    const fakeWindow = { webContents: { send: jest.fn() } } as never;

    initializeUpdater(fakeWindow);
    const afterFirst = on.mock.calls.length;
    expect(afterFirst).toBeGreaterThan(0); // listeners were registered on the first call

    // Simulate macOS 'activate' -> createWindow() -> initializeUpdater() again.
    initializeUpdater(fakeWindow);
    const afterSecond = on.mock.calls.length;

    // No additional listeners were registered on the second call.
    expect(afterSecond).toBe(afterFirst);
    // The initial update check ran only once, too.
    expect(check).toHaveBeenCalledTimes(1);
  });
});
