export const ipcMain = { handle: jest.fn() };
export const BrowserWindow = jest.fn();
export const dialog = { showOpenDialog: jest.fn() };

// `app` stand-in. DockerService reads `app.isPackaged` to phrase its
// bundle-missing error (packaged: "reinstall/rebuild" vs dev: "run from repo
// root"). Defaults to dev mode; tests that need packaged mode override it.
export const app = { isPackaged: false };

// safeStorage: default to "encryption available" with a reversible round-trip
// so AuthService's on-disk store/decrypt path is exercisable without a real
// OS keychain. Tests that need the no-keyring fallback override
// isEncryptionAvailable to return false per-case.
export const safeStorage = {
  isEncryptionAvailable: jest.fn(() => true),
  encryptString: jest.fn((value: string) => Buffer.from(value, 'utf8')),
  decryptString: jest.fn((buffer: Buffer) => buffer.toString('utf8')),
};
