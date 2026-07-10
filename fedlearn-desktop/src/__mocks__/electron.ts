export const ipcMain = { handle: jest.fn() };
export const BrowserWindow = jest.fn();
export const dialog = { showOpenDialog: jest.fn() };

// safeStorage: default to "encryption available" with a reversible round-trip
// so AuthService's on-disk store/decrypt path is exercisable without a real
// OS keychain. Tests that need the no-keyring fallback override
// isEncryptionAvailable to return false per-case.
export const safeStorage = {
  isEncryptionAvailable: jest.fn(() => true),
  encryptString: jest.fn((value: string) => Buffer.from(value, 'utf8')),
  decryptString: jest.fn((buffer: Buffer) => buffer.toString('utf8')),
};
