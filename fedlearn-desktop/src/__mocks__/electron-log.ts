const log = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
  // electron-updater assigns `autoUpdater.logger = log` and then reaches into
  // `.transports.file.level`, so the mock must expose a transports shape.
  transports: {
    file: { level: 'info' as string },
    console: { level: 'debug' as string },
  },
  initialize: jest.fn(),
  functions: {},
};
export default log;
