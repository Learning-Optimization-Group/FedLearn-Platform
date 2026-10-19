// Manual mock for react-native-device-info — the real module constructs a NativeEventEmitter on import, which needs
// the native bridge. Suites that care about specific values still override it with jest.mock(..., factory).
const DeviceInfo = {
  getModel: jest.fn(() => 'test-device'),
  getSystemVersion: jest.fn(() => '0'),
  getVersion: jest.fn(() => '0.0.0'),
  getBuildNumber: jest.fn(() => '0'),
  getTotalMemory: jest.fn(async () => 0),
  getFreeDiskStorage: jest.fn(async () => 0),
  supportedAbis: jest.fn(async () => ['arm64-v8a']),
};

export default DeviceInfo;
