import { Platform } from 'react-native';

// Mock device-info with explicit factory — global jest.setup.ts was dropped in Task 1.
// DeviceInfo is the default export; factory returns { default: {...} } so the
// `import DeviceInfo from 'react-native-device-info'` in deviceClass.ts resolves.
jest.mock('react-native-device-info', () => ({
  __esModule: true,
  default: {
    getTotalMemory: jest.fn(),
    getFreeDiskStorage: jest.fn(),
    getSystemVersion: jest.fn(),
  },
}));

jest.mock('@spec/NativeFedLearnCore', () => ({
  __esModule: true,
  default: { getDeviceMetrics: jest.fn() },
}));

// Import AFTER mocks are hoisted
import DeviceInfo from 'react-native-device-info';
import NativeFedLearnCore from '@spec/NativeFedLearnCore';
import { collectDeviceCapabilities } from '../lib/deviceClass';

const MockDeviceInfo = DeviceInfo as unknown as {
  getTotalMemory: jest.Mock;
  getFreeDiskStorage: jest.Mock;
  getSystemVersion: jest.Mock;
};
const NCore = NativeFedLearnCore as unknown as { getDeviceMetrics: jest.Mock };

describe('collectDeviceCapabilities', () => {
  beforeEach(() => jest.clearAllMocks());

  test('maps device-info + metrics into DeviceCapabilities', async () => {
    MockDeviceInfo.getTotalMemory.mockResolvedValue(8 * 1024 ** 3);
    MockDeviceInfo.getFreeDiskStorage.mockResolvedValue(20 * 1024 ** 3);
    MockDeviceInfo.getSystemVersion.mockReturnValue('17.2');
    NCore.getDeviceMetrics.mockResolvedValue({
      batteryLevel: 0.42,
      thermalState: 'nominal',
      batteryCharging: false,
      peakRssBytes: 0,
    });

    const caps = await collectDeviceCapabilities();
    expect(caps.ramGb).toBeCloseTo(8, 1);
    expect(caps.freeStorageGb).toBeCloseTo(20, 1);
    expect(caps.osName).toBe(Platform.OS); // 'ios' under the RN jest preset
    expect(caps.osVersion).toBe('17.2');
    expect(caps.batteryPct).toBe(42);
    expect(caps.npuTops).toBeUndefined();
    expect(caps.onWifi).toBeUndefined();
  });

  test('degrades gracefully when a native call fails', async () => {
    MockDeviceInfo.getTotalMemory.mockResolvedValue(4 * 1024 ** 3);
    MockDeviceInfo.getFreeDiskStorage.mockRejectedValue(new Error('nope'));
    MockDeviceInfo.getSystemVersion.mockReturnValue('14');
    NCore.getDeviceMetrics.mockRejectedValue(new Error('no metrics'));

    const caps = await collectDeviceCapabilities();
    expect(caps.ramGb).toBeCloseTo(4, 1);
    expect(caps.freeStorageGb).toBeUndefined();
    expect(caps.batteryPct).toBeUndefined();
  });
});
