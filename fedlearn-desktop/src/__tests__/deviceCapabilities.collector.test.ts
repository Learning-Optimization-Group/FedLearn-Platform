import { collectDeviceCapabilities } from '../main/deviceCapabilities.collector';

describe('collectDeviceCapabilities', () => {
  test('returns real RAM and a known OS name', () => {
    const caps = collectDeviceCapabilities();
    expect(caps.ramGb).toBeGreaterThan(0);
    expect(['android', 'ios', 'macos', 'windows', 'linux']).toContain(caps.osName);
  });

  test('freeStorageGb is a non-negative number or undefined', () => {
    const caps = collectDeviceCapabilities();
    if (caps.freeStorageGb !== undefined) {
      expect(caps.freeStorageGb).toBeGreaterThanOrEqual(0);
    }
  });

  test('desktop OS leaves phone-only and battery fields undefined', () => {
    const caps = collectDeviceCapabilities();
    // On CI/dev this runs on macos/linux/windows — never a phone.
    expect(['macos', 'windows', 'linux']).toContain(caps.osName);
    expect(caps.batteryPct).toBeUndefined();
    expect(caps.npuTops).toBeUndefined();
  });
});
