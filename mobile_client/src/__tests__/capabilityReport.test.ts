// Stage 3 D1: what a device reports about itself when it enrolls. Informational only: it never decides whether the
// device may train (the contract gate and the qualification probe do). Every fact is best-effort, and every value is
// kept within the bounds the server accepts, so a strange device string can never get an enrollment refused.
import {
  collectCapabilityReport, diagnosticsWithReport, formatCapabilityReport, type CapabilitySources,
} from '../lib/capabilityReport';

function sources(over: Partial<CapabilitySources> = {}): CapabilitySources {
  return {
    platform: 'android',
    apiLevel: async () => 27,
    systemVersion: () => '8.1.0',
    abis: async () => ['arm64-v8a', 'armeabi-v7a'],
    totalMemory: async () => 3_000_000_000,
    freeDiskStorage: async () => 20_000_000_000,
    model: () => 'vivo 1805',
    version: () => '2.1.0',
    buildNumber: () => '7',
    runtimeCompatibility: async () => ({ bridgeAbiVersion: 3, protocolVersion: 2 }),
    deviceMetrics: async () => ({ peakRssBytes: 1, thermalState: 'NOMINAL', batteryLevel: 0.8, batteryCharging: true }),
    ...over,
  };
}

test('reports the platform, OS level, ABIs, memory, storage, app, bridge, thermal and battery state', async () => {
  expect(await collectCapabilityReport(sources())).toEqual({
    platform: 'android', osVersion: '8.1.0', apiLevel: 27, abis: ['arm64-v8a', 'armeabi-v7a'],
    totalRamBytes: 3_000_000_000, freeStorageBytes: 20_000_000_000, deviceModel: 'vivo 1805', appVersion: '2.1.0',
    appBuild: '7', bridgeAbiVersion: 3, protocolVersion: 2, thermalState: 'NOMINAL', batteryPct: 80,
  });
});

test('a fact that cannot be read is left out, never guessed', async () => {
  const report = await collectCapabilityReport(sources({
    apiLevel: async () => { throw new Error('no'); },
    totalMemory: async () => { throw new Error('no'); },
    runtimeCompatibility: async () => { throw new Error('no native core'); },
    deviceMetrics: async () => { throw new Error('no'); },
  }));
  expect(report.apiLevel).toBeUndefined();
  expect(report.totalRamBytes).toBeUndefined();
  expect(report.bridgeAbiVersion).toBeUndefined();
  expect(report.batteryPct).toBeUndefined();
  expect(report.platform).toBe('android');
});

test('values are kept within the bounds the server accepts', async () => {
  const report = await collectCapabilityReport(sources({
    model: () => 'm'.repeat(200),
    abis: async () => Array.from({ length: 20 }, (_, i) => `abi-${i}`),
    deviceMetrics: async () => ({ peakRssBytes: 1, thermalState: 'T'.repeat(99), batteryLevel: 1.7,
      batteryCharging: false }),
    freeDiskStorage: async () => -1,
  }));
  expect(report.deviceModel).toHaveLength(64);
  expect(report.abis).toHaveLength(8);
  expect(report.thermalState).toHaveLength(32);
  expect(report.batteryPct).toBe(100);
  expect(report.freeStorageBytes).toBeUndefined();
});

test('an unknown battery level (negative) is left out', async () => {
  const report = await collectCapabilityReport(sources({
    deviceMetrics: async () => ({ peakRssBytes: 1, thermalState: 'NOMINAL', batteryLevel: -1, batteryCharging: false }),
  }));
  expect(report.batteryPct).toBeUndefined();
});

test('formats the report for the shared diagnostics, one fact per line', async () => {
  const text = formatCapabilityReport(await collectCapabilityReport(sources()));
  expect(text).toContain('platform: android');
  expect(text).toContain('apiLevel: 27');
  expect(text).toContain('abis: arm64-v8a, armeabi-v7a');
});

test('the shared diagnostics carry the device report after the journal, or say it could not be read', async () => {
  const report = await collectCapabilityReport(sources());
  const text = diagnosticsWithReport('FedLearn on-device diagnostics\nline', report);
  expect(text.startsWith('FedLearn on-device diagnostics\nline\n\nThis device\n')).toBe(true);
  expect(text).toContain('deviceModel: vivo 1805');
  expect(diagnosticsWithReport('journal', undefined)).toBe('journal\n\nThis device\n(the device report could not be read)');
});
