import { evaluateEligibility, eligibilitySummary } from '../shared/evaluateEligibility';
import type { DeviceCapabilities, DeviceRequirements } from '../shared/deviceCapabilities.types';

const desktop = (over: Partial<DeviceCapabilities> = {}): DeviceCapabilities => ({
  ramGb: 16, freeStorageGb: 100, osName: 'macos', ...over,
});
const phone = (over: Partial<DeviceCapabilities> = {}): DeviceCapabilities => ({
  ramGb: 8, freeStorageGb: 20, osName: 'android', osVersion: '30', ...over,
});

describe('evaluateEligibility — hard gates', () => {
  test('null requirements → eligible, no findings', () => {
    const r = evaluateEligibility(desktop(), null);
    expect(r.eligible).toBe(true);
    expect(r.hardFailures).toHaveLength(0);
  });

  test('meets all → eligible', () => {
    const req: DeviceRequirements = { minRamGb: 8, minStorageGb: 2, mobileSafe: true };
    expect(evaluateEligibility(desktop(), req).eligible).toBe(true);
  });

  test('RAM below floor → hard failure, not eligible', () => {
    const r = evaluateEligibility(desktop({ ramGb: 4 }), { minRamGb: 8 });
    expect(r.eligible).toBe(false);
    expect(r.hardFailures[0]).toMatch(/RAM/);
  });

  test('free storage below floor → hard failure', () => {
    const r = evaluateEligibility(desktop({ freeStorageGb: 1 }), { minStorageGb: 5 });
    expect(r.eligible).toBe(false);
    expect(r.hardFailures[0]).toMatch(/storage/i);
  });

  test('mobileSafe=false hard-fails on a phone', () => {
    const r = evaluateEligibility(phone(), { mobileSafe: false });
    expect(r.eligible).toBe(false);
    expect(r.hardFailures[0]).toMatch(/phone/i);
  });

  test('mobileSafe=false does NOT gate a desktop', () => {
    expect(evaluateEligibility(desktop(), { mobileSafe: false }).eligible).toBe(true);
  });

  test('android OS below floor → hard failure (phone only)', () => {
    const r = evaluateEligibility(phone({ osVersion: '27' }), { minOsAndroid: 34 });
    expect(r.eligible).toBe(false);
    expect(r.hardFailures[0]).toMatch(/Android/);
  });

  test('ios version below floor → hard failure', () => {
    const r = evaluateEligibility(phone({ osName: 'ios', osVersion: '15.0' }), { minOsIos: '16.0' });
    expect(r.eligible).toBe(false);
  });

  test('ios version meeting floor → eligible', () => {
    const r = evaluateEligibility(phone({ osName: 'ios', osVersion: '17.2' }), { minOsIos: '16.0' });
    expect(r.eligible).toBe(true);
  });
});

describe('evaluateEligibility — soft gates', () => {
  test('NPU below floor → soft warning, still eligible', () => {
    const r = evaluateEligibility(desktop({ npuTops: 5 }), { minNpuTops: 35 });
    expect(r.eligible).toBe(true);
    expect(r.softWarnings.some(w => /NPU/.test(w))).toBe(true);
  });

  test('NPU unknown → soft warning, still eligible', () => {
    const r = evaluateEligibility(desktop({ npuTops: undefined }), { minNpuTops: 35 });
    expect(r.eligible).toBe(true);
    expect(r.softWarnings.some(w => /NPU/.test(w))).toBe(true);
  });

  test('requiresWifi while off wifi → soft warning', () => {
    const r = evaluateEligibility(desktop({ onWifi: false }), { requiresWifi: true });
    expect(r.softWarnings.some(w => /Wi-?Fi/i.test(w))).toBe(true);
  });

  test('low battery → soft warning', () => {
    const r = evaluateEligibility(phone({ batteryPct: 10 }), { minBatteryPct: 30 });
    expect(r.softWarnings.some(w => /[Bb]attery/.test(w))).toBe(true);
  });
});

describe('eligibilitySummary', () => {
  test('eligible + no warnings → ✅', () => {
    expect(eligibilitySummary({ eligible: true, hardFailures: [], softWarnings: [] }).marker).toBe('✅');
  });
  test('eligible + warnings → ℹ️', () => {
    expect(eligibilitySummary({ eligible: true, hardFailures: [], softWarnings: ['slow'] }).marker).toBe('ℹ️');
  });
  test('hard failure → ⚠️ with the failure lines', () => {
    const s = eligibilitySummary({ eligible: false, hardFailures: ['needs 8 GB'], softWarnings: [] });
    expect(s.marker).toBe('⚠️');
    expect(s.lines).toContain('needs 8 GB');
  });
});
