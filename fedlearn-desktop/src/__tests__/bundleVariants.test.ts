// fedlearn-desktop/src/__tests__/bundleVariants.test.ts
//
// DE-10: the shippable-variant manifest is the single source of truth for
// platform/arch/gpu combinations we build. These tests pin the invariants that
// matter: mac is arm64-only (the x64 mac target was dropped), Windows ships both
// a cpu and a cuda build, every variant is fully described, and no (platform,
// arch, gpuVariant) tuple is duplicated.

import {
  BUNDLE_VARIANTS,
  PLATFORM_NATIVE_BINARY,
  PLATFORM_BUILD_COMMAND,
  nativeBinaryForPlatform,
  variantsForPlatform,
} from '../shared/bundleVariants';

describe('bundleVariants manifest (DE-10)', () => {
  test('mac is arm64-only — no x64 mac entry', () => {
    const macVariants = BUNDLE_VARIANTS.filter((v) => v.platform === 'mac');
    expect(macVariants.length).toBeGreaterThan(0);
    expect(macVariants.every((v) => v.arch === 'arm64')).toBe(true);
    expect(BUNDLE_VARIANTS.some((v) => v.platform === 'mac' && v.arch === 'x64')).toBe(false);
  });

  test('Windows ships both a cpu and a cuda variant', () => {
    const winVariants = BUNDLE_VARIANTS.filter((v) => v.platform === 'win');
    const gpuVariants = winVariants.map((v) => v.gpuVariant).sort();
    expect(gpuVariants).toEqual(['cpu', 'cuda']);
    expect(winVariants.every((v) => v.arch === 'x64')).toBe(true);
  });

  test('both Linux arches (x64 + arm64) ship', () => {
    const linuxArches = variantsForPlatform('linux')
      .map((v) => v.arch)
      .sort();
    expect(linuxArches).toEqual(['arm64', 'x64']);
  });

  test('every variant has a non-empty nativeBundleId and artifactName', () => {
    for (const v of BUNDLE_VARIANTS) {
      expect(typeof v.nativeBundleId).toBe('string');
      expect(v.nativeBundleId.trim().length).toBeGreaterThan(0);
      expect(typeof v.artifactName).toBe('string');
      expect(v.artifactName.trim().length).toBeGreaterThan(0);
    }
  });

  test('no duplicate (platform, arch, gpuVariant) tuples', () => {
    const tuples = BUNDLE_VARIANTS.map((v) => `${v.platform}|${v.arch}|${v.gpuVariant}`);
    expect(new Set(tuples).size).toBe(tuples.length);
  });

  test('nativeBundleId values are unique', () => {
    const ids = BUNDLE_VARIANTS.map((v) => v.nativeBundleId);
    expect(new Set(ids).size).toBe(ids.length);
  });

  test('native binary + build-command maps cover every shipped platform', () => {
    const platforms = new Set(BUNDLE_VARIANTS.map((v) => v.platform));
    for (const p of platforms) {
      expect(PLATFORM_NATIVE_BINARY[p]).toBeTruthy();
      expect(PLATFORM_BUILD_COMMAND[p]).toBeTruthy();
    }
    // Windows keeps the .exe suffix; posix targets do not.
    expect(nativeBinaryForPlatform('win')).toBe('fedlearn-client.exe');
    expect(nativeBinaryForPlatform('mac')).toBe('fedlearn-client');
    expect(nativeBinaryForPlatform('linux')).toBe('fedlearn-client');
  });
});
