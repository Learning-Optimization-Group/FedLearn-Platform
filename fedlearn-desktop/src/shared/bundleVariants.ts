// fedlearn-desktop/src/shared/bundleVariants.ts
//
// Single source of truth for the desktop app's shippable "variants".
//
// A variant is one (platform, arch, gpuVariant) combination we actually build
// and publish. Two things vary per variant and used to be scattered across
// electron-builder.yml, package.json scripts, and scripts/check-native-bundle.js:
//
//   1. The embedded native PyInstaller client bundle differs by platform+arch
//      (client-docker/packaging/build-*.{sh,ps1} emit a per-target bundle).
//   2. Windows additionally splits into a CPU-only build and a CUDA build
//      (see package.json `package:win:cpu` / `package:win:cuda`).
//
// This module is the one place that enumerates them. It intentionally has NO
// imports (no electron, no node) so it stays importable from every process,
// from jest, AND from the plain-node preflight script — scripts/check-native-bundle.js
// transpiles this file with the TypeScript compiler API and reads the maps below.
// Keep it dependency-free and self-contained.
//
// Derivation (kept in sync with electron-builder.yml + package.json):
//   - mac   → arm64 / none            (x64 mac target was dropped: unshipped,
//                                       untested on real Intel hardware)
//   - linux → x64 / none, arm64 / none
//   - win   → x64 / cpu, x64 / cuda

export type BundlePlatform = 'mac' | 'win' | 'linux';
export type BundleArch = 'arm64' | 'x64';
export type GpuVariant = 'cpu' | 'cuda' | 'none';

export interface BundleVariant {
  /** Target OS family. */
  platform: BundlePlatform;
  /** CPU architecture of the artifact. */
  arch: BundleArch;
  /** GPU flavor of the embedded client: Windows splits cpu/cuda; others 'none'. */
  gpuVariant: GpuVariant;
  /** Stable, unique id for this variant (platform-arch[-gpu]). */
  nativeBundleId: string;
  /**
   * electron-builder artifact filename for this variant's primary installer.
   * `${version}` is a literal template token (electron-builder / the npm script
   * substitutes it at build time). The mac dmg and both Windows names are pinned
   * in config (electron-builder.yml `dmg.artifactName`, package.json
   * `package:win:*` `-c.nsis.artifactName`); the linux AppImage names follow
   * electron-builder's default `${productName}-${version}-${arch}` template
   * (the linux target sets no custom artifactName).
   */
  artifactName: string;
}

/**
 * Every shippable variant. Derived from (post-x64-drop) electron-builder.yml and
 * the package.json packaging scripts. Adding/removing a build target means
 * editing THIS array (and the corresponding electron-builder target).
 */
export const BUNDLE_VARIANTS: readonly BundleVariant[] = [
  {
    platform: 'mac',
    arch: 'arm64',
    gpuVariant: 'none',
    nativeBundleId: 'mac-arm64',
    artifactName: 'FedLearn Desktop-${version}-arm64.dmg',
  },
  {
    platform: 'linux',
    arch: 'x64',
    gpuVariant: 'none',
    nativeBundleId: 'linux-x64',
    artifactName: 'FedLearn Desktop-${version}-x64.AppImage',
  },
  {
    platform: 'linux',
    arch: 'arm64',
    gpuVariant: 'none',
    nativeBundleId: 'linux-arm64',
    artifactName: 'FedLearn Desktop-${version}-arm64.AppImage',
  },
  {
    platform: 'win',
    arch: 'x64',
    gpuVariant: 'cpu',
    nativeBundleId: 'win-x64-cpu',
    artifactName: 'FedLearn-Desktop-Setup-${version}-cpu.exe',
  },
  {
    platform: 'win',
    arch: 'x64',
    gpuVariant: 'cuda',
    nativeBundleId: 'win-x64-cuda',
    artifactName: 'FedLearn-Desktop-Setup-${version}-cuda.exe',
  },
] as const;

/**
 * Entry binary that PyInstaller emits inside the native client bundle, per
 * platform. Source of truth for scripts/check-native-bundle.js.
 */
export const PLATFORM_NATIVE_BINARY: Record<BundlePlatform, string> = {
  win: 'fedlearn-client.exe',
  mac: 'fedlearn-client',
  linux: 'fedlearn-client',
};

/**
 * Packaging command that produces the native client bundle, per platform.
 * Source of truth for scripts/check-native-bundle.js's "build it first" hint.
 */
export const PLATFORM_BUILD_COMMAND: Record<BundlePlatform, string> = {
  win: 'client-docker/packaging/build-win-cpu.ps1   (or build-win-cuda.ps1)',
  mac: 'client-docker/packaging/build-mac.sh',
  linux: 'client-docker/packaging/build-linux.sh',
};

/** The native entry binary name for a platform (throws on unknown platform). */
export function nativeBinaryForPlatform(platform: BundlePlatform): string {
  return PLATFORM_NATIVE_BINARY[platform];
}

/** The bundle-build command hint for a platform. */
export function buildCommandForPlatform(platform: BundlePlatform): string {
  return PLATFORM_BUILD_COMMAND[platform];
}

/** All variants for a given platform. */
export function variantsForPlatform(platform: BundlePlatform): BundleVariant[] {
  return BUNDLE_VARIANTS.filter((v) => v.platform === platform);
}
