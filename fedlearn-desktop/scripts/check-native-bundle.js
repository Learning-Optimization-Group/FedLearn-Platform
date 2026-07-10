#!/usr/bin/env node
// =============================================================================
// Preflight: native client bundle must exist before electron-builder runs
// =============================================================================
// The zero-dependency promise (no Docker, no Python on the end-user machine)
// depends entirely on the PyInstaller bundle being shipped inside the installer
// via electron-builder's `extraResources` (see electron-builder.yml). That
// bundle is produced by a SEPARATE build step:
//
//   client-docker/packaging/build-win-cpu.ps1   (Windows CPU)
//   client-docker/packaging/build-win-cuda.ps1  (Windows CUDA)
//   client-docker/packaging/build-mac.sh        (macOS)
//   client-docker/packaging/build-linux.sh      (Linux)
//
// If a developer runs `npm run package:*` locally WITHOUT first building the
// bundle, electron-builder happily produces an installer with no native client.
// The app then launches fine but fails the instant the user clicks "Start"
// ("Native training bundle not found at <resources>/fedlearn-client"). This
// guard turns that silent shipping bug into a loud, actionable failure.
//
// The release CI (.github/workflows/release-desktop.yml) builds the bundle in a
// dedicated step and has its own verification, so it does not rely on this
// guard — this protects the LOCAL manual packaging path.
// =============================================================================

const fs = require('fs');
const path = require('path');

// Platform/arch/variant knowledge lives in ONE place: the typed manifest at
// src/shared/bundleVariants.ts. That file is TypeScript and has no build output
// available at preflight time (this script runs BEFORE `npm run build`), so we
// transpile it in-memory with the TypeScript compiler API — the same thing
// ts-node/ts-jest do — and read its exported maps. The manifest is deliberately
// import-free and self-contained so this load stays trivial. The input is a
// trusted, checked-in repo file (not user input), so the `new Function` eval
// below carries no injection surface beyond "can already edit the repo".
function loadBundleManifest() {
  const ts = require('typescript');
  const manifestPath = path.resolve(
    __dirname,
    '..',
    'src',
    'shared',
    'bundleVariants.ts',
  );
  const source = fs.readFileSync(manifestPath, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2019,
    },
  });
  const mod = { exports: {} };
  new Function('module', 'exports', 'require', outputText)(mod, mod.exports, require);
  return mod.exports;
}

const manifest = loadBundleManifest();
// Entry binary PyInstaller emits inside the bundle, per platform (from manifest).
const BINARY_BY_PLATFORM = manifest.PLATFORM_NATIVE_BINARY;
// "Build it first" hint per platform (from manifest).
const BUILD_CMD_BY_PLATFORM = manifest.PLATFORM_BUILD_COMMAND;

function targetPlatform() {
  const arg = (process.argv[2] || '').toLowerCase();
  if (arg && BINARY_BY_PLATFORM[arg]) return arg;
  // Fall back to the host platform if no explicit target was passed.
  if (process.platform === 'win32') return 'win';
  if (process.platform === 'darwin') return 'mac';
  return 'linux';
}

function main() {
  const platform = targetPlatform();
  const bundleDir = path.resolve(
    __dirname,
    '..',
    '..',
    'client-docker',
    'packaging',
    'dist',
    'fedlearn-client',
  );
  const binary = path.join(bundleDir, BINARY_BY_PLATFORM[platform]);

  if (!fs.existsSync(bundleDir) || !fs.existsSync(binary)) {
    console.error('\n  ✖ Native client bundle is MISSING.\n');
    console.error(`    Expected: ${binary}`);
    console.error('\n    electron-builder would ship an installer with no Python');
    console.error('    training client, and the app would fail on "Start training".\n');
    console.error(`    Build it first (target=${platform}):`);
    console.error(`      ${BUILD_CMD_BY_PLATFORM[platform]}\n`);
    process.exit(1);
  }

  console.log(`  ✓ Native client bundle present for ${platform}: ${bundleDir}`);
}

main();
