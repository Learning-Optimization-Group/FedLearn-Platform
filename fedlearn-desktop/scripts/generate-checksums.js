// scripts/generate-checksums.js — DE-12: emit SHA256 checksums for the release installers AND the
// embedded native client bundle, so a downloader (or an auditor) can verify integrity end-to-end.
//
// Wired as electron-builder's `afterAllArtifactBuild` hook (electron-builder.yml): after all installers
// are built it writes `<output>/SHA256SUMS.txt` in the canonical `<hex>  <name>` format and returns its
// path so electron-builder tracks it. The embedded PyInstaller client bundle (extraResources
// `../client-docker/packaging/dist/fedlearn-client`) is a one-dir tree of many files, so it's rolled up
// into ONE deterministic digest (sorted per-file hashes) rather than hundreds of lines.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// The extraResources source of the embedded native client bundle (mirrors electron-builder.yml
// `extraResources[0].from` and check-native-bundle.js's bundleDir).
const CLIENT_BUNDLE_DIR = path.resolve(
  __dirname, '..', '..', 'client-docker', 'packaging', 'dist', 'fedlearn-client',
);

// Metadata/output that must NOT be checksummed as an installer.
const NON_INSTALLER = new Set(['.yml', '.yaml']);

/** SHA-256 hex digest of a file's bytes. */
function sha256File(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

/** One SHA256SUMS line: `<hex>  <name>` (two spaces = the coreutils "binary" marker). */
function sumsLine(hexDigest, name) {
  return `${hexDigest}  ${name}`;
}

/** From electron-builder's artifactPaths, keep the user-facing installer artifacts — drop the
 *  *.yml update metadata, builder-debug.yml, and any pre-existing SHA256SUMS.txt. Blockmaps are kept
 *  (they ship next to the installer and are worth verifying). */
function installerArtifacts(artifactPaths) {
  return (artifactPaths || []).filter((p) => {
    const base = path.basename(p).toLowerCase();
    if (base === 'sha256sums.txt') return false;
    if (NON_INSTALLER.has(path.extname(base))) return false;
    return true;
  });
}

/** Every file under `dir`, as POSIX-style relative paths, sorted for determinism. */
function walkFilesSorted(dir) {
  const out = [];
  const walk = (d, rel) => {
    for (const name of fs.readdirSync(d).sort()) {
      const full = path.join(d, name);
      const r = rel ? `${rel}/${name}` : name;
      if (fs.statSync(full).isDirectory()) walk(full, r);
      else out.push(r);
    }
  };
  walk(dir, '');
  return out.sort();
}

/** One deterministic SHA-256 over an entire directory tree: hash of (relpath + file-hash) for every
 *  file, in sorted order. Any added/removed/changed file flips the digest. */
function bundleDigest(dir) {
  const h = crypto.createHash('sha256');
  for (const rel of walkFilesSorted(dir)) {
    h.update(rel);
    h.update('\0');
    h.update(sha256File(path.join(dir, rel)));
    h.update('\n');
  }
  return h.digest('hex');
}

/**
 * electron-builder afterAllArtifactBuild hook. Writes `<outDir>/SHA256SUMS.txt` and returns [its path].
 * @param buildResult electron-builder's result ({ outDir, artifactPaths }).
 * @param bundleDir   override for the embedded-bundle dir (tests); defaults to CLIENT_BUNDLE_DIR.
 */
function generateChecksums(buildResult, bundleDir = CLIENT_BUNDLE_DIR) {
  const outDir = buildResult.outDir;
  const lines = installerArtifacts(buildResult.artifactPaths)
    .map((f) => sumsLine(sha256File(f), path.basename(f)))
    .sort();

  if (fs.existsSync(bundleDir)) {
    lines.push(sumsLine(bundleDigest(bundleDir), 'embedded-client-bundle'));
  }

  const sumsPath = path.join(outDir, 'SHA256SUMS.txt');
  fs.writeFileSync(sumsPath, lines.join('\n') + '\n');
  return [sumsPath];
}

module.exports = generateChecksums; // default export = the hook (electron-builder calls the module)
module.exports.generateChecksums = generateChecksums;
module.exports.sha256File = sha256File;
module.exports.sumsLine = sumsLine;
module.exports.installerArtifacts = installerArtifacts;
module.exports.walkFilesSorted = walkFilesSorted;
module.exports.bundleDigest = bundleDigest;
