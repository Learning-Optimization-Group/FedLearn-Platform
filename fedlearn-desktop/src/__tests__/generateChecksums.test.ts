// DE-12: SHA256 checksums for the release installers + the embedded native client bundle, so a
// downloader can verify integrity. Pins the pure hashing/formatting/filtering logic and the
// SHA256SUMS.txt the electron-builder afterAllArtifactBuild hook writes.
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// The script is plain CommonJS (runs as an electron-builder hook), required from this TS test.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const gen = require('../../scripts/generate-checksums');

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'de12-'));
}
function sha256(buf: Buffer | string): string {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

describe('generate-checksums (DE-12)', () => {
  test('sha256File hashes file bytes (matches crypto over the same content)', () => {
    const dir = tmp();
    const f = path.join(dir, 'a.bin');
    fs.writeFileSync(f, 'hello world');
    expect(gen.sha256File(f)).toBe(sha256('hello world'));
  });

  test('sumsLine uses the canonical `<hex>  <name>` (two-space, binary-mode) format', () => {
    expect(gen.sumsLine('abc123', 'FedLearn.dmg')).toBe('abc123  FedLearn.dmg');
  });

  test('installerArtifacts keeps user-facing installers and drops metadata (yml/SHA256SUMS)', () => {
    const paths = [
      '/r/FedLearn-1.0-arm64.dmg',
      '/r/FedLearn-Setup-1.0-cpu.exe',
      '/r/FedLearn-1.0.AppImage',
      '/r/fedlearn_1.0_amd64.deb',
      '/r/FedLearn-1.0-mac.zip',
      '/r/FedLearn-1.0-arm64.dmg.blockmap',
      '/r/latest-mac.yml',
      '/r/builder-debug.yml',
      '/r/SHA256SUMS.txt',
    ];
    const kept = gen.installerArtifacts(paths).map((p: string) => path.basename(p));
    expect(kept).toEqual(expect.arrayContaining([
      'FedLearn-1.0-arm64.dmg', 'FedLearn-Setup-1.0-cpu.exe', 'FedLearn-1.0.AppImage',
      'fedlearn_1.0_amd64.deb', 'FedLearn-1.0-mac.zip', 'FedLearn-1.0-arm64.dmg.blockmap',
    ]));
    expect(kept).not.toContain('latest-mac.yml');
    expect(kept).not.toContain('builder-debug.yml');
    expect(kept).not.toContain('SHA256SUMS.txt');
  });

  test('bundleDigest is deterministic over the tree and changes when any file changes', () => {
    const a = tmp();
    fs.mkdirSync(path.join(a, 'sub'));
    fs.writeFileSync(path.join(a, 'client'), 'BIN');
    fs.writeFileSync(path.join(a, 'sub', 'lib.so'), 'LIB');

    const b = tmp();
    fs.mkdirSync(path.join(b, 'sub'));
    fs.writeFileSync(path.join(b, 'client'), 'BIN');
    fs.writeFileSync(path.join(b, 'sub', 'lib.so'), 'LIB');

    expect(gen.bundleDigest(a)).toBe(gen.bundleDigest(b)); // same content -> same digest

    fs.writeFileSync(path.join(b, 'sub', 'lib.so'), 'LIB-TAMPERED');
    expect(gen.bundleDigest(a)).not.toBe(gen.bundleDigest(b)); // any change flips the digest
  });

  test('the hook writes SHA256SUMS.txt covering the installers + the embedded bundle digest', () => {
    const outDir = tmp();
    const dmg = path.join(outDir, 'FedLearn-1.0-arm64.dmg');
    fs.writeFileSync(dmg, 'DMG-BYTES');
    fs.writeFileSync(path.join(outDir, 'latest-mac.yml'), 'ignore me');

    const bundleDir = path.join(tmp(), 'fedlearn-client');
    fs.mkdirSync(bundleDir);
    fs.writeFileSync(path.join(bundleDir, 'fedlearn-client'), 'CLIENT');

    const written = gen.generateChecksums(
      { outDir, artifactPaths: [dmg, path.join(outDir, 'latest-mac.yml')] },
      bundleDir,
    );

    const sumsPath = path.join(outDir, 'SHA256SUMS.txt');
    expect(written).toContain(sumsPath);
    const body = fs.readFileSync(sumsPath, 'utf8');
    expect(body).toContain(`${sha256('DMG-BYTES')}  FedLearn-1.0-arm64.dmg`);
    expect(body).toContain('embedded-client-bundle'); // the rolled-up bundle line
    expect(body).not.toContain('latest-mac.yml');     // metadata excluded
  });
});
