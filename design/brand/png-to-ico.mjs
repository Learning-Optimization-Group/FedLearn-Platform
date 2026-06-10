#!/usr/bin/env node
// Dependency-free PNG -> ICO packer (PNG-embedded ICO, Windows Vista+).
// Usage: node png-to-ico.mjs out.ico 16.png 32.png 48.png 64.png 128.png 256.png
import { readFileSync, writeFileSync } from 'node:fs';

const [, , outPath, ...pngs] = process.argv;
if (!outPath || pngs.length === 0) {
  console.error('usage: png-to-ico.mjs out.ico <png...>');
  process.exit(1);
}

const images = pngs.map((p) => {
  const data = readFileSync(p);
  // PNG IHDR: width at byte 16, height at byte 20 (big-endian uint32).
  const w = data.readUInt32BE(16);
  const h = data.readUInt32BE(20);
  return { data, w, h };
});

const HEADER = 6;
const ENTRY = 16;
const offsets = [];
let cursor = HEADER + ENTRY * images.length;
for (const img of images) {
  offsets.push(cursor);
  cursor += img.data.length;
}

const buf = Buffer.alloc(cursor);
buf.writeUInt16LE(0, 0); // reserved
buf.writeUInt16LE(1, 2); // type: icon
buf.writeUInt16LE(images.length, 4);

images.forEach((img, i) => {
  const e = HEADER + ENTRY * i;
  buf.writeUInt8(img.w >= 256 ? 0 : img.w, e + 0);
  buf.writeUInt8(img.h >= 256 ? 0 : img.h, e + 1);
  buf.writeUInt8(0, e + 2); // palette
  buf.writeUInt8(0, e + 3); // reserved
  buf.writeUInt16LE(1, e + 4); // color planes
  buf.writeUInt16LE(32, e + 6); // bits per pixel
  buf.writeUInt32LE(img.data.length, e + 8);
  buf.writeUInt32LE(offsets[i], e + 12);
});

images.forEach((img, i) => img.data.copy(buf, offsets[i]));
writeFileSync(outPath, buf);
console.log('wrote', outPath, `(${images.length} frames: ${images.map((i) => i.w).join(', ')})`);
