// fedlearn-desktop/src/main/validators.ts
// Pure validation utilities extracted for testability.
// ipc.handlers.ts imports these. Do not import 'electron' or 'electron-log' here.

import * as fs from 'fs';
import * as path from 'path';

export const ALLOWED_HARDWARE_PROFILES: ReadonlySet<string> =
  new Set(['discrete', 'jetson', 'cpu', 'mps']);

export const PROJECT_ID_PATTERN = /^[a-zA-Z0-9_-]{1,128}$/;
export const PARTITION_ID_PATTERN = /^[0-9]{1,10}$/;
export const SERVER_ADDRESS_PATTERN = /^[a-zA-Z0-9._:/-]{1,256}$/;
export const MAX_DATASET_PATH_LEN = 2048;

export function sanitizeDatasetPath(raw: unknown): string | null {
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > MAX_DATASET_PATH_LEN) {
    return null;
  }
  if (raw.includes('\0')) {
    return null;
  }
  let resolved: string;
  try {
    resolved = path.resolve(raw);
  } catch {
    return null;
  }
  if (resolved.split(path.sep).some((seg) => seg === '..')) {
    return null;
  }
  if (!path.isAbsolute(resolved)) {
    return null;
  }
  let stat: fs.Stats;
  try {
    stat = fs.statSync(resolved);
  } catch {
    return null;
  }
  if (!stat.isDirectory()) {
    return null;
  }
  return resolved;
}

export function validateHardwareProfile(profile: unknown): profile is string {
  return typeof profile === 'string' && ALLOWED_HARDWARE_PROFILES.has(profile);
}

export function validateProjectId(id: unknown): id is string {
  return typeof id === 'string' && PROJECT_ID_PATTERN.test(id);
}

export function validatePartitionId(id: unknown): id is string {
  return typeof id === 'string' && PARTITION_ID_PATTERN.test(id);
}

export function validateServerAddress(addr: unknown): addr is string {
  return typeof addr === 'string' && SERVER_ADDRESS_PATTERN.test(addr);
}

export function validateStringInput(val: unknown, maxLength: number): val is string {
  return typeof val === 'string' && val.length > 0 && val.length <= maxLength;
}
