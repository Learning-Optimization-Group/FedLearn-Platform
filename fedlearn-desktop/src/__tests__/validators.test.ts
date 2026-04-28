// fedlearn-desktop/src/__tests__/validators.test.ts
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

import {
  sanitizeDatasetPath,
  validateHardwareProfile,
  validateProjectId,
  validatePartitionId,
  validateServerAddress,
  validateStringInput,
} from '../main/validators';

// ─── sanitizeDatasetPath ────────────────────────────────────────────────────

describe('sanitizeDatasetPath', () => {
  let tempDir: string;

  beforeEach(() => {
    // Create a real temp directory so statSync can verify it exists
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fedlearn-test-'));
  });

  afterEach(() => {
    fs.rmdirSync(tempDir);
  });

  test('returns resolved path for a valid existing directory', () => {
    const result = sanitizeDatasetPath(tempDir);
    expect(result).not.toBeNull();
    expect(path.isAbsolute(result!)).toBe(true);
  });

  test('returns null for a non-string input', () => {
    expect(sanitizeDatasetPath(42)).toBeNull();
    expect(sanitizeDatasetPath(null)).toBeNull();
    expect(sanitizeDatasetPath(undefined)).toBeNull();
    expect(sanitizeDatasetPath({})).toBeNull();
  });

  test('returns null for empty string', () => {
    expect(sanitizeDatasetPath('')).toBeNull();
  });

  test('returns null when path is too long', () => {
    const longPath = '/tmp/' + 'a'.repeat(3000);
    expect(sanitizeDatasetPath(longPath)).toBeNull();
  });

  test('returns null when path contains NUL byte', () => {
    expect(sanitizeDatasetPath('/tmp/foo\0bar')).toBeNull();
  });

  test('returns null for a path pointing to a file (not directory)', () => {
    const tmpFile = path.join(tempDir, 'test.txt');
    fs.writeFileSync(tmpFile, 'hello');
    const result = sanitizeDatasetPath(tmpFile);
    expect(result).toBeNull();
    fs.unlinkSync(tmpFile);
  });

  test('returns null for a non-existent path', () => {
    expect(sanitizeDatasetPath('/this/path/does/not/exist/xyz123')).toBeNull();
  });
});

// ─── validateHardwareProfile ────────────────────────────────────────────────

describe('validateHardwareProfile', () => {
  test('accepts valid profiles', () => {
    expect(validateHardwareProfile('discrete')).toBe(true);
    expect(validateHardwareProfile('jetson')).toBe(true);
    expect(validateHardwareProfile('cpu')).toBe(true);
    expect(validateHardwareProfile('mps')).toBe(true);
  });

  test('rejects invalid profiles', () => {
    expect(validateHardwareProfile('gpu')).toBe(false);
    expect(validateHardwareProfile('')).toBe(false);
    expect(validateHardwareProfile(null)).toBe(false);
    expect(validateHardwareProfile(42)).toBe(false);
    expect(validateHardwareProfile('DISCRETE')).toBe(false); // case-sensitive
  });
});

// ─── validateProjectId ──────────────────────────────────────────────────────

describe('validateProjectId', () => {
  test('accepts alphanumeric IDs with hyphens and underscores', () => {
    expect(validateProjectId('abc')).toBe(true);
    expect(validateProjectId('project-123')).toBe(true);
    expect(validateProjectId('my_project_2')).toBe(true);
    expect(validateProjectId('a'.repeat(128))).toBe(true); // max length
  });

  test('rejects empty string', () => {
    expect(validateProjectId('')).toBe(false);
  });

  test('rejects strings longer than 128 characters', () => {
    expect(validateProjectId('a'.repeat(129))).toBe(false);
  });

  test('rejects strings with special characters', () => {
    expect(validateProjectId('project/id')).toBe(false);
    expect(validateProjectId('project id')).toBe(false);
    expect(validateProjectId('../etc')).toBe(false);
    expect(validateProjectId('project;rm -rf')).toBe(false);
  });

  test('rejects non-string input', () => {
    expect(validateProjectId(123)).toBe(false);
    expect(validateProjectId(null)).toBe(false);
  });
});

// ─── validatePartitionId ────────────────────────────────────────────────────

describe('validatePartitionId', () => {
  test('accepts numeric-only strings', () => {
    expect(validatePartitionId('0')).toBe(true);
    expect(validatePartitionId('42')).toBe(true);
    expect(validatePartitionId('9999999999')).toBe(true); // 10 digits max
  });

  test('rejects strings with letters', () => {
    expect(validatePartitionId('1a')).toBe(false);
    expect(validatePartitionId('abc')).toBe(false);
  });

  test('rejects empty string', () => {
    expect(validatePartitionId('')).toBe(false);
  });

  test('rejects strings longer than 10 digits', () => {
    expect(validatePartitionId('12345678901')).toBe(false); // 11 digits
  });

  test('rejects non-string input', () => {
    expect(validatePartitionId(5)).toBe(false);
  });
});

// ─── validateServerAddress ──────────────────────────────────────────────────

describe('validateServerAddress', () => {
  test('accepts valid server addresses', () => {
    expect(validateServerAddress('localhost:8080')).toBe(true);
    expect(validateServerAddress('192.168.1.1:9090')).toBe(true);
    expect(validateServerAddress('my-server.example.com:443')).toBe(true);
    expect(validateServerAddress('api/v1')).toBe(true);
  });

  test('rejects empty string', () => {
    expect(validateServerAddress('')).toBe(false);
  });

  test('rejects strings with disallowed characters', () => {
    expect(validateServerAddress('server;cmd')).toBe(false);
    expect(validateServerAddress('server\ninjection')).toBe(false);
    expect(validateServerAddress('server$(whoami)')).toBe(false);
  });

  test('rejects non-string input', () => {
    expect(validateServerAddress(null)).toBe(false);
  });
});

// ─── validateStringInput ────────────────────────────────────────────────────

describe('validateStringInput', () => {
  test('accepts strings within length', () => {
    expect(validateStringInput('hello', 256)).toBe(true);
    expect(validateStringInput('a', 1)).toBe(true);
  });

  test('rejects empty string', () => {
    expect(validateStringInput('', 256)).toBe(false);
  });

  test('rejects strings exceeding maxLength', () => {
    expect(validateStringInput('abcde', 4)).toBe(false);
  });

  test('rejects non-string input', () => {
    expect(validateStringInput(42, 256)).toBe(false);
    expect(validateStringInput(null, 256)).toBe(false);
  });
});
