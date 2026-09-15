// Server trust on the FL boundary. A deployment serves gRPC over TLS with a self-signed certificate, so the client
// can verify the server only if the desktop hands it that certificate. The backend sends it on the connection
// payload; these pin how it reaches the client in both launch paths, and that a plaintext deployment adds nothing.

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { buildContainerBinds, buildContainerEnv, withGrpcTlsEnv, TrainingConfig } from '../main/docker.service';
import {
  CONTAINER_ROOT_CERT_PATH, isValidServerCertPem, resolveServerTrust, writeServerCertPem,
} from '../main/grpcTls';

const PEM =
  '-----BEGIN CERTIFICATE-----\n' +
  'MIIBszCCAVmgAwIBAgIUQ2FrZUZha2VGYWtlRmFrZUZha2UwCgYIKoZIzj0EAwIw\n' +
  'EjEQMA4GA1UEAwwHZXhhbXBsZTAeFw0yNjAxMDEwMDAwMDBaFw0yNzAxMDEwMDAw\n' +
  '-----END CERTIFICATE-----\n';

const base: TrainingConfig = {
  hardwareProfile: 'cpu',
  projectId: 'proj-1',
  serverAddress: 'fl.example.org:50001',
  partitionId: '2',
  modelType: 'CNN',
  datasetPath: '/data',
};

describe('isValidServerCertPem', () => {
  test('accepts a PEM certificate', () => {
    expect(isValidServerCertPem(PEM)).toBe(true);
  });

  test('accepts an absent certificate: a plaintext deployment, or a public-CA one, sends none', () => {
    expect(isValidServerCertPem(undefined)).toBe(true);
    expect(isValidServerCertPem(null)).toBe(true);
  });

  test('rejects anything that is not a single bounded PEM certificate', () => {
    expect(isValidServerCertPem('not a certificate')).toBe(false);
    expect(isValidServerCertPem('-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n')).toBe(false);
    expect(isValidServerCertPem(`${PEM}trailing`)).toBe(false);
    expect(isValidServerCertPem(`-----BEGIN CERTIFICATE-----\n${'A'.repeat(20000)}\n-----END CERTIFICATE-----\n`))
      .toBe(false);
    expect(isValidServerCertPem(42)).toBe(false);
  });
});

describe('writeServerCertPem', () => {
  test('writes the certificate and returns the same path for the same content', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fl-cert-'));
    const first = writeServerCertPem(dir, PEM);
    const second = writeServerCertPem(dir, PEM);
    expect(second).toBe(first);
    expect(path.dirname(first)).toBe(dir);
    expect(fs.readFileSync(first, 'utf-8')).toBe(PEM);
  });
});

describe('docker path', () => {
  test('dials TLS and trusts the certificate mounted read-only into the container', () => {
    const cfg = { ...base, grpcTls: true, grpcServerCertPath: '/home/u/.fl/certs/fl-server-abc.pem' };
    expect(buildContainerEnv(cfg)).toEqual(
      expect.arrayContaining(['FEDLEARN_GRPC_USE_TLS=1', `FEDLEARN_GRPC_ROOT_CERT=${CONTAINER_ROOT_CERT_PATH}`]),
    );
    expect(buildContainerBinds(cfg)).toEqual(['/data:/data', `/home/u/.fl/certs/fl-server-abc.pem:${CONTAINER_ROOT_CERT_PATH}:ro`]);
  });

  test('dials TLS against the system roots when no certificate was sent', () => {
    const cfg = { ...base, grpcTls: true };
    const env = buildContainerEnv(cfg);
    expect(env).toContain('FEDLEARN_GRPC_USE_TLS=1');
    expect(env.some((e) => e.startsWith('FEDLEARN_GRPC_ROOT_CERT='))).toBe(false);
    expect(buildContainerBinds(cfg)).toEqual(['/data:/data']);
  });

  test('a plaintext deployment adds nothing', () => {
    expect(buildContainerEnv(base).some((e) => e.startsWith('FEDLEARN_GRPC_'))).toBe(false);
    expect(buildContainerBinds(base)).toEqual(['/data:/data']);
  });
});

describe('native path', () => {
  test('dials TLS and trusts the certificate file', () => {
    const env = withGrpcTlsEnv({ PATH: '/usr/bin' }, { ...base, grpcTls: true, grpcServerCertPath: '/tmp/c.pem' });
    expect(env).toEqual({ PATH: '/usr/bin', FEDLEARN_GRPC_USE_TLS: '1', FEDLEARN_GRPC_ROOT_CERT: '/tmp/c.pem' });
  });

  test('a plaintext deployment leaves the env as it was', () => {
    expect(withGrpcTlsEnv({ PATH: '/usr/bin' }, base)).toEqual({ PATH: '/usr/bin' });
  });
});

// What main does with the connection payload before a start: validate it, write the certificate, and hand the launch
// config a path. Kept pure (a directory in, config out) so it is testable without Electron.
describe('resolveServerTrust', () => {
  const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'fl-trust-'));

  test('writes the certificate and points the launch config at it', () => {
    const dir = tmp();
    const trust = resolveServerTrust({ grpcTls: true, grpcServerCertPem: PEM }, dir);
    expect(trust.grpcTls).toBe(true);
    expect(trust.grpcServerCertPath).toBeDefined();
    expect(fs.readFileSync(trust.grpcServerCertPath as string, 'utf-8')).toBe(PEM);
  });

  test('TLS without a certificate dials against the system roots and writes nothing', () => {
    const dir = tmp();
    expect(resolveServerTrust({ grpcTls: true }, dir)).toEqual({ grpcTls: true });
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  test('a plaintext deployment ignores any certificate and writes nothing', () => {
    const dir = tmp();
    expect(resolveServerTrust({ grpcTls: false, grpcServerCertPem: PEM }, dir)).toEqual({ grpcTls: false });
    expect(resolveServerTrust({}, dir)).toEqual({ grpcTls: false });
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  test('refuses a start whose certificate is malformed rather than dialing without it', () => {
    expect(() => resolveServerTrust({ grpcTls: true, grpcServerCertPem: 'not a certificate' }, tmp())).toThrow(
      /certificate/i,
    );
  });

  test('refuses a grpcTls value that is not a boolean', () => {
    expect(() => resolveServerTrust({ grpcTls: 'yes' }, tmp())).toThrow(/grpcTls/);
  });
});
