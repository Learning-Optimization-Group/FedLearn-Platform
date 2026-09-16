import { createHash } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';

// Server trust on the FL boundary. A deployment serves gRPC over TLS with a self-signed certificate
// (deploy/TLS.md), and the backend sends that certificate on the connection payload. The client verifies the
// server against it, so main writes it to a file the client process can read.

/** Where the certificate is mounted inside the client container. */
export const CONTAINER_ROOT_CERT_PATH = '/run/fedlearn/grpc-root.pem';

const MAX_PEM_LENGTH = 16 * 1024;
const PEM_CERTIFICATE = /^-----BEGIN CERTIFICATE-----\r?\n(?:[A-Za-z0-9+/=]+\r?\n)+-----END CERTIFICATE-----\r?\n?$/;

/**
 * A single PEM certificate of bounded size, or absent. Absent is valid: a plaintext deployment sends none, and so
 * does one whose certificate chains to a public root.
 */
export function isValidServerCertPem(val: unknown): boolean {
  if (val === undefined || val === null) {
    return true;
  }
  return typeof val === 'string' && val.length <= MAX_PEM_LENGTH && PEM_CERTIFICATE.test(val);
}

/**
 * Writes the certificate under `dir`, named by its content hash so the same certificate always lands at the same
 * path, and returns that path. The file holds a public certificate, not a secret.
 */
export function writeServerCertPem(dir: string, pem: string): string {
  const name = `fl-server-${createHash('sha256').update(pem).digest('hex').slice(0, 16)}.pem`;
  const file = path.join(dir, name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(file, pem, { mode: 0o644 });
  return file;
}

export interface ServerTrust {
  grpcTls: boolean;
  grpcServerCertPath?: string;
}

/**
 * Resolves the connection payload's server trust for a launch: validates it, writes the certificate under `certDir`,
 * and returns what the launch config needs. A malformed value refuses the start with a clear message, instead of
 * dialing TLS without the certificate the backend meant to send and failing the handshake less clearly.
 */
export function resolveServerTrust(
  payload: { grpcTls?: unknown; grpcServerCertPem?: unknown },
  certDir: string,
): ServerTrust {
  const { grpcTls, grpcServerCertPem } = payload;
  if (grpcTls !== undefined && grpcTls !== null && typeof grpcTls !== 'boolean') {
    throw new Error('Invalid grpcTls in the connection payload: expected true or false.');
  }
  if (grpcTls !== true) {
    return { grpcTls: false };
  }
  if (grpcServerCertPem === undefined || grpcServerCertPem === null || grpcServerCertPem === '') {
    return { grpcTls: true };
  }
  if (!isValidServerCertPem(grpcServerCertPem)) {
    throw new Error(
      'The FL server certificate in the connection payload is not a valid PEM certificate; refusing to start.',
    );
  }
  return { grpcTls: true, grpcServerCertPath: writeServerCertPem(certDir, grpcServerCertPem as string) };
}
