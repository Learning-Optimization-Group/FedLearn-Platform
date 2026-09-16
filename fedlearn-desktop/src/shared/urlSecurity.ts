// fedlearn-desktop/src/shared/urlSecurity.ts
// Pure transport-security helpers for user-supplied server URLs, shared by the
// Main process (auth:set-server-url policy) and the Renderer (honest security
// copy on the login screen). No electron imports — this module must stay
// importable from every process and from jest.

/**
 * True for hosts where plaintext HTTP never leaves the machine:
 * `localhost`, the whole IPv4 loopback block (127.0.0.0/8), and IPv6 `::1`
 * (with or without the brackets the WHATWG URL parser keeps on `hostname`).
 * Suffix lookalikes such as `localhost.evil.com` do NOT qualify.
 */
export function isLoopbackHost(hostname: string): boolean {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (host === 'localhost' || host === '::1') {
    return true;
  }
  return /^127(\.\d{1,3}){3}$/.test(host);
}

/**
 * True when `url` is plaintext `http://` to a non-loopback host — i.e. any
 * credentials or tokens sent to it would traverse the network unencrypted.
 * `https://` and loopback `http://` return false. Strings that are not
 * `http://` URLs at all return false (the protocol gate rejects those
 * separately). An `http://` URL that fails to parse is treated as remote
 * (fail closed).
 */
export function isPlaintextRemoteUrl(url: string): boolean {
  const trimmed = url.trim();
  if (!/^http:\/\//i.test(trimmed)) {
    return false;
  }
  try {
    return !isLoopbackHost(new URL(trimmed).hostname);
  } catch {
    return true;
  }
}

/** Refusal message returned when a remote http:// URL is set without the override. */
export const PLAINTEXT_HTTP_REFUSAL =
  'This server uses plaintext http:// — your username, password, and session token ' +
  'would cross the network unencrypted. Use https://, or explicitly allow insecure HTTP.';

/** Persistent warning attached when a remote http:// URL is accepted via override. */
export const PLAINTEXT_HTTP_WARNING =
  'Insecure server URL: credentials and session tokens are sent over unencrypted HTTP.';
