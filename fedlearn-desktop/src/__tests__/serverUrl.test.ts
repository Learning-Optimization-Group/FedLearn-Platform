// fedlearn-desktop/src/__tests__/serverUrl.test.ts
//
// DE-13: the desktop client sends credentials + a Bearer JWT to whatever server
// URL the user types. Plaintext http:// to a remote host means those secrets
// traverse the network unencrypted, while the login screen claims "secure".
// Policy under test: refuse remote http:// by default; accept it only with an
// explicit override (and then carry a persistent warning); loopback http:// and
// any https:// stay accepted warning-free.

import { isLoopbackHost, isPlaintextRemoteUrl } from '../shared/urlSecurity';
import { evaluateServerUrl } from '../main/validators';

// ─── isLoopbackHost ─────────────────────────────────────────────────────────

describe('isLoopbackHost', () => {
  test('accepts localhost and IPv4/IPv6 loopback', () => {
    expect(isLoopbackHost('localhost')).toBe(true);
    expect(isLoopbackHost('LOCALHOST')).toBe(true);
    expect(isLoopbackHost('127.0.0.1')).toBe(true);
    expect(isLoopbackHost('127.1.2.3')).toBe(true); // whole 127.0.0.0/8 block
    expect(isLoopbackHost('::1')).toBe(true);
    expect(isLoopbackHost('[::1]')).toBe(true); // WHATWG URL keeps the brackets
  });

  test('rejects remote hosts, including lookalikes', () => {
    expect(isLoopbackHost('example.com')).toBe(false);
    expect(isLoopbackHost('192.168.1.10')).toBe(false);
    expect(isLoopbackHost('localhost.evil.com')).toBe(false);
    expect(isLoopbackHost('127.0.0.1.evil.com')).toBe(false);
    expect(isLoopbackHost('')).toBe(false);
  });
});

// ─── isPlaintextRemoteUrl ───────────────────────────────────────────────────

describe('isPlaintextRemoteUrl', () => {
  test('true for http:// to a remote host', () => {
    expect(isPlaintextRemoteUrl('http://example.com:8081')).toBe(true);
    expect(isPlaintextRemoteUrl('http://192.168.1.10:8081')).toBe(true);
    expect(isPlaintextRemoteUrl('HTTP://EXAMPLE.COM')).toBe(true);
  });

  test('false for loopback http://', () => {
    expect(isPlaintextRemoteUrl('http://localhost:8081')).toBe(false);
    expect(isPlaintextRemoteUrl('http://127.0.0.1:8081')).toBe(false);
    expect(isPlaintextRemoteUrl('http://[::1]:8081')).toBe(false);
  });

  test('false for https:// anywhere', () => {
    expect(isPlaintextRemoteUrl('https://example.com:8081')).toBe(false);
    expect(isPlaintextRemoteUrl('https://localhost:8081')).toBe(false);
  });

  test('false for non-URL strings (protocol gate handles those separately)', () => {
    expect(isPlaintextRemoteUrl('example.com:8081')).toBe(false);
    expect(isPlaintextRemoteUrl('')).toBe(false);
  });

  test('fails closed on an unparseable http:// URL', () => {
    expect(isPlaintextRemoteUrl('http://')).toBe(true);
  });
});

// ─── evaluateServerUrl (the auth:set-server-url decision) ──────────────────

describe('evaluateServerUrl (DE-13)', () => {
  test('REFUSES a remote http:// URL by default with code INSECURE_HTTP', () => {
    const result = evaluateServerUrl('http://example.com:8081');
    expect(result.ok).toBe(false);
    expect(result.code).toBe('INSECURE_HTTP');
    expect(result.error).toMatch(/unencrypted|plaintext/i);
    expect(result.url).toBeUndefined(); // nothing was accepted or normalized
  });

  test('accepts a remote http:// URL only with the explicit override — and keeps a warning attached', () => {
    const result = evaluateServerUrl('http://example.com:8081', true);
    expect(result.ok).toBe(true);
    expect(result.url).toBe('http://example.com:8081/api');
    expect(result.warning).toMatch(/unencrypted|plaintext/i);
  });

  test('accepts http://localhost:8081 warning-free', () => {
    const result = evaluateServerUrl('http://localhost:8081');
    expect(result).toEqual({ ok: true, url: 'http://localhost:8081/api' });
    expect(result.warning).toBeUndefined();
  });

  test('accepts http://127.0.0.1:8081 warning-free', () => {
    const result = evaluateServerUrl('http://127.0.0.1:8081');
    expect(result).toEqual({ ok: true, url: 'http://127.0.0.1:8081/api' });
    expect(result.warning).toBeUndefined();
  });

  test('accepts an https:// URL warning-free', () => {
    const result = evaluateServerUrl('https://fedlearn.example.com');
    expect(result).toEqual({ ok: true, url: 'https://fedlearn.example.com/api' });
    expect(result.warning).toBeUndefined();
  });

  test('the override flag does not add a warning to already-secure URLs', () => {
    expect(evaluateServerUrl('https://example.com', true).warning).toBeUndefined();
    expect(evaluateServerUrl('http://localhost:8081', true).warning).toBeUndefined();
  });

  // Pre-existing validation behavior must be preserved (TE-2).
  test('still rejects non-string, empty, and oversized input', () => {
    expect(evaluateServerUrl(42).ok).toBe(false);
    expect(evaluateServerUrl(null).ok).toBe(false);
    expect(evaluateServerUrl('').ok).toBe(false);
    expect(evaluateServerUrl('https://' + 'a'.repeat(512)).ok).toBe(false);
  });

  test('still rejects URLs without an http(s):// protocol', () => {
    const result = evaluateServerUrl('example.com:8081');
    expect(result.ok).toBe(false);
    expect(result.error).toBe('URL must start with http:// or https://');
    expect(result.code).toBeUndefined();
  });

  test('still normalizes by stripping trailing slashes and appending /api', () => {
    expect(evaluateServerUrl('https://example.com///').url).toBe('https://example.com/api');
    expect(evaluateServerUrl('http://localhost:8081/api').url).toBe('http://localhost:8081/api');
  });
});
