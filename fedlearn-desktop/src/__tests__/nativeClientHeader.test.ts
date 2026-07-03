// fedlearn-desktop/src/__tests__/nativeClientHeader.test.ts
//
// SE-9: the backend honors `Authorization: Bearer` only when the request also
// carries the X-FedLearn-Client marker header — browsers stay strictly
// cookie-only. Every desktop REST call goes through the shared `http` instance,
// so pinning the marker as an instance-wide default pins the whole client.

import { http, NATIVE_CLIENT_HEADER, NATIVE_CLIENT_VALUE } from '../main/http';

describe('native-client marker header (SE-9)', () => {
  test('exposes the exact header name and desktop client identifier', () => {
    expect(NATIVE_CLIENT_HEADER).toBe('X-FedLearn-Client');
    expect(NATIVE_CLIENT_VALUE).toBe('fedlearn-desktop');
  });

  test('the shared http instance carries the marker as a default header', () => {
    expect(http.defaults.headers.common[NATIVE_CLIENT_HEADER]).toBe('fedlearn-desktop');
  });

  test('an outbound request actually carries the marker (stub adapter, no network)', async () => {
    let seen: { get(name: string): unknown } | null = null;
    await http.get('http://localhost:8081/api/client/projects', {
      headers: { Authorization: 'Bearer jwt-xyz' },
      adapter: async (config) => {
        seen = config.headers as unknown as { get(name: string): unknown };
        return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
      },
    });
    expect(seen!.get(NATIVE_CLIENT_HEADER)).toBe(NATIVE_CLIENT_VALUE);
    // Per-call headers (the Bearer token) still ride alongside the shared default.
    expect(seen!.get('Authorization')).toBe('Bearer jwt-xyz');
  });
});
