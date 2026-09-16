import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';
import type { AxiosError } from 'axios';
import api from './axiosConfig';

// The interceptor's rejection handler is what encodes the auth contract. Pull it off the instance
// and call it directly with synthetic AxiosErrors — no network/adapter mocking needed, and it tests
// the exact function the app registers.
interface RejectionHandler {
  rejected: (error: AxiosError) => Promise<never>;
}
const handlers = (api.interceptors.response as unknown as { handlers: RejectionHandler[] }).handlers;
const onRejected = handlers[0].rejected;

function axiosError(status: number | undefined, url: string): AxiosError {
  return { response: status ? { status } : undefined, config: { url } } as AxiosError;
}

describe('axios response interceptor (auth contract)', () => {
  let dispatch: MockInstance<(event: Event) => boolean>;

  beforeEach(() => {
    dispatch = vi.spyOn(window, 'dispatchEvent');
  });
  afterEach(() => {
    dispatch.mockRestore();
  });

  it('dispatches authError on a 401 for a data route', async () => {
    await expect(onRejected(axiosError(401, '/projects'))).rejects.toBeDefined();
    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch.mock.calls[0][0]).toMatchObject({ type: 'authError' });
  });

  it('stays silent on a 401 from the /auth/me probe', async () => {
    await expect(onRejected(axiosError(401, '/auth/me'))).rejects.toBeDefined();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('stays silent on a 401 from the /auth/login attempt', async () => {
    await expect(onRejected(axiosError(401, '/auth/login'))).rejects.toBeDefined();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('does NOT log out on a 403 (authorized-elsewhere, not session-expired)', async () => {
    await expect(onRejected(axiosError(403, '/users'))).rejects.toBeDefined();
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('always rejects with the original error so callers still see it', async () => {
    const err = axiosError(500, '/projects');
    await expect(onRejected(err)).rejects.toBe(err);
  });
});
