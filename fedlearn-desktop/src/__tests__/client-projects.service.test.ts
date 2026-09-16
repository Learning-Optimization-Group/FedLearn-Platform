// getConnection must JOIN the project (idempotently) before asking for the FL connection
// details: the backend's /connection endpoint enrolls only owner-or-CLIENT members, so a PUBLIC
// project the user merely *discovered* (never joined) 403s "Access denied". The mobile client joins
// before it connects; the desktop must do the same. Regression guard for that live-test fix.
import { AuthService } from '../main/auth.service';
import { ClientProjectService } from '../main/client-projects.service';
import { http } from '../main/http';

describe('ClientProjectService.getConnection — join before connect', () => {
  const auth = {
    getApiUrl: () => 'http://backend/api',
    getAuthHeader: () => 'Bearer tok',
  } as unknown as AuthService;

  let svc: ClientProjectService;
  let postSpy: jest.SpyInstance;
  let getSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.restoreAllMocks();
    postSpy = jest.spyOn(http, 'post');
    getSpy = jest.spyOn(http, 'get');
    svc = new ClientProjectService(auth);
  });

  it('POSTs /join before GETting /connection and returns the connection', async () => {
    postSpy.mockResolvedValue({ status: 200, data: {} });
    getSpy.mockResolvedValue({
      status: 200,
      data: { serverAddress: '10.0.0.130:50000', partitionId: 0, connectionToken: 'jwt' },
    });

    const res = await svc.getConnection('proj-1');

    expect(res.success).toBe(true);
    expect(res.connection?.serverAddress).toBe('10.0.0.130:50000');
    expect(postSpy).toHaveBeenCalledWith(
      'http://backend/api/client/projects/proj-1/join',
      {},
      expect.anything(),
    );
    // join must precede the connection request
    expect(postSpy.mock.invocationCallOrder[0]).toBeLessThan(getSpy.mock.invocationCallOrder[0]);
  });

  it('surfaces a join failure and never requests the connection', async () => {
    postSpy.mockResolvedValue({ status: 403, data: { message: 'Join not allowed for this project' } });

    const res = await svc.getConnection('proj-1');

    expect(res.success).toBe(false);
    expect(res.error).toContain('Join not allowed');
    expect(getSpy).not.toHaveBeenCalled();
  });
});
