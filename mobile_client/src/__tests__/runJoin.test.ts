import { api } from '../lib/restClient';
import nativeCore from '../lib/nativeCore';
import { getOrCreateClientId } from '../lib/clientId';
import { joinRun } from '../lib/runJoin';

jest.mock('../lib/restClient', () => ({ api: { get: jest.fn(), post: jest.fn() } }));
jest.mock('../lib/nativeCore', () => ({ __esModule: true, default: { registerClient: jest.fn() } }));
jest.mock('../lib/clientId', () => ({ getOrCreateClientId: jest.fn() }));

const mApi = api as unknown as { get: jest.Mock; post: jest.Mock };
const mCore = nativeCore as unknown as { registerClient: jest.Mock };
const mClientId = getOrCreateClientId as jest.Mock;

const MANIFEST = {
  runId: 'r1', projectId: 'p1', recipeKey: 'CNN', strategy: 'DeComFL',
  numRounds: 5, clientsPerRound: 4, partitioningMode: 'SHARDED', seed: 42, torchVersion: '2.9.1',
};

function stubHappyPath() {
  mClientId.mockResolvedValue('device-uuid');
  // GET /api/client/projects/p1 → activeRun; GET /api/runs/r1/status → RUNNING
  mApi.get.mockImplementation((url: string) => {
    if (url === '/api/client/projects/p1') return Promise.resolve({ data: { activeRun: { runId: 'r1', status: 'RUNNING' } } });
    if (url === '/api/runs/r1/status') return Promise.resolve({ data: { status: 'RUNNING', grpcEndpoint: 'host:50001', caFingerprint: null } });
    return Promise.reject(new Error('unexpected GET ' + url));
  });
  // POST /api/runs/r1/enroll
  mApi.post.mockResolvedValue({ data: {
    runId: 'r1', projectId: 'p1', grpcEndpoint: 'host:50001', partitionId: 2, clientKind: 'SHARD',
    caFingerprint: null, connectionToken: 'tok-123', expiresAt: '2026-06-24T00:02:00Z', manifest: MANIFEST,
  }});
  mCore.registerClient.mockResolvedValue({ accepted: true, message: 'ok', assignedRound: 3, serverProtocolVersion: 2 });
}

describe('joinRun (slice 1b connect/enroll/register)', () => {
  beforeEach(() => jest.clearAllMocks());

  test('resolves run, enrolls, registers, returns JoinedRun', async () => {
    stubHappyPath();
    const out = await joinRun({ projectId: 'p1' });
    expect(mApi.get).toHaveBeenCalledWith('/api/client/projects/p1');
    expect(mApi.get).toHaveBeenCalledWith('/api/runs/r1/status');
    expect(mApi.post).toHaveBeenCalledWith('/api/runs/r1/enroll');
    expect(mCore.registerClient).toHaveBeenCalledWith('host:50001', 'r1', 'device-uuid', 'tok-123', false);
    expect(out.runId).toBe('r1');
    expect(out.partitionId).toBe(2);
    expect(out.assignedRound).toBe(3);
    expect(out.grpcEndpoint).toBe('host:50001');
    expect(out.manifest.torchVersion).toBe('2.9.1');
  });

  test('NEVER posts the dead /api/projects/{id}/runs route', async () => {
    stubHappyPath();
    await joinRun({ projectId: 'p1' });
    expect(mApi.post).not.toHaveBeenCalledWith('/api/projects/p1/runs', expect.anything());
  });

  test('throws when the project has no active run', async () => {
    mApi.get.mockResolvedValue({ data: { activeRun: null } });
    await expect(joinRun({ projectId: 'p1' })).rejects.toThrow(/active run/i);
    expect(mApi.post).not.toHaveBeenCalled();
  });

  test('throws when the run is FAILED during polling', async () => {
    mApi.get.mockImplementation((url: string) => {
      if (url === '/api/client/projects/p1') return Promise.resolve({ data: { activeRun: { runId: 'r1', status: 'STARTING' } } });
      if (url === '/api/runs/r1/status') return Promise.resolve({ data: { status: 'FAILED', grpcEndpoint: null } });
      return Promise.reject(new Error('unexpected'));
    });
    await expect(joinRun({ projectId: 'p1' })).rejects.toThrow(/FAILED/);
  });

  test('throws when the server rejects registration', async () => {
    stubHappyPath();
    mCore.registerClient.mockResolvedValue({ accepted: false, message: 'run full', assignedRound: -1, serverProtocolVersion: 2 });
    await expect(joinRun({ projectId: 'p1' })).rejects.toThrow(/run full|rejected/i);
  });

  test('passes useTls=true when explicitly requested', async () => {
    stubHappyPath();
    await joinRun({ projectId: 'p1', useTls: true });
    expect(mCore.registerClient).toHaveBeenCalledWith('host:50001', 'r1', 'device-uuid', 'tok-123', true);
  });
});
