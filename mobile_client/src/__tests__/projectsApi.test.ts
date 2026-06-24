import { api } from '../lib/restClient';
import { listProjects, joinProject, annotateEligibility } from '../lib/projectsApi';
import type { DeviceCapabilities } from '../lib/deviceCapabilities.types';

jest.mock('../lib/restClient', () => ({ api: { get: jest.fn(), post: jest.fn() } }));
const mApi = api as unknown as { get: jest.Mock; post: jest.Mock };

describe('projectsApi', () => {
  beforeEach(() => jest.clearAllMocks());

  test('listProjects GETs /api/client/projects', async () => {
    mApi.get.mockResolvedValue({ data: [{ projectId: 'p1', name: 'P1', modelType: 'CNN', status: 'RUNNING', visibility: 'PUBLIC' }] });
    const out = await listProjects();
    expect(mApi.get).toHaveBeenCalledWith('/api/client/projects');
    expect(out[0]!.projectId).toBe('p1');
  });

  test('joinProject POSTs the join endpoint', async () => {
    await joinProject('p1');
    expect(mApi.post).toHaveBeenCalledWith('/api/client/projects/p1/join');
  });

  test('annotateEligibility pairs each project with an eligibility result', () => {
    const caps: DeviceCapabilities = { ramGb: 4, osName: 'android', osVersion: '30' };
    const projects = [
      { projectId: 'a', name: 'A', modelType: 'CNN', status: 'RUNNING', visibility: 'PUBLIC', requirements: { minRamGb: 8 } },
      { projectId: 'b', name: 'B', modelType: 'MLP', status: 'RUNNING', visibility: 'PUBLIC', requirements: { minRamGb: 2 } },
    ];
    const rows = annotateEligibility(projects, caps);
    expect(rows.find(r => r.project.projectId === 'a')!.result.eligible).toBe(false); // 4 < 8
    expect(rows.find(r => r.project.projectId === 'b')!.result.eligible).toBe(true);  // 4 >= 2
  });
});
