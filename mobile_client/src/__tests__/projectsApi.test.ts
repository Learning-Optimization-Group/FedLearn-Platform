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

  describe('a model not yet cleared for phones', () => {
    const caps: DeviceCapabilities = { ramGb: 8, osName: 'android', osVersion: '30' };
    const projects = [{ projectId: 'r', name: 'R', modelType: 'CIFAR_RESNET18', status: 'RUNNING', visibility: 'PRIVATE',
                        requirements: { mobileSafe: false, minRamGb: 4 } }];
    const g = globalThis as unknown as { __DEV__: boolean };
    let dev: boolean;
    beforeEach(() => { dev = g.__DEV__; });
    afterEach(() => { g.__DEV__ = dev; });

    test('is refused by a release build', () => {
      g.__DEV__ = false;
      const { result } = annotateEligibility(projects, caps)[0]!;
      expect(result.eligible).toBe(false);
      expect(result.hardFailures).toEqual(['This model is not supported on phones']);
    });

    test('can be joined by a development build, which says it is not cleared, so it can be measured', () => {
      g.__DEV__ = true;
      const { result } = annotateEligibility(projects, caps)[0]!;
      expect(result.eligible).toBe(true);
      expect(result.hardFailures).toEqual([]);
      expect(result.softWarnings).toEqual(['Not cleared for phones yet: a development build may train it to measure it']);
    });

    test('still fails a development build on every other requirement', () => {
      g.__DEV__ = true;
      const { result } = annotateEligibility(projects, { ...caps, ramGb: 2 })[0]!;
      expect(result.eligible).toBe(false);
      expect(result.hardFailures).toEqual(['Needs 4 GB RAM, device has 2.0 GB']);
    });
  });
});
