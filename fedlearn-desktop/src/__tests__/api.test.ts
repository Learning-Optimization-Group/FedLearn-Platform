import { fetchMyProjects, fetchDiscoverProjects, fetchMyRequests, requestAccess, trainProject } from '../renderer/lib/api';

beforeEach(() => {
  (global as any).window = {
    fedLearnAPI: {
      listClientProjects: jest.fn().mockResolvedValue({ success: true, projects: [{ projectId: 'p1' }] }),
      listDiscover: jest.fn().mockResolvedValue({ success: true, projects: [{ id: 'p2' }] }),
      listMyRequests: jest.fn().mockResolvedValue({ success: true, requests: [{ id: 7 }] }),
      requestAccess: jest.fn().mockResolvedValue({ success: true, status: 'PENDING' }),
      trainProject: jest.fn().mockResolvedValue({ success: true }),
    },
  };
});

test('fetchMyProjects returns projects array', async () => {
  await expect(fetchMyProjects()).resolves.toEqual([{ projectId: 'p1' }]);
});

test('fetchDiscoverProjects returns projects array', async () => {
  await expect(fetchDiscoverProjects()).resolves.toEqual([{ id: 'p2' }]);
});

test('fetchMyRequests returns requests array', async () => {
  await expect(fetchMyRequests()).resolves.toEqual([{ id: 7 }]);
});

test('requestAccess passes projectId and message to bridge', async () => {
  await requestAccess('abc', 'please');
  expect((window as any).fedLearnAPI.requestAccess).toHaveBeenCalledWith('abc', 'please');
});

test('trainProject passes projectId and datasetPath to bridge', async () => {
  const r = await trainProject('abc', '/data');
  expect(r.success).toBe(true);
  expect((window as any).fedLearnAPI.trainProject).toHaveBeenCalledWith('abc', '/data');
});

test('fetchMyProjects returns [] on bridge failure', async () => {
  (window as any).fedLearnAPI.listClientProjects.mockResolvedValueOnce({ success: false });
  await expect(fetchMyProjects()).resolves.toEqual([]);
});
