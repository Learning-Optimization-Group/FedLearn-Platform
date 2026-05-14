import { registerClientIpcHandlers, __resetClientHandlerStateForTest } from '../main/ipc.handlers';

const mockIpc = { handle: jest.fn() };
jest.mock('electron', () => ({
  ipcMain: { handle: (...args: any[]) => mockIpc.handle(...args) },
}));

const mockAuth = {
  authenticatedGet: jest.fn(),
  authenticatedPost: jest.fn(),
};

const mockDocker = {
  startTraining: jest.fn().mockResolvedValue({ success: true }),
};

beforeEach(() => {
  mockIpc.handle.mockClear();
  mockAuth.authenticatedGet.mockReset();
  mockAuth.authenticatedPost.mockReset();
  mockDocker.startTraining.mockClear();
  __resetClientHandlerStateForTest();
});

test('client:list-projects returns project array on success', async () => {
  mockAuth.authenticatedGet.mockResolvedValue([{ projectId: 'p1', name: 'X', status: 'RUNNING' }]);
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:list-projects')![1];
  const result = await handler({});
  expect(result).toEqual({ success: true, projects: [{ projectId: 'p1', name: 'X', status: 'RUNNING' }] });
});

test('client:list-projects returns success=false when auth helper returns null', async () => {
  mockAuth.authenticatedGet.mockResolvedValue(null);
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:list-projects')![1];
  const result = await handler({});
  expect(result).toEqual({ success: false });
});

test('client:train-project rejects malformed projectId', async () => {
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:train-project')![1];
  const result = await handler({}, 'bad id!!', '/tmp');
  expect(result).toEqual({ success: false, error: 'Invalid project ID' });
});

test('client:train-project fetches connection then starts docker', async () => {
  mockAuth.authenticatedGet.mockResolvedValue({
    projectId: 'p1', name: 'X', modelType: 'CNN-CIFAR10', modelName: 'resnet8',
    serverAddress: 'localhost:50000', partitionId: 3, status: 'RUNNING',
  });
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:train-project')![1];
  const result = await handler({}, 'p1', '/data');
  expect(result.success).toBe(true);
  expect(mockDocker.startTraining).toHaveBeenCalledWith(expect.objectContaining({
    projectId: 'p1',
    serverAddress: 'localhost:50000',
    partitionId: '3',
    modelType: 'CNN-CIFAR10',
    datasetPath: '/data',
  }));
});

test('client:train-project fails when project not running (no connection)', async () => {
  mockAuth.authenticatedGet.mockResolvedValue(null);
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:train-project')![1];
  const result = await handler({}, 'p1', '/data');
  expect(result.success).toBe(false);
});

test('client:request-access POSTs and surfaces backend status', async () => {
  mockAuth.authenticatedPost.mockResolvedValue({ status: 201, data: { status: 'PENDING' } });
  registerClientIpcHandlers(mockAuth as any, mockDocker as any);
  const handler = mockIpc.handle.mock.calls.find((c) => c[0] === 'client:request-access')![1];
  const result = await handler({}, 'p1', 'hello');
  expect(result).toEqual({ success: true, status: 'PENDING' });
  expect(mockAuth.authenticatedPost).toHaveBeenCalledWith('/projects/p1/access-requests', { message: 'hello' });
});
