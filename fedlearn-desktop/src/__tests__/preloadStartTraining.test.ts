// The preload is the only path from the renderer's start request to main, and it rebuilds the payload field by
// field. A field it leaves out never reaches the client, however carefully main validates it. These drive the real
// preload with Electron's bridge mocked and check what it hands to main.
jest.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: jest.fn() },
  ipcRenderer: {
    invoke: jest.fn().mockResolvedValue({ success: true }),
    on: jest.fn(),
    removeListener: jest.fn(),
    removeAllListeners: jest.fn(),
    send: jest.fn(),
  },
}));

import { contextBridge, ipcRenderer } from 'electron';
import '../preload/preload';

type StartTraining = (config: Record<string, unknown>) => Promise<{ success: boolean; error?: string }>;

function exposedApi(): { startTraining: StartTraining } {
  const call = (contextBridge.exposeInMainWorld as jest.Mock).mock.calls.find(([name]) => name === 'fedLearnAPI');
  if (!call) {
    throw new Error('the preload did not expose fedLearnAPI');
  }
  return call[1];
}

const BASE = {
  hardwareProfile: 'cpu',
  projectId: 'proj-1',
  serverAddress: 'fl.example.org:50001',
  partitionId: '2',
  modelType: 'CNN',
  datasetPath: '/Users/me/data',
  connectionToken: 'a.b.c',
  strategy: 'FedAvg',
};

beforeEach(() => (ipcRenderer.invoke as jest.Mock).mockClear());

describe('preload startTraining forwarding', () => {
  test('forwards the training arm, so a FROZEN_HEAD project does not train as FULL', async () => {
    await exposedApi().startTraining({ ...BASE, trainingArm: 'FROZEN_HEAD' });
    expect(ipcRenderer.invoke).toHaveBeenCalledWith(
      'docker:start-training',
      expect.objectContaining({ trainingArm: 'FROZEN_HEAD' }),
    );
  });
});
