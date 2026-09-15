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

const PEM =
  '-----BEGIN CERTIFICATE-----\n' +
  'MIIBszCCAVmgAwIBAgIUQ2FrZUZha2VGYWtlRmFrZUZha2UwCgYIKoZIzj0EAwIw\n' +
  '-----END CERTIFICATE-----\n';

// Server trust: the backend sends whether the FL server serves TLS and the certificate to verify it with. Main
// validates the certificate fully and writes it to a file; the preload has to pass both through and reject junk.
describe('preload startTraining — server trust', () => {
  test('forwards whether to dial TLS and the certificate to verify the server with', async () => {
    await exposedApi().startTraining({ ...BASE, grpcTls: true, grpcServerCertPem: PEM });
    expect(ipcRenderer.invoke).toHaveBeenCalledWith(
      'docker:start-training',
      expect.objectContaining({ grpcTls: true, grpcServerCertPem: PEM }),
    );
  });

  test('a plaintext deployment, whose payload carries a null certificate, still starts', async () => {
    const res = await exposedApi().startTraining({ ...BASE, grpcTls: false, grpcServerCertPem: null });
    expect(res.success).toBe(true);
    expect(ipcRenderer.invoke).toHaveBeenCalled();
  });

  test('refuses a grpcTls that is not a boolean', async () => {
    const res = await exposedApi().startTraining({ ...BASE, grpcTls: 'yes' });
    expect(res.success).toBe(false);
    expect(ipcRenderer.invoke).not.toHaveBeenCalled();
  });

  test('refuses a certificate that is not a PEM certificate', async () => {
    const res = await exposedApi().startTraining({ ...BASE, grpcTls: true, grpcServerCertPem: 'not a certificate' });
    expect(res.success).toBe(false);
    expect(ipcRenderer.invoke).not.toHaveBeenCalled();
  });
});
