// Regression: DockerService must NOT probe the Docker daemon on construction.
//
// An eager constructor ping() previously fired a "docker:daemon-unavailable"
// banner on every launch, so Windows/macOS users with no Docker (running the
// bundled native client on CPU/CUDA/MPS) saw a spurious
//   "Docker is not running: connect ENOENT \\.\pipe\docker_engine"
// error even though Docker is only needed for the Jetson path. The daemon is
// now probed lazily inside startDockerTraining() only. This test pins that.

import Docker from 'dockerode';
import { DockerService } from '../main/docker.service';

jest.mock('dockerode');

describe('DockerService startup', () => {
  it('does not probe the Docker daemon on construction', () => {
    const ping = jest.fn().mockResolvedValue(undefined);
    (Docker as unknown as jest.Mock).mockImplementation(() => ({ ping }));

    // Minimal BrowserWindow stand-in — the constructor only stores it.
    const fakeWindow = {
      isDestroyed: () => false,
      webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
    } as never;

    // eslint-disable-next-line no-new
    new DockerService(fakeWindow);

    expect(ping).not.toHaveBeenCalled();
  });
});

// DE-3: hardwareProfile is the sole dispatcher. Only 'jetson' uses the Docker
// path; every other profile (discrete/cpu/mps) runs the bundled native client.
// The dead discrete/cpu Docker switch-cases were removed, so a non-jetson
// profile must never reach createContainer.
describe('DockerService training routing (DE-3)', () => {
  const fakeWindow = {
    isDestroyed: () => false,
    webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
  } as never;

  const baseConfig = {
    projectId: 'p1',
    serverAddress: 'localhost:50000',
    partitionId: '0',
    modelType: 'CNN',
    datasetPath: '/data',
  };

  type PrivateDockerService = {
    startNativeProcess: (config: unknown) => Promise<void>;
    startDockerTraining: (config: unknown) => Promise<void>;
  };

  it("routes a non-jetson profile ('discrete') to the native path, never createContainer", async () => {
    const createContainer = jest.fn();
    const ping = jest.fn().mockResolvedValue(undefined);
    (Docker as unknown as jest.Mock).mockImplementation(() => ({
      ping,
      createContainer,
      getContainer: jest.fn(),
    }));

    const service = new DockerService(fakeWindow);
    // Stub the native path so no real process is spawned; we only assert routing.
    const nativeSpy = jest
      .spyOn(service as unknown as PrivateDockerService, 'startNativeProcess')
      .mockResolvedValue(undefined);

    await service.startTraining({ ...baseConfig, hardwareProfile: 'discrete' } as never);

    expect(nativeSpy).toHaveBeenCalledTimes(1);
    expect(createContainer).not.toHaveBeenCalled();
    expect(ping).not.toHaveBeenCalled();
  });

  it("routes 'jetson' to the Docker path (startDockerTraining), not native", async () => {
    const createContainer = jest.fn();
    const ping = jest.fn().mockResolvedValue(undefined);
    (Docker as unknown as jest.Mock).mockImplementation(() => ({
      ping,
      createContainer,
      getContainer: jest.fn(),
    }));

    const service = new DockerService(fakeWindow);
    const nativeSpy = jest
      .spyOn(service as unknown as PrivateDockerService, 'startNativeProcess')
      .mockResolvedValue(undefined);
    const dockerSpy = jest
      .spyOn(service as unknown as PrivateDockerService, 'startDockerTraining')
      .mockResolvedValue(undefined);

    await service.startTraining({ ...baseConfig, hardwareProfile: 'jetson' } as never);

    expect(dockerSpy).toHaveBeenCalledTimes(1);
    expect(nativeSpy).not.toHaveBeenCalled();
  });
});
