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
