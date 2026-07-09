// Regression: DockerService must NOT probe the Docker daemon on construction.
//
// An eager constructor ping() previously fired a "docker:daemon-unavailable"
// banner on every launch, so Windows/macOS users with no Docker (running the
// bundled native client on CPU/CUDA/MPS) saw a spurious
//   "Docker is not running: connect ENOENT \\.\pipe\docker_engine"
// error even though Docker is only needed for the Jetson path. The daemon is
// now probed lazily inside startDockerTraining() only. This test pins that.

import { EventEmitter } from 'events';
import { spawn } from 'child_process';
import Docker from 'dockerode';
import { DockerService } from '../main/docker.service';

jest.mock('dockerode');
jest.mock('child_process');

describe('DockerService startup', () => {
  it('does not probe the Docker daemon on construction', () => {
    const ping = jest.fn().mockResolvedValue(undefined);
    (Docker as unknown as jest.Mock).mockImplementation(() => ({ ping }));

    // Minimal BrowserWindow stand-in — the constructor only stores it.
    const fakeWindow = {
      isDestroyed: () => false,
      webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
    } as never;

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

// DE-9: a native respawn must fully DRAIN the previous native client before the
// new one spawns. startNativeProcess previously killed the old process
// fire-and-forget (`kill('SIGTERM'); nativeProcess = null`) and immediately
// spawned the replacement, so a new FL client could connect to the server on the
// same partition while the old one was still alive — a double-client race. The
// fix awaits stopTraining()'s SIGTERM → grace → SIGKILL drain, which only clears
// nativeProcess once the old child has exited. This test pins the ordering: the
// first process's SIGTERM + 'exit' must both land BEFORE the second spawn, so the
// two native processes never overlap.
describe('DockerService native respawn drain (DE-9)', () => {
  const fakeWindow = {
    isDestroyed: () => false,
    webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
  } as never;

  const baseConfig = {
    hardwareProfile: 'cpu' as const,
    projectId: 'p1',
    serverAddress: 'localhost:50000',
    partitionId: '0',
    modelType: 'CNN',
    datasetPath: '/data',
  };

  // Minimal ChildProcess stand-in backed by a real EventEmitter so on()/once()/emit
  // behave like the real thing. Stays "running" (exitCode/signalCode null) until the
  // test explicitly emits 'exit'.
  type FakeChild = EventEmitter & {
    stdout: { on: jest.Mock };
    stderr: { on: jest.Mock };
    exitCode: number | null;
    signalCode: string | null;
    pid: number;
    kill: jest.Mock;
  };

  type PrivateDockerService = {
    resolveNativeInvocation: () => {
      command: string;
      baseArgs: string[];
      cwd: string;
      env: NodeJS.ProcessEnv;
    } | null;
  };

  it('awaits the previous process exit (SIGTERM then exit) before the second spawn', async () => {
    const events: string[] = [];
    const children: FakeChild[] = [];

    const spawnMock = spawn as unknown as jest.Mock;
    spawnMock.mockReset();
    spawnMock.mockImplementation(() => {
      const emitter = new EventEmitter() as FakeChild;
      emitter.stdout = { on: jest.fn() };
      emitter.stderr = { on: jest.fn() };
      emitter.exitCode = null;
      emitter.signalCode = null;
      emitter.pid = 1000 + children.length;
      emitter.kill = jest.fn((signal?: string) => {
        events.push(`kill:${signal}`);
        return true;
      });
      children.push(emitter);
      events.push('spawn');
      return emitter;
    });

    (Docker as unknown as jest.Mock).mockImplementation(() => ({
      ping: jest.fn().mockResolvedValue(undefined),
      getContainer: jest.fn(),
    }));

    const service = new DockerService(fakeWindow);
    // Bypass the real invocation resolver (electron app.isPackaged + fs.existsSync)
    // so the real startNativeProcess spawn/drain logic runs against the mocked spawn.
    jest
      .spyOn(service as unknown as PrivateDockerService, 'resolveNativeInvocation')
      .mockReturnValue({ command: 'python3', baseArgs: [], cwd: '/tmp', env: {} });

    // First client: no prior process, so it spawns immediately.
    await service.startTraining(baseConfig as never);
    expect(spawnMock).toHaveBeenCalledTimes(1);
    expect(events).toEqual(['spawn']);

    // Second client: kick it off but DON'T await yet. The drain must be in-flight —
    // SIGTERM already delivered to child[0], but the second spawn must NOT have
    // happened, because the old child hasn't exited.
    const second = service.startTraining(baseConfig as never);
    expect(children[0].kill).toHaveBeenCalledWith('SIGTERM');
    expect(events).toEqual(['spawn', 'kill:SIGTERM']);
    expect(spawnMock).toHaveBeenCalledTimes(1); // <-- no overlap: second not spawned yet

    // Release the drain: the old child finally exits.
    children[0].exitCode = 0;
    events.push('exit');
    children[0].emit('exit', 0, null);

    await second;

    // Ordering proof: first spawn, its SIGTERM, its exit, THEN the second spawn.
    expect(events).toEqual(['spawn', 'kill:SIGTERM', 'exit', 'spawn']);
    expect(spawnMock).toHaveBeenCalledTimes(2);
    expect(children[1]).not.toBe(children[0]);
  });
});
