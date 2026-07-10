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

// DE-2: the user-selected "Local Dataset Path" is bound to /data on the Jetson
// Docker path but was silently DROPPED on the native path — the field looked
// wired but the native client never received it. It must be forwarded as
// `--dataset-path` when non-empty (parity with the Docker /data bind), and
// omitted entirely when blank so the client falls back to its recipe-default
// data source. This pins the spawn argv on both branches.
describe('DockerService native dataset-path forwarding (DE-2)', () => {
  const fakeWindow = {
    isDestroyed: () => false,
    webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
  } as never;

  const baseConfig = {
    hardwareProfile: 'cpu' as const,
    projectId: 'p1',
    serverAddress: 'localhost:50000',
    partitionId: '0',
    modelType: 'MLP',
    datasetPath: '',
  };

  type PrivateDockerService = {
    resolveNativeInvocation: () => {
      command: string;
      baseArgs: string[];
      cwd: string;
      env: NodeJS.ProcessEnv;
    } | null;
  };

  function makeService(): { service: DockerService; spawnMock: jest.Mock } {
    const spawnMock = spawn as unknown as jest.Mock;
    spawnMock.mockReset();
    spawnMock.mockImplementation(() => {
      const emitter = new EventEmitter() as EventEmitter & {
        stdout: { on: jest.Mock };
        stderr: { on: jest.Mock };
        exitCode: number | null;
        pid: number;
        kill: jest.Mock;
      };
      emitter.stdout = { on: jest.fn() };
      emitter.stderr = { on: jest.fn() };
      emitter.exitCode = null;
      emitter.pid = 4242;
      emitter.kill = jest.fn(() => true);
      return emitter;
    });

    (Docker as unknown as jest.Mock).mockImplementation(() => ({
      ping: jest.fn().mockResolvedValue(undefined),
      getContainer: jest.fn(),
    }));

    const service = new DockerService(fakeWindow);
    jest
      .spyOn(service as unknown as PrivateDockerService, 'resolveNativeInvocation')
      .mockReturnValue({ command: 'python3', baseArgs: [], cwd: '/tmp', env: {} });
    return { service, spawnMock };
  }

  it('forwards --dataset-path <path> when a dataset path is provided', async () => {
    const { service, spawnMock } = makeService();

    await service.startTraining({ ...baseConfig, datasetPath: '/home/me/ecg.csv' } as never);

    const args = spawnMock.mock.calls[0][1] as string[];
    const idx = args.indexOf('--dataset-path');
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(args[idx + 1]).toBe('/home/me/ecg.csv');
  });

  it('omits --dataset-path entirely when the dataset path is blank', async () => {
    const { service, spawnMock } = makeService();

    await service.startTraining({ ...baseConfig, datasetPath: '' } as never);

    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args).not.toContain('--dataset-path');
  });

  it('omits --dataset-path when the path is whitespace-only', async () => {
    const { service, spawnMock } = makeService();

    await service.startTraining({ ...baseConfig, datasetPath: '   ' } as never);

    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args).not.toContain('--dataset-path');
  });
});

// DE-11: stop + status LIFECYCLE across the native and Docker execution paths. The routing/drain/
// dataset tests above pin START; these pin STOP (idempotent no-op, the native-already-exited fast path,
// Docker stop+remove+clear, and the SIGKILL escalation) and the getStatus() state matrix.
describe('DockerService stop + status lifecycle (DE-11)', () => {
  const fakeWindow = {
    isDestroyed: () => false,
    webContents: { send: jest.fn(), isDestroyed: () => false, isLoading: () => false },
  } as never;

  type Privates = {
    nativeProcess: unknown;
    activeContainerId: string | null;
  };

  function makeService(getContainer?: jest.Mock): DockerService {
    (Docker as unknown as jest.Mock).mockImplementation(() => ({
      ping: jest.fn().mockResolvedValue(undefined),
      getContainer: getContainer ?? jest.fn(),
    }));
    return new DockerService(fakeWindow);
  }

  function fakeNative(exitCode: number | null, signalCode: string | null = null) {
    const emitter = new EventEmitter();
    return Object.assign(emitter, { exitCode, signalCode, kill: jest.fn() });
  }

  // ---- getStatus() state matrix ----
  it("getStatus: native running -> 'running', clean exit -> 'completed', non-zero exit -> 'error'", async () => {
    const s = makeService();
    (s as unknown as Privates).nativeProcess = fakeNative(null);
    expect(await s.getStatus()).toBe('running');
    (s as unknown as Privates).nativeProcess = fakeNative(0);
    expect(await s.getStatus()).toBe('completed');
    (s as unknown as Privates).nativeProcess = fakeNative(1);
    expect(await s.getStatus()).toBe('error');
  });

  it("getStatus: nothing active -> 'idle'", async () => {
    expect(await makeService().getStatus()).toBe('idle');
  });

  it("getStatus: Docker container running -> 'running' (inspected by id)", async () => {
    const inspect = jest.fn().mockResolvedValue({ State: { Running: true } });
    const getContainer = jest.fn().mockReturnValue({ inspect });
    const s = makeService(getContainer);
    (s as unknown as Privates).activeContainerId = 'c1';
    expect(await s.getStatus()).toBe('running');
    expect(getContainer).toHaveBeenCalledWith('c1');
  });

  it("getStatus: Docker container exited 0 -> 'completed'", async () => {
    const inspect = jest.fn().mockResolvedValue({ State: { Running: false, ExitCode: 0 } });
    const s = makeService(jest.fn().mockReturnValue({ inspect }));
    (s as unknown as Privates).activeContainerId = 'c1';
    expect(await s.getStatus()).toBe('completed');
  });

  it("getStatus: a vanished Docker container ('No such container') -> 'idle' and clears the id", async () => {
    const inspect = jest.fn().mockRejectedValue(new Error('No such container: c1 (404)'));
    const s = makeService(jest.fn().mockReturnValue({ inspect }));
    (s as unknown as Privates).activeContainerId = 'c1';
    expect(await s.getStatus()).toBe('idle');
    expect((s as unknown as Privates).activeContainerId).toBeNull();
  });

  // ---- stopTraining() ----
  it('stopTraining: no active process -> resolves without throwing (idempotent)', async () => {
    await expect(makeService().stopTraining()).resolves.toBeUndefined();
  });

  it('stopTraining: an already-exited native process is cleared immediately (no kill, no 5s grace)', async () => {
    const s = makeService();
    const proc = fakeNative(0);
    (s as unknown as Privates).nativeProcess = proc;
    await s.stopTraining();
    expect(proc.kill).not.toHaveBeenCalled();
    expect((s as unknown as Privates).nativeProcess).toBeNull();
  });

  it('stopTraining: Docker path stops + force-removes the container and clears the id', async () => {
    const stop = jest.fn().mockResolvedValue(undefined);
    const remove = jest.fn().mockResolvedValue(undefined);
    const s = makeService(jest.fn().mockReturnValue({ stop, remove }));
    (s as unknown as Privates).activeContainerId = 'c1';
    await s.stopTraining();
    expect(stop).toHaveBeenCalled();
    expect(remove).toHaveBeenCalledWith({ force: true });
    expect((s as unknown as Privates).activeContainerId).toBeNull();
  });

  it('stopTraining: a live native process gets SIGTERM, then SIGKILL after the 5s grace if it ignores it', async () => {
    jest.useFakeTimers();
    try {
      const s = makeService();
      const proc = fakeNative(null, null); // never exits
      (s as unknown as Privates).nativeProcess = proc;
      const pending = s.stopTraining();
      expect(proc.kill).toHaveBeenCalledWith('SIGTERM');
      jest.advanceTimersByTime(5000); // grace elapses with no 'exit' event
      await pending;
      expect(proc.kill).toHaveBeenCalledWith('SIGKILL');
      expect((s as unknown as Privates).nativeProcess).toBeNull();
    } finally {
      jest.useRealTimers();
    }
  });
});
