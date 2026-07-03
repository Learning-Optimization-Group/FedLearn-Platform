// =============================================================================
// FedLearn Desktop — Training Orchestration Service
// =============================================================================
// Manages the federated-learning training lifecycle. Two execution paths:
//
//   1. Native binary (Mac arm64 MPS, Windows x64 CUDA/CPU)
//      In packaged mode the Electron app ships a PyInstaller bundle of the
//      Python client at <resourcesPath>/fedlearn-client/. No Docker, no
//      system Python, no repo checkout required on the end-user's machine.
//      In dev mode we fall back to spawning system python3 against the
//      repo's scripts/ directory with framework/src on PYTHONPATH.
//
//   2. Docker container (Jetson SoC)
//      Jetson hardware uses direct /dev/nvhost-* device mounts; bundling a
//      self-contained Jetson exe is impractical because NVIDIA's L4T torch
//      wheel is pinned to a specific JetPack firmware stack.
//
// Docker socket access is confined exclusively to this Main Process service.
// =============================================================================

import Docker from 'dockerode';
import { app, BrowserWindow } from 'electron';
import { ChildProcess, spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import log from 'electron-log';

export type HardwareProfile = 'discrete' | 'jetson' | 'cpu' | 'mps';

export interface TrainingConfig {
  hardwareProfile: HardwareProfile;
  projectId: string;
  serverAddress: string;
  partitionId: string;
  modelType: string;
  datasetPath: string;
  // Backend-minted FL connection token (from GET /client/projects/{id}/connection).
  // Optional so the legacy no-auth flow still type-checks; required in practice once
  // the FL server is fail-closed (app.fl.require-client-auth=true).
  connectionToken?: string;
}

/**
 * Container env for the docker client path. The framework client reads
 * FEDLEARN_CONNECTION_TOKEN straight from its process environment (see
 * fedlearn/security/client_interceptor.maybe_wrap_channel), so the FL connection
 * token travels as a container env var rather than a CLI arg. Omitted entirely
 * when absent, so a gate-off server still accepts the legacy no-token flow.
 */
export function buildContainerEnv(config: TrainingConfig): string[] {
  const env = [
    `PROJECT_ID=${config.projectId}`,
    `SERVER_ADDRESS=${config.serverAddress}`,
    `PARTITION_ID=${config.partitionId}`,
    `MODEL_TYPE=${config.modelType}`,
    `DATASET_PATH=/data`,
  ];
  if (config.connectionToken) {
    env.push(`FEDLEARN_CONNECTION_TOKEN=${config.connectionToken}`);
  }
  return env;
}

/**
 * Injects the FL connection token into a spawn env for the native client path.
 * Same rationale as buildContainerEnv — the framework reads it from the env.
 * Returns the base env unchanged when no token is set.
 */
export function withConnectionTokenEnv(
  base: NodeJS.ProcessEnv,
  config: TrainingConfig,
): NodeJS.ProcessEnv {
  return config.connectionToken
    ? { ...base, FEDLEARN_CONNECTION_TOKEN: config.connectionToken }
    : base;
}

// Full list of Jetson SoC device nodes required for GPU access inside containers.
// Incomplete mounts cause the container to hang or crash.
const JETSON_DEVICE_MOUNTS: Docker.DeviceMapping[] = [
  { PathOnHost: '/dev/nvhost-ctrl', PathInContainer: '/dev/nvhost-ctrl', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-ctrl-gpu', PathInContainer: '/dev/nvhost-ctrl-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-dbg-gpu', PathInContainer: '/dev/nvhost-dbg-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-prof-gpu', PathInContainer: '/dev/nvhost-prof-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvmap', PathInContainer: '/dev/nvmap', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-gpu', PathInContainer: '/dev/nvhost-gpu', CgroupPermissions: 'rwm' },
];

const CONTAINER_NAME = 'fedlearn-training-client';
const DEFAULT_DOCKER_IMAGE = 'fedlearn-client:latest';
const DOCKER_IMAGE = process.env.FEDLEARN_CLIENT_IMAGE || DEFAULT_DOCKER_IMAGE;

// Name of the PyInstaller bundle directory shipped via electron-builder
// extraResources. Matches the `name` attribute in fedlearn-client.spec.
const NATIVE_BUNDLE_DIR = 'fedlearn-client';

export class DockerService {
  private docker: Docker;
  private mainWindow: BrowserWindow;
  private activeContainerId: string | null = null;
  private logStream: NodeJS.ReadableStream | null = null;
  private nativeProcess: ChildProcess | null = null;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;

    const socketPath =
      process.platform === 'win32'
        ? '//./pipe/docker_engine'
        : '/var/run/docker.sock';

    // dockerode opens the socket lazily — constructing the client does NOT connect.
    // We deliberately do NOT probe the daemon on startup. Docker is required ONLY for
    // the Jetson training path, which probes on demand in startDockerTraining() and
    // surfaces an actionable error at the moment it's needed. Probing eagerly on every
    // launch produced a spurious "Docker is not running: connect ENOENT \\.\pipe\docker_engine"
    // banner for the overwhelming majority of users (Windows/macOS on CPU/CUDA/MPS), who
    // run the bundled native client and never touch Docker at all.
    this.docker = new Docker({ socketPath });
    log.info(`[DockerService] Initialized with socket: ${socketPath} (daemon probed lazily — Jetson path only)`);
  }

  /**
   * Routes the training request to the native bundled binary (mps/cuda/cpu)
   * or to Docker (jetson). The hardware profile is the sole dispatcher.
   */
  async startTraining(config: TrainingConfig): Promise<void> {
    if (config.hardwareProfile === 'jetson') {
      await this.startDockerTraining(config);
      return;
    }
    await this.startNativeProcess(config);
  }

  /**
   * Stops whichever execution path is currently active. Safe to call even
   * when nothing is running (logs a warning and returns).
   */
  async stopTraining(): Promise<void> {
    if (this.nativeProcess) {
      const proc = this.nativeProcess;
      // Already exited? (exitCode set on normal exit; signalCode set when killed by a signal.)
      // proc.killed is NOT an "exited" signal — it's true the instant .kill() delivers a signal —
      // so we must not use it to detect liveness.
      if (proc.exitCode !== null || proc.signalCode !== null) {
        this.nativeProcess = null;
        return; // nothing to wait for (avoids a spurious 5s grace on quit)
      }
      log.info('[DockerService] Stopping native process (SIGTERM, then SIGKILL after 5s grace)');
      proc.kill('SIGTERM');
      // WAIT for the child to actually exit (up to a 5s grace), then escalate to SIGKILL. Keep
      // this.nativeProcess SET during the drain: clearing it early would let a concurrent
      // docker:get-status poll observe `idle` and re-enable the Start button, allowing a second
      // spawn onto the same partition (startNativeProcess's dedup guard also keys on nativeProcess).
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          // Gate the escalation on EXIT status, not proc.killed (which is already true from the
          // SIGTERM above) — otherwise SIGKILL would never be sent to a SIGTERM-ignoring child.
          if (proc.exitCode === null && proc.signalCode === null) {
            log.warn('[DockerService] native process did not exit on SIGTERM; sending SIGKILL');
            try { proc.kill('SIGKILL'); } catch { /* already gone */ }
          }
          resolve();
        }, 5000);
        proc.once('exit', () => { clearTimeout(timer); resolve(); });
      });
      this.nativeProcess = null; // clear only AFTER the drain completes
      return;
    }

    if (this.activeContainerId) {
      await this.stopDockerContainer();
      return;
    }

    log.warn('[DockerService] No active training process to stop');
  }

  /**
   * Unified status across native + Docker paths.
   */
  async getStatus(): Promise<string> {
    if (this.nativeProcess) {
      if (this.nativeProcess.exitCode === null) return 'running';
      return this.nativeProcess.exitCode === 0 ? 'completed' : 'error';
    }

    if (!this.activeContainerId) return 'idle';

    try {
      const container = this.docker.getContainer(this.activeContainerId);
      const info = await container.inspect();

      if (info.State.Running) return 'running';
      if (info.State.Restarting) return 'restarting';
      if (info.State.Paused) return 'paused';
      if (info.State.Dead) return 'error';

      return info.State.ExitCode === 0 ? 'completed' : 'error';
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '';
      if (message.includes('No such container') || message.includes('404')) {
        this.activeContainerId = null;
        return 'idle';
      }
      log.error(`[DockerService] Status check failed: ${message}`);
      return 'error';
    }
  }

  // ==========================================================================
  // Native execution path (Mac MPS, Windows CUDA/CPU)
  // ==========================================================================

  /**
   * Resolves the native client invocation for the current runtime.
   *
   * Packaged mode: the PyInstaller bundle is shipped inside the app's
   * resources directory via electron-builder extraResources. We spawn the
   * bundle's entry-point binary directly — no python, no PYTHONPATH, no
   * external deps.
   *
   * Dev mode: fall back to the system python3 + scripts/client.py so the
   * developer workflow keeps working without rebuilding the bundle after
   * every Python edit.
   *
   * Returns null if packaged mode is detected but the bundle is missing —
   * the caller surfaces a clear error to the renderer rather than silently
   * attempting the dev fallback on an end-user machine.
   */
  private resolveNativeInvocation(): { command: string; baseArgs: string[]; cwd: string; env: NodeJS.ProcessEnv } | null {
    const binaryName = process.platform === 'win32' ? 'fedlearn-client.exe' : 'fedlearn-client';

    if (app.isPackaged) {
      const bundleDir = path.join(process.resourcesPath, NATIVE_BUNDLE_DIR);
      const binary = path.join(bundleDir, binaryName);

      if (!fs.existsSync(binary)) {
        log.error(`[Native] Packaged bundle missing at ${binary}`);
        return null;
      }

      return {
        command: binary,
        baseArgs: [],
        cwd: bundleDir,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
      };
    }

    // Dev fallback: spawn python3 against the source tree.
    const repoRoot = path.resolve(__dirname, '..', '..', '..');
    const clientScript = path.join(repoRoot, 'client-docker', 'scripts', 'client.py');
    const frameworkSrc = path.join(repoRoot, 'framework', 'src');

    if (!fs.existsSync(clientScript)) {
      log.error(`[Native] Dev-mode script missing at ${clientScript}`);
      return null;
    }

    const pythonPathSep = process.platform === 'win32' ? ';' : ':';
    const existingPythonPath = process.env.PYTHONPATH || '';

    return {
      command: process.platform === 'win32' ? 'python' : 'python3',
      baseArgs: ['-u', clientScript],
      cwd: path.dirname(clientScript),
      env: {
        ...process.env,
        PYTHONPATH: existingPythonPath ? `${frameworkSrc}${pythonPathSep}${existingPythonPath}` : frameworkSrc,
        PYTHONUNBUFFERED: '1',
      },
    };
  }

  private async startNativeProcess(config: TrainingConfig): Promise<void> {
    if (this.nativeProcess) {
      this.nativeProcess.kill('SIGTERM');
      this.nativeProcess = null;
    }

    const invocation = this.resolveNativeInvocation();
    if (!invocation) {
      const msg = app.isPackaged
        ? `Native training bundle not found at <resources>/${NATIVE_BUNDLE_DIR}. Reinstall the app or rebuild with the client bundle.`
        : 'Dev-mode client.py not found. Run the app from the repo root.';
      this.sendLog(`[System] ${msg}\n`);
      throw new Error(msg);
    }

    const args = [
      ...invocation.baseArgs,
      '--project-id', config.projectId,
      '--server-address', config.serverAddress,
      '--partition-id', config.partitionId,
    ];

    // Forward the recipe key so the client trains the right architecture. The
    // value comes from the backend connection payload (the project's modelType),
    // not a hardcoded dropdown. The client derives USE_LLM from --model-type
    // TRANSFORMER itself, so --use-llm is no longer passed separately.
    if (config.modelType) {
      args.push('--model-type', config.modelType);
    }

    log.info(`[Native] Profile=${config.hardwareProfile} command=${invocation.command} args=${args.join(' ')}`);
    log.info(`[Native] cwd=${invocation.cwd}`);

    const child = spawn(invocation.command, args, {
      env: withConnectionTokenEnv(invocation.env, config),
      cwd: invocation.cwd,
    });

    this.nativeProcess = child;

    child.stdout?.on('data', (data: Buffer) => this.sendLog(data.toString('utf-8')));
    child.stderr?.on('data', (data: Buffer) => this.sendLog(`[stderr] ${data.toString('utf-8')}`));

    child.on('error', (err: Error) => {
      log.error(`[Native] Process error: ${err.message}`);
      this.sendLog(`[System] Native process error: ${err.message}\n`);
      this.nativeProcess = null;
    });

    child.on('exit', (code: number | null, signal: string | null) => {
      log.info(`[Native] Process exited code=${code} signal=${signal}`);
      this.sendLog(`[System] Native process exited (code=${code}, signal=${signal})\n`);
      this.nativeProcess = null;
    });

    log.info(`[Native] Process started (PID: ${child.pid})`);
  }

  // ==========================================================================
  // Docker execution path (Jetson)
  // ==========================================================================

  private async startDockerTraining(config: TrainingConfig): Promise<void> {
    try {
      await this.docker.ping();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      const hint = process.platform === 'win32'
        ? 'Start Docker Desktop and wait for the whale icon to say "Engine running."'
        : 'Start the Docker daemon.';
      const full = `Docker daemon unreachable: ${message}. ${hint}`;
      this.sendLog(`[System] ${full}\n`);
      throw new Error(full);
    }

    await this.cleanupExistingContainer();

    const hostConfig: Docker.HostConfig = {
      // Principle of least privilege — never mount the host Docker socket
      // into the training container.
      AutoRemove: false,
      Binds: [`${config.datasetPath}:/data`],
    };

    switch (config.hardwareProfile) {
      case 'jetson':
        // The --runtime nvidia flag is PROHIBITED on Jetson — it searches
        // for PCIe discrete-GPU metadata in the kernel device tree and
        // hangs indefinitely. Direct device mounts are the supported path.
        hostConfig.Devices = JETSON_DEVICE_MOUNTS;
        log.info('[Docker] Profile: Jetson SoC (direct device mounts)');
        break;
      case 'mps':
        // MPS is native-only; it can never run under Docker.
        throw new Error('MPS profile cannot run under Docker');
      default:
        // Jetson is the ONLY profile that uses the Docker path — startTraining()
        // routes every other profile (discrete/cpu/mps) to the bundled native
        // client and never calls this method for them. Reaching here with a
        // non-jetson profile means a routing regression, so fail loudly instead
        // of silently building a Docker container for a native profile.
        throw new Error(
          `Profile '${config.hardwareProfile}' does not use the Docker path — only 'jetson' runs under Docker`,
        );
    }

    const env = buildContainerEnv(config);

    log.info(`[Docker] Creating container: image=${DOCKER_IMAGE}, project=${config.projectId}`);

    try {
      const container = await this.docker.createContainer({
        Image: DOCKER_IMAGE,
        name: CONTAINER_NAME,
        Env: env,
        HostConfig: hostConfig,
        Tty: false,
        AttachStdout: true,
        AttachStderr: true,
      });

      this.activeContainerId = container.id;
      log.info(`[Docker] Container created: ${container.id}`);

      await container.start();
      log.info(`[Docker] Container started: ${container.id}`);

      this.attachLogStream(container);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      if (message.includes('No such image') || message.includes('404')) {
        const full = `Docker image '${DOCKER_IMAGE}' not found locally. Build it with: docker build -t ${DOCKER_IMAGE} -f client-docker/Dockerfile .`;
        this.sendLog(`[System] ${full}\n`);
        throw new Error(full);
      }
      throw err;
    }
  }

  private async stopDockerContainer(): Promise<void> {
    if (!this.activeContainerId) return;

    try {
      const container = this.docker.getContainer(this.activeContainerId);

      try {
        await container.stop({ t: 10 });
        log.info(`[Docker] Container stopped: ${this.activeContainerId}`);
      } catch (stopErr: unknown) {
        const msg = stopErr instanceof Error ? stopErr.message : '';
        if (!msg.includes('is not running') && !msg.includes('304')) {
          log.warn(`[Docker] Stop warning: ${msg}`);
        }
      }

      try {
        await container.remove({ force: true });
        log.info(`[Docker] Container removed: ${this.activeContainerId}`);
      } catch (rmErr: unknown) {
        const msg = rmErr instanceof Error ? rmErr.message : '';
        if (!msg.includes('No such container') && !msg.includes('404')) {
          log.warn(`[Docker] Remove warning: ${msg}`);
        }
      }
    } finally {
      this.activeContainerId = null;
      if (this.logStream) {
        (this.logStream as NodeJS.ReadableStream & { destroy?: () => void }).destroy?.();
        this.logStream = null;
      }
    }
  }

  private async attachLogStream(container: Docker.Container): Promise<void> {
    try {
      const stream = await container.logs({
        follow: true,
        stdout: true,
        stderr: true,
        timestamps: true,
      });

      this.logStream = stream;

      // Docker multiplexed stream: each frame has an 8-byte header
      //   [stream_type(1)] [0(3)] [size(4)] [payload(size)]
      const demuxBuffer = { partial: '' };

      stream.on('data', (chunk: Buffer) => {
        const text = this.demuxDockerStream(chunk, demuxBuffer);
        if (text) this.sendLog(text);
      });

      stream.on('end', () => {
        log.info('[Docker] Log stream ended');
        this.sendLog('[System] Container log stream ended.\n');
        this.logStream = null;
      });

      stream.on('error', (err: Error) => {
        log.error(`[Docker] Log stream error: ${err.message}`);
        this.sendLog(`[System] Log stream error: ${err.message}\n`);
      });

      log.info('[Docker] Log stream attached');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[Docker] Failed to attach log stream: ${message}`);
    }
  }

  private demuxDockerStream(chunk: Buffer, state: { partial: string }): string {
    let output = state.partial;
    let offset = 0;

    while (offset < chunk.length) {
      if (offset + 8 > chunk.length) {
        output += chunk.slice(offset).toString('utf-8');
        break;
      }

      const payloadSize = chunk.readUInt32BE(offset + 4);

      if (payloadSize === 0) {
        offset += 8;
        continue;
      }

      if (offset + 8 + payloadSize > chunk.length) {
        output += chunk.slice(offset + 8).toString('utf-8');
        break;
      }

      output += chunk.slice(offset + 8, offset + 8 + payloadSize).toString('utf-8');
      offset += 8 + payloadSize;
    }

    state.partial = '';
    return output;
  }

  private async cleanupExistingContainer(): Promise<void> {
    try {
      const container = this.docker.getContainer(CONTAINER_NAME);
      const info = await container.inspect();

      if (info) {
        try {
          await container.stop({ t: 5 });
        } catch {
          // May already be stopped
        }
        await container.remove({ force: true });
        log.info(`[Docker] Cleaned up existing container: ${CONTAINER_NAME}`);
      }
    } catch {
      // No existing container — expected path
    }

    this.activeContainerId = null;
    this.logStream = null;
  }

  private sendLog(text: string): void {
    if (!this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send('docker:training-log', text);
    }
  }
}
