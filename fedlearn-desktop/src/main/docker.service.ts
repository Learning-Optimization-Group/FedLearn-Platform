// =============================================================================
// FedLearn Desktop — Docker Service
// =============================================================================
// Manages the federated learning Docker container lifecycle via dockerode.
// Docker socket access is confined exclusively to this Main Process service.
// Hardware profile routing per Section 4.2:
//   - 'discrete' → --gpus all (DeviceRequests)
//   - 'jetson'   → --device /dev/nvhost-* (SoC device mounts)
//   - 'cpu'      → no GPU configuration
// =============================================================================

import Docker from 'dockerode';
import { BrowserWindow } from 'electron';
import { ChildProcess, spawn } from 'child_process';
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
}

// Full list of Jetson SoC device nodes required for GPU access inside containers.
// Incomplete mounts cause the container to hang or crash (Section 4.2).
const JETSON_DEVICE_MOUNTS: Docker.DeviceMapping[] = [
  { PathOnHost: '/dev/nvhost-ctrl', PathInContainer: '/dev/nvhost-ctrl', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-ctrl-gpu', PathInContainer: '/dev/nvhost-ctrl-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-dbg-gpu', PathInContainer: '/dev/nvhost-dbg-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-prof-gpu', PathInContainer: '/dev/nvhost-prof-gpu', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvmap', PathInContainer: '/dev/nvmap', CgroupPermissions: 'rwm' },
  { PathOnHost: '/dev/nvhost-gpu', PathInContainer: '/dev/nvhost-gpu', CgroupPermissions: 'rwm' },
];

const CONTAINER_NAME = 'fedlearn-training-client';
// Default to :latest (matches `docker build -t fedlearn-client .` in DEPLOYMENT_GUIDE.md).
// Pin to a specific version per-environment via FEDLEARN_CLIENT_IMAGE (e.g. fedlearn-client:0.1.0).
const DEFAULT_DOCKER_IMAGE = 'fedlearn-client:latest';
const DOCKER_IMAGE = process.env.FEDLEARN_CLIENT_IMAGE || DEFAULT_DOCKER_IMAGE;

export class DockerService {
  private docker: Docker;
  private mainWindow: BrowserWindow;
  private activeContainerId: string | null = null;
  private logStream: NodeJS.ReadableStream | null = null;
  private nativeProcess: ChildProcess | null = null;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;

    // Platform-aware socket path (Section 4.3)
    const socketPath =
      process.platform === 'win32'
        ? '//./pipe/docker_engine'
        : '/var/run/docker.sock';

    this.docker = new Docker({ socketPath });
    log.info(`[DockerService] Initialized with socket: ${socketPath}`);

    // Verify Docker daemon is reachable. Non-blocking — fires IPC event on failure
    // so the renderer can show a warning without blocking startup.
    this.docker.ping().then(() => {
      log.info('[DockerService] Docker daemon is reachable');
    }).catch((err: Error) => {
      log.error(`[DockerService] Docker daemon unreachable: ${err.message}`);
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('docker:daemon-unavailable', err.message);
      }
    });
  }

  /**
   * Starts a federated learning training container with hardware-aware configuration.
   * Per Section 4.2: discrete GPUs use DeviceRequests, Jetson uses direct device mounts,
   * CPU mode uses no GPU configuration.
   */
  async startTraining(config: TrainingConfig): Promise<void> {
    // Clean up any existing container first
    await this.cleanupExistingContainer();

    const hostConfig: Docker.HostConfig = {
      // Do NOT mount the Docker socket into the training container — principle of least privilege
      AutoRemove: false,
      Binds: [`${config.datasetPath}:/data`],
    };

    // ========== Hardware Profile Routing (Section 4.2) ==========

    // MPS (Apple Silicon): bypass Docker entirely — run client.py natively
    if (config.hardwareProfile === 'mps') {
      log.info('[DockerService] Hardware profile: MPS (Apple Silicon) — launching native process');
      await this.startNativeTraining(config);
      return;
    }

    switch (config.hardwareProfile) {
      case 'discrete':
        // Standard NVIDIA GPU workstation: --gpus all equivalent
        hostConfig.DeviceRequests = [
          {
            Count: -1,
            Capabilities: [['gpu']],
          },
        ];
        log.info('[DockerService] Hardware profile: discrete GPU (DeviceRequests --gpus all)');
        break;

      case 'jetson':
        // NVIDIA Jetson SoC: direct device node mounts (Section 4.2)
        // The standard --runtime nvidia flag is PROHIBITED on Jetson — it searches
        // for PCIe discrete GPU metadata in the kernel device tree and hangs indefinitely.
        hostConfig.Devices = JETSON_DEVICE_MOUNTS;
        log.info('[DockerService] Hardware profile: Jetson SoC (direct device mounts)');
        break;

      case 'cpu':
        // CPU-only mode: no GPU configuration needed
        log.info('[DockerService] Hardware profile: CPU only (no GPU configuration)');
        break;
    }

    // Environment variables matching the entrypoint.sh contract
    const env = [
      `PROJECT_ID=${config.projectId}`,
      `SERVER_ADDRESS=${config.serverAddress}`,
      `PARTITION_ID=${config.partitionId}`,
      `MODEL_TYPE=${config.modelType}`,
      `DATASET_PATH=/data`,
    ];

    log.info(`[DockerService] Creating container: image=${DOCKER_IMAGE}, project=${config.projectId}`);

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
    log.info(`[DockerService] Container created: ${container.id}`);

    // Start the container
    await container.start();
    log.info(`[DockerService] Container started: ${container.id}`);

    // Attach log stream and forward to Renderer via IPC
    this.attachLogStream(container);
  }

  /**
   * Attaches to the container's stdout/stderr and forwards each log line
   * to the Renderer process via 'docker:training-log' IPC channel.
   */
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
      // Byte 0: stream type (1=stdout, 2=stderr)
      // Bytes 4-7: payload size (big-endian uint32)
      // We demux the stream manually for plain-text forwarding.
      const demuxBuffer = { partial: '' };

      stream.on('data', (chunk: Buffer) => {
        // Attempt to extract plain text from the multiplexed stream
        const text = this.demuxDockerStream(chunk, demuxBuffer);
        if (text && !this.mainWindow.isDestroyed()) {
          // Send plain text lines to Renderer (LogPanel renders text only — no HTML)
          this.mainWindow.webContents.send('docker:training-log', text);
        }
      });

      stream.on('end', () => {
        log.info('[DockerService] Log stream ended');
        if (!this.mainWindow.isDestroyed()) {
          this.mainWindow.webContents.send('docker:training-log', '[System] Container log stream ended.\n');
        }
        this.logStream = null;
      });

      stream.on('error', (err: Error) => {
        log.error(`[DockerService] Log stream error: ${err.message}`);
        if (!this.mainWindow.isDestroyed()) {
          this.mainWindow.webContents.send('docker:training-log', `[System] Log stream error: ${err.message}\n`);
        }
      });

      log.info('[DockerService] Log stream attached');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[DockerService] Failed to attach log stream: ${message}`);
    }
  }

  /**
   * Demultiplexes Docker's multiplexed stream format.
   * Docker uses an 8-byte header per frame when Tty=false:
   *   [stream_type(1)] [0(3)] [size(4)] [payload(size)]
   */
  private demuxDockerStream(chunk: Buffer, state: { partial: string }): string {
    let output = state.partial;
    let offset = 0;

    while (offset < chunk.length) {
      // Need at least 8 bytes for the header
      if (offset + 8 > chunk.length) {
        // Incomplete header — treat remainder as raw text
        output += chunk.slice(offset).toString('utf-8');
        break;
      }

      const payloadSize = chunk.readUInt32BE(offset + 4);

      // Validate payload size to prevent buffer overread
      if (payloadSize === 0) {
        offset += 8;
        continue;
      }

      if (offset + 8 + payloadSize > chunk.length) {
        // Incomplete payload — extract what we can
        output += chunk.slice(offset + 8).toString('utf-8');
        break;
      }

      output += chunk.slice(offset + 8, offset + 8 + payloadSize).toString('utf-8');
      offset += 8 + payloadSize;
    }

    state.partial = '';
    return output;
  }

  /**
   * Stops and removes the active training container.
   */
  async stopTraining(): Promise<void> {
    // Handle native MPS process
    if (this.nativeProcess) {
      log.info('[DockerService] Stopping native MPS process');
      this.nativeProcess.kill('SIGTERM');
      // Give it 5s to shut down gracefully, then force kill
      setTimeout(() => {
        if (this.nativeProcess && !this.nativeProcess.killed) {
          this.nativeProcess.kill('SIGKILL');
        }
      }, 5000);
      return;
    }

    if (!this.activeContainerId) {
      log.warn('[DockerService] No active container to stop');
      return;
    }

    try {
      const container = this.docker.getContainer(this.activeContainerId);

      // Attempt graceful stop with 10s timeout before SIGKILL
      try {
        await container.stop({ t: 10 });
        log.info(`[DockerService] Container stopped: ${this.activeContainerId}`);
      } catch (stopErr: unknown) {
        // Container may already be stopped
        const msg = stopErr instanceof Error ? stopErr.message : '';
        if (!msg.includes('is not running') && !msg.includes('304')) {
          log.warn(`[DockerService] Stop warning: ${msg}`);
        }
      }

      // Remove the container
      try {
        await container.remove({ force: true });
        log.info(`[DockerService] Container removed: ${this.activeContainerId}`);
      } catch (rmErr: unknown) {
        const msg = rmErr instanceof Error ? rmErr.message : '';
        if (!msg.includes('No such container') && !msg.includes('404')) {
          log.warn(`[DockerService] Remove warning: ${msg}`);
        }
      }
    } finally {
      this.activeContainerId = null;

      // Clean up log stream
      if (this.logStream) {
        (this.logStream as NodeJS.ReadableStream & { destroy?: () => void }).destroy?.();
        this.logStream = null;
      }
    }
  }

  /**
   * Returns the current status of the training container.
   */
  async getStatus(): Promise<string> {
    // Handle native MPS process status
    if (this.nativeProcess) {
      if (this.nativeProcess.exitCode === null) return 'running';
      return this.nativeProcess.exitCode === 0 ? 'completed' : 'error';
    }

    if (!this.activeContainerId) {
      return 'idle';
    }

    try {
      const container = this.docker.getContainer(this.activeContainerId);
      const info = await container.inspect();

      if (info.State.Running) return 'running';
      if (info.State.Restarting) return 'restarting';
      if (info.State.Paused) return 'paused';
      if (info.State.Dead) return 'error';

      // Container exists but is not running — it has exited
      const exitCode = info.State.ExitCode;
      if (exitCode === 0) return 'completed';
      return 'error';
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

  /**
   * Starts client.py as a native child process for MPS (Apple Silicon).
   * Docker cannot pass through Metal GPUs, so we bypass it entirely.
   * The log stream is forwarded to the Renderer via the same IPC channel
   * used by Docker containers, so the UI doesn't need to know the difference.
   */
  private async startNativeTraining(config: TrainingConfig): Promise<void> {
    // Kill any existing native process
    if (this.nativeProcess) {
      this.nativeProcess.kill('SIGTERM');
      this.nativeProcess = null;
    }

    // Resolve paths relative to the packaged or dev app directory.
    // In dev: <project-root>/client-docker/scripts/client.py
    // In production: the user must have the client-docker directory alongside the app.
    const appRoot = path.resolve(__dirname, '..', '..', '..');
    const clientScript = path.join(appRoot, 'client-docker', 'scripts', 'client.py');
    const frameworkSrc = path.join(appRoot, 'framework', 'src');

    const args = [
      '-u', clientScript,
      '--project-id', config.projectId,
      '--server-address', config.serverAddress,
      '--partition-id', config.partitionId,
    ];

    // Add model-type arguments based on configuration
    if (config.modelType === 'OPT-125M') {
      args.push('--use-llm');
    }

    log.info(`[DockerService:MPS] Spawning: python3 ${args.join(' ')}`);
    log.info(`[DockerService:MPS] PYTHONPATH=${frameworkSrc}`);

    const child = spawn('python3', args, {
      env: {
        ...process.env,
        PYTHONPATH: `${frameworkSrc}:${process.env.PYTHONPATH || ''}`,
        PYTHONUNBUFFERED: '1',
      },
      cwd: path.join(appRoot, 'client-docker', 'scripts'),
    });

    this.nativeProcess = child;

    // Forward stdout to renderer
    child.stdout.on('data', (data: Buffer) => {
      const text = data.toString('utf-8');
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('docker:training-log', text);
      }
    });

    // Forward stderr to renderer
    child.stderr.on('data', (data: Buffer) => {
      const text = data.toString('utf-8');
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('docker:training-log', `[stderr] ${text}`);
      }
    });

    child.on('error', (err: Error) => {
      log.error(`[DockerService:MPS] Process error: ${err.message}`);
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('docker:training-log', `[System] Native process error: ${err.message}\n`);
      }
      this.nativeProcess = null;
    });

    child.on('exit', (code: number | null, signal: string | null) => {
      log.info(`[DockerService:MPS] Process exited with code=${code}, signal=${signal}`);
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send(
          'docker:training-log',
          `[System] Native process exited (code=${code}, signal=${signal})\n`
        );
      }
      this.nativeProcess = null;
    });

    log.info(`[DockerService:MPS] Native process started (PID: ${child.pid})`);
  }

  /**
   * Cleans up any existing container with the same name before starting a new one.
   */
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
        log.info(`[DockerService] Cleaned up existing container: ${CONTAINER_NAME}`);
      }
    } catch {
      // No existing container — this is the expected path
    }

    this.activeContainerId = null;
    this.logStream = null;
  }
}
