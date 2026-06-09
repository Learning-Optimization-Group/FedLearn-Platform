# FedLearn Desktop — Hardware Profiles & Training Execution

> **Part of:** [FedLearn Platform Docs](../../README.md) → [Desktop Wiki](./README.md)

---

## Table of Contents

1. [Hardware Profile Overview](#hardware-profile-overview)
2. [Apple Silicon MPS (Native)](#apple-silicon-mps-native)
3. [Windows CUDA — Discrete GPU (Native)](#windows-cuda--discrete-gpu-native)
4. [CPU-Only (Native)](#cpu-only-native)
5. [NVIDIA Jetson SoC (Docker)](#nvidia-jetson-soc-docker)
6. [Model Architecture Selection](#model-architecture-selection)
7. [Training Configuration Parameters](#training-configuration-parameters)
8. [Training Lifecycle State Machine](#training-lifecycle-state-machine)
9. [Hardware Auto-Detection Logic](#hardware-auto-detection-logic)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Hardware Profile Overview

FedLearn Desktop supports four hardware execution profiles. Each profile maps to a specific hardware architecture and execution backend:

| Profile ID | Label | Execution | Docker | GPU Runtime |
|---|---|---|---|---|
| `mps` | Apple Silicon | Native binary | ❌ No | Apple Metal Performance Shaders |
| `discrete` | Discrete GPU | Native binary | ❌ No | NVIDIA CUDA (via PyTorch) |
| `cpu` | CPU Only | Native binary | ❌ No | CPU (no GPU) |
| `jetson` | Jetson SoC | Docker container | ✅ Yes | Tegra GPU (direct device mounts) |

The profile selection drives all downstream decisions: how the training process is launched, what environment variables or device mounts are configured, and which binary/image is used.

---

## Apple Silicon MPS (Native)

### Why Native, Not Docker?

Docker Desktop on macOS runs Linux containers inside a Linux virtual machine (using Apple Hypervisor framework). This VM has **no access to the Apple GPU hardware** — Metal Performance Shaders only work from processes running natively on macOS. Therefore, MPS training **cannot** run in a Docker container on macOS.

The solution: ship a PyInstaller bundle that includes a macOS-native PyTorch build with MPS support. This binary runs as a direct child process of Electron.

### Execution Flow

```
User selects "Apple Silicon" profile
    │
    ▼
HardwareSelector → handleStart()
    │
    ▼
window.fedLearnAPI.startTraining({ hardwareProfile: 'mps', ... })
    │
    ▼
IPC: 'docker:start-training'
    │
    ▼
DockerService.startTraining() → startNativeProcess()
    │
    ▼
resolveNativeInvocation():
    ├── app.isPackaged = true  → binary: <resourcesPath>/fedlearn-client/fedlearn-client
    └── app.isPackaged = false → python3 client-docker/scripts/client.py
    │
    ▼
spawn(binary, ['--project-id', ..., '--server-address', ..., '--partition-id', ...])
    │
    env: { PYTHONUNBUFFERED: '1' }
    │
    ▼
Python client runs training:
    ├── Downloads model from FedLearn server (gRPC/HTTP)
    ├── Trains on local dataset (using torch.device('mps'))
    └── Sends updated weights back to server
    │
stdout/stderr → sendLog() → IPC push → LogPanel
```

### MPS-Specific PyTorch Behavior

The Python client detects MPS availability at runtime:
```python
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

MPS requires macOS 12.3+ and Apple Silicon. On Intel Macs with `mps` profile selected, PyTorch falls back to CPU automatically.

### Binary Path at Runtime

```typescript
// In packaged app:
const bundleDir = path.join(process.resourcesPath, 'fedlearn-client');
// e.g., /Applications/FedLearn Desktop.app/Contents/Resources/fedlearn-client/

const binary = path.join(bundleDir, 'fedlearn-client');
// e.g., /Applications/FedLearn Desktop.app/Contents/Resources/fedlearn-client/fedlearn-client
```

---

## Windows CUDA — Discrete GPU (Native)

### Execution Flow

Same as MPS but uses the Windows binary and CUDA PyTorch:

```
User selects "Discrete GPU" profile
    │
    ▼
resolveNativeInvocation():
    binary: <resourcesPath>/fedlearn-client/fedlearn-client.exe
    │
    ▼
spawn(binary.exe, ['--project-id', ..., ...])
    │
    env: { PYTHONUNBUFFERED: '1' }
    │
    ▼
Python client: torch.device('cuda')
```

### CUDA Detection

Before recommending the `discrete` profile, the hardware probe runs `nvidia-smi`:
```typescript
execFile('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'], (err, stdout) => {
  if (err) { resolve({ available: false }); return; }
  const info = stdout.trim().split('\n')[0]; // e.g., "NVIDIA GeForce RTX 4090"
  resolve({ available: true, info });
});
```

If `nvidia-smi` is not found or fails, `cudaAvailable` is `false` and the probe defaults to `cpu`.

### Two Windows Installer Variants

The Windows installer comes in two variants because the PyInstaller bundle includes PyTorch:

- **CPU variant** (`build-win-cpu.ps1`): `torch` without CUDA. Smaller file size. For users with no NVIDIA GPU or who only want CPU training.
- **CUDA variant** (`build-win-cuda.ps1`): `torch` with CUDA. Larger file. For users with an NVIDIA GPU.

Shipping the wrong variant means the GPU either won't be used (CPU variant on a CUDA machine) or the install is unnecessarily large (CUDA variant on a CPU-only machine).

---

## CPU-Only (Native)

The simplest profile — same native binary path as MPS/CUDA, but no GPU configuration. The Python client falls through to CPU:

```python
device = torch.device('cpu')
```

CPU training is significantly slower than GPU training but is compatible with **any machine**, including those without NVIDIA drivers or Apple Silicon.

**Typical use case:** Development machines, CI/CD testing, machines behind enterprise IT restrictions.

---

## NVIDIA Jetson SoC (Docker)

Jetson is the most complex profile because it requires Docker and specific device node configuration.

### Why Docker?

NVIDIA Jetson runs JetPack OS, which includes a specific version of the L4T (Linux for Tegra) PyTorch wheel. This wheel is compiled against a specific CUDA Compute Architecture (e.g., SM 8.7 for Orin), specific GCC version, and specific Linux kernel. It cannot be PyInstaller-bundled cross-platform.

Docker allows us to ship a pre-built container image (`fedlearn-client:latest`) with the exact L4T environment already configured. The Jetson host's device nodes are bind-mounted into the container so PyTorch inside the container can access the GPU.

### Jetson Device Node Requirements

```typescript
const JETSON_DEVICE_MOUNTS: Docker.DeviceMapping[] = [
  // Tegra GPU control interface
  { PathOnHost: '/dev/nvhost-ctrl',     PathInContainer: '/dev/nvhost-ctrl',     CgroupPermissions: 'rwm' },
  // GPU-specific control interface
  { PathOnHost: '/dev/nvhost-ctrl-gpu', PathInContainer: '/dev/nvhost-ctrl-gpu', CgroupPermissions: 'rwm' },
  // GPU debug interface
  { PathOnHost: '/dev/nvhost-dbg-gpu',  PathInContainer: '/dev/nvhost-dbg-gpu',  CgroupPermissions: 'rwm' },
  // GPU profiling interface
  { PathOnHost: '/dev/nvhost-prof-gpu', PathInContainer: '/dev/nvhost-prof-gpu', CgroupPermissions: 'rwm' },
  // NVIDIA video memory map
  { PathOnHost: '/dev/nvmap',           PathInContainer: '/dev/nvmap',           CgroupPermissions: 'rwm' },
  // GPU kernel interface
  { PathOnHost: '/dev/nvhost-gpu',      PathInContainer: '/dev/nvhost-gpu',      CgroupPermissions: 'rwm' },
];
```

`CgroupPermissions: 'rwm'` = read + write + mknod. All three are required for the GPU to work inside the container.

**Critical:** The `--runtime nvidia` flag (used for discrete NVIDIA GPUs) is **explicitly NOT used** for Jetson:
```typescript
case 'jetson':
  // DO NOT use runtime: 'nvidia' — it hangs on Jetson
  hostConfig.Devices = JETSON_DEVICE_MOUNTS;
  break;
case 'discrete':
  // Standard NVIDIA Container Toolkit
  hostConfig.DeviceRequests = [{ Count: -1, Capabilities: [['gpu']] }];
  break;
```

### Docker Container Configuration

```typescript
const container = await this.docker.createContainer({
  Image: 'fedlearn-client:latest',
  name: 'fedlearn-training-client',
  Env: [
    `PROJECT_ID=${config.projectId}`,
    `SERVER_ADDRESS=${config.serverAddress}`,
    `PARTITION_ID=${config.partitionId}`,
    `MODEL_TYPE=${config.modelType}`,
    `DATASET_PATH=/data`,
  ],
  HostConfig: {
    AutoRemove: false,
    Binds: [`${config.datasetPath}:/data`],  // Dataset bind mount
    Devices: JETSON_DEVICE_MOUNTS,           // GPU access
  },
  Tty: false,
  AttachStdout: true,
  AttachStderr: true,
});
```

### Building the Jetson Docker Image

The Docker image must be built on the Jetson itself (or an ARM64 Linux machine with Jetson's software stack):
```bash
docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

If the image is missing, `DockerService` surfaces a helpful error:
```
Docker image 'fedlearn-client:latest' not found locally.
Build it with: docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

### Log Stream Demultiplexing

Docker's `container.logs()` returns a multiplexed stream. The `demuxDockerStream` method strips the 8-byte frame headers:

```
Raw Docker log chunk:
┌──────────┬───────────┬──────────────────────────────────┐
│ type (1B)│ pad (3B)  │ size (4B, BE) │ payload (size B) │
└──────────┴───────────┴──────────────────────────────────┘

After demux:
"[2026-04-28T01:15:23.456Z] Epoch 1/10 — loss: 0.4321\n"
```

---

## Model Architecture Selection

The model type is selected via a dropdown in the `HardwareSelector` component:

```typescript
<select id="config-model-type" className="form-input" value={modelType} onChange={...}>
  <option value="CNN">CNN</option>
  <option value="OPT-125M">OPT-125M</option>
  <option value="Transformer">Transformer</option>
</select>
```

### How Model Type Affects the Launch Command

```typescript
const args = [
  ...invocation.baseArgs,
  '--project-id', config.projectId,
  '--server-address', config.serverAddress,
  '--partition-id', config.partitionId,
  // NOTE: modelType is NOT passed as an argument for CNN
];

// LLM architectures need special flag
if (config.modelType === 'OPT-125M' || config.modelType === 'Transformer') {
  args.push('--use-llm');
}
```

The `--use-llm` flag tells the Python training client to load the language model variant of the federated learning pipeline instead of the default CNN pipeline.

### For Docker-Based Training (Jetson)

Model type is passed as an environment variable instead:
```typescript
Env: [
  `MODEL_TYPE=${config.modelType}`,
  // ...
]
```

The container's entrypoint reads `MODEL_TYPE` and selects the appropriate training script.

---

## Training Configuration Parameters

### Full Parameter Reference

| Parameter | UI Field | Validation | Purpose |
|---|---|---|---|
| `hardwareProfile` | Profile card selection | `ALLOWED_HARDWARE_PROFILES` enum | Determines execution path and GPU config |
| `projectId` | "Project ID" text input | `/^[a-zA-Z0-9_-]{1,128}$/` | Identifies which federated learning project on the server |
| `serverAddress` | "Server Address" text input | `/^[a-zA-Z0-9._:/-]{1,256}$/` | Host:port of the FedLearn gRPC/REST server |
| `partitionId` | "Partition ID" text input | `/^[0-9]{1,10}$/` | Which data partition this client represents |
| `modelType` | "Model Architecture" dropdown | `/^[a-zA-Z0-9_\-\.]{1,128}$/` | CNN, OPT-125M, or Transformer |
| `datasetPath` | "Local Dataset Path" (browse button) | Existing directory, absolute path | Local dataset directory mounted into training |

### `serverAddress` Format

The server address is passed directly to the Python training client as the gRPC endpoint. Format:
```
host:port           → 192.168.1.100:8080
hostname:port       → fedlearn-server.local:8080
domain:port         → api.fedlearn.company.com:8080
```

Note: This is **not** the same as the backend HTTP URL configured in the auth settings. That URL is for the Spring Boot REST API. This address is for the federated learning training coordinator (Flower/gRPC).

### `partitionId` and Data Partitioning

In federated learning, the global dataset is divided into partitions — one per client (participant). Each client trains only on its local partition:

```
Global Dataset
    ├── Partition 0 → Client A (partitionId=0)
    ├── Partition 1 → Client B (partitionId=1)
    ├── Partition 2 → Client C (partitionId=2)
    └── Partition N → Client N+1 (partitionId=N)
```

The `partitionId` tells the training client which subset of the global dataset it should use. In a real deployment, each hospital/institution has a unique partition ID.

---

## Training Lifecycle State Machine

The training lifecycle from the UI perspective:

```
IDLE
  │
  │ [User clicks Start Training]
  ▼
PULLING / INITIALIZING
  │ (status set optimistically to 'pulling' before IPC resolves)
  │
  ├── IPC fails → ERROR
  │
  └── IPC succeeds
        │
        ▼
      RUNNING
        │ (container/process active, logs streaming)
        │
        ├── [User clicks Stop Training]
        │     └── → IDLE
        │
        ├── Process exits code 0 → COMPLETED
        │
        └── Process exits non-0 → ERROR
```

### Status Transitions in Code

```typescript
// Start button clicked:
setContainerStatus('pulling');          // Optimistic
const result = await startTraining();
if (!result.success) setContainerStatus('error');
// If success: polling will detect 'running' within 3 seconds

// Stop button clicked:
const result = await stopTraining();
if (result.success) setContainerStatus('idle');

// Status poll:
const result = await getDockerStatus();
setContainerStatus(result.status as ContainerStatus);
```

---

## Hardware Auto-Detection Logic

The detection probe runs once on `HardwareSelector` mount and pre-selects the appropriate profile:

```
detectHardware() called
    │
    ▼
probeNvidiaSmi() ──── timeout 2s ────► { available: false }
    │                                           │
    └── available: true → cudaAvailable = true  │
                                                │
platform === 'darwin' && arch === 'arm64'?      │
    YES → recommendedProfile = 'mps'            │
    NO  → cudaAvailable && platform !== 'linux'?│
          YES → recommendedProfile = 'discrete' │
          NO  → recommendedProfile = 'cpu'      │
                                                │
nativeBundleExists()?                           │
    YES (packaged) → check <resourcesPath>/fedlearn-client/
    NO  (dev mode) → always true (uses python3)
    │
    ▼
Return { platform, arch, recommendedProfile, nativeBundleAvailable, cudaAvailable, cudaInfo }
```

### Detection Result Examples

| Machine | Detection Output |
|---|---|
| MacBook Air M2 | `{ platform: 'darwin', arch: 'arm64', recommendedProfile: 'mps', cudaAvailable: false }` |
| Windows PC + RTX 4090 | `{ platform: 'win32', arch: 'x64', recommendedProfile: 'discrete', cudaAvailable: true, cudaInfo: 'NVIDIA GeForce RTX 4090' }` |
| Windows VM (no GPU) | `{ platform: 'win32', arch: 'x64', recommendedProfile: 'cpu', cudaAvailable: false }` |
| Ubuntu Server (no display) | `{ platform: 'linux', arch: 'x64', recommendedProfile: 'cpu', cudaAvailable: false }` |
| Jetson Orin (Linux arm64) | `{ platform: 'linux', arch: 'arm64', recommendedProfile: 'cpu', cudaAvailable: false }` → User manually selects Jetson |

Note: Jetson is detected as `linux/arm64` + no CUDA (nvidia-smi isn't the right tool for Jetson), so it defaults to `cpu`. The user must manually select the Jetson profile. This is intentional — Jetson requires Docker to be running, which we shouldn't assume.

---

## Troubleshooting Guide

### "Docker daemon unreachable"

**Symptom:** Docker warning banner appears; Jetson profile fails to start.

**Cause:** Docker Desktop is not running, or the Docker socket path is wrong.

**Fix:**
- macOS/Windows: Start Docker Desktop and wait for the status to show "Engine running"
- Linux: `sudo systemctl start docker`

**Note:** For MPS, CPU, and Discrete GPU profiles, Docker is not required. Only Jetson needs Docker.

---

### "Native training bundle not found"

**Symptom:** `[System] Native training bundle not found at <resources>/fedlearn-client.`

**Cause:** The PyInstaller bundle was not built before packaging, or the wrong installer variant was used.

**Fix:**
- Run the appropriate build script before packaging:
  - Mac: `client-docker/packaging/build-mac.sh`
  - Windows CPU: `client-docker/packaging/build-win-cpu.ps1`
- Re-run `npm run package:mac` or `npm run package:win:cpu`

---

### "Docker image 'fedlearn-client:latest' not found locally"

**Symptom:** Training fails immediately after starting with this message in the log panel.

**Cause:** The Docker image hasn't been built on this machine.

**Fix:**
```bash
# From the repository root:
docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

---

### Logs not appearing in real-time

**Symptom:** LogPanel stays empty during training, then dumps all logs when training finishes.

**Cause:** Python stdout is buffered (default behavior when not connected to a TTY).

**Fix:** The app sets `PYTHONUNBUFFERED=1` and uses `-u` flag. If logs are still delayed:
1. Verify `PYTHONUNBUFFERED=1` is in the spawned process's environment (check electron-log output)
2. Check that the Python script uses `print(..., flush=True)` or `sys.stdout.flush()` for critical output

---

### "MPS profile cannot run under Docker"

**Symptom:** Error thrown immediately when selecting Jetson profile on macOS.

**Cause:** This is a guard — if somehow `startDockerTraining` is called with `mps` profile, it throws this error.

**Fix:** This shouldn't happen in normal usage. The `mps` profile always goes to `startNativeProcess`. If you see this, check `DockerService.startTraining()` routing logic.

---

### Log panel performance degrading over long training runs

**Symptom:** UI becomes sluggish after hours of training.

**Cause:** The log buffer has exceeded `MAX_LOG_LINES = 10_000` and the string joining + re-render is taking too long.

**Fix:** The buffer cap should handle this automatically. If performance still degrades:
1. Reduce `MAX_LOG_LINES` in `App.tsx`
2. Consider implementing virtual scrolling in `LogPanel` for very long runs

---

*Next: [08 — Developer Guide & Contributing](./08-developer-guide.md)*  
*Previous: [06 — Build, Packaging & Distribution](./06-build-and-packaging.md)*
