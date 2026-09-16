# FedLearn Desktop — Hardware Profiles & Training Execution

> **Part of:** [FedLearn Platform Docs](../README.md) → [Desktop Wiki](./README.md)

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
| `discrete` | Discrete GPU | Native binary | ❌ No | NVIDIA CUDA (via PyTorch in the bundle) |
| `cpu` | CPU Only | Native binary | ❌ No | CPU (no GPU) |
| `jetson` | Jetson SoC | Docker container | ✅ Yes | Tegra GPU — see the [Jetson section](#nvidia-jetson-soc-docker) |

`hardwareProfile` is the **sole** dispatcher in `DockerService.startTraining()`: `jetson` → Docker, everything else → the bundled native client. There is no Docker fallback for the other three; if the native bundle is missing, training does not run.

> `discrete` does **not** go through Docker. An earlier version of this page (and of the profile card copy) described it as using `DeviceRequests: --gpus all`, i.e. the `--gpus all` container equivalent. That branch no longer exists in `startDockerTraining`'s switch — a `discrete` value reaching that method now throws, because it means a routing regression (`2b02173`).

---

## Apple Silicon MPS (Native)

### Why Native, Not Docker?

Docker Desktop on macOS runs Linux containers inside a Linux virtual machine (using Apple Hypervisor framework). This VM has **no access to the Apple GPU hardware** — Metal Performance Shaders only work from processes running natively on macOS. Therefore, MPS training **cannot** run in a Docker container on macOS.

The solution: ship a PyInstaller bundle that includes a macOS-native PyTorch build with MPS support. This binary runs as a direct child process of Electron.

### Execution Flow

```
User selects "Apple Silicon" (or leaves it preselected by detection)
    │
    ▼
TrainSection → handleStart() → getProjectConnection(projectId)
    │  (backend returns serverAddress, partitionId, modelType, strategy,
    │   trainingArm, connectionToken)
    ▼
window.fedLearnAPI.startTraining({ hardwareProfile: 'mps', ... })
    │
    ▼
IPC: 'docker:start-training'   (validate + canonicalize + consent-check)
    │
    ▼
DockerService.startTraining() → startNativeProcess()
    │
    ▼
resolveNativeInvocation():
    ├── app.isPackaged = true  → binary: <resourcesPath>/fedlearn-client/fedlearn-client
    └── app.isPackaged = false → python3 -u fl-runtime/client.py   (PYTHONPATH=framework/src)
    │
    ▼
spawn(binary, ['--project-id', …, '--server-address', …, '--partition-id', …,
               '--model-type', …, '--strategy', …, '--dataset-path', …])
    │
    env: { PYTHONUNBUFFERED: '1', FEDLEARN_CONNECTION_TOKEN: <token> }
    │
    ▼
Python client runs training:
    ├── Downloads the global model from the FL server (gRPC)
    ├── Trains on the local dataset (using torch.device('mps'))
    └── Sends the updated parameters back
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

Jetson is the most complex profile: it is the only one that requires Docker, and the mechanism by which the container gets GPU access is both version-sensitive and — in the shipped code — wrong for JetPack 6. Read the correction below before deploying to a device.

### Why Docker?

NVIDIA Jetson runs JetPack OS, which includes a specific version of the L4T (Linux for Tegra) PyTorch wheel. That wheel is compiled against a specific CUDA compute architecture (SM 8.7 for Orin), a specific GCC, and a specific kernel. It cannot be PyInstaller-bundled cross-platform.

Docker lets us ship a pre-built container image (`fedlearn-client:latest`) with the exact L4T environment already configured. The container then needs GPU access from the host — and **how** that access is granted is the part that is version-sensitive and that this page corrects below.

### What the Shipped Code Does

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

case 'jetson':
  // Source comment: "The --runtime nvidia flag is PROHIBITED on Jetson — it
  // searches for PCIe discrete-GPU metadata in the kernel device tree and
  // hangs indefinitely. Direct device mounts are the supported path."
  hostConfig.Devices = JETSON_DEVICE_MOUNTS;
  break;
```

`CgroupPermissions: 'rwm'` = read + write + mknod.

---

### The `--runtime nvidia` Prohibition Was Measured Wrong on JetPack 6

**Read this before trusting the recipe above on any specific device.**

The blanket ban on `--runtime nvidia` above is **withdrawn**. It was tested directly on an NVIDIA AGX Orin running **L4T R36.5.0 / JetPack 6.2**, with `nvidia-container-toolkit 1.19.0-1` installed and `docker info` listing `nvidia` among its runtimes. The result is the reverse of what the comment claims:

| Approach | Result on that Orin |
|---|---|
| `docker run --runtime nvidia` (image `fedbench:orin`) | **Works — 7.9 s, `torch.cuda.is_available()` True, device reported as "Orin".** No hang. |
| `docker run --runtime nvidia` (image `vllm:0.20.0-orin`) | Works, CUDA True |
| Device mounts, **no** `--runtime nvidia`, host driver libs bind-mounted | **Fails** — `cuInit → 801` (`CUDA_ERROR_NOT_SUPPORTED`), then a segfault. The in-container `libcuda.so.1` is a stub (`file too short`). |

So on JetPack 6 the flag the code prohibits is the one that works, and the hand-rolled device-mount path the code implements is the one that does not.

Two further problems with the shipped list on that L4T version:

1. **`/dev/nvhost-ctrl` does not exist on L4T R36.5.** It is the first entry in `JETSON_DEVICE_MOUNTS`, and Docker **hard-errors** (`no such file or directory`) if you pass a device node that isn't there. The set above is a JetPack-5-era node list.
2. **The documented base image tag is JetPack-5-era.** `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` is the `--build-arg BASE_IMAGE=` value shown for Jetson builds in the root `README.md`, `framework/docs/installation.md` and the client-docker wiki. It is two major L4T generations behind an R36.5 device. (The Dockerfile's own `ARG BASE_IMAGE` default is `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` — an x86 image — so the L4T tag must be passed explicitly on a Jetson build regardless.)

#### Honest scope of this correction

- **What was measured:** container GPU access on one AGX Orin at L4T R36.5.0 / JetPack 6.2. Both mechanisms were exercised; the results above are observations, not inference.
- **What was *not* measured:** the original hang. It is plausible the ban was genuinely correct on the older JetPack 5 / `nvidia-container-runtime` it was written against — but **that was not re-tested**, because no JetPack 5 hardware was available. Treat the "it hung on JetPack 5" half as an inference, not a finding.
- **What was also not measured:** the desktop `DockerService` Jetson flow **end to end**. Only the container GPU-access mechanism it depends on was exercised, with plain `docker run`. Nothing here says the desktop app's Jetson path has been validated on JetPack 6.
- **What follows from it:** on JetPack 6+, treat `--runtime nvidia` as the default to try and keep the device-mount path as a fallback. On any device, **re-verify against the L4T version that device actually runs** rather than trusting either rule blindly. `cat /etc/nv_tegra_release` and `docker info | grep -i runtime` are the two things to check first.

The shipped code has not been changed to match; this is documentation of a measurement, not of a fix.

---

### Docker Container Configuration

```typescript
const container = await this.docker.createContainer({
  Image: DOCKER_IMAGE,               // FEDLEARN_CLIENT_IMAGE ?? 'fedlearn-client:latest'
  name: 'fedlearn-training-client',
  Env: buildContainerEnv(config),    // see below
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

`buildContainerEnv(config)` produces:

```
PROJECT_ID, SERVER_ADDRESS, PARTITION_ID, MODEL_TYPE, DATASET_PATH=/data
STRATEGY=<strategy>                     (only when set)
TRAINING_ARM=<arm>                      (only when set)
FEDLEARN_CONNECTION_TOKEN=<token>       (only when set)
```

`client-docker/entrypoint.sh` converts `MODEL_TYPE`, `STRATEGY` and `TRAINING_ARM` back into `--model-type` / `--strategy` / `--training-arm` flags — omitting each entirely when its variable is unset — giving the container path argv parity with the native path. `FEDLEARN_CONNECTION_TOKEN` needs no flag: the framework reads it straight from the process environment.

### Building the Jetson Docker Image

The Docker image must be built on the Jetson itself (or an ARM64 Linux machine with the matching L4T stack). **The build context is the repository root**, because the Dockerfile copies both `framework/` and `fl-runtime/`:

```bash
docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

If the image is missing, `DockerService` surfaces the command verbatim:
```
Docker image 'fedlearn-client:latest' not found locally.
Build it with: docker build -t fedlearn-client:latest -f client-docker/Dockerfile .
```

Set `FEDLEARN_CLIENT_IMAGE` to point the app at a different image tag.

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

**There is no model-architecture dropdown any more.** `619a5a5` replaced manual entry with "pick a model to train": the user selects a *project* from the list the backend says they may train, and `modelType` arrives in that project's connection payload as a **recipe key** (the catalog lives in `fl-runtime/recipes.py` and is served to the platform at `GET /api/model-recipes`). The old three-option `<select>` of `CNN` / `OPT-125M` / `Transformer` is gone, along with the hardcoded values.

### How Model Type Reaches the Client

```typescript
const args = [
  ...invocation.baseArgs,
  '--project-id', config.projectId,
  '--server-address', config.serverAddress,
  '--partition-id', config.partitionId,
];

// Forward the recipe key so the client trains the right architecture. The value
// comes from the backend connection payload, not a hardcoded dropdown. The client
// derives USE_LLM from --model-type TRANSFORMER itself, so --use-llm is no longer
// passed separately.
if (config.modelType)   args.push('--model-type', config.modelType);
if (config.trainingArm) args.push('--training-arm', config.trainingArm);
if (config.strategy)    args.push('--strategy', config.strategy);
if (datasetPath)        args.push('--dataset-path', datasetPath);
```

Three changes from the old behaviour, all of them corrections:

- **`--model-type` is now passed.** It previously was not, and the container client silently defaulted to CNN.
- **`--use-llm` is no longer passed.** The client derives it from `--model-type TRANSFORMER` (the flag itself survives in `fl-runtime/client.py` only as a deprecated alias).
- **`--strategy` is passed** (`e6fae1a`) so a non-MLP DeComFL project runs the DeComFL client path instead of a mismatched FedAvg-path client. Only DeComFL actually changes client behaviour; the other strategy strings use the same first-order path, so passing them is a safe no-op.

`--dataset-path` (DE-2, `d8f1b60`) was meant to mirror the Jetson path's `${datasetPath}:/data` bind for the native client, and is omitted when blank. **It no longer reaches a receiver:** `fl-runtime/client.py` — the dev-mode target and the PyInstaller bundle's entry point — declares no `--dataset-path` argument and uses strict `parse_args()`, so a non-blank dataset path makes the native client exit 2 on an unrecognised argument. DE-2 added the flag to `client-docker/scripts/client.py`, which DA-5 (`d2cc757`) deleted when it consolidated onto `fl-runtime/`. See [03 → Starting a Native Process](./03-main-process.md#starting-a-native-process).

**`--training-arm` is constructed here but never actually emitted from the desktop app today**, because the value is dropped one layer earlier at the preload bridge — see [04 → the `startTraining` validation flow](./04-preload-ipc-bridge.md#the-starttraining-validation-flow). The argv and container-env construction shown here are both correct and unit-tested; the gap is upstream of them.

### For Docker-Based Training (Jetson)

Everything travels as environment variables instead of argv, and `client-docker/entrypoint.sh` converts them back into the same flags — see [Docker Container Configuration](#docker-container-configuration) above.

---

## Training Configuration Parameters

### Full Parameter Reference

| Parameter | Where it comes from | Validation | Purpose |
|---|---|---|---|
| `hardwareProfile` | Detected, overridable under **Advanced** | `ALLOWED_HARDWARE_PROFILES` enum | Determines the execution path and GPU config |
| `projectId` | Connection payload (user picked the project) | `/^[a-zA-Z0-9_-]{1,128}$/` | Which FL project on the server |
| `serverAddress` | Connection payload | `/^[a-zA-Z0-9._:/-]{1,256}$/` | `host:port` of the FL gRPC server |
| `partitionId` | Connection payload (**server-assigned**) | `/^[0-9]{1,10}$/` | Which data partition this client represents |
| `modelType` | Connection payload | `/^[a-zA-Z0-9_\-\.]{1,128}$/` | The recipe key to train |
| `strategy` | Connection payload | `/^[a-zA-Z0-9_\-\.]{1,64}$/`; bad value ⇒ dropped | Aggregation strategy (DeComFL vs the first-order path) |
| `trainingArm` | Connection payload | Strict: `FULL` \| `FROZEN_HEAD`; bad value ⇒ **throws** | Which parameter subset the client federates |
| `connectionToken` | Connection payload | `/^[A-Za-z0-9._-]+$/`, ≤ 8192 | FL connection token (SE-14) |
| `datasetPath` | **User**, via the native dialog only | Existing absolute directory **and** on the consent allowlist; `''` allowed | Local dataset, bind-mounted or passed as `--dataset-path` |

Only `hardwareProfile` and `datasetPath` are user-supplied, and `datasetPath` only through the OS picker. Everything else is resolved server-side by `GET /api/client/projects/{id}/connection` after an idempotent join.

### `serverAddress` Format

The server address is passed directly to the Python training client as the gRPC endpoint:
```
host:port           → 192.168.1.100:50000
hostname:port       → fedlearn-server.local:50000
domain:port         → api.fedlearn.company.com:50000
```

Note: this is **not** the backend HTTP URL configured in Settings. That URL is the Spring Boot REST API; this address is the FL training coordinator (gRPC), on a port the backend reserved for the run.

### `partitionId` and Data Partitioning

In federated learning, the global dataset is divided into partitions — one per client. Each client trains only on its local partition:

```
Global Dataset
    ├── Partition 0 → Client A (partitionId=0)
    ├── Partition 1 → Client B (partitionId=1)
    ├── Partition 2 → Client C (partitionId=2)
    └── Partition N → Client N+1 (partitionId=N)
```

The backend **assigns** this id when the client requests a connection; the user never types it. That is also what makes the "drain before respawn" behaviour in `DockerService.stopTraining()` matter — two live clients on the same partition would both train and both submit.

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

**Symptom:** the log panel shows `[System] Docker daemon unreachable: … Start the Docker daemon.` when starting a Jetson run.

**Cause:** the Docker daemon is not running, or the socket path is wrong.

**Fix:**
- macOS/Windows: start Docker Desktop and wait for "Engine running"
- Linux: `sudo systemctl start docker`

**Note:** there is **no startup Docker warning banner any more.** The eager constructor ping and the `docker:daemon-unavailable` push channel were removed in `4d7d3a4` because they produced a false alarm for every MPS/CPU/CUDA user who never touches Docker. The daemon is probed only when the Jetson path needs it, and the error appears in the log panel at that moment. For MPS, CPU and Discrete GPU profiles Docker is not required at all.

---

### "Native training bundle not found"

**Symptom:** `[System] Native training bundle not found at <resources>/fedlearn-client.`

**Cause:** The PyInstaller bundle was not built before packaging, or the wrong installer variant was used.

**Fix:**
- Run the appropriate build script before packaging:
  - Mac: `client-docker/packaging/build-mac.sh`
  - Windows CPU: `client-docker/packaging/build-win-cpu.ps1`
  - Windows CUDA: `client-docker/packaging/build-win-cuda.ps1`
  - Linux: `client-docker/packaging/build-linux.sh`
- Re-run `npm run package:mac` (or the platform equivalent)

The `npm run package:*` scripts run `scripts/check-native-bundle.js` first, so this should now fail at *packaging* time with the exact build command rather than at *runtime* on a user's machine. If you see it at runtime, either the installer was produced by invoking `electron-builder` directly (bypassing the preflight) or the bundle was removed after packaging.

In the UI, the same condition also shows up before you press Start: the **Hardware detected** readiness row goes to `warn` with "Native client bundle missing — reinstall to enable training". There is no Docker fallback for the non-Jetson profiles.

---

### Jetson: `docker: Error response from daemon: … /dev/nvhost-ctrl: no such file or directory`

**Symptom:** container creation fails immediately on a Jetson.

**Cause:** `/dev/nvhost-ctrl` is in `JETSON_DEVICE_MOUNTS` but does not exist on L4T R36.5 (JetPack 6). Docker hard-errors on a missing device node rather than skipping it. That node set is JetPack-5-era.

**Fix / next step:** check what your device actually runs (`cat /etc/nv_tegra_release`) and read [the correction above](#the---runtime-nvidia-prohibition-was-measured-wrong-on-jetpack-6). On JetPack 6 the working mechanism measured here was `--runtime nvidia`, not the device-mount list.

---

### Jetson: `cuInit → 801` (`CUDA_ERROR_NOT_SUPPORTED`) then a segfault

**Symptom:** the container starts, then dies during CUDA initialization.

**Cause:** measured on an AGX Orin at L4T R36.5.0 — with device mounts and no `--runtime nvidia`, the in-container `libcuda.so.1` is a stub (`file too short`), so `cuInit` fails and the process segfaults.

**Fix / next step:** same as above — on JetPack 6, `--runtime nvidia` with `nvidia-container-toolkit` installed is the path that worked. Also check the base image: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` is two major L4T generations behind an R36.5 device.

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

**Symptom:** the run fails immediately with this message in the log panel.

**Cause:** a routing guard. `startDockerTraining`'s switch throws for `mps`, and its `default` branch throws for any other non-`jetson` profile (`Profile 'X' does not use the Docker path — only 'jetson' runs under Docker`).

**Fix:** this cannot happen through the UI — `DockerService.startTraining()` sends everything except `jetson` to `startNativeProcess`. Seeing either message means the dispatch was bypassed or regressed; check `startTraining()`'s routing.

---

### Log panel performance degrading over long training runs

**Symptom:** UI becomes sluggish after hours of training.

**Cause:** log rendering cost. Two independent caps already bound it — `MAX_LOG_LINES = 10_000` on the buffer in `App.tsx`, and `MAX_RENDERED_LINES = 2_000` on the DOM in `LogPanel.tsx` — and `logView.ts` parses only newly-appended entries while `LogLineRow` is memoized, so a steady stream should re-render only the new rows.

**Fix:** if it still degrades:
1. Use the log filter box — filtering runs over the full buffer but renders far fewer rows
2. Reduce `MAX_RENDERED_LINES` in `LogPanel.tsx` (cheaper than reducing the buffer, which loses history)
3. Consider virtual scrolling in `LogPanel` for very long runs

---

*Next: [08 — Developer Guide & Contributing](./08-developer-guide.md)*  
*Previous: [06 — Build, Packaging & Distribution](./06-build-and-packaging.md)*
