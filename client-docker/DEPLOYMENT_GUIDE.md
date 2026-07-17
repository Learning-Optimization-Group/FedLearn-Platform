# FedLearn Client — Deployment Guide

> **Audience**: Developers deploying the FedLearn client on edge devices (NVIDIA Jetson) or local machines (Mac/Linux).
> **Last Updated**: April 2, 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Deployment on NVIDIA Jetson AGX Orin (Docker)](#2-deployment-on-nvidia-jetson-agx-orin-docker)
3. [Deployment on Local Mac / Linux (Native)](#3-deployment-on-local-mac--linux-native)
4. [Challenges & Fixes — Complete Log](#4-challenges--fixes--complete-log)
5. [Building Cross-Platform Docker Images](#5-building-cross-platform-docker-images)
6. [Troubleshooting Playbook](#6-troubleshooting-playbook)

---

## 1. Architecture Overview

```
┌──────────────────────────┐      gRPC (HTTP/2)       ┌───────────────────────┐
│   FedLearn Client        │ ◄──────────────────────► │   FedLearn Server     │
│   (Jetson / Mac / Linux) │                          │   (Mac / Cloud)       │
│                          │                          │                       │
│  • CNN (CIFAR-10)        │   RegisterClient ──►     │  • Spring Boot (8081) │
│  • OPT-125M (NLP)       │   GetGlobalModel ──►     │  • Python gRPC Server │
│  • Local Training        │   SubmitModelUpdate ──►  │  • FedAvg Aggregation │
│  • gRPC Streaming        │   Heartbeat ──►          │  • React Dashboard    │
└──────────────────────────┘                          └───────────────────────┘
```

**Components**:
- **Backend**: Spring Boot API (`./gradlew bootRun` on port 8081) + React frontend (port 5173)
- **gRPC Server**: Python process spawned by backend on a dynamic port (e.g., 50181)
- **Client**: Python script (`client.py`) that trains locally and communicates via gRPC

---

## 2. Deployment on NVIDIA Jetson AGX Orin (Docker)

### 2.1 Prerequisites

| Requirement | Version |
|---|---|
| JetPack | 5.1.1 (L4T R35.2.1) |
| Docker | 20.10+ |
| NVIDIA Container Runtime | Pre-installed with JetPack |
| Network | Client must reach server IP on gRPC port |

### 2.2 Build the Docker Image

The Dockerfile `COPY`s `framework/` and `fl-runtime/` from the repo root, so the
**build context must be the repo root** — clone the whole repo to the Jetson, not
just `client-docker/`.

```bash
cd ~/codebase/FedLearn-Platform

# Build (use --no-cache on first build or after dependency changes).
# Always tag BOTH :latest (default resolved by the Electron orchestrator)
# AND a version tag (consumed via FEDLEARN_CLIENT_IMAGE overrides).
sudo docker build --no-cache \
  -f client-docker/Dockerfile \
  -t fedlearn-client:latest \
  -t fedlearn-client:0.1.0 \
  .
```

`client-docker/test_docker_build.sh` runs this exact build from the project root and
then validates the image (framework import, PyTorch, device backends).

**Expected build time**: ~10-15 minutes (first build downloads ~2GB of dependencies).

> **Why two tags?** `fedlearn-desktop` (`src/main/docker.service.ts`) resolves
> `fedlearn-client:latest` by default. Pinning a specific release
> (`fedlearn-client:0.1.0`) lets you override per-environment with
> `FEDLEARN_CLIENT_IMAGE=fedlearn-client:0.1.0 npm run dev`. Bump the version
> tag every release; `:latest` moves with it.

### 2.3 Run the Client Container

```bash
sudo docker run --rm -it \
  -e PROJECT_ID="<your-project-uuid>" \
  -e SERVER_ADDRESS="<server-ip>:<grpc-port>" \
  -e PARTITION_ID=0 \
  fedlearn-client
```

**Example**:
```bash
sudo docker run --rm -it \
  -e PROJECT_ID="20334813-017d-49b3-ae58-27982069e782" \
  -e SERVER_ADDRESS="192.168.0.7:50181" \
  -e PARTITION_ID=0 \
  fedlearn-client
```

### 2.4 Expected Startup Output

```
[entrypoint] Container started successfully.
[entrypoint] PROJECT_ID=20334813-017d-49b3-ae58-27982069e782
[entrypoint] SERVER_ADDRESS=192.168.0.7:50181
[entrypoint] PARTITION_ID=0
[entrypoint] Launching python3 client.py ...
============================================================
FedLearn Client — Starting up...
Python: 3.8.10
============================================================
[BOOT] Importing argparse...
[BOOT] Importing torch... (this can take 1-3 min on Jetson)
[BOOT] ✓ torch 2.0.0a0+ec3941ad.nv23.02 loaded in 1.8s | CUDA: False
[BOOT] Importing numpy...
[BOOT] ✓ numpy 1.24.4
[BOOT] Importing psutil...
[BOOT] Importing fedlearn...
[BOOT] ✓ fedlearn loaded
...
[BOOT] ✓ All imports complete in 4.3s
```

### 2.5 Important: Do NOT Use `--runtime nvidia`

On Jetson devices, **do not** pass `--runtime nvidia` to `docker run`. Unlike discrete GPU workstations, the Jetson's GPU is integrated into the SoC. The `--runtime nvidia` flag triggers the NVIDIA Container Toolkit's GPU isolation logic, which can silently hang on Jetson because it expects a discrete GPU driver model.

```bash
# ✗ WRONG — will silently hang
sudo docker run --runtime nvidia ...

# ✓ CORRECT — GPU is accessible by default on Jetson
sudo docker run --rm -it ...
```

> **Note**: `CUDA: False` is expected without `--runtime nvidia`. To enable CUDA inside containers on Jetson, use device mounting (see Section 5.3).

### 2.6 Network Considerations

If the Jetson is on a **university/lab network with a VPN or HTTP proxy**, the gRPC connection will fail with:
```
ERROR: Could not register with server: Failed parsing HTTP/2
```

The `entrypoint.sh` script already handles this by unsetting all proxy variables:
```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export no_proxy="*"
```

If you still have issues, pass `--network host` to bypass Docker's network bridge:
```bash
sudo docker run --rm -it --network host \
  -e PROJECT_ID="..." \
  -e SERVER_ADDRESS="..." \
  -e PARTITION_ID=0 \
  fedlearn-client
```

---

## 3. Deployment on Local Mac / Linux (Native)

### 3.1 Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ (`framework/setup.py` floor; repo pins 3.12.9) |
| pip | 24.0+ |
| torch | Pre-installed or auto-installed |

### 3.2 Setup

```bash
cd ~/codebase/personalProjects/FedLearn-Platform/client-docker

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the FL framework (the canonical implementation the client wraps)
pip install -e ../framework

# Install client dependencies (torchvision and lz4 are already listed here)
pip install -r requirements.txt
```

### 3.3 Regenerate Protobuf Files

The checked-in protobuf files may have been compiled with a different `protoc` version. Regenerate them to match your installed `protobuf` runtime:

```bash
cd ~/codebase/personalProjects/FedLearn-Platform

# Regenerate
python3 -m grpc_tools.protoc \
    -I framework/src/fedlearn/communication/protos \
    --python_out=framework/src/fedlearn/communication/generated \
    --grpc_python_out=framework/src/fedlearn/communication/generated \
    framework/src/fedlearn/communication/protos/fedlearn.proto

# Fix the import path in the generated gRPC file
sed -i '' 's/^import fedlearn_pb2/from fedlearn.communication.generated import fedlearn_pb2/' \
    framework/src/fedlearn/communication/generated/fedlearn_pb2_grpc.py
```

### 3.4 Run the Client

```bash
cd ~/codebase/personalProjects/FedLearn-Platform/client-docker
source venv/bin/activate
export PYTHONPATH="../framework/src"

python3 ../fl-runtime/client.py \
  --project-id "<your-project-uuid>" \
  --server-address "127.0.0.1:<grpc-port>" \
  --partition-id 1
```

### 3.5 Start the Backend Server

The backend must be running before clients can connect:

```bash
cd ~/codebase/personalProjects/FedLearn-Platform/backend/fl-platform-api
./gradlew clean bootRun
```

Then start a project from the React dashboard at `http://localhost:5173`. This spawns the Python gRPC server on a dynamic port. Find the port:

```bash
lsof -nP -iTCP -sTCP:LISTEN | grep python
```

---

### 3.6 Apple Silicon GPU Acceleration (MPS)

On Apple Silicon Macs (M1/M2/M3/M4), the client automatically detects and uses the **Metal Performance Shaders (MPS)** backend for GPU-accelerated training. This happens transparently — no extra flags needed.

> **Important**: MPS acceleration only works when running the client **natively** (outside Docker). Docker Desktop for Mac does not support Metal GPU pass-through.

#### How It Works

The client's device selection logic:
```python
if torch.cuda.is_available():
    DEVICE = "cuda"          # NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = "mps"           # Apple Silicon GPU
else:
    DEVICE = "cpu"           # Fallback
```

#### Verify MPS Is Available

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expected: MPS available: True
```

#### Run with MPS

```bash
cd ~/codebase/personalProjects/FedLearn-Platform/client-docker
source venv/bin/activate
export PYTHONPATH="../framework/src"

python3 ../fl-runtime/client.py \
  --project-id "<your-project-uuid>" \
  --server-address "127.0.0.1:<grpc-port>" \
  --partition-id 1
```

Or use the smart launcher (recommended):

```bash
./run-client.sh "<project-uuid>" "127.0.0.1:<grpc-port>" 1 --use-llm --dataset sst2
```

The launcher auto-detects macOS and runs natively instead of Docker.

#### Expected Output

```
Client operating on mps
[Usage] after model init CPU RAM 1024.32 MB GPU alloc 502.1 MB GPU util None
```

> **Note**: MPS does not expose GPU utilization percentage or reserved memory — only allocated memory is tracked. This is a PyTorch/Metal limitation.

#### MPS Known Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| No GPU utilization % | Telemetry shows `None` | Monitor via Activity Monitor → GPU History |
| No reserved memory tracking | Only allocated memory reported | Sufficient for debugging OOM issues |
| Some ops fall back to CPU | Minor perf hit on unsupported ops | PyTorch handles this transparently |
| Docker cannot use MPS | Must run natively | Use `run-client.sh` (auto-detects) |

---

## 4. Challenges & Fixes — Complete Log

### 4.1 Docker Container Silent Hang

| | |
|---|---|
| **Symptom** | `docker run` produces zero output and hangs indefinitely |
| **Root Cause** | The original `Dockerfile` generated `entrypoint.sh` inline using `echo '...\n...'` with **single quotes**. In bash, single-quoted strings don't interpret `\n` as newlines — they write literal `\n` characters, creating a garbled script that bash couldn't execute. |
| **Fix** | Created a standalone `entrypoint.sh` file and `COPY` it into the Docker image instead of generating it inline. |

### 4.2 `--runtime nvidia` Causes Silent Hang on Jetson

| | |
|---|---|
| **Symptom** | Container hangs before any output when using `--runtime nvidia` |
| **Root Cause** | The NVIDIA Container Toolkit on Jetson checks for discrete GPU metadata during container creation. Jetson uses an integrated SoC GPU (Tegra) which doesn't match the expected driver model, causing the runtime to hang. |
| **Fix** | Remove `--runtime nvidia` from the `docker run` command. The Jetson's GPU is accessible by default without it. |

### 4.3 Python 3.8 Type Hint Incompatibility

| | |
|---|---|
| **Symptom** | `TypeError: 'type' object is not subscriptable` on `OrderedDict[str, torch.Tensor]` |
| **Root Cause** | The Jetson L4T base image ships Python 3.8. In Python 3.8, built-in generic types like `OrderedDict[K, V]`, `list[T]`, `tuple[T]` cannot be used as runtime type hints — this syntax was only added in Python 3.9 (PEP 585). |
| **Fix** | Added `from __future__ import annotations` to the top of **every** `.py` file in the `fedlearn/` package (21 files total). This makes all type annotations lazy strings evaluated at definition time, not runtime. |

```python
# Must be the FIRST line in every .py file (after shebang)
from __future__ import annotations
```

### 4.4 UTF-8 BOM (Byte Order Mark) Characters

| | |
|---|---|
| **Symptom** | `SyntaxError: invalid character in identifier` pointing at a `#` comment |
| **Root Cause** | 6 files in `fedlearn/server/` contained invisible UTF-8 BOM characters (`EF BB BF`, 3 bytes) at the start. These were likely inserted by Windows text editors. When `from __future__ import annotations` was prepended, the BOM moved to line 3, causing Python 3.8 to reject it as an invalid character mid-file. |
| **Fix** | Stripped BOM characters from all affected files: |

```bash
LC_ALL=C sed -i '' $'s/\xef\xbb\xbf//g' file.py
```

### 4.5 Protobuf Gencode/Runtime Version Mismatch

| | |
|---|---|
| **Symptom** | `VersionError: gencode 6.31.1 runtime 5.29.6` |
| **Root Cause** | The `.pb2.py` files checked into version control were compiled on a Mac with `protobuf 6.31.1`. The Jetson Docker container (Python 3.8) can only install `protobuf 5.x` (no aarch64 wheels exist for protobuf 6.x on Python 3.8). The runtime refuses to load gencode from a newer major version. |
| **Fix** | Added a `Dockerfile` build step that **regenerates** the protobuf files from the `.proto` source during `docker build`, ensuring they always match the installed runtime: |

```dockerfile
RUN python3 -m grpc_tools.protoc \
    -I /app/fedlearn/communication/protos \
    --python_out=/app/fedlearn/communication/generated \
    --grpc_python_out=/app/fedlearn/communication/generated \
    /app/fedlearn/communication/protos/fedlearn.proto \
    && sed -i 's/^import fedlearn_pb2/from fedlearn.communication.generated import fedlearn_pb2/' \
       /app/fedlearn/communication/generated/fedlearn_pb2_grpc.py
```

### 4.6 gRPC Generated Import Path Wrong

| | |
|---|---|
| **Symptom** | `ModuleNotFoundError: No module named 'fedlearn_pb2'` |
| **Root Cause** | `grpc_tools.protoc` generates bare imports (`import fedlearn_pb2`) instead of package-qualified imports (`from fedlearn.communication.generated import fedlearn_pb2`). The bare import only works if the generated directory is on `sys.path`, which it isn't in a package layout. |
| **Fix** | A `sed` command in the Dockerfile (shown above) rewrites the import to the fully qualified path after generation. |

### 4.7 `Failed parsing HTTP/2` — Lab Proxy Interception

| | |
|---|---|
| **Symptom** | `ERROR: Could not register with server: Failed parsing HTTP/2` |
| **Root Cause** | The Jetson lab machine had system-wide `http_proxy` / `https_proxy` environment variables set (inherited from the university/lab network configuration). When Python's `grpcio` detects these, it routes all TCP traffic through the HTTP proxy. The proxy expects `HTTP/1.1` but gRPC sends pure `HTTP/2` frames, causing the proxy to return an HTML error page that the gRPC client can't parse. |
| **Fix** | Added proxy cleanup to `entrypoint.sh`: |

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export no_proxy="*"
export GRPC_ENABLE_FORK_SUPPORT=0
```

### 4.8 Python 3.12 Can't Build Old Packages from Source

| | |
|---|---|
| **Symptom** | `error: can't find Rust compiler` when installing `tokenizers<0.15.0` and `safetensors<0.4.0` on Mac (Python 3.12) |
| **Root Cause** | The `requirements.txt` had strict upper bounds (`tokenizers<0.15.0`, `safetensors<0.4.0`) to ensure compatibility with the Jetson's Python 3.8. However, these old versions don't have pre-built binary wheels for Python 3.12 — pip tries to compile them from source, which requires a Rust compiler. |
| **Fix** | Removed all upper bounds from `requirements.txt`. Pip's dependency resolver automatically picks the newest version that has a compatible binary wheel for the target platform: |

```
# Before (breaks on Python 3.12)
tokenizers>=0.13.0,<0.15.0
safetensors>=0.3.1,<0.4.0

# After (works everywhere)
tokenizers>=0.13.0
safetensors>=0.3.1
```

> **Why this works for both platforms**: On the Jetson (Python 3.8), pip will still resolve to the highest version that supports Python 3.8 (e.g., `tokenizers 0.14.1`). On Mac (Python 3.12), pip installs the latest version with pre-built wheels (e.g., `tokenizers 0.19.x`). No Rust compiler needed on either platform.

### 4.9 `numpy<1.25.0` Fails on Python 3.12

| | |
|---|---|
| **Symptom** | `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` |
| **Root Cause** | Python 3.12 removed the deprecated `pkgutil.ImpImporter` API. Numpy versions below 1.26.0 rely on this API during their build process, causing a crash when pip attempts to build from source. |
| **Fix** | Changed `numpy>=1.21.0,<1.25.0` to `numpy>=1.21.0,<2.0.0`. This allows numpy 1.26.x (which supports Python 3.12) while preventing numpy 2.x (which breaks PyTorch). |

---

## 5. Building Cross-Platform Docker Images

### 5.1 Key Principles

1. **Use minimum version bounds (`>=`), avoid strict upper bounds (`<X.Y.Z`)** unless there's a known breaking change. Pip's resolver will find the newest compatible version for each platform.

2. **Keep the protobuf gencode and runtime on the same major version.** The framework ships pre-generated stubs; if a base image pins a protobuf runtime older than the gencode, regenerate from the `.proto` source (§3.3) rather than pinning the runtime forward.

3. **Add `from __future__ import annotations`** to every Python file. This ensures compatibility from Python 3.7 through 3.13+ with zero runtime cost.

4. **Strip BOM characters** from all source files before packaging. Windows editors silently insert these, and they cause `SyntaxError` on Linux/Mac.

5. **Disable proxies in the entrypoint**, not in the Dockerfile. Environment variables from the host system can leak into containers and break gRPC.

### 5.2 Dockerfile Anatomy

This mirrors the committed `client-docker/Dockerfile`. Note that every `COPY` path is
relative to the **repo root**, which is why the build context is the repo root and the
Dockerfile is selected with `-f client-docker/Dockerfile`.

```dockerfile
# 1. Base image: selected per target via a build arg
#    - Jetson:         nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
#    - Newer CUDA x86: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# 2. System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-dev python3-pip build-essential git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy and pip-install the framework (single source of truth)
COPY framework/ /app/framework/
RUN pip3 install --no-cache-dir /app/framework/

# 4. Client deps (copied separately for layer caching)
COPY client-docker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. Copy the canonical client runtime + entrypoint
COPY fl-runtime/ /app/fl-runtime/
COPY client-docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 6. Drop to an unprivileged user (GPU access does not require root)
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid ${APP_GID} fedlearn \
    && useradd --system --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash fedlearn \
    && chown -R fedlearn:fedlearn /app /home/fedlearn

ENV PYTHONPATH=/app:${PYTHONPATH}
WORKDIR /app/fl-runtime
USER fedlearn
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []
```

> **Note**: the image no longer regenerates protobuf at build time — the framework
> ships pre-generated stubs under
> `framework/src/fedlearn/communication/generated/`. Keep the regeneration recipe in
> §3.3 and §4.5 in mind if you ever hit a gencode/runtime mismatch on a new base
> image; the `.proto` source now declares `package fedlearn.v2`.

### 5.3 Enabling GPU on Jetson (device mounts)

To access the Jetson's GPU inside Docker without `--runtime nvidia`, mount the Tegra
device nodes directly. This is the same set the desktop orchestrator applies for the
Jetson hardware profile (`fedlearn-desktop/src/main/docker.service.ts`):

```bash
sudo docker run --rm -it \
  --device /dev/nvhost-ctrl \
  --device /dev/nvhost-ctrl-gpu \
  --device /dev/nvhost-dbg-gpu \
  --device /dev/nvhost-prof-gpu \
  --device /dev/nvmap \
  --device /dev/nvhost-gpu \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -e PROJECT_ID="..." \
  -e SERVER_ADDRESS="..." \
  -e PARTITION_ID=0 \
  fedlearn-client
```

---

## 6. Troubleshooting Playbook

| Symptom | Likely Cause | Fix |
|---|---|---|
| Container hangs, zero output | Broken entrypoint.sh (literal `\n`) or `--runtime nvidia` | Remove `--runtime nvidia`; use standalone entrypoint.sh |
| `TypeError: 'type' object is not subscriptable` | Python 3.8 + modern type hints | Add `from __future__ import annotations` |
| `SyntaxError: invalid character` | UTF-8 BOM in source files | Strip with `sed -i $'s/\xef\xbb\xbf//g'` |
| `VersionError: gencode X runtime Y` | Protobuf version mismatch | Regenerate proto at build time (see Dockerfile) |
| `ModuleNotFoundError: fedlearn_pb2` | Bare import in generated gRPC | Fix with `sed` after protoc (see Dockerfile) |
| `Failed parsing HTTP/2` | Lab/VPN proxy intercepting gRPC | Unset `http_proxy` in entrypoint |
| `can't find Rust compiler` | Old tokenizers/safetensors + Python 3.12 | Remove upper bounds from requirements.txt |
| `pkgutil.ImpImporter` error | numpy < 1.26 + Python 3.12 | Use `numpy>=1.21.0,<2.0.0` |
| `Socket closed` on connect | Server not running or wrong port | Check `lsof -i :<port>` on server machine |
| `Server is still in round X` | Waiting for more clients | Start additional client with different `PARTITION_ID` |

---

## Quick Reference

### Build & Run (Jetson)
```bash
# from the repo root — the Dockerfile COPYs framework/ and fl-runtime/
sudo docker build --no-cache -f client-docker/Dockerfile -t fedlearn-client .
sudo docker run --rm -it \
  -e PROJECT_ID="<uuid>" \
  -e SERVER_ADDRESS="<ip>:<port>" \
  -e PARTITION_ID=0 \
  fedlearn-client
```

### Run (Mac/Linux Native)
```bash
source venv/bin/activate
export PYTHONPATH="../framework/src"
python3 ../fl-runtime/client.py \
  --project-id "<uuid>" \
  --server-address "127.0.0.1:<port>" \
  --partition-id 1
```

### Start Backend
```bash
cd backend/fl-platform-api
./gradlew clean bootRun
# Then start a project from http://localhost:5173
```
