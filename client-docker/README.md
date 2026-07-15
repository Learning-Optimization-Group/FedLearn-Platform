# FedLearn Client — Docker Package

Containerized FL client. Bundles the `framework/` library + `scripts/client.py` so distributed clients can join training without a Python toolchain on the host. End users typically don't run this directly — the **Electron desktop orchestrator** (`fedlearn-desktop/`) launches it via `dockerode`. Devs and Jetson operators run it manually.

For the full operational procedure (Jetson, native Mac/Linux, troubleshooting, multi-arch builds), see **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)**.

## Layout

```
client-docker/
├── Dockerfile                 # Multi-arch via BASE_IMAGE build arg
├── entrypoint.sh              # Single CLI surface; parses flags + env
├── run-client.sh              # Helper wrapper for ad-hoc local runs
├── requirements.txt           # Pinned client deps (framework already installs its own)
├── scripts/
│   ├── client.py              # FL client entry point (thin — logic lives in framework/)
│   ├── config.py
│   ├── data_loaders/
│   ├── ecg_data/
│   └── models/
├── packaging/                 # PyInstaller bundling for the desktop installer (Mac / Win)
└── DEPLOYMENT_GUIDE.md        # Canonical Jetson + native deployment procedure
```

## Hardware variants (Dockerfile `BASE_IMAGE` build arg)

| Target | Base image |
|---|---|
| **Default** (x86 + CUDA 11.7) | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| **Jetson** (ARM64, L4T) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| **Newer x86 hosts** (CUDA 12.4) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| **CPU-only** | any standard CPU PyTorch image |

`framework/` is `pip install`-ed inside the image so the client stays a thin wrapper around the canonical implementation. Don't duplicate framework logic into `scripts/`.

## Build

```bash
cd client-docker

# Default (x86 CUDA 11.7)
docker build -t fedlearn-client:latest -t fedlearn-client:0.1.0 .

# Jetson AGX Orin
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:jetson .
```

Always tag both `:latest` and a version (`:0.1.0`) — the Electron orchestrator (`fedlearn-desktop/src/main/docker.service.ts`) resolves `:latest` by default; pinned tags are consumed via `FEDLEARN_CLIENT_IMAGE` overrides for per-environment testing.

## Run

```bash
docker run --rm -it \
  -v /path/to/data:/data \
  -e FEDLEARN_CONNECTION_TOKEN=<token> \
  fedlearn-client:latest \
  --server-address <server-host>:<grpc-port> \
  --client-id 0
```

**`FEDLEARN_CONNECTION_TOKEN` (SE-14).** When the FL server enforces client auth
(`app.fl.require-client-auth=true`, the default in deployed profiles once rolled
out), every client must present a backend-minted connection token on its gRPC
calls or the server rejects it `UNAUTHENTICATED`. This container reads that token
from the `FEDLEARN_CONNECTION_TOKEN` env var (the framework client attaches it as
`x-connection-token` on every call). Obtain the token from the backend, over your
authenticated web session, at `GET /api/client/projects/{projectId}/connection`
(field `connectionToken`) — the same DTO that carries the gRPC endpoint. The token
is sized to the run's length and expires after it, so fetch a fresh one per run.
The **desktop launcher sets this automatically** when it starts the container; a
**standalone `docker run` must pass it explicitly** (omit it only against a dev
server with auth off).

For Jetson with CUDA, add `--runtime nvidia` (NVIDIA Container Runtime is pre-installed with JetPack).

For multi-platform image distribution (`buildx`, registry pushes, offline export), see `DEPLOYMENT_GUIDE.md`.

## How it connects to the rest of the system

- The backend (`fl-platform-api`) spawns a Python FL server on a dynamic port in `50000-50010` when a project is started.
- The port is logged to the dashboard's live telemetry view; FL clients use `<server-host>:<port>` to connect.
- gRPC is currently `insecure_channel` — gradients fly plaintext over the WAN. See repo audit item #37 before adding behaviour that assumes encryption.
- For demos across networks (e.g. classroom Jetson + home Mac), use Tailscale to skip NAT/firewall headaches. See [`docs/guides/pneumonia_demo_plan.md`](../docs/guides/pneumonia_demo_plan.md).

## Adjacent docs

- **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)** — full Jetson + native deployment procedure with troubleshooting playbook
- **[`packaging/README.md`](packaging/README.md)** — PyInstaller bundling for the desktop app
- **[`CLAUDE.md`](CLAUDE.md)** — AI assistant guidance (gitignored, on-disk only)
- **`../framework/README.md`** — the FL framework that powers the client
- **`../fedlearn-desktop/README.md`** — the Electron orchestrator that launches this image for end users
