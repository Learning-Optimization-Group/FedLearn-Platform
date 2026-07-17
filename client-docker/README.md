# FedLearn Client — Docker Package

Containerized FL client. Bundles the `framework/` library + the canonical `fl-runtime/client.py` so distributed clients can join training without a Python toolchain on the host. End users typically don't run this directly — the **Electron desktop orchestrator** (`fedlearn-desktop/`) launches it via `dockerode`. Devs and Jetson operators run it manually.

For the full operational procedure (Jetson, native Mac/Linux, troubleshooting, multi-arch builds), see **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)**.

## Layout

```
client-docker/
├── Dockerfile                 # Multi-arch via BASE_IMAGE build arg; build context = REPO ROOT
├── entrypoint.sh              # Env-driven launcher (PROJECT_ID / SERVER_ADDRESS / PARTITION_ID)
├── run-client.sh              # Smart launcher: detects platform, picks Docker vs native
├── requirements.txt           # Client deps (the framework installs its own via pip)
├── test_docker_build.sh       # Build + import + device-backend validation suite
├── packaging/                 # PyInstaller bundling for the desktop installer (Mac / Win / Linux)
└── DEPLOYMENT_GUIDE.md        # Canonical Jetson + native deployment procedure
```

There is **no `scripts/` directory here** — the FL client has one canonical source,
`fl-runtime/client.py` at the repo root (DA-5). The Dockerfile copies `framework/`
and `fl-runtime/` from the repo root, which is why the **build context is the repo
root, not this directory**.

## Hardware variants (Dockerfile `BASE_IMAGE` build arg)

| Target | Base image |
|---|---|
| **Default** (x86 + CUDA 11.7) | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| **Jetson** (ARM64, L4T) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| **Newer x86 hosts** (CUDA 12.4) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| **CPU-only** | any standard CPU PyTorch image |

`framework/` is `pip install`-ed inside the image so the client stays a thin wrapper around the canonical implementation. Don't duplicate framework logic into `fl-runtime/`.

## Build

Run these from the **repo root** — the Dockerfile `COPY`s `framework/` and
`fl-runtime/`, so a `docker build .` from inside `client-docker/` cannot see them.

```bash
cd /path/to/FedLearn-Platform

# Default (x86 CUDA 11.7)
docker build -f client-docker/Dockerfile \
  -t fedlearn-client:latest -t fedlearn-client:0.1.0 .

# Jetson AGX Orin
docker build -f client-docker/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:jetson .
```

Always tag both `:latest` and a version (`:0.1.0`) — the Electron orchestrator (`fedlearn-desktop/src/main/docker.service.ts`) resolves `:latest` by default; pinned tags are consumed via `FEDLEARN_CLIENT_IMAGE` overrides for per-environment testing.

## Run

`entrypoint.sh` is **env-driven**: it requires `PROJECT_ID`, `SERVER_ADDRESS` and
`PARTITION_ID`, and exits with a usage message if any is missing. It builds the
`--project-id` / `--server-address` / `--partition-id` flags itself and forwards any
extra `docker run` arguments straight to `client.py`.

```bash
docker run --rm -it \
  -e PROJECT_ID=<project-uuid> \
  -e SERVER_ADDRESS=<server-host>:<grpc-port> \
  -e PARTITION_ID=0 \
  -e FEDLEARN_CONNECTION_TOKEN=<token> \
  fedlearn-client:latest
```

Extra client flags (`--model-type`, `--dataset`, `--strategy`, …) go after the image
name and are appended to the generated command line.

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

**On Jetson, do NOT pass `--runtime nvidia`** — it searches for PCIe discrete-GPU
metadata in the device tree and hangs indefinitely on the Tegra SoC. Expose the GPU
with direct `/dev/nvhost-*` device mounts instead (the same approach the desktop
app's `DockerService` takes for the Jetson profile); see `DEPLOYMENT_GUIDE.md` §5.3.

For multi-platform image distribution (`buildx`, registry pushes, offline export), see `DEPLOYMENT_GUIDE.md`.

## How it connects to the rest of the system

- The backend (`fl-platform-api`) spawns a Python FL server on a dynamic port in `50000-50010` when a project is started.
- The port is logged to the dashboard's live telemetry view; FL clients use `<server-host>:<port>` to connect.
- gRPC is **plaintext by default** — with `FEDLEARN_GRPC_USE_TLS` unset, the client uses `insecure_channel` and gradients cross the network unencrypted. That default is fine for a local dev server; don't assume encryption is on.
- TLS **is** implemented (SE-2) and is **required on deployed servers**: the `ec2demo` and `production` profiles set `app.fl.require-tls=true`, so the FL server serves TLS and fails closed rather than binding plaintext. Against such a server a plaintext client is rejected — pass `-e FEDLEARN_GRPC_USE_TLS=1` and mount the server's public cert as `FEDLEARN_GRPC_ROOT_CERT`, and dial the DNS name in the cert SAN (not a raw IP). `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` adds mTLS. See `../deploy/TLS.md`.
- For demos across networks (e.g. classroom Jetson + home Mac), use Tailscale to skip NAT/firewall headaches.

## Adjacent docs

- **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)** — full Jetson + native deployment procedure with troubleshooting playbook
- **[`packaging/README.md`](packaging/README.md)** — PyInstaller bundling for the desktop app
- **`../framework/README.md`** — the FL framework that powers the client
- **`../fedlearn-desktop/README.md`** — the Electron orchestrator that launches this image for end users
