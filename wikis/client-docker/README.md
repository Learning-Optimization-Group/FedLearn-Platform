# FedLearn Client (Docker) — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)  
> **Stack:** Docker, multi-arch base images, thin wrapper around `framework/`

`client-docker/` is the **containerised FL client**. It bundles the `framework/` library plus a thin `scripts/client.py` entry point so a client can join training **without a Python toolchain on the host**. End users don't usually run it directly — the **Electron desktop orchestrator** (`fedlearn-desktop/`) launches it via `dockerode`. Developers and Jetson operators run it manually.

> **Thin wrapper, by design.** `framework/` is `pip install`-ed inside the image. The container carries **no FL logic of its own** — don't duplicate framework code into `client-docker/scripts/`.

---

## Layout

```
client-docker/
├── Dockerfile                 # Multi-arch via the BASE_IMAGE build arg
├── entrypoint.sh              # Single CLI surface; parses flags + env
├── run-client.sh              # Helper wrapper for ad-hoc local runs
├── requirements.txt           # Pinned client deps (framework installs its own)
├── scripts/
│   ├── client.py              # FL client entry point (thin — logic lives in framework/)
│   ├── config.py
│   ├── data_loaders/  ecg_data/  models/
├── packaging/                 # PyInstaller bundling for the desktop installer (Mac / Win)
└── DEPLOYMENT_GUIDE.md        # Canonical Jetson + native deployment procedure
```

---

## Hardware variants (`BASE_IMAGE` build arg)

| Target | Base image |
|---|---|
| **Default** (x86 + CUDA 11.7) | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| **Jetson** (ARM64, L4T) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| **Newer x86 hosts** (CUDA 12.4) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| **CPU-only** | any standard CPU PyTorch image |

### ⚠️ Jetson path

Jetson clients **must** use the L4T base image (`nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3`) and **must not** pass the `--runtime nvidia` Docker flag — on Jetson it hangs indefinitely searching the device tree for discrete-GPU PCIe metadata. The desktop app's `DockerService` uses direct `/dev/nvhost-*` **device mounts** for the Jetson profile instead of the NVIDIA runtime. (Note: the in-repo `client-docker/DEPLOYMENT_GUIDE.md` is the canonical operational reference; follow the device-mount approach the platform uses.)

---

## Build

```bash
cd client-docker

# Default (x86 CUDA 11.7) — tag both :latest and a version
docker build -t fedlearn-client:latest -t fedlearn-client:0.1.0 .

# Jetson AGX Orin (ARM64, L4T)
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:jetson .
```

Always tag both `:latest` and a version — the desktop orchestrator (`fedlearn-desktop/src/main/docker.service.ts`) resolves `:latest` by default; pinned tags are consumed via `FEDLEARN_CLIENT_IMAGE` overrides for per-environment testing. The image tag mirrors the **framework** version it bundles (client-docker is not independently versioned — see [`VERSIONS.md`](../VERSIONS.md)).

## Run

```bash
docker run --rm -it \
  -v /path/to/data:/data \
  -e FEDLEARN_CONNECTION_TOKEN=<token> \
  fedlearn-client:latest \
  --server-address <server-host>:<grpc-port> \
  --client-id 0
```

> **`FEDLEARN_CONNECTION_TOKEN` (SE-14).** When the FL server enforces client auth
> (`app.fl.require-client-auth=true`), every client must present a backend-minted
> connection token or the server rejects it `UNAUTHENTICATED`. Fetch it from
> `GET /api/client/projects/{projectId}/connection` (`connectionToken`) over your
> authenticated web session; the framework client attaches it as `x-connection-token`
> on every gRPC call. The desktop launcher injects it automatically — a **standalone
> `docker run` must pass it explicitly** (omit only against a dev server with auth off).

For multi-platform image distribution (`buildx`, registry pushes, offline export) see the in-repo `DEPLOYMENT_GUIDE.md`.

---

## How it connects to the rest of the system

- The **Backend** (`fl-platform-api`) spawns a Python FL server on a dynamic port in `50000-50010` when a project starts; the port is surfaced in the dashboard's live telemetry. Clients connect to `<server-host>:<port>`.
- gRPC currently uses `insecure_channel` — **gradients fly plaintext over the WAN** (repo audit item #37). Do not add behaviour that assumes encryption between the FL server and clients.
- For cross-network demos (e.g. a classroom Jetson + a home Mac), use **Tailscale** to skip NAT/firewall issues.

## Related documentation

- [Framework Wiki](../framework/README.md) — the FL engine this container wraps
- [Desktop Wiki](../desktop/README.md) — the Electron orchestrator that launches this image for end users
- [Desktop: Hardware Profiles](../desktop/07-hardware-profiles.md) — the Jetson Docker device-mount path in detail
