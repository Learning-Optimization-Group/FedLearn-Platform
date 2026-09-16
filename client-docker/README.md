# FedLearn Client — Docker Package

Containerized FL client. The image bundles the `framework/` library **and** the canonical
`fl-runtime/client.py`, so a distributed client can join training without a Python toolchain on the
host. End users normally don't run this directly — the **Electron desktop orchestrator**
(`fedlearn-desktop/`) launches it via `dockerode`. Devs and Jetson operators run it by hand.

For the full operational procedure (Jetson, native Mac/Linux, troubleshooting, multi-arch builds),
see **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)**.

---

## The two things that trip everyone up

1. **The build context is the repo root**, not this directory. The `Dockerfile` `COPY`s
   `framework/` and `fl-runtime/`, which live above it.
2. **The container is configured by environment variables, not CLI flags.** `entrypoint.sh` builds
   `--project-id` / `--server-address` / `--partition-id` itself from the environment.
   **There is no `--client-id` flag anywhere** — `fl-runtime/client.py` has never had one.

```bash
# ✅ correct — run from the repo root
docker build -f client-docker/Dockerfile -t fedlearn-client:latest .
docker run --rm -it -e PROJECT_ID=<uuid> -e SERVER_ADDRESS=<host>:<port> -e PARTITION_ID=0 \
  fedlearn-client:latest

# ❌ wrong — exits 1 in entrypoint.sh before client.py ever starts
docker run --rm -it fedlearn-client:latest --project-id <uuid> --client-id 0
```

## Layout

```
client-docker/
├── Dockerfile                 # Multi-arch via BASE_IMAGE build arg; build context = REPO ROOT
├── entrypoint.sh              # Env-driven launcher (PROJECT_ID / SERVER_ADDRESS / PARTITION_ID)
├── run-client.sh              # Smart launcher: detects platform, picks Docker vs native
├── requirements.txt           # Client-side deps (framework/ installs its own via pip)
├── test_docker_build.sh       # Build + import + device-backend validation suite
├── packaging/                 # PyInstaller bundling for the desktop installer (Mac / Win / Linux)
└── DEPLOYMENT_GUIDE.md        # Canonical Jetson + native deployment procedure
```

There is **no committed `scripts/` directory here.** The FL client has exactly one canonical
source, `fl-runtime/client.py` at the repo root (DA-5) — the Docker image, the desktop PyInstaller
bundle and local dev runs all consume that same file. Don't reintroduce a fork.

> **`flwr` is no longer a dependency of this project.** A few comments under `requirements.txt` and
> `packaging/` still reason about `flwr` / `flwr-datasets` version conflicts (the `pyarrow` pin, the
> PyInstaller `hiddenimports`). That is leftover text, not a live constraint — nothing in the image
> installs Flower any more, and CIFAR-10 partitioning is reproduced natively in `fl-runtime/`.

## Hardware variants (`BASE_IMAGE` build arg)

The `Dockerfile` defaults to a standard x86 PyTorch/CUDA image and takes the target as a build arg:

| Target | `BASE_IMAGE` |
|---|---|
| **Default** (x86, CUDA 11.7) | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| **Jetson** (ARM64, L4T) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| **Newer x86 hosts** (CUDA 12.4) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| **CPU-only** | any standard CPU PyTorch image |

> The pinned Jetson tag is **JetPack 5-era (L4T r35)**. A JetPack 6 device (L4T r36.x) runs a much
> newer stack — pick a matching `l4t-pytorch`/`l4t-jetpack` tag for it rather than assuming r35
> works, and re-verify GPU access on the L4T the device actually runs.

`framework/` is `pip install`-ed inside the image, so the client stays a thin wrapper around the
canonical implementation. Don't duplicate framework logic into `fl-runtime/`.

## Build

Run from the **repo root** — a `docker build .` from inside `client-docker/` cannot see
`framework/` or `fl-runtime/` and will fail on the first `COPY`.

```bash
cd /path/to/FedLearn-Platform

# Default (x86, CUDA 11.7)
docker build -f client-docker/Dockerfile \
  -t fedlearn-client:latest -t fedlearn-client:0.1.0 .

# Jetson AGX Orin (or any ARM64 L4T host)
docker build -f client-docker/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:jetson .
```

Tag both `:latest` and a version. The Electron orchestrator
(`fedlearn-desktop/src/main/docker.service.ts`) resolves `fedlearn-client:latest` by default and
honours a `FEDLEARN_CLIENT_IMAGE` override for pinned per-environment testing.

The image runs as an **unprivileged user** (`fedlearn`, uid/gid `10001`, overridable via the
`APP_UID`/`APP_GID` build args) with `WORKDIR=/app/fl-runtime`. GPU access through `--gpus` or
device mounts does not need root; any host directory you bind-mount must be readable by that uid.

## Run

`entrypoint.sh` is **env-driven**. It requires `PROJECT_ID`, `SERVER_ADDRESS` and `PARTITION_ID`,
prints a usage line and **exits 1** if any is missing, then execs
`python3 -u client.py --project-id … --server-address … --partition-id …` and forwards `"$@"` —
everything after the image name — as extra client flags.

```bash
docker run --rm -it \
  -e PROJECT_ID=<project-uuid> \
  -e SERVER_ADDRESS=<server-host>:<grpc-port> \
  -e PARTITION_ID=0 \
  -e FEDLEARN_CONNECTION_TOKEN=<token> \
  fedlearn-client:latest
```

Extra client flags go **after the image name** and are appended to the generated command line —
for example `--use-llm`, `--dataset cb|sst2|ecg`, `--model-type <recipe-key>`, `--strategy`,
`--training-arm`, `--device auto|cpu|cuda|mps`.

Three optional env vars are translated into flags by `entrypoint.sh` (this is how the desktop's
`buildContainerEnv` configures a run — an explicit flag in `"$@"` still wins, since argparse takes
the last occurrence):

| Env var | Becomes | Why it matters |
|---|---|---|
| `MODEL_TYPE` | `--model-type` | Without it the client silently defaults to `CNN`. |
| `STRATEGY` | `--strategy` | Without it a DeComFL project runs the FedAvg client path against a DeComFL server — a silent mismatch. |
| `TRAINING_ARM` | `--training-arm` | The arm the FL server was spawned with. Without it the client trains and uploads every parameter while a `FROZEN_HEAD` server expects only the head. |

Model-type keys come from the recipe catalog (`fl-runtime/recipes.py --describe`), and each recipe
declares which arms it supports — an unsupported arm is rejected at client startup.

### `FEDLEARN_CONNECTION_TOKEN` (SE-14)

When the FL server enforces client auth, every client must present a backend-minted connection
token on its gRPC calls or the server rejects it `UNAUTHENTICATED`. This container reads the token
from `FEDLEARN_CONNECTION_TOKEN`; the framework client attaches it as the `x-connection-token`
metadata header on every call.

Fetch it from the backend over your authenticated web session at
`GET /api/client/projects/{projectId}/connection` (field `connectionToken`) — the same DTO that
carries the gRPC endpoint. The token is sized to the run's length and expires after it, so fetch a
fresh one per run.

The **desktop launcher passes this automatically**; a standalone `docker run` must set it
explicitly. Enforcement is off by default (`app.fl.require-client-auth` defaults to `false` in
every profile today), so you can omit it against a server with auth off — but pass it whenever you
don't control the server's setting.

> `run-client.sh` does **not** forward `FEDLEARN_CONNECTION_TOKEN` into the container. Against a
> server with client auth on, use a direct `docker run` (or add `-e FEDLEARN_CONNECTION_TOKEN` to
> the script's invocation).

### GPU access on Jetson

**On JetPack 6+, try `--runtime nvidia` first.** The long-standing advice here and in
`DEPLOYMENT_GUIDE.md` was the opposite — that the flag hangs indefinitely on Tegra and you must use
`/dev/nvhost-*` device mounts instead. Measured on an AGX Orin running **JetPack 6.2 / L4T R36.5**
with `nvidia-container-toolkit` 1.19.0, the reverse holds: `docker run --runtime nvidia` works and
reports `torch.cuda.is_available() == True`, while the device-mount path fails there with
`cuInit → 801 (CUDA_ERROR_NOT_SUPPORTED)` followed by a segfault, because the in-container
`libcuda.so.1` is a stub. Two follow-on details:

- `/dev/nvhost-ctrl` **does not exist on L4T R36.5** — passing it makes Docker hard-error with
  `no such file or directory`. The JetPack 5-era device-node set is not valid there, which also
  means `run-client.sh`'s Jetson branch and the desktop app's `DockerService` device list need
  adjusting before they will start a container on such a host.
- Device mounts remain the documented fallback for older L4T.

Honest scope: the original hang was plausibly real on the older JetPack 5 /
`nvidia-container-runtime` the advice was written against, and that was **not** re-tested (no
JetPack 5 hardware available). Treat `--runtime nvidia` as the first thing to try on JetPack 6+,
keep device mounts as the fallback, and **re-verify on whatever L4T the target device actually
runs.** `DEPLOYMENT_GUIDE.md` §2.5 / §4.2 / §5.3 still carry the blanket ban and are stale on this
point.

For multi-platform image distribution (`buildx`, registry pushes, offline export), see
`DEPLOYMENT_GUIDE.md`.

## How it connects to the rest of the system

- The backend (`fl-platform-api`) spawns a Python FL server on a dynamic port in `50000-50010` when
  a project is started.
- The port is logged to the dashboard's live telemetry view; FL clients dial `<server-host>:<port>`.
- gRPC is **plaintext by default** — with `FEDLEARN_GRPC_USE_TLS` unset the client uses
  `insecure_channel` and gradients cross the network unencrypted. Fine for a local dev server;
  don't assume encryption is on.
- TLS **is** implemented (SE-2) and is **required on deployed servers**: the `ec2demo` and
  `production` profiles set `app.fl.require-tls=true`, which the backend turns into
  `FEDLEARN_REQUIRE_TLS=1` on the spawned FL server so it fails closed rather than binding
  plaintext. Against such a server, pass `-e FEDLEARN_GRPC_USE_TLS=1`, mount the server's public
  cert and point `FEDLEARN_GRPC_ROOT_CERT` at it, and dial the DNS name in the cert SAN (not a raw
  IP). mTLS is turned on **server-side** (`FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` on the FL server —
  setting it on this container does nothing); the client's half is `FEDLEARN_GRPC_CLIENT_CERT` +
  `FEDLEARN_GRPC_CLIENT_KEY`. See [`../deploy/TLS.md`](../deploy/TLS.md).
- Model updates travel as a **deterministic safetensors blob**, never pickle. The wire is
  float32-only so the libtorch-free mobile C++ client can decode it; non-float32 buffers (a
  BatchNorm `num_batches_tracked`, say) are excluded from the federated set and stay local.
- For demos across networks (e.g. a lab Jetson and a laptop elsewhere), a mesh VPN such as
  Tailscale avoids NAT/firewall work.

## CI

`ci.yml` has a path-filtered `client-docker` job that runs whenever this directory changes: it
builds the image for `linux/amd64` with the default `BASE_IMAGE`, then smoke-tests the real
entrypoint —

```bash
docker run --rm -e PROJECT_ID=smoke -e SERVER_ADDRESS=localhost:50000 -e PARTITION_ID=0 \
  fedlearn-client:ci --help
```

`--help` is forwarded through `"$@"`, so argparse prints and exits 0 before any gRPC connect. That
proves the framework + client imports load and the entrypoint wiring is sound. The Jetson lane is
arm64 and needs real CUDA hardware, so it is not built in CI.

## Adjacent docs

- **[`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md)** — full Jetson + native deployment procedure with troubleshooting playbook
- **[`packaging/README.md`](packaging/README.md)** — PyInstaller bundling for the desktop app
- **[`../framework/README.md`](../framework/README.md)** — the FL framework the client wraps
- **[`../fedlearn-desktop/README.md`](../fedlearn-desktop/README.md)** — the Electron orchestrator that launches this image for end users
- **[`../wikis/client-docker/README.md`](../wikis/client-docker/README.md)** — deeper reference for this unit
