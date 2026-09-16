# FedLearn Client (Docker) — Wiki

> **Part of:** [FedLearn Platform Docs](../README.md)  
> **Stack:** Docker, multi-arch base images, thin wrapper around `framework/` + `fl-runtime/`

`client-docker/` is the **containerised FL client**. It bundles the `framework/` library plus the canonical `fl-runtime/client.py` entry point (shared with the desktop native bundle and dev/local runs) so a client can join training **without a Python toolchain on the host**. End users don't usually run it directly — the **Electron desktop orchestrator** (`fedlearn-desktop/`) launches it via `dockerode`. Developers and Jetson operators run it manually.

> **Thin wrapper, by design.** `framework/` is `pip install`-ed and the canonical `fl-runtime/client.py` is copied into the image. The container carries **no forked FL logic of its own** — DA-5 removed the old `client-docker/scripts/` client fork; don't reintroduce one.

---

## The two facts that are misunderstood most often

**1. The build context is the REPO ROOT, not this directory.** The `Dockerfile` `COPY`s `framework/` *and* `fl-runtime/`, both of which live above `client-docker/`. A `docker build .` from inside the directory fails on the first `COPY`.

**2. The container is configured by ENV VARS, not CLI flags.** `entrypoint.sh` **hard-fails** if `PROJECT_ID`, `SERVER_ADDRESS` or `PARTITION_ID` is unset, then builds the `--project-id` / `--server-address` / `--partition-id` flags *itself* and forwards `"$@"` for extras. Passing those three as CLI flags exits 1 before `client.py` is ever exec'd. And there is **no `--client-id`** anywhere — `fl-runtime/client.py` has never had one; the per-client index is `--partition-id`.

```bash
# correct — from the repo root, config in the environment
docker build -f client-docker/Dockerfile -t fedlearn-client:latest .
docker run --rm -it \
  -e PROJECT_ID=<uuid> -e SERVER_ADDRESS=<host>:<port> -e PARTITION_ID=0 \
  fedlearn-client:latest

# wrong — exits 1 in entrypoint.sh before client.py starts
docker run --rm -it fedlearn-client:latest --project-id <uuid> --client-id 0
```

---

## Layout

```
client-docker/
├── Dockerfile                 # Multi-arch via the BASE_IMAGE build arg; context = REPO ROOT
├── entrypoint.sh              # Env-driven launcher; the only CLI surface
├── run-client.sh              # Smart launcher: detects platform, picks Docker vs native
├── requirements.txt           # Client-side deps (framework/ installs its own via pip)
├── test_docker_build.sh       # Build + import + device-backend validation suite
├── packaging/                 # PyInstaller bundling for the desktop installer (Mac / Win / Linux)
└── DEPLOYMENT_GUIDE.md        # Canonical Jetson + native deployment procedure
```

There is **no committed `scripts/` directory here** — DA-5 deleted that client fork (`git ls-files client-docker/` confirms it; an empty untracked `scripts/models/` may exist locally as build residue). The image is assembled from two trees *outside* this directory.

### Image anatomy

The `Dockerfile` is short and every step matters:

| Step | What it does |
|---|---|
| `ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` | the only multi-arch knob; see the table below |
| `apt-get install python3-venv python3-dev python3-pip build-essential git wget curl ca-certificates` | build toolchain for source-only wheels |
| `COPY framework/ /app/framework/` → `pip3 install /app/framework/` | the FL library, installed as `fedlearn` — **single source of truth** |
| `COPY client-docker/requirements.txt .` → `pip3 install -r requirements.txt` | client-side extras only (transformers/datasets, grpc, torchvision, monitoring, CLI) |
| `COPY fl-runtime/ /app/fl-runtime/` | the executable layer: `client.py`, `recipes.py`, `config.py`, `data.py`, `device.py` … |
| `COPY client-docker/entrypoint.sh /app/entrypoint.sh` + `chmod +x` | the launcher |
| `groupadd/useradd fedlearn` (`APP_UID`/`APP_GID`, default `10001`) + `chown -R /app /home/fedlearn` | the container runs **unprivileged** — GPU access via `--gpus` or device mounts does not need root |
| `ENV PYTHONPATH=/app` · `WORKDIR /app/fl-runtime` · `USER fedlearn` | `client.py` resolves its siblings by being *in* `/app/fl-runtime` |
| `ENTRYPOINT ["/app/entrypoint.sh"]` · `CMD []` | everything after the image name lands in `"$@"` |

Because the working directory is `/app/fl-runtime` and the process is uid `10001`, **any host directory you bind-mount must be readable by that uid**, and any relative path a recipe writes (e.g. torchvision's `root="./data"`) resolves to `/app/fl-runtime/data` inside the container — not to `/data`.

> **`flwr` is no longer a dependency.** Comments in `requirements.txt` (the `pyarrow`-uncapped note at line 23) and in `packaging/` still reason about `flwr` / `flwr-datasets` conflicts. Those are leftover strings, not live constraints: nothing in the image installs Flower any more, and the CIFAR-10 IID shard is reproduced natively in `fl-runtime/recipes.py` (`_cnn_iid_shard`, `CNN_SHUFFLE_SEED = 42`). See **Packaging** below for the remaining traces and what each one now costs.

---

## Hardware variants (`BASE_IMAGE` build arg)

| Target | Base image |
|---|---|
| **Default** (x86 + CUDA 11.7) | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| **Jetson** (ARM64, L4T) | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |
| **Newer x86 hosts** (CUDA 12.4) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| **CPU-only** | any standard CPU PyTorch image |

The pinned Jetson tag is **JetPack 5-era (L4T r35)**. A JetPack 6 device runs L4T r36.x — two major generations newer — so pick a matching `l4t-pytorch` / `l4t-jetpack` tag rather than assuming r35 works, and re-verify GPU access on whatever L4T the device actually runs.

`requirements.txt` still carries its original Jetson framing ("Python 3.8, aarch64, CUDA 11.4"; `numpy<2.0`, `scikit-learn<1.4`, `fsspec<2024.3`). Those caps exist for cp38-aarch64 wheel availability on the r35 base image and are conservative-but-harmless on a modern x86 base — worth knowing before you assume a version floor is a deliberate security pin. The one genuine floor is `protobuf>=5.29.0,<6.0.0` (P0-2b: it tracks the newest committed gencode, `fot_pb2.py` at 5.29.0).

### Jetson GPU access — the old blanket ban is withdrawn

This page used to state that `--runtime nvidia` must never be passed on Jetson because it hangs searching the device tree for discrete-GPU PCIe metadata. **That was measured false on JetPack 6.** On an AGX Orin running **L4T R36.5 / JetPack 6.2** with `nvidia-container-toolkit` 1.19.0:

| Approach | Result |
|---|---|
| `docker run --runtime nvidia` | **works** — `torch.cuda.is_available()` is `True`, device reports as `Orin`. No hang. |
| `/dev/nvhost-*` device mounts, no `--runtime nvidia` | **fails** — `cuInit → 801 (CUDA_ERROR_NOT_SUPPORTED)` then a segfault; the in-container `libcuda.so.1` is a stub |

Recorded in `research/results/decomfl/device_ab_cpu_vs_gpu.json` (`orin_cuda_added_2026_07_28`) and used for the Orin CUDA rows of `research/results/benchmark/pytorch_crossdevice_matrix.json`. Two further JetPack 6 details:

- The device-node list itself is JetPack-5-era. Both `run-client.sh` (its `jetson` branch, line 133) and the desktop's `DockerService` (`JETSON_DEVICE_MOUNTS`) pass `/dev/nvhost-ctrl`, which is **reported absent on L4T R36.5** — Docker hard-errors with `no such file or directory` when a device node does not exist. That specific node's absence was not re-verified while writing this page; the measured fact is that the whole device-mount path failed on the JetPack 6.2 box above.
- The device-mount path stays the documented fallback and is still what the desktop app uses for its Jetson profile.

**Honest scope:** the original hang was plausibly real on the older JetPack 5 / `nvidia-container-runtime` the note was written against, but that was *not* re-tested — no JetPack 5 hardware was available. Treat `--runtime nvidia` as the default to *try* on JetPack 6+, keep device mounts as the fallback, and re-verify per device. `DEPLOYMENT_GUIDE.md` §2.5 / §4.2 / §5.3 still carry the older blanket ban and are stale on this point.

---

## Build

```bash
cd /path/to/FedLearn-Platform          # the build context is the repo root

# Default (x86 CUDA 11.7) — tag both :latest and a version
docker build -f client-docker/Dockerfile -t fedlearn-client:latest -t fedlearn-client:0.1.0 .

# Jetson AGX Orin (ARM64, L4T)
docker build -f client-docker/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 \
  -t fedlearn-client:jetson .
```

Always tag both `:latest` and a version — the desktop orchestrator (`fedlearn-desktop/src/main/docker.service.ts`) resolves `fedlearn-client:latest` by default; pinned tags are consumed via the `FEDLEARN_CLIENT_IMAGE` override for per-environment testing. The image tag mirrors the **framework** version it bundles (client-docker is not independently versioned — see [`VERSIONS.md`](../VERSIONS.md)).

`test_docker_build.sh` builds from the repo root and then validates the image: the framework imports, PyTorch loads, and the available device backends are reported. Note its step 3 asserts `/app/scripts` exists — a path the current `Dockerfile` never creates (it copies `fl-runtime/` to `/app/fl-runtime/`), so that assertion fails on a correctly built image and the script is stale on that point.

---

## Run

`entrypoint.sh` does four things, in order:

1. **Echoes** `PROJECT_ID` / `SERVER_ADDRESS` / `PARTITION_ID` so a container log always begins with its identity.
2. **Hard-fails** (`exit 1` with a usage line) if any of the three is empty.
3. **Clears proxy state** — `unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY`, `export no_proxy="*"`, `export GRPC_ENABLE_FORK_SUPPORT=0`. A VPN or lab proxy that intercepts the gRPC connection otherwise surfaces as `Failed parsing HTTP/2`, which looks like a server bug and is not.
4. **Execs** `python3 -u client.py --project-id … --server-address … --partition-id … "${EXTRA_ARGS[@]}" "$@"`.

```bash
docker run --rm -it \
  -e PROJECT_ID=<project-uuid> \
  -e SERVER_ADDRESS=<server-host>:<grpc-port> \
  -e PARTITION_ID=0 \
  -e FEDLEARN_CONNECTION_TOKEN=<token> \
  fedlearn-client:latest \
  --model-type CNN --training-arm FROZEN_HEAD
```

### Env vars the entrypoint translates into flags

`EXTRA_ARGS` is a bash **array**, so empty or space-containing values stay safe, and it is placed *before* `"$@"` — an explicit flag you pass after the image name still wins, because argparse takes the last occurrence.

| Env var | Becomes | Why it exists |
|---|---|---|
| `MODEL_TYPE` | `--model-type` | It was previously dropped here, and the container client then silently defaulted to `CNN`. |
| `STRATEGY` | `--strategy` | Without it a non-MLP DeComFL project ran the **FedAvg client path against a DeComFL server** — a silent mismatch. |
| `TRAINING_ARM` | `--training-arm` | The arm the FL server was spawned with. Without it the client trains and uploads *every* parameter while a `FROZEN_HEAD` server expects only the head. |

These three are exactly what the desktop's `buildContainerEnv` emits (`docker.service.ts`), alongside `PROJECT_ID` / `SERVER_ADDRESS` / `PARTITION_ID` / `DATASET_PATH` and, when present, `FEDLEARN_CONNECTION_TOKEN`.

### The flag surface `client.py` actually accepts

Anything after the image name is forwarded verbatim. The full parser (`fl-runtime/client.py`, `build_arg_parser`):

| Flag | Notes |
|---|---|
| `--project-id`, `--server-address`, `--partition-id` | required; supplied by the entrypoint. `--partition-id` is bounded by `NUM_PARTITIONS` |
| `--model-type` | choices are **data-driven from the recipe catalog** (`recipes.catalog_keys()`): `PNEUMONIA_CNN`, `CNN`, `CIFAR_RESNET18`, `MLP`, `TRANSFORMER`, `LLM_LORA`, `TINYNET_GOLDEN` |
| `--training-arm` | `FULL` (default) · `FROZEN_HEAD` · `OVA_LP`; validated against the recipe's `supported_arms` at startup and rejected otherwise. An arm carries an **objective**, not just a trainable subset (`OVA_LP` → one-vs-all) |
| `--strategy` | free-form string, default `FedAvg`; the client compares it case-insensitively (`args.strategy.lower() == 'decomfl'`) to pick the DeComFL client path. **`MLP` overrides it to `DeComFL` unconditionally** (`ECG_STRATEGY`), so an `MLP` project ignores whatever you pass |
| `--dataset` | `cb` · `sst2` · `ecg`. Also overridden per model type: `MLP` forces `ecg`, `PNEUMONIA_CNN` forces `pneumonia` |
| `--model-name`, `--aggregation` (`FFA_LORA`/`FEDIT`), `--task-type` (`SEQ_CLASSIFICATION`/`CAUSAL_LM`) | `LLM_LORA` only |
| `--use-llm` | deprecated; use `--model-type TRANSFORMER` |
| `--device` | `auto` (default, `cuda > mps > cpu`) · `cpu` · `cuda` · `mps`; falls back to the `FEDLEARN_DEVICE` env var |

There is **no `--client-id`**.

### Data and volumes — read this before mounting

`-v /path/to/data:/data` on its own does **nothing**: neither `client.py` nor `fl-runtime/data.py` reads `/data` or the `DATASET_PATH` env var the desktop sets. Datasets are otherwise fetched at runtime (HuggingFace `load_dataset`, torchvision `download=True`). To point a recipe at a mounted directory, use the recipe's own env var — for `PNEUMONIA_CNN`:

| Env var | Effect |
|---|---|
| `FEDLEARN_PNEUMONIA_DIR` | use a local `ImageFolder` layout: `<dir>/train` and `<dir>/test` (or `/val`), each with `NORMAL/` and `PNEUMONIA/`. Missing split ⇒ a clear `FileNotFoundError`, not a silent fallback |
| `FEDLEARN_PNEUMONIA_DATASET` / `_CONFIG` / `_REVISION` | HuggingFace fallback repo, config, and a commit pin |
| `FEDLEARN_PNEUMONIA_TRUST_REMOTE_CODE=1` | **SE-19** — remote dataset-loader execution is OFF by default. Enabling it downloads and runs the dataset repo's loader script on this host; pair it with `_REVISION` so an opted-in run executes a known immutable revision |
| `FEDLEARN_PNEUMONIA_ALPHA` / `_BATCH` / `_SUBSET`, `FEDLEARN_NUM_CLIENTS` | Dirichlet non-IID α, batch size, subset cap, cohort size |

### `FEDLEARN_CONNECTION_TOKEN` (SE-14)

When the FL server enforces client auth (`app.fl.require-client-auth=true`), every client must present a backend-minted connection token or the server rejects it `UNAUTHENTICATED`. Fetch it from `GET /api/client/projects/{projectId}/connection` (field `connectionToken`) over your authenticated web session — the same DTO that carries the gRPC endpoint. The framework's client interceptor reads it straight from the process environment and attaches it as the `x-connection-token` metadata header on every gRPC call, which is why it travels as a container **env var** and not a CLI arg. The token is sized to the run's length and expires with it, so fetch a fresh one per run.

The desktop launcher injects it automatically; a **standalone `docker run` must pass it explicitly**. The property defaults to `false` (`app.fl.require-client-auth=${APP_FL_REQUIRE_CLIENT_AUTH:false}`), so you can omit it against a server with auth off — but pass it whenever you don't control the server's setting. Note the backend refuses to boot a deployed profile with client auth on and TLS off (`FlBoundaryAuthPolicyValidator`), precisely because the token would then ride plaintext gRPC and be replayable.

> `run-client.sh` does **not** forward `FEDLEARN_CONNECTION_TOKEN` (nor `MODEL_TYPE` / `STRATEGY` / `TRAINING_ARM`) into the container — it only passes the three required vars and appends extra args. Against a server with client auth on, or a project on a non-default recipe/arm, use a direct `docker run` or extend the script's invocation.

### `run-client.sh` — what it actually does

A convenience launcher, not a supported deployment path: `./run-client.sh <PROJECT_ID> <SERVER_ADDRESS> <PARTITION_ID> [extra args...]`. It detects the platform and branches:

- **macOS** → runs `fl-runtime/client.py` **natively** (with `framework/src` on `PYTHONPATH`), because Docker on Mac has no Metal pass-through, so MPS is unreachable from a container.
- **Linux + `/etc/nv_tegra_release`** → Docker with the JetPack-5-era `/dev/nvhost-*` device mounts (see the Jetson section — `/dev/nvhost-ctrl` is not valid on L4T R36.5).
- **Linux + `nvidia-smi`** → Docker with `--gpus all`.
- **otherwise** → Docker, CPU only.

---

## How it connects to the rest of the system

- The **Backend** (`fl-platform-api`) spawns a Python FL server on a dynamic port in `50000-50010` when a project starts; the port is surfaced in the dashboard's live telemetry. Clients dial `<server-host>:<port>`.
- gRPC is **plaintext by default, TLS opt-in (SE-2)**. Unset, the client builds an `insecure_channel` and **gradients fly plaintext**; set `FEDLEARN_GRPC_USE_TLS=1` (plus `FEDLEARN_GRPC_ROOT_CERT`, and `FEDLEARN_GRPC_CLIENT_CERT`/`_KEY` for mTLS) and it uses `ssl_channel_credentials` + `secure_channel` (`framework/src/fedlearn/client/grpc_client.py:55-76`). Server-side this is fail-closed: `app.fl.require-tls=true` on the `ec2demo`/`production` profiles becomes `FEDLEARN_REQUIRE_TLS=1` on the spawned FL server, which then refuses to serve rather than bind plaintext. See [`deploy/TLS.md`](../../deploy/TLS.md). Dial the DNS name in the cert SAN, not a raw IP.
- Model updates travel as a **deterministic safetensors blob**, never pickle. The wire is **float32-only** so the libtorch-free mobile C++ client can decode it; non-float32 buffers (BatchNorm's int64 `num_batches_tracked`, for instance) are excluded from the federated set and stay local. On the *receive* path the server sniffs the legacy `torch.save` zip / raw pickle magic bytes and rejects loudly.
- Uploads are **unary by default and streamed when large**: `GrpcClient.submit_update()` picks the chunked streaming path only for a transformer (`ALWAYS_STREAM_TRANSFORMERS = True`) or a blob over `STREAMING_THRESHOLD_MB = 100`; once streaming, it always chunks at `FEDLEARN_CHUNK_SIZE_MB` (default 4 MB). A container federating an LLM is on the streaming path.
- The client holds **two gRPC stubs** — training and heartbeat — so a long round can't be timed out; a `HeartbeatResponse` with `should_stop=True` latches a stop the fit loop polls (FR-10).
- For cross-network demos (e.g. a lab Jetson and a laptop elsewhere), a mesh VPN such as Tailscale avoids NAT/firewall work.

---

## CI

`ci.yml` has a path-filtered **`client-docker`** job (TE-6) that runs whenever this directory changes:

1. Build the image with `docker/build-push-action` — `context: .` (the repo root), `file: client-docker/Dockerfile`, `platforms: linux/amd64`, with a `type=gha` layer cache.
2. Smoke-test the **real entrypoint**:
   ```bash
   docker run --rm -e PROJECT_ID=smoke -e SERVER_ADDRESS=localhost:50000 -e PARTITION_ID=0 \
     fedlearn-client:ci --help
   ```
   `--help` rides through `"$@"`, so argparse prints and exits 0 before any gRPC connect — proving the framework + client imports load and the entrypoint wiring is sound.

amd64 only: the default `BASE_IMAGE` is x86 PyTorch/CUDA and hosted runners have no GPU (CUDA runtime libs load lazily, so a CPU-only build is fine). The Jetson lane is arm64 and needs real hardware, so it is deliberately not built here. The job feeds the aggregate `ci-gate`, which is the single required status check.

---

## Packaging — the native sibling (`packaging/`)

The same `fl-runtime/client.py` is also frozen into a standalone binary for the desktop installer, so a Mac/Windows/Linux user never needs Docker or Python. `packaging/fedlearn-client.spec` is a PyInstaller **onedir** spec that:

- resolves the entry point as `../../fl-runtime/client.py` and hard-fails if it is missing;
- `collect_all()`s the dynamic-dispatch libraries (`transformers`, `tokenizers`, `datasets`, `huggingface_hub`, `safetensors`, `accelerate`, `scipy`, `sklearn`, `fedlearn`);
- bundles the sibling modules `client.py` imports by name — `config.py`, `data.py`, `recipes.py`, `device.py`, `models/`, `data_loaders/`, `architecture/` — a missing entry surfaces only as a runtime `ModuleNotFoundError` in the frozen binary, never at build time;
- disables UPX deliberately (it breaks torch dylib loading on macOS and trips Windows AV heuristics).

The per-platform wrappers (`build-mac.sh`, `build-win-cpu.ps1`, `build-win-cuda.ps1`, `build-linux.sh`) create a venv, install the right torch wheel for the target, install `requirements-client.txt`, then `pip install -e ../../framework --no-deps`.

> **The spec still carries dead `flwr` wiring — noisy, not fatal.** `FULL_COLLECT` still lists `'flwr_datasets'`, but `requirements-client.txt` no longer installs it (its "FL Datasets" section is now empty). The spec wraps each `collect_all` in a `try/except … raise`, which reads as "a missing package aborts the build" — it does **not**, because on the pinned PyInstaller 6.11.1 `collect_all()` on an absent package logs two `WARNING: … not a package` lines and returns empty tuples rather than raising (verified against the pinned version). So the entry is a silent no-op: remove it, but it is not blocking a release.
>
> Three related leftovers, all now false: the `excludes` comment claiming matplotlib **cannot** be excluded "because `flwr_datasets/__init__.py` eagerly imports its visualization submodule" — with Flower gone, that exclusion is now available and is a real (unclaimed) size win; the `--no-deps` rationale repeated in `build-mac.sh` / `build-win-cpu.ps1` / `build-win-cuda.ps1` ("the framework's `requirements.txt` pulls in flwr + friends that downgrade protobuf/numpy/transformers"); and the `ResolutionImpossible` troubleshooting entry in `packaging/README.md`, which names the same dead cause. `--no-deps` is still the right flag — it just needs a current reason.

---

## Related documentation

- [`client-docker/README.md`](../../client-docker/README.md) — the unit's own entry point
- [`client-docker/DEPLOYMENT_GUIDE.md`](../../client-docker/DEPLOYMENT_GUIDE.md) — full Jetson + native deployment procedure and troubleshooting playbook (its `--runtime nvidia` sections are stale on JetPack 6 — see above)
- [`client-docker/packaging/README.md`](../../client-docker/packaging/README.md) — PyInstaller bundling for the desktop app
- [Framework Wiki](../framework/README.md) — the FL engine this container wraps
- [Desktop Wiki](../desktop/README.md) — the Electron orchestrator that launches this image for end users
- [Desktop: Hardware Profiles](../desktop/07-hardware-profiles.md) — the Jetson Docker device-mount path in detail
- [`deploy/TLS.md`](../../deploy/TLS.md) — certificate provisioning for the gRPC boundary
