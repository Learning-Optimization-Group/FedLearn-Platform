# A4 — `client-docker` Audit (v2 Greenfield)

**Unit:** `client-docker/` — the multi-arch FL client container, a thin wrapper around `framework/`.
**Date:** 2026-05-29
**Branch:** `main-clean`
**Builds on:** `docs/audit/2026-05-27/03-framework.md` (the 2026-05-27 framework+client-docker report). That report's **H6** ("`flwr` is still a runtime dep"), **C2** (pickle RCE on cache), and the Dockerfile/`entrypoint.sh` Low items are extended and re-scoped here with deeper verification. I do **not** re-litigate the framework-internal findings (C1 chunked-upload `KeyError`, heartbeat, DeComFL aggregation) — those belong to the framework unit (A3).

---

## Executive summary

The container is structurally sound as a thin wrapper — `Dockerfile` installs `framework/` via `pip` (line 26) and copies only entry scripts; the Jetson L4T device-mount path is implemented correctly and matches the platform invariant (no `--runtime nvidia`). But three packaging defects undermine production readiness:

1. **The `flwr`/`flwr-datasets` leak is worse than the prior audit stated.** It is present in *three* dependency surfaces (`framework/requirements.txt`, `client-docker/requirements.txt`, `packaging/requirements-client.txt`) and is **functionally trivial to remove** — `flwr_datasets.FederatedDataset` is used on exactly **one** code path (CIFAR-10 demo, `client.py:363`) while two hand-rolled Dirichlet splitters already exist in the same repo (`client.py:248`, `ecg_loader.py:30`). It violates a stated platform invariant for a demo-only convenience and forces matplotlib + a heavy transitive tree into the native bundle.

2. **The Docker dependency resolution is internally contradictory and silently corrupts pins.** `Dockerfile:26` installs the framework *with* its transitive deps (`numpy==2.1.2`, `transformers==4.55.2`, `flwr==1.20.0`), then `Dockerfile:30` installs `client-docker/requirements.txt` which pins `numpy>=1.21.0,<2.0.0`. These conflict; pip resolves last-wins and downgrades numpy across a torch ABI boundary. The PyInstaller build scripts already know this and use `--no-deps` — the Docker build does not.

3. **Zero supply-chain pinning of provenance.** Base images are tags, not digests; `requirements.txt` is a floor-only (`>=`) spec; no SBOM, no `pip-audit`, no image scan. The default base (`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`) is an EOL Aug-2023 build with known CVEs in its CUDA/Python layers.

**Verdict spread:** the wrapper concept and Jetson path are **salvage**; the dependency manifests and Dockerfile build are **rebuild**; `flwr_datasets` usage is **kill**.

---

## Module verdicts

| Module / concern | Verdict | One-line rationale |
|---|---|---|
| Thin-wrapper discipline (`framework` via pip, scripts only) | **salvage** | Correctly installs `framework/`; no duplicated aggregation/serialization logic — only data-loading + CLI glue lives in `scripts/`. |
| `flwr_datasets` runtime dependency | **kill** | Used on one demo path; two in-repo Dirichlet splitters already exist; violates the no-Flower invariant and bloats every artifact. |
| `Dockerfile` build (single-stage, dep-resolution order) | **rebuild** | Conflicting pins silently downgrade numpy across torch ABI; single-stage ships `build-essential`+`git`; tag-pinned EOL base. |
| Dependency manifests (`requirements.txt`, `requirements-client.txt`) | **rebuild** | Three divergent specs; floor-only `>=` in Docker vs exact `==` in native; numpy/transformers/protobuf disagree across surfaces. |
| Jetson L4T path (`run-client.sh`, docs) | **salvage** | Device mounts correct, `--runtime nvidia` correctly avoided. One device-list drift between script and docs to reconcile. |
| Multi-arch story (`BASE_IMAGE` build-arg) | **refactor** | "Multi-arch" is really "swap the base by hand"; no `buildx` manifest, no digest pinning, default tag has no arm64 variant. |
| `entrypoint.sh` / `run-client.sh` hygiene | **refactor** | `set -e` only (no `-u`/`-o pipefail`); `--gpus`/device flags hand-assembled per host. |
| Pickle cache (`client.py:328`, `ecg_loader.py:114`) | **rebuild** | Carried over from prior C2 — `pickle.load` on a writable bind-mount path is RCE; replace with `npz`/`json`. |
| Supply-chain posture (provenance, scanning, SBOM) | **rebuild** | No digest pinning, no `pip-audit`, no Trivy/Grype, no SBOM; EOL base image. |

---

## Findings

### CD1 (Critical) — `flwr-datasets` runtime leak: violates the no-Flower invariant, and is trivially removable

**Evidence.**
- Lazy import: `client-docker/scripts/client.py:84-87` (`from flwr_datasets import FederatedDataset`).
- The **only** call site: `client.py:363` — `fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_PARTITIONS})`. This is the CNN/CIFAR-10 demo path only.
- Pinned in three places:
  - `client-docker/requirements.txt:10` → `flwr-datasets>=0.3.0`
  - `client-docker/packaging/requirements-client.txt:25` → `flwr-datasets==0.5.0` (the exact `==0.5.0` the assignment flagged — it lives in the *native* manifest, not the Docker one)
  - `framework/requirements.txt:7-8` → `flwr==1.20.0` **and** `flwr-datasets==0.5.0` (this is the full `flwr` runtime, pulled transitively into the Docker image via `Dockerfile:26`)
- The dependency is load-bearing for unrelated cost: `packaging/fedlearn-client.spec:92` documents that **matplotlib cannot be excluded** from the native bundle because `flwr_datasets/__init__.py` eagerly imports its visualization submodule. So one demo data loader forces matplotlib + its transitive tree into every desktop installer.
- The repo **already contains** two hand-rolled Dirichlet splitters that do exactly what `FederatedDataset` partitioning does: `client.py:248` (`dirichlet_split`, used on the LLM path at `client.py:331`) and `ecg_loader.py:30` (used at `ecg_loader.py:141`). The LLM and ECG paths do not touch `flwr_datasets` at all.

**Why this matters for a startup.** It is a stated, non-negotiable platform rule (`CLAUDE.md`: "No `flwr` dependency. Custom protobuf only."). Beyond principle: `flwr` 1.20 pulls `grpcio`, `protobuf`, `cryptography`, `pyarrow==16.1.0` (the exact pin called out at `requirements.txt:24`), and matplotlib pins that the build scripts already fight with `--no-deps`. Carrying a second FL framework's runtime to partition CIFAR-10 is pure liability — larger images, larger attack surface, more CVE exposure, and a contradiction any technical-DD reviewer will catch.

**Recommendation (v2).** Delete `flwr_datasets` entirely.
1. Replace `client.py:363-369` CIFAR-10 loading with: load CIFAR-10 via HuggingFace `datasets` (`datasets.load_dataset("cifar10")`, already a dep at `requirements.txt:16`), then partition with the **existing** `dirichlet_split` (`client.py:248`). This is the same Dirichlet non-IID semantics `FederatedDataset` provides with `DirichletPartitioner`, minus the framework.
2. Remove `flwr==1.20.0` and `flwr-datasets==0.5.0` from `framework/requirements.txt:7-8`, `client-docker/requirements.txt:10`, `packaging/requirements-client.txt:25`.
3. Remove `'flwr_datasets'` from `fedlearn-client.spec:37` `FULL_COLLECT`, then re-test whether matplotlib can finally be added to `excludes` (`spec:89-102`) — expected saving ~150-250 MB off the native bundle.
4. Add a regression test asserting the new in-repo CIFAR-10 Dirichlet partition reproduces the prior per-client class histogram for a fixed seed (the prior audit's H6 also asked for this).
- *Uncertainty flagged:* I have not executed `FederatedDataset` vs `dirichlet_split` side-by-side; both use a class-conditional Dirichlet, but `flwr_datasets` `DirichletPartitioner` differs in min-partition-size handling and shuffling. Treat the histogram regression test as the acceptance gate, not an a-priori equivalence claim.

**Verdict: kill** (the dependency). The CIFAR-10 loader path itself is **refactor**.

---

### CD2 (Critical) — `Dockerfile` dependency resolution is self-contradictory; silently downgrades numpy across the torch ABI

**Evidence.**
- `Dockerfile:26` — `RUN pip3 install --no-cache-dir /app/framework/` — **no `--no-deps`**. `framework/setup.py:53` does `install_requires=read_requirements()`, reading `framework/requirements.txt`. So this step pulls `numpy==2.1.2` (`framework/requirements.txt:20`), `transformers==4.55.2`, `tokenizers==0.21.4`, `flwr==1.20.0`, `protobuf` per flwr's constraint.
- `Dockerfile:30` — `RUN pip3 install --no-cache-dir -r requirements.txt` — installs `client-docker/requirements.txt`, which pins `numpy>=1.21.0,<2.0.0` (`requirements.txt:31`), `transformers>=4.30.0` (floor), `protobuf>=4.21.0,<6.0.0`, `grpcio>=1.60.0`.
- These are **mutually exclusive** on numpy: framework wants `==2.1.2`, client wants `<2.0.0`. Two separate `pip install` invocations means pip does not jointly solve; the second silently **downgrades numpy to <2.0**, rebuilding/relinking against a torch wheel that may have been built against numpy 2.x. This is exactly the class of "torch + numpy ABI" breakage the native build scripts pre-empt with `pip install -e framework --no-deps` (`build-mac.sh:46-48`, `build-win-cpu.ps1`). **The Docker build does not apply that lesson.**

**Why this matters.** The container's actual installed dependency set is undefined-by-construction and order-dependent — it "works" today only because the conflict resolves to a numpy that happens to import. That is not a production posture; a base-image bump or a pip resolver change can flip it to a hard `ImportError` at client start, which in this system surfaces only as a dead STOMP log line on `/topic/logs/{projectId}`.

**Recommendation (v2).** Single source of truth for deps + deterministic install:
1. Install the framework **`--no-deps`** in Docker too (mirror the native scripts): `RUN pip3 install --no-cache-dir --no-deps /app/framework/`.
2. Make `client-docker/requirements.txt` the *complete, fully-pinned* runtime lock for the container (exact `==`, not `>=`), generated from a resolved environment (`pip-compile`/`uv pip compile`). One manifest, one solve.
3. After removing `flwr` (CD1), reconcile numpy/transformers/protobuf to a single set that has cp-aarch64 wheels for the Jetson Python (see CD4).

**Verdict: rebuild.**

---

### CD3 (High) — Supply-chain: no provenance pinning, EOL base, no scanning

**Evidence.**
- Base images are **tags, not digests**: `Dockerfile:9` `ARG BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`; Jetson `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (README:30). Tags are mutable — no `@sha256:` digest anywhere.
- The default base is **EOL**: PyTorch 2.0.1 / CUDA 11.7 is an Aug-2023 image (prior audit Low item already flagged the staleness; I extend it to a *security* finding). Its Ubuntu/Python/CUDA userland carries unpatched CVEs.
- `client-docker/requirements.txt` is **floor-only** (`>=`) for nearly every line — `grpcio>=1.60.0`, `transformers>=4.30.0`, `aiohttp>=3.8.0`, etc. `aiohttp` and `transformers` floors that old admit known CVEs (e.g. aiohttp <3.9 advisories). A floor-only spec means any rebuild can silently pull a different transitive graph.
- No `pip-audit`/`safety`, no image scanner (Trivy/Grype), no SBOM emitted in any build script (`test_docker_build.sh` checks import + torch only).
- `Dockerfile:14-15` installs `build-essential git wget curl` into the **runtime** image — extra binaries that are pure attack surface and never needed at FL-client runtime.

**Why this matters.** This client runs on third-party / classroom / edge hardware and speaks **plaintext gRPC over WAN** (audit item #37; README:99). A compromised or CVE-laden client image is the soft underbelly of a federation. For a startup pursuing customer pilots, "we can't tell you what's in the client image or whether it has known CVEs" fails the first security questionnaire.

**Recommendation (v2).**
1. Pin base images by digest (`FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:...`). Bump default off EOL 2.0.1.
2. Fully pin `requirements.txt` (exact `==`) via lockfile; rebuild the lock on a schedule with `pip-audit` as a gate.
3. Add `trivy image --severity HIGH,CRITICAL --exit-code 1` (or Grype) to CI on every image build; fail the build on new criticals.
4. Emit an SBOM (`syft` / `docker buildx --sbom=true`) and attach it to the published image.
5. Multi-stage build (see CD4) so `build-essential`/`git` never reach the runtime layer.

**Verdict: rebuild** (the supply-chain posture).

---

### CD4 (High) — "Multi-arch" is manual base-swapping, not a true multi-arch build; default base has no arm64

**Evidence.**
- The mechanism is a single `BASE_IMAGE` build-arg (`Dockerfile:9`, README:25-32). To produce an arm64 image you must *manually* pass the L4T base; there is no `docker buildx build --platform linux/amd64,linux/arm64` manifest list. README:93 punts true multi-platform distribution to `DEPLOYMENT_GUIDE.md`.
- The default base `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` is **x86-only** — it has no `linux/arm64` manifest, so a naive `docker build` on an arm64 host (or `buildx --platform linux/arm64`) without overriding `BASE_IMAGE` fails or silently emulates. This is a footgun for anyone building on Apple Silicon / Jetson without reading the README.
- Python-version skew: `client-docker/requirements.txt:2` is written for **Jetson Python 3.8 / cp38-aarch64** wheels ("ALL versions verified for cp38-aarch64", `requirements.txt:4`), while the cached bytecode in the tree is `client.cpython-312.pyc` (built on a 3.12 host) and the newer x86 base ships 3.10/3.11. The same floor-only manifest is being asked to resolve across **three** Python versions (3.8 Jetson, 3.10/3.11 x86, 3.12 dev). `DEPLOYMENT_GUIDE.md:327-397` documents the resulting breakage history (PEP 585 hints, protobuf gencode major mismatch, tokenizers/safetensors source-build on 3.12). This is the root cause of the manifest divergence.

**Why this matters.** "Multi-arch" in the README implies a single `docker pull` resolves the right arch. It does not. For a startup shipping to heterogeneous edge clients, the deployment story must be `buildx` manifest lists pushed to a registry, with per-arch locked dependency sets — not "edit the build-arg and hope the floors resolve."

**Recommendation (v2).**
1. Adopt `docker buildx` with explicit per-arch base images and **per-arch pinned lockfiles** (one for cp38-aarch64/L4T, one for cp311-x86). Push as a single manifest-list tag so clients pull the right arch transparently.
2. Pick **one** Python per arch and freeze it; stop trying to satisfy 3.8 + 3.12 from one floor-only file.
3. Track NVIDIA's newer JetPack 6 / L4T r36 images (Ubuntu 22.04, Python 3.10) to retire the Python 3.8 constraint that drives most of the `DEPLOYMENT_GUIDE.md` breakage log — *flag uncertainty:* this is a hardware-dependent migration (requires JetPack 6 on the target Jetson) and must be validated on the actual device before committing.

**Verdict: refactor.**

---

### CD5 (High) — Pickle cache deserialization is RCE on a writable path (carried from prior C2, re-confirmed)

**Evidence.**
- `client.py:328` — `client_indices_list = pickle.load(f)` from `./data_splits/{dataset}_clients{n}_alpha{a}.pkl` (`client.py:323-328`).
- `ecg_loader.py:114` — `split_data = pickle.load(f)` from `./data_splits/ecg_clients...pkl` (`ecg_loader.py:105,113`).
- `./data_splits` is created relative to CWD inside the container (`WORKDIR /app/scripts`, `Dockerfile:44`). Any user who bind-mounts a shared host volume over `/app/scripts/data_splits` (the README's run examples already bind-mount `/data`) or co-resides on the host can plant a malicious pickle → arbitrary code execution at client start, running as the `fedlearn` user (`Dockerfile:39-45`).

**Why this matters.** The container correctly drops to a non-root `fedlearn` user (good — `Dockerfile:37-45`), but RCE-as-`fedlearn` on an edge box still gives an attacker the gRPC client identity, the project ID, and a foothold to poison model updates. The payloads being cached are **lists of integer indices** and small numpy arrays — there is zero reason to use pickle.

**Recommendation (v2).** Replace pickle with `np.savez_compressed` (for the index arrays + ECG splits) or JSON (for the index lists in `client.py`). This was prior audit C2/Low; it remains unfixed in the current tree. Cheap, high-value.

**Verdict: rebuild** (the cache serialization).

---

### CD6 (Medium) — Shell hygiene and GPU-flag assembly

**Evidence.**
- `entrypoint.sh:2` — `set -e` only (prior audit Low). No `-u` (undefined vars silently empty — `PROJECT_ID` checks at line 9 mitigate the obvious case but not `$@` mishandling) and no `-o pipefail`.
- `run-client.sh:22` — same `set -e` only.
- `run-client.sh:133` device list for Jetson: `--device /dev/nvhost-ctrl --device /dev/nvhost-ctrl-gpu --device /dev/nvhost-prof-gpu --device /dev/nvmap --device /dev/nvhost-gpu` — **omits `/dev/nvhost-dbg-gpu`**, which the README (line 80) and `DEPLOYMENT_GUIDE.md:82` both include. The script and the docs disagree on the canonical Jetson device set.
- `EXTRA_ARGS="$@"` then unquoted `$EXTRA_ARGS` (`run-client.sh:53,110,152`) — word-splitting bug if any forwarded arg contains spaces.

**Recommendation (v2).** `set -euo pipefail` in both scripts; reconcile the Jetson device list to a single sourced constant shared by script + docs (or better, drive it from the desktop `DockerService` which already encodes the canonical Jetson mounts per `CLAUDE.md`); use `"$@"` arrays instead of a flattened `EXTRA_ARGS` string.

**Verdict: refactor.**

---

### CD7 (Medium) — Thin-wrapper discipline holds, but `scripts/` carries a duplicated `dirichlet_split` and stale `.pyc`

**Evidence.**
- Discipline is largely respected: `Dockerfile:22-26` copies `framework/` and `pip install`s it as the single source of truth; `scripts/` contains only data loaders (`ecg_loader.py`), models (`ecg_mlp.py`, `CnnNet`), config, and the CLI entry. No aggregation/serialization/gRPC logic is duplicated — that all lives in `framework/` (imported at `client.py:64-65`). This satisfies the `CLAUDE.md` rule "Don't duplicate framework logic into `client-docker/scripts/`."
- **However:** `dirichlet_split` exists **twice** — `client.py:248` and `ecg_loader.py:30` — with subtly different signatures (`client.py` uses global `np.random.seed`; `ecg_loader.py` uses a local `np.random.RandomState`/`rng`). Data-partitioning *is* arguably FL logic. This is duplication waiting to drift.
- A committed build artifact leaks into the tree: `scripts/__pycache__/client.cpython-312.pyc`. `.dockerignore` excludes `__pycache__/` from the image (good), but it should not be in git at all.

**Recommendation (v2).** Hoist a single `dirichlet_partition(labels, num_clients, alpha, seed) -> list[np.ndarray]` into `framework/` (it is FL-domain logic, used by every client variant) using a `np.random.Generator` (not global `np.random.seed`, which the prior framework audit M5 already flagged as a cross-run RNG hazard). Have both `client.py` and `ecg_loader.py` import it. Add `__pycache__/` and `*.pyc` to `.gitignore`; the stray `._.DS_Store`/`.DS_Store` files in the dir should also be purged.

**Verdict: salvage** (wrapper concept) with a **refactor** on the duplicated splitter.

---

### CD8 (Low) — Default `BASE_IMAGE` is EOL PyTorch (re-confirm of prior Low)

`Dockerfile:9` defaults to `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`. Prior audit (03-framework.md Low + Quick win 8) already recommended bumping to a 2024-era image; the in-file comment at `Dockerfile:7` even *suggests* `2.5.1-cuda12.4-cudnn9-runtime` but the default was never changed. Make `2.5.1-cuda12.4-cudnn9-runtime@sha256:...` the default (folded into CD3/CD4). **Verdict: refactor.**

---

## v2 client-packaging recommendation

**Keep Docker as the primary distribution; add distroless for the runtime layer; standardize on `buildx` manifest lists.**

### Target architecture
```
Stage 1 (builder):  python:3.11-slim  (x86)  /  l4t-base + py3  (arm64)
   - install build-essential, git ONLY here
   - pip install --no-deps framework/  + fully-pinned requirements lock
   - produce a wheelhouse or a populated venv

Stage 2 (runtime):  distroless or *-slim, NO build tooling
   - copy the venv / site-packages from builder
   - copy scripts/ + entrypoint
   - USER nonroot (already done)
```

**Why distroless (or at minimum multi-stage `-slim`):** the current single-stage image carries `build-essential`, `git`, `wget`, `curl`, dev headers (`Dockerfile:14-15`) into the runtime — all attack surface, all CVE feed. Distroless (`gcr.io/distroless/python3`) removes the shell and package manager entirely. *Caveat / uncertainty:* distroless has **no shell**, so `entrypoint.sh` (bash) cannot run as-is — the entry must become a Python entrypoint (`ENTRYPOINT ["python3","client.py", ...]`) with env parsing moved into `client.py`. The GPU path also requires the CUDA userland; full distroless may not be viable for the GPU image — recommend distroless for a **CPU-only** client variant and a slim multi-stage CUDA image for GPU. Validate CUDA-lib presence on the chosen runtime base before committing; do not assume distroless ships CUDA.

### Replacing the `flwr-datasets` Dirichlet split — concrete plan
1. CIFAR-10: `ds = datasets.load_dataset("cifar10", split="train")` (HF `datasets`, already pinned). Extract labels, call the hoisted `framework`-owned `dirichlet_partition(labels, NUM_PARTITIONS, alpha, seed)`, then `ds.select(indices_for_partition)`. Apply the existing `apply_transforms` (`client.py:371-375`).
2. This is the same class-conditional Dirichlet non-IID scheme `flwr_datasets.DirichletPartitioner` implements — *but verify equivalence empirically* via a fixed-seed per-client class-histogram regression test, not by assertion (uncertainty flagged in CD1).
3. Net effect: `flwr` and `flwr-datasets` deleted from all three manifests; matplotlib droppable from the PyInstaller `excludes`; one Dirichlet implementation in the codebase instead of three.

### Dependency management
- One `pyproject.toml`/lock per arch, exact-pinned, regenerated by `uv pip compile`/`pip-compile`; `pip-audit` gate in CI.
- Framework installed `--no-deps` in *all* package paths (Docker + PyInstaller) — the lockfile is the single solver.

### Supply chain
- Digest-pin base images; Trivy/Grype scan on every build (fail on HIGH/CRITICAL); emit + publish SBOM; rebuild lock on a cadence to absorb CVE patches.

### Observability hook (ties to 04-observability.md)
- The container's only telemetry today is stdout → STOMP. For production FL-run observability, the v2 client should emit structured (JSON) logs and per-round metrics (loss, samples, round latency, GPU util from the already-present `pynvml`/`psutil`) to the same backend the framework uses (the 2026-05-27 observability report recommends self-hosted MLflow). Keep the human-readable STOMP stream, add a machine-readable metrics channel.

---

## Prioritized recommendations

| # | Action | Effort | Payoff | Refs |
|---|---|---|---|---|
| 1 | Delete `flwr_datasets`; swap CIFAR-10 to HF `datasets` + existing `dirichlet_split`; add histogram regression test | M | Removes invariant violation + 2nd FL framework runtime + unblocks matplotlib exclude (~150-250MB native) | CD1 |
| 2 | Install framework `--no-deps` in Docker; make `requirements.txt` a complete exact-pinned lock | S | Stops silent numpy downgrade across torch ABI | CD2 |
| 3 | Replace pickle cache load with `npz`/JSON in `client.py:328` + `ecg_loader.py:114` | S | Closes RCE on writable bind-mount | CD5 |
| 4 | Digest-pin bases; bump default off EOL 2.0.1; add Trivy + `pip-audit` + SBOM in CI | M | Customer-DD-grade supply chain | CD3, CD8 |
| 5 | Multi-stage (builder/runtime) + distroless CPU variant; drop `build-essential`/`git` from runtime | M | Smaller, lower-CVE images | CD3, v2 plan |
| 6 | `buildx` manifest lists, per-arch locked deps, freeze one Python per arch | L | Real multi-arch `docker pull`; ends 3.8/3.12 skew | CD4 |
| 7 | `set -euo pipefail`; reconcile Jetson device list (add `/dev/nvhost-dbg-gpu`) script↔docs; quote `"$@"` | S | Script robustness; consistent Jetson GPU access | CD6 |
| 8 | Hoist single `dirichlet_partition` into `framework/`; `.gitignore` `__pycache__`/`.DS_Store` | S | One Dirichlet impl; clean tree | CD7 |

---

## What is genuinely good (keep)

- **Thin-wrapper discipline is real** — `framework/` is the single source of truth, installed via pip; no FL logic copied into `scripts/` (`Dockerfile:22-26`, `client.py:64-65`).
- **Non-root runtime user** (`Dockerfile:37-45`) — already best practice; preserve.
- **Lazy heavy-import pattern** (`client.py:81-115`) — `flwr_datasets`/`transformers`/`torchvision` imported only on the path that needs them; meaningfully faster boot and lets the `flwr` removal be surgical.
- **Jetson L4T correctness** — device mounts not `--runtime nvidia` (`run-client.sh:133`, `DEPLOYMENT_GUIDE.md:130-139`), matching the platform invariant exactly. The `DEPLOYMENT_GUIDE.md` troubleshooting playbook (sections 4.2-4.5) is high-quality institutional knowledge — keep and version it.
- **Build smoke tests** (`test_docker_build.sh`) verify framework import + torch backend — a good base to extend into a CI gate.
