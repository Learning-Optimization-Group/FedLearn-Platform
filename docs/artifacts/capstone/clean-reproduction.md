# Clean CPU reproduction

This profile verifies the capstone's canonical CPU training path using generated data. It does not require the medical bundle, the old shared Python environment, GPUs or an inference service. It covers both mean512 and projected128, real gRPC, final-head simulator comparisons and journal/selection rescoring.

## Prerequisites and fresh installation

Use Python 3.12, Git and uv 0.11.9. Supported targets are macOS arm64, Linux x86_64 and Linux aarch64. The Linux locks select CPU-only Torch wheels. Run from this capstone worktree. These are minimal qualification dependencies, not the full platform/NLP/image-training environment.

```sh
python3.12 docs/artifacts/capstone/repro/reproduce_capstone.py environment \
  --output .temp/clean-capstone-env

.temp/clean-capstone-env/bin/python \
  docs/artifacts/capstone/repro/reproduce_capstone.py doctor
```

`environment` refuses an existing directory, creates a venv without system packages, installs complete hash-locked dependencies with `uv pip sync --require-hashes --torch-backend cpu`, checks dependency compatibility, and records its environment report. It does not inherit the old `.venv` or change shared packages. A download cache may be reused; package imports must resolve inside the new environment.

`doctor` requires the exact complete platform lock and checks direct and loaded transitive dependency origins, user-site isolation, Python version and CPU wheel selection. It rejects inherited site packages, symlink escapes and an externally supplied locked transitive module. It allows the base interpreter's standard library and the checkout's source files. This is a reproducibility check on trusted machines, not a sandbox or supply-chain security certification.

The locks retain Torch 2.12.0, NumPy 2.1.2 and SciPy 1.17.1. The profile uses cryptography 50.0.1 to meet the repository's current floor. It does not rewrite earlier studies' recorded environments. Do not add plotting or unrelated packages to the qualification venv; use a separate environment for them.

## One-command local qualification

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .temp/clean-capstone-env/bin/python \
  docs/artifacts/capstone/repro/reproduce_capstone.py qualify \
  --output .temp/clean-local-qualification
```

This builds deterministic generated training/selection inputs, freezes preparation, runs two Mac-only five-round cells through the actual canonical client/server, compares both final heads to the real simulator HeadClient at maximum absolute error <=1e-5, independently rescores selection AUROC/loss, and reconciles raw journals. Only a fully successful run writes `receipt.json`. Failed or partial artifacts remain available for diagnosis; there is no replacement fitting.

## Three-host qualification

On each remote host, copy `reproduce_capstone.py`, `requirements-capstone-cpu.in` and all three `requirements-capstone-*.lock` files together into a task-owned directory. Run its `environment` command with Python 3.12 and a fresh remote output. Use `--uv` if the pinned uv executable is not on PATH. Do not install into an inference or shared system environment.

Copy [the launch template](repro/capstone-launch.example.json) into an ignored private location. Supply the actual clean interpreter paths, existing noninteractive SSH routes and task-owned storage roots. Keep the routes and machine paths private.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .temp/clean-capstone-env/bin/python \
  docs/artifacts/capstone/repro/reproduce_capstone.py qualify \
  --output .temp/clean-three-host-qualification \
  --launch .temp/private-launch.json
```

The command checks the actual selected interpreters on all hosts, then runs both methods in Mac-only and three-host placements: four cells, twenty rounds, sixty client fits. Remote hosts receive generated training packs only. The Mac retains generated selection. SSH wraps loopback gRPC. Each fresh output receives a distinct remote execution namespace, including when two local output directories have the same basename.

After execution, the command independently compares all four heads to simulator references, rescores metrics, and runs the existing audit that regenerates the entire public analysis. The success receipt binds every platform lock, reproduction source, actual environments, generated input hash, manifest and numerical comparison. Host routes and absolute private paths are excluded from that receipt.

## Bounded profile tests

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=framework/src:fl-runtime:docs/artifacts/capstone/repro \
  .temp/clean-capstone-env/bin/python -m pytest -q \
  fl-runtime/tests/test_effusion_runtime.py \
  fl-runtime/tests/test_effusion_host.py \
  fl-runtime/tests/test_effusion_server.py \
  fl-runtime/tests/test_effusion_transport.py \
  framework/tests/test_multihost_cohort_contract.py \
  docs/artifacts/capstone/repro/test_multihost_manifest.py \
  docs/artifacts/capstone/repro/test_run_multihost_benchmark.py \
  docs/artifacts/capstone/repro/test_audit_multihost_results.py \
  docs/artifacts/capstone/repro/test_reproduce_capstone.py
```

The real gRPC tests require local socket permission. Failure exercises use generated data and a bounded deadline; they do not modify the medical benchmark's 600-second deadline. These checks do not represent the unrelated full platform dependency suite.

The completed 52-cell medical benchmark and report v5 remain pinned historical evidence. This procedure performs no new medical fitting or holdout/test evaluation. A fresh clinical cohort, privacy extension, accelerator study, advisor communication or submission remains a separate task.
