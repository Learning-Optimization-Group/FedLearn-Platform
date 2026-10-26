# Reproducing the three-host CPU benchmark

The protocol is [multihost-protocol-2026-10-26.md](multihost-protocol-2026-10-26.md). The prospective source and input commitments are [multihost-public-manifest-2026-10-26.json](multihost-public-manifest-2026-10-26.json). The initial reviewed implementation is checkpoint `8ac59c4`.

Use compatible isolated CPU environments on the Mac coordinator and both Linux hosts. Keep the qualified training/selection bundle, launch configuration and outputs private. The launch JSON has exactly `local_python` and `hosts`; `hosts` contains `g14` and `orin`, each with an `ssh` argument array, absolute `python` interpreter path and fresh task-owned `root` directory. SSH routes must already authenticate without prompts. The runner wraps loopback gRPC in private reverse forwarding and refuses duplicate remote SSH targets. It does not configure public networking, install packages or alter existing services.

Set `PYTHONPATH=framework/src:fl-runtime:docs/artifacts/capstone/repro`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`. The commands below use `python` from the qualified isolated environment. Replace bracketed arguments with private paths; fresh directories are required.

```text
python docs/artifacts/capstone/repro/run_multihost_benchmark.py preflight \
  --prepared <fresh-preparation-directory> \
  --bundle <qualified-training-selection-bundle> \
  --bundle-sha256 dec54dfb580232ed6bf8da16ea9466248db58de4925285c8fb0d07a381562ffa \
  --launch <private-launch.json>
```

Preparation performs no fitting. Publish/checkpoint its path-free `public-manifest.json` before medical execution. Its input commitments bind host payloads and server/selection plans; changing a private payload and refreshing its local hashes does not preserve the public seal. Runtime probes bind the actual selected interpreters. Generated-data qualification and real transport failure checks must pass first; see the qualification receipt and tests.

```text
python docs/artifacts/capstone/repro/run_multihost_benchmark.py execute \
  --prepared <preparation-directory> --launch <private-launch.json> \
  --output <fresh-execution-directory>
```

The fixed 52-cell budget and prospective order come from preparation. There is no failed-cell retry or silent replacement mode. A failed run remains failed and later declared cells remain unrun. Do not edit source or packs after preparation. Preserve all private logs, including failures.

```text
python docs/artifacts/capstone/repro/run_multihost_benchmark.py analyze \
  --prepared <preparation-directory> --execution <execution-directory> \
  --historical <pinned-selection-stage-finalist-study> --output <fresh-public-analysis.json>

python docs/artifacts/capstone/repro/audit_multihost_results.py \
  --prepared <preparation-directory> --execution <execution-directory> \
  --analysis <public-analysis.json> --checkpoint <source-checkpoint> \
  --historical <pinned-selection-stage-finalist-study> \
  --output <fresh-verification.json>

python docs/artifacts/capstone/repro/render_multihost_results.py \
  --source <public-analysis.json> --output <fresh-figure-directory>
```

The first analysis independently rescores final-head loss/AUROC and compares placement pairs plus historical anchors. The separate audit reconciles raw JSONL with summaries, exact client-round events, batch budgets, example counts, every evaluated aggregate and independent final-head digests. Plotting is descriptive: three paired training seeds, median/minimum/maximum timing ratios, no confidence-interval claim.

Generated qualification used NumPy default RNG seed911, 1,040 training rows and80 selection rows of512 independent standard-normal float32 features. Labels are `x[:,0]>0`; consecutive groups of four rows share a generated patient ID. Training and selection use disjoint generated ID ranges, with selection draws following training draws. The four selected qualification cells are both methods and placements at three clients, seed14001 and five rounds. `execute --qualification` requires a generated preparation and selects only those four cells. The simulator comparison uses the actual `HeadClient` recipe and independently managed round/client RNG; it is a float32 execution comparison, not a formal numerical analysis or float64 oracle.

Do not interpret three physical machines as eighty independent devices, the five-round scores as the original primary endpoint, application protobuf bytes as wire traffic, or this nonprivate CPU benchmark as DP/GPU/hospital qualification.
