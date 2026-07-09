# Research — working notes for a paper on this FL platform

This folder is the **durable, versioned home** for anything the paper needs. The repo's
`benchmarks/results/` dirs are gitignored, so committed result copies live here under
`research/results/`; experiment logs live under `research/notes/`.

> Standing directive (see the project `CLAUDE.md` → "Research paper"): salvage publishable
> material here as it is produced; Claude may run extra honest, seeded, logged experiments that
> advance the paper.

## Working thesis (evolving)

A full-stack federated-learning platform used as a vehicle for a *systems + honest-empirics* study
of the tensions in privacy-preserving, robust, on-device federated adaptation. The emphasis is on
**measured, reproducible, honestly-caveated** results — including negative ones — rather than a
single headline number.

## Candidate contributions (with code + result pointers)

| # | Contribution | Where | Result / artifact |
|---|---|---|---|
| C1 | **Central-DP FedLoRA** on the FFA-LoRA adapter-B/head channel (clip → uniform-avg → Gaussian), calibrated by a **from-scratch pure-Python RDP accountant** for the Sampled Gaussian Mechanism. | `framework/src/fedlearn/privacy/{dp_mechanism,dp_accountant}.py`; `server/strategy.py::FedLoRA` | Accountant verified vs Opacus 1.6.0 to **1.76e-10** across 154 RDP orders; ε-vs-accuracy benchmark (`FR-13`) → `results/dp_epsilon_accuracy.*` |
| C2 | **Honest utility–SNR analysis** of DP-FL: `SNR = N/(z·√d)` (clip-independent) quantifies the high-dimension / small-cohort **utility collapse** — a *negative result*, measured not hand-waved. | `framework/benchmarks/dp_epsilon_accuracy.py` | Utility collapses to chance at all ε (SNR 0.023/0.012/0.003 at ε=8/4/1, d=26112); accounted ε exact to ~9 dp |
| C3 | **Byzantine-robust aggregation** (coordinate-wise median, β-trimmed-mean, δ-space L2 clip, Byzantine-fraction guard; unweighted by design) with a **measured breakdown point**. | `framework/src/fedlearn/server/robust_aggregation.py` | `FR-12` benchmark → `results/robust_aggregation_attack.*`: under a 20% sign-flipped gradient-scaling attack, trimmed-mean(β=.2)/median **retain 100%** vs FedAvg **3.7%**; textbook breakdown at f=β |
| C4 | **Deterministic, byte-identical cross-language safetensors wire** replacing `torch.save`/pickle on both the upload and (FR-8) download paths, with a version gate + per-chunk sha256 integrity. | `framework/src/fedlearn/communication/{safetensors_codec,serializer}.py`; mobile C++ decoder | Python↔C++ byte format pinned by a committed golden; tamper-rejection tests |
| C5 | **Content-addressed adapter-bundle registry** with a **cross-language sha256 provenance contract**: a LoRA run emits a `fedlearn.bundle` manifest whose `artifact_sha256` resolves to the registry row (`DA-9`). | `framework/src/fedlearn/bundle/manifest.py`; `fl_server.py`; backend `ArtifactRegistryService` | Python `sha256_hex` == Java `blobSha256` golden-pinned |
| C6 | **Cross-language contract testing** for a Java(JJWT)↔Python(PyJWT) security token via a golden fixture with value-level checks (`SE-13`). | `.../GoldenConnectionTokenFixtureTest.java`; `test_token_verify_golden.py` | golden pins alg + all 9 claims; RED-proven both sides |
| C7 | **On-device zeroth-order DeComFL** path (ExecuTorch + native C++ via a TurboModule bridge). | `mobile_client/`; `framework/.../decomfl_*` | (systems contribution; convergence pinned by `TE-1`) |

## Salvaged results index (`research/results/`)

- `dp_epsilon_accuracy.{json,md}` — **FR-13** DP ε-vs-accuracy on FedLoRA (C1/C2). Reproduce: `cd framework && PYTHONPATH=src python benchmarks/dp_epsilon_accuracy.py`.
- `robust_aggregation_attack.{json,md}` — **FR-12** Byzantine-robustness under a gradient-scaling attack (C3). Reproduce: `cd framework && PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --rounds 25 --clients 10 --attack-scale -10 --sweep-fractions 0.1,0.2,0.3`.

(Copies are committed here because `benchmarks/results/` is gitignored. Re-copy after a re-run to keep them current.)

## Honesty ledger (caveats that must survive into the paper)

- **FR-13**: the DP utility collapse is real at every laptop-tractable scale; a usable privacy–utility gradient needs N≈√d clients, subsampling amplification, or a lower-dim adapter. Do **not** present a hand-tuned curve.
- **FR-12**: the attack is a *sign-flipped* (−10×) gradient scaling (amplifying an honest gradient on a separable task is not adversarial — disclosed). The synthetic task is separable, so absolute accuracies (100%) demonstrate the *contrast*, not a production number. A weight confound (20% of clients = 36% of FedAvg's weighted mass) is reported via an explicit column; RobustAggregator is unweighted by design.

## Open experiments / paper TODO

- [ ] FR-12 on a **real** non-IID dataset (not synthetic) + more attack types (IPM, "Fall of Empires", label-flip) to strengthen C3 beyond a demonstration.
- [ ] FR-13 at N≈√d clients and/or with client **subsampling amplification** to show the SNR crossing 1 (a positive privacy–utility curve to complement the collapse).
- [ ] DeComFL zeroth-order vs first-order convergence/communication trade-off (C7) — a measured comparison.
- [ ] End-to-end system evaluation (latency, communication rounds, chunked-wire overhead) for the systems framing.

## Method note

All benchmarks are seeded + committed harnesses. Findings are logged in `research/notes/`. Keep this
README's contribution table and results index in sync as work lands.
