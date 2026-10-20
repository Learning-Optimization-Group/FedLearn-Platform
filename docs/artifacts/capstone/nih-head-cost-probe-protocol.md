# Prospective NIH head-cost qualification

19 October 2026. Fixed development-only execution/cost probe, before any of its
real-data cells are run. This is not the confirmatory privacy/robustness protocol.

## Purpose and prior information

The [v3 linear-head baseline](nih-schedule-pilot-v3-report.md) established useful
eighty-client validation utility. A tiny-head run cannot establish the runtime,
memory or numerical stability of larger heads. Qualify the four already proposed
head sizes before budgeting secondary experiments. Do not select a head, tune a
learning rate or claim convergence from this short cost probe.

All representations are the same frozen pretrained ResNet-18 features. Changing
head width changes capacity and optimization as well as trainable dimension;
this is not a causal isolation of dimension alone.

## Fixed cells

| Name | Head | Hidden width | Trainable parameters |
| --- | --- | ---: | ---: |
| `linear` | Linear 512→2 | — | 1,026 |
| `mlp8` | 512→8→2, ReLU | 8 | 4,122 |
| `mlp128` | 512→128→2, ReLU | 128 | 65,922 |
| `mlp512` | 512→512→2, ReLU | 512 | 263,682 |

Execute in this order, one isolated child at a time, with no concurrent regression
suite or other planned benchmark. Ordinary background load remains uncontrolled.
The fixed recipe is eighty whole-patient IID clients, all eighty each round,
two rounds, seven local epochs, SGD LR 0.003, minibatch size 256, mean aggregation,
joint L2 clipping C 0.4, no server optimizer, no DP and no bad/faulty clients.
Base seed 20261019 with the existing deterministic seed derivation. There are
four cells and no adaptive follow-ons or retries under a new seed.

Use CPU with four torch/OMP/MKL threads. The qualified feature report SHA-256 is
`6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
Evaluation is validation, natural prevalence, `margin_v2` plus the probability
diagnostic. Official-test classifier performance must remain unseen.

## Bounds and execution qualification

Each child has a 300-second wall limit; combined child execution has a 600-second
limit. Parent preparation/source checking is measured separately and is not
charged to the child budget. A cell with no remaining child budget is retained as
`not_run`, not omitted. A timeout is recorded with any start proof/log/artifacts.
Do not restart solely because observation of a live child temporarily fails.

Each completed record must pass the [core execution checks](core-execution-qualification.md)
with the pinned cache, 80/80 cohort and validation identity; model parameter count
must match its declared cell. All utility metrics must be finite and in range,
but there is **no utility threshold or best-head selection** for this short probe.
Peak child RSS above 2 GiB disqualifies resource readiness; keep the record and the
reason. This is a measured post-run threshold, not an OS memory isolation limit.
Training loss may be poor after only fourteen nominal image passes.

Retain all four planned cells as qualified, failed, timed out or not run. Continue
after a cell-level failure while the budget permits. Source/import mismatch stops
further execution and marks remaining cells not run. Never select successful
survivors and describe the whole four-head sweep as qualified.

## Provenance and outputs

The runner and this protocol must be committed before execution. Refuse dirty or
foreign source/imports. Store exact resolved configurations and hashes, protocol
digest and source fingerprints before the first child; verify imported source
before engine access and after return, retaining separate atomic start/end proofs.
Retain complete round records and failed postflight records, logs, child/control
timing, per-child RSS/scope, parameter count, communication estimates, analytic
and measured FLOPs, and stage timing. Do not overwrite previous pilot directories.

Report measured end-to-end child and engine times separately. Engine execution
includes evaluation and diagnostic/counterfactual computations; it is not a pure
training-kernel benchmark. Model-payload estimates are not physical traffic.
RSS is a child-process peak, not whole-machine memory. No phone, Jetson training
or physical network/battery claim follows from this simulation.

If reporting a twenty-round planning estimate, label it as an extrapolation from
two rounds, separate process startup/control overhead, and state that convergence,
thermal behavior and longer-run resource stability were not tested. Failure to
qualify any head triggers an explicit scope/budget decision before that head enters
the confirmatory protocol; do not quietly remove its failed cell.

## Status

Protocol prepared before execution. Runner implementation and all four real-data
cells remain pending. The final confirmatory hypotheses, matched cells/seeds,
uncertainty method and test-release rules must be frozen separately afterward.
