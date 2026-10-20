# Privacy and secure-aggregation readiness for the follow-up

20 October 2026; source inspected at `23c213e`. The stronger disease baseline remains
first priority, and the condition is not yet selected. This audit checks which
mechanisms the later study can actually compare. It does not launch a new NIH
experiment, change the completed 360-cell study or establish a private deployment.

## Verdict

Central DP is qualified in the existing NIH research path. Local DP exists in the
generic simulation engine but is deliberately excluded from that qualified path
and the dashboard worker. Secure aggregation exists for DeComFL scalar updates,
not the NIH head simulator. These are different readiness levels, not four ready
checkboxes that can be combined arbitrarily.

| Comparison | Implemented path | Qualified NIH/Lab status | Work needed before the follow-up |
| --- | --- | --- | --- |
| Matched no-DP control | Clipped uniform mean or coordinate trimming through `ComposedStrategy` | Supported for the historical any-finding workload | Qualify the new condition, representation and learning recipe first |
| Central DP | Server clips individual updates, reduces, then adds sensitivity-calibrated Gaussian noise | Supported for mean/trimming, fixed accepted cohort, replace-one adjacency | Requalify new head/clip settings and prospectively declare the release/trust boundary |
| Local DP | Honest clients clip their own update, add Gaussian noise, then submit it to the simulator | Generic engine only; core and Lab reject it | Versioned local-DP qualification, client/server noise reporting and clean controls |
| Secure aggregation | Encrypted share relay and masked sums on DeComFL's scalar RPC channel | Not wired into the head simulator or Lab | Choose and qualify an explicitly matched secure/plain learning path |
| Secure aggregation plus ordinary trimming | Trimming needs individual-coordinate ordering; the existing secure path reveals a sum | No implementation in this path | Mark unavailable; a specialized secure-computation protocol would be different work |
| Secure aggregation plus DP | Conceptually composable under declared assumptions | No qualified combined NIH path | Define clipping enforcement, noise location, collusion/dropout assumptions and all released information |

## What source and fresh controls establish

- `simulation/engine.py:130` calibrates a Gaussian multiplier with `q=1` and the
  declared rounds. It does not claim subsampling amplification in this path.
  `ComposedStrategy._pipeline` applies joint clipping before reduction, with
  rule-specific replace-one sensitivity from `privacy/dp_mechanism.py:49`.
  Mean uses `2C/n`; trimming uses `2C/(n−2 floor(trim_ratio*n))`.
- `privacy/local_dp.py:16` clips an honest client's delta to `C` and adds noise of
  coordinate SD `2zC`. The engine applies this before upload, exempts malicious
  clients from honest randomization, and expands the server clipping threshold
  to `C + 2zC(sqrt(d)+4)`. The second clip is post-processing, but changes the
  numerical/attack setting; it is not the same server clip as the no-DP control.
- `simulation/core_qualification.py:35` explicitly permits only `none` and `central`.
  `simulation/lab_contract.py:26` similarly rejects `privacy=local`. Direct checks
  returned the expected controlled errors without opening NIH features. A generic
  `secure_aggregation` configuration key is rejected as unknown.
- `client/secure_agg_client.py:92` uses fresh secret mask/share seeds and encrypted
  share relay. `server/secure_agg_session.py:37` reconstructs an aggregate mask;
  `DeComFL.aggregate_secure` applies the recovered scalar sum. The backend service
  and Flyway V25 restrict the platform feature to DeComFL. Deterministic mask
  helpers used by primitive tests must not be mistaken for the live client path.
- The secure-client/key-registry documentation limits the threat model to an
  honest-but-curious relaying server. Server-side identity binding does not stop
  a malicious server substituting peer keys. Passing correctness controls is not
  a cryptographic review or an active-adversary guarantee.

The legacy `dp_aggregate` function uses a different `zC/n` convention. The qualified
capstone path uses the explicit replace-one `2C/n` mechanism above. Do not substitute
one function or its reported epsilon for the other without an adjacency audit.

## Confirmed local-noise reporting gap

In a three-round synthetic engine control, local DP used `C=0.5` and
`z=1.1957273033325988`, so client coordinate SD was `1.1957273033325988`.
The saved round field `privacy.noise_std` was nevertheless `0.0` in every round,
and `noise_to_signal` was null: these fields come from server-added noise, which
is absent in local mode. This is not evidence that local randomization was off.
It is a reporting gap that would mislead a comparison labelled simply “noise”.

Before qualifying local DP, retain explicit, versioned client-noise SD and
server-noise SD, the client clip, server post-processing clip, protected unit and
accounted release count. Keep existing record semantics and historical artifacts;
do not silently reinterpret old zeros. The existing central-only NIH study is
unaffected by this local-mode observation.

For a uniform mean of `n` honest clients with independent Gaussian components, before
server re-clipping, the local-noise contribution has coordinate SD `2zC/sqrt(n)`;
central mean noise has SD `2zC/n`. Their ratio is `sqrt(n)`, holding `z,C` fixed.
This arithmetic explains a possible utility cost, not a prediction of AUROC or a
fairness claim about equal epsilon across different trust models. It is not the
post-clipping/trimming distribution, and changing the server clip also changes
what attacks are permitted numerically.

## Study decisions to make after the baseline gate

1. Keep matched data, model, patient allocation, initialization and exposure
   controls. Changing from FedAvg heads to DeComFL and calling the difference
   “secure-aggregation cost” would confound the learning algorithm with masking.
   Either qualify a matched DeComFL secure/plain pair or build and qualify the
   intended head-update secure path. Do not select a path solely because a small
   protocol fixture already passes.
2. Declare who is trusted and who sees each release. Server-added central noise
   protects an output boundary; adding it after secure-sum recovery still leaves
   the server seeing the unnoised aggregate. Protection from that server needs a
   different noise/trust construction. Local noise and secure aggregation protect
   different views. [NIST threat-model explanation](https://www.nist.gov/blogs/cybersecurity-insights/threat-models-differential-privacy).
3. Secure summation reveals an aggregate rather than the individual updates.
   Ordinary coordinate trimming cannot be performed from that sum alone. It also
   cannot inspect individual norms to enforce an honest client's clip bound;
   client-side clipping needs explicit honesty or verification assumptions.
   [Flower protocol documentation](https://flower.ai/docs/framework/explanation-ref-secure-aggregation-protocols.html),
   [LightSecAgg paper](https://arxiv.org/abs/2109.14236).
4. Compare secure/plain results using the same accepted cohort and an explicit
   quantized plaintext control. Retain reconstruction thresholds, refused rounds,
   quantization bounds, protocol bytes and actual crypto runtime. Modeled head
   payload alone does not measure encrypted shares or a real network deployment.
5. Keep raw-update oracle attacks separate from protected-output claims. The
   completed core allows own-data/own-update sign flip and label flip; existing
   IPM/ALIE stress-test access is broader. Declare adversary access and whether
   malicious clients can skip local noise before choosing heatmap cells.
6. Freeze a bounded comparison after the viable baseline. Display unavailable
   combinations explicitly. Report absolute utility and matched attack drops,
   not only whether an attack was defeated or whether epsilon is present.

## Research randomness and export boundary

The simulation records reproducible experiment seeds and raw diagnostics. Their
noise is not a protected production randomness source, and record export is not
DP-safe. The earlier [release-boundary audit](privacy-output-boundary-audit.md) and
[matched-noise analysis](matched-noise-release-boundary.md) remain applicable.
A production claim needs independently secret per-release randomness, joint or
conditional accounting and a reviewed output boundary. No patient-level or
whole-matrix release qualification is established here.

## Verification and reproduction

Fresh focused run: **218 passed in 33.19 seconds**, with thread exceptions treated
as errors and no coverage claim for this selected subset. Controls cover accountant
oracles, sensitivity/noise arithmetic, generic local-DP execution, core restrictions,
secure clients, threshold/dropout handling and secure/plain DeComFL equivalence.
They use synthetic/in-process fixtures, not NIH disease outcomes, physical-device
qualification or measured deployment security/performance.

From `framework/`, with its development dependencies installed:

```sh
PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q --no-cov \
  -W error::pytest.PytestUnhandledThreadExceptionWarning \
  tests/test_local_dp.py tests/test_dp_accountant.py \
  tests/test_dp_robust_aggregate.py tests/test_composed_strategy.py \
  tests/test_simulation_engine.py tests/test_simulation_core_qualification.py \
  tests/test_secure_agg_client.py tests/test_secure_round_loop.py \
  tests/test_secure_round_deadline.py tests/test_secure_agg_server_config.py \
  tests/test_decomfl_secure_aggregation.py
```

The reporting observation can be repeated on the existing synthetic fixture, with
`PYTHONPATH=src` from `framework/`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from fedlearn.simulation.engine import run
from tests.test_simulation_engine import _cfg, _features

with TemporaryDirectory() as directory:
    record = run(_cfg(_features(Path(directory)), privacy={"mode": "local", "epsilon": 8.0}))
    z = record["meta"]["noise_multiplier"]
    c = record["meta"]["config"]["server"]["clip_norm"]
    print("Client coordinate SD:", 2 * z * c)
    print("Saved server-noise fields:", [row["privacy"]["noise_std"] for row in record["rounds"]])
```

Private receipts retain source hashes and the explicit policy/reporting checks.
Test log SHA256: `faaf53c28b3e2f287b2838598214d8a3f416b1275642de23df8ea700236465cf`.
Readiness-check receipt SHA256:
`1e6e1d49e86800933813efa33f7c7153c23e3223a328c6608cece87f53175f6d`.
No mechanism, training algorithm, old record or protected-export policy was changed.
