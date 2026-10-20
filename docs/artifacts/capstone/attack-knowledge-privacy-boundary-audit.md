# Attack knowledge and the privacy-accounting boundary

19 October 2026. Code-backed audit before the confirmatory NIH protocol.
The clean [v3 baseline](nih-schedule-pilot-v3-report.md) is unaffected: it has no
attacks or DP noise. Historical utility records remain unchanged.

## Verdict

The clipped-aggregate sensitivity function is correct for replacing **one final
upload while the others remain fixed**. That premise is not preserved when the
simulation recomputes IPM or ALIE attackers from all honest clients' current raw
updates. Replacing one raw honest update can then change many attacker uploads.
Do not equate the existing one-upload noise calibration with an end-to-end
client-data privacy guarantee for that oracle-driven system.

This is separate from the [research-output audit](privacy-output-boundary-audit.md):
published noise seeds and raw diagnostics already prevent treating the complete
research record as a protected deployment export. Secret randomness and an output
allowlist alone would not repair the upstream oracle-dependence issue.

## Execution path and protected input

- `framework/src/fedlearn/simulation/behaviors.py`, `BehaviorPlan.upload_overrides`:
  IPM/ALIE calculate mean/std from `records[c]["honest_delta"]` over honest clients.
  These are raw pre-noise, pre-server-clipping deltas. There is an explicit
  attacker-own-data fallback only when no honest client exists.
- `framework/src/fedlearn/simulation/engine.py`, `round_hook`: local DP changes
  `uploads` for honest clients, but attack generation still reads the separate raw
  `records`. The oracle is not restricted to already-privatized uploads.
- `framework/src/fedlearn/server/composed_strategy.py`, `_pipeline`: server
  clipping bounds each final upload, then aggregation/noise uses
  `replace_one_sensitivity` from `privacy/dp_mechanism.py`.
- That function bounds one changed clipped input by `2C/n` for uniform mean and
  `2C/(n-2k)` for coordinate-wise trimmed mean. It does not claim a bound for
  replacing multiple inputs after a data-dependent preprocessing stage.

For an end-to-end honest-client data claim, fix cohort/assignment, the accepted
set/cardinality, other clients' data, attacker state and permitted prior public
transcript. The strategy rejects invalid/nonpositive-count submissions and the
engine can drop late submissions; merely fixing the selected cohort does not
establish a fixed accepted set. Audit that path and its sensitivity denominator.
In historical round JSON without a participation schema version, `participants.aggregated` was populated from
`len(submissions)` before the strategy's final validation, whereas the strategy's
own `aggregated` statistic counts accepted updates. Those counters must be
distinguished in the confirmatory execution checks; the headline alone does not
prove the actual denominator stayed fixed. The subsequent
[version 2 participation contract](participation-accounting-contract.md) separates
submission, admission, strategy acceptance and successful aggregation, and fixes
the simulator's handling of rejected admissions. Prospective runner enforcement
of a fixed accepted cohort is still required.
A current raw update
from the protected honest client must not secretly enter many additional input
slots without accounting for that dependence. Adapting to previously released
private models is different from reading current unprotected honest updates.

## Production-code upload-domain counterexample

The synthetic probe uses the actual `BehaviorPlan` and `ComposedStrategy.preview`;
no Gaussian sample is needed to measure the pre-noise query shift.

| Fixed item | Value |
| --- | --- |
| Clients / attackers | 80 / 16 (20%) |
| Aggregation / clipping | Uniform mean / joint L2 C=0.4 |
| IPM scale | 4 |
| Global parameter | One scalar, zero |
| All other honest and attacker-owned raw updates | Zero in both inputs |
| Only changed honest raw update | +40 versus −40 |

For each neighboring raw-update table, recompute the oracle. IPM sets all sixteen
attacker uploads to the negative scaled honest mean; after server clipping each
is −0.4 or +0.4. The changed honest upload clips to +0.4 or −0.4. Thus the two
pre-noise means are **−0.075 and +0.075**: an L2 shift **0.15**. The nominal
single-upload bound is **2×0.4/80 = 0.01**, a factor **15** smaller. One raw input
row changed, but **seventeen final uploads** changed.

The negative control freezes the sixteen attacker uploads across the two inputs.
Now only the honest final upload changes, and the observed shift is 0.0099999905,
matching 0.01 within float32 arithmetic. The aggregator bound is not the defect;
applying it through the oracle while retaining the same adjacency is unsupported.

This is a counterexample in the raw-client-update domain. It is **not** a measured
NIH patient-neighbor example, an estimate of practical information leakage, or a
claim that real v3 deltas attain those magnitudes. Clipping is meant to provide a
bound without relying on an unproved raw-update range; the current code provides
no end-to-end sensitivity proof for the oracle pipeline. This probe demonstrates
an IPM/uniform-mean violation of the single-upload bound on the unrestricted
raw-update domain. It does **not** demonstrate a corresponding bound violation
for trimmed mean or ALIE; their shared oracle dependence requires separate
analysis. Multiple changed slots alone do not prove a violation for every
aggregate, because cancellation and trimming can matter.

Minimal reproduction from this checkout's framework environment, with
`PYTHONPATH=framework/src:framework` set when starting Python (rather than an
editable installation pointing to a different checkout):

```python
from collections import OrderedDict
import torch
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.server.composed_strategy import ComposedStrategy

plan = BehaviorPlan([{"type": "ipm", "fraction": 0.2, "scale": 4.0}], 80, 20262019, 2)
changed = next(c for c in range(80) if c not in plan.malicious_ids())
zero = OrderedDict(w=torch.zeros(1))
server = ComposedStrategy(zero, rule="mean", clip_norm=0.4)

def uploads(value):
    records = {c: {"honest_delta": OrderedDict(w=torch.tensor([value if c == changed else 0.])),
                   "client_state": OrderedDict(w=torch.tensor([value if c == changed else 0.])),
                   "n_examples": 1} for c in range(80)}
    result = {c: r["client_state"] for c, r in records.items()}
    result.update(plan.upload_overrides(records, zero, 1))
    return result

def mean(values):
    return float(server.preview([(str(c), p, 1) for c, p in sorted(values.items())])["w"][0])

positive, negative = uploads(40.), uploads(-40.)
fixed = {c: negative[c] if c == changed else positive[c] for c in range(80)}
assert abs(mean(positive) - mean(negative)) > 0.149
assert abs(abs(mean(positive) - mean(fixed)) - 0.01) < 1e-7
```

## Relation to accounting

Gaussian/RDP calibration depends on the sensitivity of the query actually
released; adaptive composition requires each conditional mechanism to satisfy
its assumed bound. These are the relevant premises in Mironov's
[Gaussian mechanism and adaptive-composition results](https://arxiv.org/html/1702.07476)
(Proposition 1, Proposition 7 and Corollary 3). Those results do not supply a missing
bound for this project's oracle preprocessing.

For the scalar counterexample, at the same Gaussian standard deviation, the
squared mean-shift term is 225 times the nominal one-upload term. This does **not**
mean epsilon is simply 15 or 225 times larger, nor does one example compute the
system's full worst-case privacy parameter. It shows that reusing the quoted
single-upload RDP argument does not establish the intended raw-client claim.

## Consequences for phase two

1. **Use a permitted-information attack in the formal-accounting core.** Own-update
   sign flip or own-data label flipping can be evaluated with other clients'
   private inputs fixed and no access to current raw honest uploads. State the
   protected honest-client unit, fixed assignment, accepted set/cardinality and
   trusted central server. Conditional on fixed permitted history and external
   randomness, changing one protected client's data must affect at most that
   client's current accepted upload. Uploads need not be statistically independent;
   they can share public model history. Verify the entire execution/release path
   rather than treat the attack choice alone as a privacy guarantee.
2. **Keep oracle IPM/ALIE as separately labelled stress tests.** Their utility
   measurements remain informative. If Gaussian noise is included, describe the
   configured epsilon/noise as nominal calibration at the aggregate-input boundary,
   not a proven client-data guarantee through the oracle. Do not mix their privacy
   claims with the permitted-information core.
3. **If oracle-like adaptation is required, define its allowed observations.**
   Estimates from attacker-owned/public data or already-accounted private releases
   need explicit designs. Do not silently swap the algorithm or multiply noise by
   a guessed factor and claim the issue solved. An attacker given raw honest
   updates has already received information outside a model-only release boundary.
4. **Preserve history and disclose the limitation.** The 945 Milestone2 runs stay
   exploratory utility evidence. Their numerical values/configured budgets are
   not retroactively corrected or relabelled as new experiments; the applicable
   privacy claims are narrowed explicitly. Local-mode oracle attacks also read
   raw records despite honest upload noise, so the local-DP contrast needs this
   caveat in addition to its existing trust/geometry limitations.

No historical algorithm or result is changed by this audit. Before confirmatory
execution, the runner must enforce the declared attack-knowledge boundary and
retain mechanism versus complete-research-record privacy scopes separately.
