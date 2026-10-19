# Research records and the privacy release boundary

19 October 2026; inspected implementation at `4a8d72d`.

**Verdict:** the simulation JSON is a reproducible research record, not an
end-to-end differentially private export. A model accountant does not cover its
unnoised diagnostics or make published noise randomness safe for a later model
release. This distinction must be explicit in the phase-two protocol and GUI.

## Observed execution paths

| Output or behavior | Code evidence | Release implication |
| --- | --- | --- |
| Actual noise seed and resolved configuration are recorded | `simulation/engine.py`, `run`, metadata `config` and `seeds` (line 292) | A recipient can reproduce the research noise sequence. Do not expose this seed with a private-data model. |
| Default noise seed is derived from public experiment seed | `simulation/config.py`, `default_seeds` (lines 68–72) | Redacting only the `noise` key is insufficient: the base or other related seeds recover it. |
| Central noise uses a persistent seeded CPU generator | `server/composed_strategy.py`, constructor (line 121), `_pipeline` noise loop (lines 171–175) | Given the seed, state shapes/order and standard deviations, its samples can be reconstructed. |
| Local noise also uses a derived seeded generator | `simulation/engine.py`, `local_gen` (line 160), `round_hook` local-DP loop | The same seed-disclosure concern applies; its exact draw history also depends on honest participating clients. |
| Raw update summaries, including honest-update cosine | `simulation/engine.py`, `round_hook`, `updates` | These query uploads or honest unprivatized deltas, rather than only the released model. |
| Clip rate, mean shrink and noise-to-signal ratio | `simulation/engine.py`, `to_line`, `privacy`; `server/composed_strategy.py`, `_pipeline` | Clipping diagnostics and the pre-noise aggregate norm are not separately privatized. Placing them under `privacy` does not protect them. |
| Deviation and exclusion summaries | `simulation/engine.py`, `round_hook` previews and `to_line`; `composed_strategy.py`, `kept_share` | These reveal data-dependent information about pre-noise aggregates or accepted coordinates. They need their own release analysis. |
| Per-client training accuracy summaries | `simulation/workloads.py`, `per_client_accuracy` (lines 129–138) | Evaluating a privatized model on private training data accesses private data again; this is not solely model postprocessing. |
| Initial upload digest | `simulation/engine.py`, `round_hook`, `round0_digest` | A deterministic fingerprint of uploads is not a DP release just because it is hashed. |
| Validation/test metrics | `simulation/workloads.py`, `evaluate_fn` | Metrics on a declared public evaluation set can be model postprocessing; metrics on private evaluation records require an additional policy/analysis. |
| Participation, refusals, timings, sample-derived costs and counts | `simulation/engine.py`, round extras and metadata | No blanket DP claim: declare which are public/fixed assumptions and which depend on private data. |

The current `run` return value contains metadata and round metrics, **not raw model
parameters** (`simulation/engine.py`, line 310). Seed disclosure alone is therefore
not evidence that this JSON currently contains a recoverable aggregate. It is a
concrete blocker to treating the same research randomness as safe when publishing
the corresponding model, and the other raw fields already prevent a blanket
full-record privacy claim.

## Checked synthetic demonstration

A one-round check used four fixed valid submissions, three float32 coordinates,
uniform mean, joint clip norm 1, noise multiplier 1, no server optimizer, and seed
20263019. The strategy reported noise standard deviation 0.5, consistent with
replace-one sensitivity `2 × 1 / 4`. A separate CPU generator given that seed
reconstructed each Gaussian sample in state-key order. Subtracting those samples
from the released parameters recovered the noise-free preview with maximum
absolute error **1.4901161193847656e-08**.

This verifies a narrow arithmetic consequence in the actual strategy code. It
does not establish an attack against every optimizer, unknown participation
history or deployment, and it does not measure patient reidentification.

The check can be repeated with the framework installed:

```python
from collections import OrderedDict
import torch
from fedlearn.server.composed_strategy import ComposedStrategy

initial = OrderedDict(weight=torch.zeros(2), bias=torch.zeros(1))
updates = [(str(i), OrderedDict(weight=torch.tensor([0.1 + i * 0.01, 0.2]),
                                bias=torch.tensor([0.01])), 1) for i in range(4)]
strategy = ComposedStrategy(initial, rule="mean", clip_norm=1.0,
                            central_noise_multiplier=1.0, noise_seed=20263019)
expected = strategy.preview(updates)
released = strategy.aggregate_fit(1, updates)
generator = torch.Generator().manual_seed(20263019)
for key in initial:
    noise = torch.empty_like(released[key]).normal_(
        mean=0.0, std=strategy.last_round_stats["noise_std"], generator=generator)
    torch.testing.assert_close(released[key] - noise, expected[key], rtol=0, atol=1e-7)
```

## Required decisions before protocol and product export

1. Label retained public-dataset experiment outputs as research records with
   nominal mechanism/accounting settings. Retain their reproducible seeds and
   diagnostics; do not rewrite historical evidence or call those exports DP-safe.
2. State the intended release and trust boundary: simulated-client replace-one
   adjacency, trusted central aggregator for central DP, and the exact model
   trajectory/accounted release count. No patient-level or clinical guarantee.
3. For a future private-data deployment, design an independently secret noise
   source and a strict output allowlist. Public training/init seeds must not derive
   the noise seed. Merely hiding one JSON field does not fix the current derivation.
4. Exclude raw diagnostics, training-based fairness and upload fingerprints from
   that protected view unless separately justified or privatized/accounted. Declare
   the status of evaluation data, participation and operational side channels.
5. Test the protected export against the allowlist and seed derivation, and verify
   the actual deployed randomness/release path before making a production claim.

The capstone can still compare utility under fixed seeded research mechanisms.
Those empirical curves and the production release guarantee are different outputs.
The immediate project recommendation is to make the research/deployment boundary
visible, rather than equating the presence of an epsilon field with private logs.
