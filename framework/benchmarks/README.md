# FR-13 — central-DP ε-vs-accuracy benchmark

`dp_epsilon_accuracy.py` measures the privacy–utility trade-off of the FedLoRA central-DP mechanism
(`fedlearn/privacy/dp_mechanism.py` clip → uniform-average → Gaussian on the adapter-B/head keys,
calibrated by the RDP accountant in `fedlearn/privacy/dp_accountant.py`). It closes FR-13 done-when
#3: a committed ε-vs-accuracy benchmark over ≥3 ε on the LoRA recipe.

## What it does

Runs the **real** in-process federated-LoRA loop (the same one exercised by
`tests/test_fedlora_e2e.py`) — a tiny from-scratch Qwen2 sequence classifier + LoRA adapters on an
SST-2 subset — at a no-DP baseline plus several target-ε budgets. Everything except the privacy
setting is fixed and deterministically seeded (same data partition, same initial adapter, same
per-client local training), so any accuracy difference is the effect of the DP noise alone. Each run
records the held-out accuracy, the solved noise multiplier `z`, and the **accounted** (ε, δ) the RDP
accountant certifies for that `z` (compare against the requested target ε).

## Run

```bash
cd framework
PYTHONPATH=src python benchmarks/dp_epsilon_accuracy.py \
    --hidden 256 --layers 4 --subset 4000 --rounds 10 --clients 8 \
    --local-epochs 3 --lr 3e-3 --clip 6.0 --epsilons 8,4,1
```

Artifacts land in `benchmarks/results/dp_epsilon_accuracy.{json,md}`. It is CPU-only and intentionally
small (finishes in ~30 min on a laptop); scale it up with the flags. The DP noise generator is
seeded from the explicit `--dp-seed` here so the benchmark is reproducible — in production
(`dp_seed=None`) it draws fresh OS entropy, independent of the run seed, per the FR-13 privacy fix
(`strategy.py`; do not confuse the benchmark control with the production path).

## The result (and why it is honest, not a tuning miss)

The mechanism and accountant are **validated end-to-end**: DP solves `z` from each target ε, the RDP
accountant certifies the accounted ε back to the requested budget, and the clip→average→noise path
runs on the real recipe.

Utility **collapses across all tested ε at this small-cohort scale**, and the report quantifies why
rather than hiding it. In the committed run the clipped aggregate has L2 norm ≤ S spread over
`d = 26112` aggregatable coordinates (√d ≈ 162), so the per-coordinate signal (~`S/√d`) sits far
below the DP noise floor (`z·S/N`) — measured utility SNR `N/(z·√d)` = 0.023 / 0.012 / 0.003 at
ε = 8 / 4 / 1, all ≪ 1, so held-out accuracy sits at chance (~0.49) for every ε while the no-DP
baseline reaches ~0.58. The SNR is **independent of the clip S** (it cancels), so no clip tuning
helps. A usable privacy–utility gradient needs the SNR near 1 — i.e. **many more clients**
(`N ≈ √d ≈ 162`, so `noise/N ≈ signal`), client **subsampling** amplification (a large enrolled
population sampled per round shrinks `z`), or a **lower-dimensional adapter**. The accountant
certifies the accounted ε back to the requested budget exactly (8.000 / 4.000 / 1.000).

This is the well-documented high-dimension / small-cohort DP-FL tension (the roadmap's own DP-utility
risk note). The benchmark measures the constraint honestly, on the real mechanism, instead of
reporting a hand-tuned curve. See `results/dp_epsilon_accuracy.md` for the measured table.

# FR-12 #2 — Byzantine-robustness accuracy benchmark

`robust_aggregation_attack.py` measures whether the real `RobustAggregator` (coordinate-wise median
/ beta-trimmed-mean, `fedlearn/server/robust_aggregation.py`) retains held-out accuracy under a
gradient-scaling attack where the real `FedAvg` strategy (`fedlearn/server/strategy.py`) collapses.
It closes FR-12 done-when #2: a committed accuracy benchmark under a 20% gradient-scaling attack on a
non-IID split, with numbers published.

## What it does

Runs a small, self-contained, deterministic 4-class Gaussian-cluster classification task in R^20
(a tiny MLP), federated non-IID (Dirichlet split, alpha=0.5, via the same `_dirichlet_indices` helper
`recipes.py` uses) across 10 clients. Every configuration shares the same seeded data partition,
model init, and per-client local training — only the aggregator and the attack vary. A Byzantine
fraction `f` of clients replace their honest upload with their own delta-from-global scaled by a
large **negative** factor (default -10x): this sign is deliberate (see the module docstring) — a
same-direction amplification of an honest client's own gradient is not actually adversarial on a
well-separated task, it just overshoots; the literature's "gradient/large-deviation scaling" Byzantine
attack (Yin et al. 2018; Xie et al. 2019; Fang et al. 2020) is adversarial because it pushes the wrong
direction at large magnitude. Both strategies are driven through their real `aggregate_fit`, no
aggregation math is reimplemented.

## Run

```bash
cd framework
PYTHONPATH=src python benchmarks/robust_aggregation_attack.py \
    --rounds 25 --clients 10 --attack-scale -10 --attack-fraction 0.2 \
    --trim-beta 0.2 --sweep-fractions 0.1,0.2,0.3
```

Artifacts land in `benchmarks/results/robust_aggregation_attack.{json,md}`. CPU-only, finishes in well
under a minute at the defaults.

## The result (and the honesty caveat)

The committed run is a clean effect: the clean FedAvg baseline reaches 100% held-out accuracy (the
task is learnable); under a 20%-of-clients gradient-scaling attack, FedAvg collapses to 3.7% while
trimmed-mean (beta=0.2) and coordinate-wise median both hold at 100% (retention 100%). The f-sweep
shows the textbook breakdown pattern: trimmed-mean(beta=0.2) holds at f=0.1 and f=0.2 (at/below its
proven tolerance) and degrades at f=0.3 (past it), while FedAvg has no tolerance at any tested f.

Honesty caveat published alongside the numbers: the deterministic (lowest-client-id) Byzantine set,
combined with the non-IID Dirichlet split's unequal client sizes, means a 20%-of-clients attack is
also a ~36%-of-weighted-mass attack under FedAvg's num_examples weighting (RobustAggregator is
unweighted by design, so this does not affect its rows). The report's "attacker weight share" column
makes this explicit rather than letting a "20%" headline imply a smaller weighted share than what was
actually run. See `results/robust_aggregation_attack.md` for the full measured tables.

## `--matrix` — multi-attack ablation (attack family x aggregator)

The same harness also supports a family of Byzantine attacks, selectable via `--attack` for a single
run, or all five at once via `--matrix`:

- `sign_flip_scale` — the baseline attack above (`-10x` an honest delta).
- `same_dir_scale` — control (`+10x`); expected NOT adversarial, confirms the harness responds to
  attack *direction*, not just magnitude.
- `label_flip` — attackers retrain locally on label-permuted data (`y -> num_classes-1-y`; Tolpegin
  et al. 2020), then upload the result honestly (pure data poisoning, no post-hoc scaling).
- `ipm` — inner-product manipulation (Xie et al. 2019, "Fall of Empires"): colluding attackers upload
  the identical `-epsilon * mean(honest deltas)`.
- `alie` — "A Little Is Enough" (Baruch et al. 2019): colluding attackers upload the identical
  `mean(honest deltas) - z * std(honest deltas)`, designed to stay embedded in the honest per-
  coordinate range and survive median/trimmed-mean.

```bash
cd framework
PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --matrix \
    --rounds 25 --clients 10 --attack-fraction 0.2 --trim-beta 0.2
```

Artifacts land in `benchmarks/results/robust_aggregation_multiattack.{json,md}`, with a retention
matrix (rows = attacks, cols = FedAvg / trimmed-mean / median) plus per-attack commentary.

**The result, including the honesty-critical one**: trimmed-mean and median retain 100% against
every attack tested here — `sign_flip_scale`, `label_flip` (FedAvg retains 77.4%, a genuine but
partial poisoning), `ipm` (FedAvg collapses to 0.1%), and a calibrated `alie` (FedAvg collapses to
19.3%). The `alie` result needs unpacking rather than taken at face value: the textbook tail-
probability z* is negligible against undefended FedAvg at this benchmark's scale (FedAvg retains
≥99% up to z=10), so a pre-registered calibration ladder (tested against undefended FedAvg *only*,
before ever running the robust aggregators) found z=20 as the smallest value with material effect —
but at that z the corrupted upload's L2 norm is ~13x the honest median, i.e. it stops looking
"embedded" and becomes a plain magnitude/rank outlier that both estimators reject just like the
gross attacks. This is a genuine property of this benchmark's low inter-client gradient variance
(a well-separated, near-noiseless synthetic task), **not** evidence that median/trimmed-mean defeat
ALIE in general — the attack's stealth premise needs a regime with real inter-client heterogeneity
to have budget to exploit, which a real non-IID dataset would exercise better than this synthetic
one — a real non-IID dataset is the natural next experiment. See the generated
`results/robust_aggregation_multiattack.md` (regenerated by `--matrix`) for the full disclosure,
including the calibration ladder and every honesty caveat.
