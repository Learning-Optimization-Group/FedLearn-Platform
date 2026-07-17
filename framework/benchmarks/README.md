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

## FR-12 #3 — MEASURED breakdown point (`robust_breakdown_point.py`)

The ablation above defends a single sub-breakdown fraction (f=0.2). This benchmark holds the attack
fixed and **sweeps** the actual Byzantine fraction f from 0 to 0.5, measuring held-out-accuracy
retention at each f, to LOCATE where each aggregator fails and compare against the classical bounds
(Yin et al. 2018): FedAvg → 0⁺, β-trimmed-mean → f>β, median → f≥0.5. Same seeded partition/init as
the ablation; only f moves; attacker ids nested across fractions; `clip_norm` OFF.

```
PYTHONPATH=src python benchmarks/robust_breakdown_point.py                    # synthetic, ipm, f in {0..0.5}
PYTHONPATH=src python benchmarks/robust_breakdown_point.py --dataset digits   # REAL sklearn 8x8 digits, no download
```
Artifacts: `results/robust_breakdown_point.{json,md}` (synthetic) / `..._digits.{json,md}` (~20–33s CPU).

Reports TWO metrics per f: **accuracy retention** (practical) and **estimator deviation**
`‖agg(all) − agg(honest-only)‖ / ‖agg(honest-only) − global‖` (the quantity the classical breakdown is
*defined on*).

**The result (honest, incl. the gap):** FedAvg collapses at the first non-zero fraction (f=0.1) — both
metrics agree, breakdown 0⁺ (deviation jumps 0→1.6 and stays ~1.4). median and trimmed-mean(β=0.2) both
hold **100% accuracy through f=0.3**, degrade at f=0.4 (81%/73%), collapse at the **majority threshold
0.5**. The honesty-critical finding is where the two metrics DISAGREE for trimmed-mean: accuracy holds
past β (→ would read breakdown ~0.5), but the **estimator deviation stays small for f≤β (0.33 at f=0.2)
and roughly doubles just past β (0.74 at f=0.3)** — the β onset the theory predicts IS visible in the
estimator and is exactly what the forgiving accuracy metric hides. Below-breakdown corruption is bounded
(as theory guarantees) and too small to move the decision boundary until the attacker share nears a
majority. Sharpness is attack-dependent: the default ipm onset is gradual, but `--attack sign_flip_scale` / `alie`
make trimmed-mean's β onset near-vertical (~0.3 at β → ~2.4–3.0 just past it) and median's 0.5 onset
sharp — the β/0.5/0⁺ structure reproduces across the strong-attack families (a weak `label_flip` is too
mild to reach any breakdown). Reproduced ordering FedAvg(0⁺) < trimmed-mean(β) < median(0.5). See
`results/robust_breakdown_point.md`.

**On REAL data (`--dataset digits`, ~94.5% clean) the accuracy-vs-estimate gap CLOSES:** trimmed-mean's
*accuracy* collapses sharply at f=0.3 (97.9% → 7.4%, just past β) — tracking the estimator, unlike the
synthetic task where accuracy held to 0.5. A non-separable decision boundary is fragile enough that the
residual post-trim corruption past β collapses accuracy directly. So the accuracy-vs-estimate gap is a
property of the *separable* synthetic task; the estimator breakdown (β/0.5/0⁺) is the task-invariant
signal. FedAvg 0⁺ and median 0.5 hold on real data too. See `results/robust_breakdown_point_digits.md`.

## TE-15 — fair algorithm comparison (`algo_comparison.py`)

Same task · same fixed non-IID partition · same seed through each algorithm's real `aggregate_fit`,
recording per round: test accuracy, loss, wall-clock, and **truthful cumulative wire bytes**
(`wire_bytes.py` measures the real serialized payloads — the one axis the platform never measured).

```bash
# The professor's table on MNIST (first-order family), ~29k-param small CNN:
python benchmarks/algo_comparison.py --task mnist --algos fedavg,fedprox,fedopt \
  --rounds 100 --clients 8 --alpha 1.0 --lr 0.03 --target 0.9 \
  --out benchmarks/results/algo_comparison
# -> benchmarks/results/algo_comparison.{json,md}: rounds-to-90%, bytes-to-90%, final acc, and the
#    DeComFL byte projection (seeds+scalars are ~100x smaller/round than a full-model upload).
```

**DeComFL convergence** (zeroth-order, a different client loop) is produced over thousands of rounds
by the existing gRPC harness (`run_full_test_suite.py` Test 3); its per-round byte cost is projected
analytically here. ZO plateauing below first-order at fixed rounds is **expected** (variance ∝ d/P):
shrink d, raise P on a schedule, run thousands of rounds — see `docs/artifacts` benchmark brief.

# DA-14 — frozen-backbone derivation benchmarks (head-only federation)

Three seeded, deterministic benchmarks for the DA-11 trainable-subset contract (a frozen backbone
is shared across peers; only the head is federated). Communication + byte numbers are REAL
(production `wire_bytes` codecs); the utility tasks are seeded-synthetic separable targets
(disclosed — the mechanism/trade-off is real, the accuracy is by construction).

- **`frozen_backbone_fl.py`** — communication + utility of head-only FedAvg. Head-vs-full-model wire
  bytes (`15.8x → 77x → 102x`, then **down to 41.9x** once the head itself is large — an honest
  non-monotonicity: the win is backbone-size vs head-size) + a converging head-only run
  (`0.17 → 0.72`, frozen backbone byte-identical throughout).
  Run: `PYTHONPATH=src python benchmarks/frozen_backbone_fl.py --rounds 15 --clients 3`.
- **`dp_on_head.py`** — central-DP on a small head vs the FR-13 high-dimension collapse. Same real
  DP mechanism + RDP accountant as FR-13, on the head (`d=99` vs FedLoRA's `d=26112` → **16.24× SNR**).
  The small head **escapes** the collapse: ε=1/4/8 retain 98–100% where FedLoRA went to chance;
  breaks only at ε=0.1. The positive privacy–utility curve complementing FR-13's negative result.
  Run: `PYTHONPATH=src python benchmarks/dp_on_head.py`.
- **`comms_regimes.py`** — the three wire regimes in one table: full-model FedAvg vs head-only FedAvg
  vs DeComFL, real codecs. At an 8192→1024→10 model: full 33.6 MB, head 41 KB, DeComFL 986 B
  (`full/head=816x`, `full/decomfl=34,077x`); DeComFL upload is O(K·P), **constant across model size**
  (its one-shot O(d) initial download is reported separately). Unifies TE-15 + the C8 head-only win.
  Run: `PYTHONPATH=src python benchmarks/comms_regimes.py`.
- **`dp_subsampling_amplification.py`** — the second DP privacy lever on the head: Poisson client
  subsampling q<1 via the accountant's subsampled RDP. Fixed z, sweep q → certified ε tightens
  **6.9×**; fixed ε, sweep q → solved z drops **3.7×** (less noise → more utility). Stacks with the
  small-d head. Run: `PYTHONPATH=src python benchmarks/dp_subsampling_amplification.py`.
- **`dp_head_cohort_sweep.py`** — the cohort-size complement of `dp_on_head`: fix ε/d, sweep N. On the
  d=99 head the SNR=N/(z·√d)=1 crossing is at **N≈36.5** (laptop-reachable) vs **N≈592** for FedLoRA
  d=26112. Run: `PYTHONPATH=src python benchmarks/dp_head_cohort_sweep.py`.

Together with `dp_epsilon_accuracy.py` (FR-13 collapse) and `dp_on_head.py`, these map the full DP-FL
privacy–utility space around `SNR = N/(z·√d)` and its three knobs (shrink d / grow N / subsample) —
synthesized in `research/notes/2026-07-17-dp-fl-privacy-utility-story.md`.
