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
