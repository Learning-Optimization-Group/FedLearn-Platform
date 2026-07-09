# FR-13 — central-DP ε-vs-accuracy benchmark (FedLoRA)

Task: **SST-2 (GLUE) sentiment, held-out validation accuracy** · Model: **tiny-Qwen2 (h=256, 4 layers, from-scratch)** · Aggregation: **FFA_LORA**
Rounds: 10 · Clients: 8 · Train subset: 4000 · Clip S: 6.0 · δ: 1e-05 · seed: 1234 (dp_seed 777)
torch 2.9.1 · total 1694.1s

Everything except the privacy setting is fixed and seeded, so accuracy differences are the
effect of the DP noise alone. `accounted ε` is what the RDP accountant certifies for the
solved noise multiplier z (compare against the requested target ε).

| setting | target ε | accounted ε | z | noise std/coord | utility SNR | final acc | best acc |
|---|---|---|---|---|---|---|---|
| no-DP baseline | — | — | — | — | — | 0.5791 | 0.5826 |
| ε=8 | 8 | 8.000 | 2.183 | 1.6373 | 0.023 | 0.4920 | 0.5275 |
| ε=4 | 4 | 4.000 | 4.099 | 3.0739 | 0.012 | 0.4874 | 0.5401 |
| ε=1 | 1 | 1.000 | 15.500 | 11.6250 | 0.003 | 0.4920 | 0.5367 |

Adapter dimension d = **26112** aggregatable coords (√d ≈ 162); round-0 median per-client delta L2 ≈ 5.42911 (vs clip S = 6.0).

## What this shows

**The mechanism + accountant are validated end-to-end**: DP solves a noise multiplier z from
each target ε, the RDP accountant certifies the accounted ε back to the requested budget, and
the clip→uniform-average→Gaussian path runs on the real FedLoRA recipe.

**Utility collapses across all tested ε at this scale — and that is the honest, expected
result, not a tuning miss.** The clipped aggregate has L2 norm ≤ S spread over d = 26112 coords,
so the per-coordinate signal (~S/√d ≈ S/162) is far below the DP noise floor (z·S/N).
The utility SNR = N/(z·√d) is **independent of S** (the clip cancels), so no clip tuning helps;
with few clients it is ≪ 1 for every ε, so the noise swamps the signal. A usable privacy–utility
gradient needs the SNR near 1 — i.e. **many more clients** (N ≈ √d ≈ 162, so noise/N ≈ signal), client **subsampling** amplification (a large enrolled
population sampled per round shrinks z), or a **lower-dimensional adapter**. This is the
well-documented high-dimension / small-cohort DP-FL tension (see the roadmap's DP-utility risk
note); the benchmark measures it rather than hiding it.

Reproduce: `PYTHONPATH=src python benchmarks/dp_epsilon_accuracy.py --hidden 256 --layers 4 --rounds 10 --clients 8 --subset 4000 --epsilons 8,4,1 --clip 6.0`

