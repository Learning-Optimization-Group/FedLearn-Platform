# Experiment log

Chronological log of paper-relevant experiments (honest numbers, seeds, caveats). Newest first.

## 2026-07-09 — FR-12 Byzantine-robustness under a gradient-scaling attack (C3)

- **Harness**: `framework/benchmarks/robust_aggregation_attack.py` (seeded, committed). Non-IID Dirichlet(α=0.5) synthetic 4-class classification, tiny MLP, N=10 clients, 25 rounds, driving the **real** `FedAvg` + `RobustAggregator` strategies.
- **Attack**: Byzantine fraction f of clients upload `global + attack_scale·(client−global)` with `attack_scale=−10` (sign-flipped ×10). Round-0 check: attacker upload L2 ≈ 43.8 vs honest ≈ 3.2 (~13.6× — a real attack, not a token perturbation).
- **Result (independently re-run + mechanism-verified)**:
  - clean FedAvg (no attack): **100%**
  - FedAvg, f=0.2: **3.7%** (collapse; attacker weighted mass 36%)
  - trimmed-mean(β=0.2), f=0.2: **100%**; median, f=0.2: **100%**
  - f-sweep: FedAvg 0.1/0.2/0.3 → 25%/3.7%/0.8%; trimmed-mean 0.1/0.2/0.3 → 100%/100%/**5.9%** (breakdown once f>β).
- **Caveats (must survive to the paper)**: sign-flip attack choice (disclosed; +10× same-direction does not collapse a separable task); synthetic separable task so 100%s show the *contrast* not production accuracy; weight confound reported via an explicit column.
- **Status**: committed `ff494a1`. Next: real dataset + more attack types (see README TODO).

## 2026-07-06 — FR-13 central-DP ε-vs-accuracy on FedLoRA (C1/C2)

- **Harness**: `framework/benchmarks/dp_epsilon_accuracy.py` (seeded, committed). Real in-process federated-LoRA loop (tiny Qwen2 + LoRA on an SST-2 subset), no-DP baseline + ε∈{8,4,1}.
- **Result**: accountant certifies accounted ε back to budget to ~9 dp (8.000/4.000/1.000); utility **collapses to chance (~0.49)** at every ε vs ~0.58 no-DP baseline. Quantified: utility SNR `N/(z·√d)` = 0.023/0.012/0.003 (d=26112), clip-independent → collapse is fundamental at this scale, not a tuning miss.
- **Privacy correctness**: a regression audit caught DA-3's global `manual_seed` making the DP noise deterministic from the disclosed eval-card seed (strippable → voided (ε,δ)-DP); fixed with a dedicated fresh-entropy generator (`0b422b6`).
- **Caveat**: negative result on utility — a positive privacy–utility curve needs N≈√d / subsampling / lower-dim adapter (see README TODO).
- **Status**: committed `b70c2a8` (+ privacy fix `0b422b6`).
