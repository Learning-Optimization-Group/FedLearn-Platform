# FR-12 #2 — Byzantine-robustness accuracy benchmark (median/trimmed-mean vs FedAvg)

Task: **4-class Gaussian clusters in R^20 (sep=6.0, sigma=1.0)** · Model: **MLP: Linear(20,64)->ReLU->Linear(64,4)**
Clients: 10 (non-IID Dirichlet alpha=0.5, sizes [555, 524, 420, 272, 104, 176, 180, 259, 218, 292]) · Rounds: 25 · local epochs: 2 · lr: 0.01 · attack scale: x-10 · seed: 1234
torch 2.9.1 · total 14.2s

Everything except the aggregator and the attack is fixed and seeded — same data partition,
same model init, same per-client local training — so accuracy differences are the effect of
the aggregator's response to the attack alone. `clip_norm` is left OFF for these headline
numbers so the result isolates the estimator (median / trimmed-mean), not a clipping assist.

## Headline

| configuration | attackers (by count) | attacker weight share | final acc | best acc | retention vs clean |
|---|---|---|---|---|---|
| clean baseline (FedAvg, no attack) | 0 | — | 1.0000 | 1.0000 | — |
| FedAvg, f=0.2 attackers | 2/10 (x-10) | 36.0% | 0.0370 | 0.1980 | 3.7% |
| trimmed-mean (beta=0.2), f=0.2 | 2/10 (x-10) | 36.0% | 1.0000 | 1.0000 | 100.0% |
| median, f=0.2 | 2/10 (x-10) | 36.0% | 1.0000 | 1.0000 | 100.0% |

Round-0 mechanism check: median honest upload delta L2 ≈ 3.2159, median attacker upload delta L2 ≈ 43.7825 — the attack is genuinely ~13.6x the honest signal magnitude, not a token perturbation.

**Attacker-selection disclosure**: the Byzantine set is the deterministic lowest-numbered
client ids (nested across f — never chosen by data volume). Because the split is non-IID
(Dirichlet alpha=0.5), client sizes vary a lot, so a client-COUNT fraction f
does not equal the same WEIGHT fraction under FedAvg's num_examples-weighted mean — the
'attacker weight share' column above reports the real weighted mass so FedAvg's collapse
isn't read as worse than it is without that context (RobustAggregator is unweighted by
design, so weight share does not affect the median/trimmed-mean rows).

## f-sweep (breakdown point)

| aggregator | f (by count) | attackers | attacker weight share | final acc | retention vs clean |
|---|---|---|---|---|---|
| fedavg | 0.1 | 1/10 | 18.5% | 0.2500 | 25.0% |
| fedavg | 0.2 | 2/10 | 36.0% | 0.0370 | 3.7% |
| fedavg | 0.3 | 3/10 | 50.0% | 0.0080 | 0.8% |
| trimmed_mean | 0.1 | 1/10 | 18.5% | 1.0000 | 100.0% |
| trimmed_mean | 0.2 | 2/10 | 36.0% | 1.0000 | 100.0% |
| trimmed_mean | 0.3 | 3/10 | 50.0% | 0.0590 | 5.9% |

## What this shows

**The effect is clean.** The clean FedAvg baseline reaches **100.0%** held-out
accuracy (the task is learnable). Under the 20% gradient-scaling
attack (x-10), plain FedAvg collapses to **3.7%**
(retention 3.7%), while trimmed-mean (beta=0.2) holds at **100.0%** (retention 100.0%) and coordinate-wise
median holds at **100.0%** (retention 100.0%).

**Breakdown point**: trimmed-mean (beta=0.2) is only proven to tolerate a
Byzantine fraction <= beta. The f-sweep above runs f in [0.1, 0.2, 0.3] for both FedAvg and
trimmed-mean at the same beta: FedAvg degrades at every tested f (it has no tolerance), while
trimmed-mean is expected to hold while f <= beta and degrade once f exceeds beta — read the
sweep table's own numbers above for whether that boundary shows up cleanly at this N and attack
scale (small-cohort estimators are noisier near the exact breakdown point than the asymptotic
theory predicts).

Reproduce: `PYTHONPATH=src python benchmarks/robust_aggregation_attack.py --rounds 25 --clients 10 --attack-scale -10 --attack-fraction 0.2 --trim-beta 0.2 --sweep-fractions 0.1,0.2,0.3`

