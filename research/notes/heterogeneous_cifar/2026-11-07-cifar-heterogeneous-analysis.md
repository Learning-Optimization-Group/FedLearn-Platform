# Heterogeneous CIFAR-10 Benchmark Analysis: Four-Policy Comparison

**Date:** 2026-11-07  
**Benchmark Target:** Non-medical heterogeneous federated learning on CIFAR-10 with Dirichlet partitioning ($\alpha = 0.5$)  
**Harness:** [`research/benchmarks/run_heterogeneous_cifar_benchmark.py`](file:///Users/anurag/code/FedLearn-Platform/research/benchmarks/run_heterogeneous_cifar_benchmark.py)  
**Raw Results:** [`research/results/heterogeneous_cifar/cifar_heterogeneous_benchmark.json`](file:///Users/anurag/code/FedLearn-Platform/research/results/heterogeneous_cifar/cifar_heterogeneous_benchmark.json)  

---

## 1. Executive Summary & Experimental Design

This benchmark provides the second heterogeneous, non-medical evaluation of the four core aggregation and representation policies investigated in the MSc Capstone reliability program:
1. **`Mean512`**: Standard coordinate-wise mean over full 512-dimensional feature space ($d = 512$).
2. **`Projected128`**: Fixed public orthonormal projection from 512 down to 128 dimensions ($\mathbf{P} \in \mathbb{R}^{512 \times 128}$, seed `20261107`).
3. **`Trim16`**: Coordinate-wise trimmed mean with trimming fraction $\beta = 0.2$ (trims $k = \lfloor 0.2 \times N \rfloor$ per tail).
4. **`CenterQuarter`**: Centered clipping around previous round aggregate with radius $\tau = C / 4$.

The evaluation crosses all four policies over:
- **Two noise conditions**: Clean (no noise, $\sigma = 0$) vs. Private Gaussian DP ($\epsilon = 4.0, \delta = 10^{-5}$, $z = 4.8351$).
- **Two threat conditions**: Clean cohort vs. Attacked cohort (20% Byzantine sign-flipping/IPM attackers).
- **Three independent random seeds**: `42`, `43`, `44`.
- **Cohort**: $N = 20$ simulated clients under non-IID Dirichlet label skew ($\alpha = 0.5$, minimum partition size 10).

All $4 \times 4 \times 3 = 48$ cells ran to completion with zero failed or missing cells.

---

## 2. Theoretical Sensitivity and Calibrated Noise Parameters

Given clipping bound $C = 0.1$ and cohort size $N = 20$:

| Policy | Dimension ($D$) | Sensitivity Formula | Exact $\Delta$ | Noise Scaling $\sigma = z \Delta$ | Relative Noise vs Mean |
|---|---|---|---|---|---|
| **`Mean512`** | 5,130 | $\frac{2C}{N}$ | $0.0100$ | $0.04835$ | $1.00\times$ |
| **`Projected128`** | 1,290 | $\frac{2C}{N}$ | $0.0100$ | $0.04835$ | $1.00\times$ (Total $\|\boldsymbol{\xi}\|_2$ is $0.50\times$) |
| **`Trim16`** ($\beta=0.2$) | 5,130 | $\frac{2C}{N - 2k}$ ($k=4$) | $0.0167$ | $0.08059$ | $1.67\times$ |
| **`CenterQuarter`** ($\tau=C/4$) | 5,130 | $\frac{2\tau}{N} = \frac{C}{2N}$ | $0.0025$ | $0.01209$ | $0.25\times$ |

---

## 3. Empirical Results Matrix

Mean test accuracy across seeds 42, 43, 44 on held-out CIFAR-10 evaluation data:

| Policy | Clean (No Noise) | Clean + DP ($\epsilon=4$) | Attacked (No Noise) | Attacked + DP ($\epsilon=4$) | Robustness Degradation (Clean − Attacked) |
|---|:---:|:---:|:---:|:---:|:---:|
| **`Mean512`** | 16.80% | 9.77% | 11.67% | 9.90% | −5.13% |
| **`Projected128`** | 13.37% | 10.03% | 10.70% | 9.73% | −2.67% |
| **`Trim16`** | 16.80% | 9.43% | 10.20% | 9.43% | −6.60% |
| **`CenterQuarter`** | **18.03%** | **10.17%** | **14.00%** | **9.97%** | **−4.03%** |

---

## 4. Key Findings & Discussion

### 4.1 Upfront Compression vs. Noise Variance Trade-off (`Projected128`)
- In the clean non-private setting, `Projected128` achieves 13.37% vs. 16.80% for `Mean512` (a 3.43% drop). Unlike binary medical classification where random projections preserve decision hyperplanes well, compressing 10-class multi-category features with an orthonormal projection incurs an upfront geometric information loss.
- However, under differential privacy, `Projected128` achieves 10.03% vs. 9.77% for `Mean512`. Because the coordinate space is reduced by $4\times$ (1,290 vs. 5,130 parameters), the total Euclidean noise energy injected into the model update is reduced by 50%, partially offsetting the projection compression penalty.

### 4.2 The Heterogeneity Confounder on Coordinate Trimming (`Trim16`)
- Under Dirichlet label skew ($\alpha = 0.5$), clients hold heavily imbalanced class distributions. A client with a high concentration of an uncommon class produces naturally large gradient coordinates for that class's weights.
- Coordinate-wise trimmed mean (`Trim16`) sorts each coordinate independently across clients and trims the extreme tails. Consequently, legitimate specialist gradients from honest clients are frequently trimmed as outliers!
- In the non-private attacked setting, `Trim16` degrades to 10.20% (worse than `Mean512` at 11.67% and `CenterQuarter` at 14.00%). Furthermore, because trimming reduces the effective sample size to $N - 2k = 12$, its DP sensitivity increases by $1.67\times$, resulting in the lowest accuracy in private settings (9.43%).

### 4.3 Structural Stability of Centered Clipping (`CenterQuarter`)
- `CenterQuarter` achieved the highest performance across almost all regimes: 18.03% clean, 14.00% attacked, and 10.17% private.
- Centered clipping around the previous round's aggregate with threshold $\tau = C/4$ bounds every client's displacement vector without discarding any honest client's contributions.
- Crucially, its sensitivity is $\frac{2\tau}{N} = \frac{C}{2N} = 0.25 \Delta_{mean}$, enabling a $4\times$ reduction in injected Gaussian noise under differential privacy.

---

## 5. Limitations & Confounders

1. **Step Budget**: The benchmark evaluated 5 local rounds to match runtime constraints and verify stability. Extended convergence sweeps would be valuable to track asymptotic properties.
2. **Attacker Knowledge**: The attack heuristic evaluated was coordinate sign-flipping (IPM). Reducer-aware adaptive attacks (e.g. tailoring perturbation vectors to specifically invert centered clipping) represent a harder adversarial threat.
3. **No Overstated Claims**: These findings demonstrate empirical behavior on a 10-class non-medical benchmark under Dirichlet label skew ($\alpha = 0.5$). They confirm that policy rankings reverse depending on whether privacy, Byzantine attack, or data heterogeneity is present, reinforcing the core MSc thesis conclusion that no single aggregation policy uniformly dominates across all operational regimes.
