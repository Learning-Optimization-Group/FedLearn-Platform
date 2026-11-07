# FedLearn Research & Empirical Benchmarks

This directory houses the experimental harnesses, raw machine-readable results, and dated analytical notes for the MSc Capstone research program.

---

## Directory Structure

- **[`benchmarks/`](file:///Users/anurag/code/FedLearn-Platform/research/benchmarks/)**: Execution harnesses and benchmarking drivers.
  - `run_heterogeneous_cifar_benchmark.py`: Second non-medical benchmark comparing `Mean512`, `Projected128`, `Trim16`, and `CenterQuarter` under Dirichlet label skew ($\alpha = 0.5$).
  - `capstone_final.sh` & `analyze_capstone_final.py`: The confirmatory capstone study driver and analysis.
  - `nih_learning.py`, `nih_finetune.py`, `nih_pretrained.py`: NIH ChestX-ray14 evaluation scripts.
  - `resnet_client_rates.py`: Client-side forward/backward compute and networking rate benchmarks.
- **[`results/`](file:///Users/anurag/code/FedLearn-Platform/research/results/)**: Machine-readable JSON outputs with full `meta` blocks and per-round trajectories.
  - `heterogeneous_cifar/cifar_heterogeneous_benchmark.json`: Raw results for the 4-policy CIFAR-10 Dirichlet study.
  - `capstone/`: Results from the 360-cell study and pilot evaluations.
  - `on-device/`: Physical mobile client telemetry and qualification receipts.
  - `robust-aggregation/`: Fixed IPM and ByzFL adapter evaluation matrices.
- **[`notes/`](file:///Users/anurag/code/FedLearn-Platform/research/notes/)**: Dated scientific interpretation notes.
  - `heterogeneous_cifar/2026-11-07-cifar-heterogeneous-analysis.md`: Empirical analysis of the second non-medical benchmark.
  - `capstone/2026-10-05-final-study-results.md`: Capstone final study results against pre-registration.
  - `capstone/2026-10-05-pilot-and-preregistration.md`: Pre-registered analysis plan.

---

## The Four Aggregation & Representation Policies

Across both medical (Kermany / ChestX-ray14) and non-medical (CIFAR-10) benchmarks, the research compares four exact policies:

1. **`Mean512`**: Standard coordinate-wise mean over full feature space ($d = 512$). Sensitivity $\Delta = \frac{2C}{N}$.
2. **`Projected128`**: Fixed public orthonormal projection from 512 to 128 dimensions ($\mathbf{P} \in \mathbb{R}^{512 \times 128}$). Sensitivity $\Delta = \frac{2C}{N}$, reducing Euclidean noise energy $\|\boldsymbol{\xi}\|_2$ by $50\%$.
3. **`Trim16`**: Coordinate-wise trimmed mean with trimming fraction $\beta = 0.2$. Sensitivity $\Delta = \frac{2C}{N - 2k}$, resulting in $1.67\times$ higher noise under Gaussian DP.
4. **`CenterQuarter`**: Centered clipping around previous release $\mathbf{c}_{t-1}$ with radius $\tau = C/4$. Sensitivity $\Delta = \frac{C}{2N} = 0.25 \Delta_{mean}$.

---

## Reproducing the CIFAR-10 Dirichlet Benchmark

To execute the 48-cell CIFAR-10 Dirichlet benchmark ($\alpha = 0.5$):

```bash
cd research/benchmarks
PYTHONPATH=../../framework/src python run_heterogeneous_cifar_benchmark.py
```

The script automatically executes all 4 policies across clean, attacked, non-private, and private ($\epsilon=4, \delta=10^{-5}$) regimes over seeds `42`, `43`, and `44`, writing the complete report to `research/results/heterogeneous_cifar/cifar_heterogeneous_benchmark.json`.

---

## Integrity & Freeze Boundaries

- Historical medical evidence (360-cell study, Kermany dataset, Effusion A1–A23 finalists) is strictly frozen and preserved.
- All benchmark artifacts are machine-readable, containing exact platform, version, seed, and timing metadata.
