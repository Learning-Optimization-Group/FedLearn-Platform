# B1 — DeComFL Paper-Alignment Audit

**Date:** 2026-05-29
**Branch:** `main-clean` (Python framework) + `origin/fed-mobile` (C++ mobile port)
**Scope:** Fidelity of the platform's DeComFL implementation to the published algorithm, across the Python framework (`framework/src/fedlearn/`) and the native C++ port (`mobile_client/shared/src/`).
**Builds on:** prior framework audit [`docs/audit/2026-05-27/03-framework.md`](../2026-05-27/03-framework.md), specifically H4 (O(KP·N) aggregation loop) and M5 (global numpy RNG mutation). This report **extends** those with correctness findings they did not cover; it does not re-litigate them.

---

## The paper (ground truth)

**Paper:** *Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization* — Li, Ying, Liu, Dong, **Yang (RIT)**. ICLR 2025.
- arXiv: https://arxiv.org/abs/2405.15861 (HTML v4: https://arxiv.org/html/2405.15861v4)
- OpenReview: https://openreview.net/forum?id=Vy9ltlTXXd
- Reference code: https://github.com/ZidongLiu/DeComFL

What the paper actually specifies (verified against arXiv v4 and the reference repo, not the platform's wiki):

| Element | Paper / reference | Source |
|---|---|---|
| ZO estimator (main text) | Forward SPSA: `g = (f(x+μz) − f(x))/μ` (Eq. 3 / Alg. 2 L12-16) | arXiv html v4 |
| ZO estimator (**reference default**) | **Central** `g = (f(x+μz) − f(x−μz))/(2μ)` (`rge_central`) | `random_gradient_estimator.py:130-176` |
| Perturbation | `z ∼ N(0, I_d)` | Alg. 2 |
| Multi-perturbation (P>1) update | **Averaged over P**: `g_full = avg_p(g_p · z_p)`, then `x ← x − η·g_full`. The reference returns `grad.div_(self.num_pert)`. | `random_gradient_estimator.py:176` (`return grad.div_(self.num_pert), ...`); per-pert `mul_(dir_grad / num_pert)` at `:93` |
| Single-step update | `x_{r}^{k+1} = x_r^k − η·g_r^k·z_r^k` (and the multi-perturbation version "is what all theorems and experiments use", Appendix A.4) | Alg. 2 L16; §3 note |
| Server update | `x_{r+1} = x_r − η Σ_k g_r^k z_r^k`, where `g_r^k = (1/|C_r|) Σ_i g_{i,r}^k` | Alg. 1 L10-12 |
| Seed history `{s_r^k}` + gradient history `{g_r^k}` + client last-round `{t_i}` | Server-maintained (Alg. 1 L2) | arXiv html v4 |
| Rebuild (missed rounds) | Replay `r' = t_i … r−1`: regenerate `z` from `s`, apply `x ← x − η g_{r'}^k z_{r'}^k` | Alg. 2 L3-8 |
| Seed→RNG derivation (reference) | `torch.Generator(device).manual_seed(seed*(perturb_index+17)+perturb_index)` | `random_gradient_estimator.py:55-57` |
| Convergence assumptions | L-smooth, bounded variance σ², bounded dissimilarity σ_G²; **dimension-free only under low-effective-rank** (Assumption 4 / Theorem 2) | arXiv html v4 |
| **Byzantine robustness** | **Not claimed anywhere.** Honest-client, honest-server model. | arXiv html v4 (full text) |

The platform wiki [`docs/wikis/framework/06_decomfl.md`](../../wikis/framework/06_decomfl.md) labels these "Algorithm 3 / Algorithm 4," cites a fabricated BibTeX (`@article{yang2024decomfl, title={DeComFL: Decomposed Federated Learning}, journal={[Journal/Conference]}}` — `06_decomfl.md:595-602`), and even mis-expands the acronym ("Decomposed" — the paper title contains no such word; "DeComFL" stands for **Dimension-Free Communication FL**). The algorithm numbering and citation in the wiki should not be trusted as paper references; the math below is verified against the real paper.

---

## Verdict summary

| Component | Verdict | One-line rationale |
|---|---|---|
| ZO estimator — Python (`zeroth_order.py`) | **salvage** | Forward-difference math is correct; missing central-difference option and per-call `model.eval()` side-effect are quality, not correctness. |
| ZO estimator — C++ (`ZerothOrderEstimator.cpp`) | **refactor** | Math mirrors Python, but the "bit-identical to Python" RNG claim is unproven and load-bearing; must be test-pinned or the port is unsafe to ship. |
| Server update `aggregate_fit` (`decomfl_strategy.py:180-208`) | **rebuild** | **Missing 1/P factor** → global model takes a step P× too large and diverges from the client/rebuild trajectory. Breaks paper update + internal consistency. |
| Server perturbation RNG device (`decomfl_strategy.py:77,212`) | **rebuild** | Server uses `cuda` if available; clients run CPU/MPS/CUDA. `torch.randn(seed)` differs across devices → reconstructed `z` ≠ client `z` → silent total divergence. |
| Client `fit` (`decomfl_client.py:121-231`) | **salvage** | Update `(η/P)·δ` matches the reference; the subtract-cumulative revert is FP-lossy vs. the C++ snapshot-restore. |
| Rebuild — Python & C++ | **salvage** | Both apply `(lr/P)·δ`, internally consistent with the client and with each other; but they replay the *client* formula, not the (buggy) server formula — so rebuild ≠ actual global trajectory. |
| Seed history / gradient history / last-round tracking | **salvage** | Round-keyed dicts, lock-guarded seed generation; correct shape and lifecycle. Unbounded growth is a scaling, not fidelity, issue. |
| Byzantine-robustness claim (READMEs) | **kill** | The paper makes no such claim; the platform implements no robust aggregator on the DeComFL path. Remove the claim. |

---

## Critical findings

### B1-C1 — Server update is missing the `1/P` factor: global model steps P× too far, and diverges from the rebuild trajectory

**Evidence.**
- Reference: multi-perturbation gradient is **averaged over P** — `random_gradient_estimator.py:176` returns `grad.div_(self.num_pert)`; the paramwise path scales each contribution by `dir_grad / num_pert` (`:93`). Canonical update: `x ← x − η·(1/P)·Σ_p g_p z_p` (per client, then averaged over clients).
- Platform **client** (`decomfl_client.py:208`): `step_update = (eta / P) * delta` — **correct**, has the `1/P`.
- Platform **rebuild** (`decomfl_client.py:115`, and C++ `DeComFLClient.cpp` rebuild loop): `x -= (learning_rate / P) * delta` — **correct**.
- Platform **server** (`decomfl_strategy.py:197-200`):
  ```python
  delta = delta / (num_clients * self.P)      # divides by N·P
  x_current = x_current - self.eta * delta * self.P   # then multiplies by P again
  ```
  Net: `x ← x − η·(1/N)·Σ_p g·z`. The `1/P` has been cancelled by the explicit `* self.P`. The inline comment `# ×P cancels in derivation` (and wiki line 333 `x_t = x_{t-1} − η × P × Δ (note: P factor cancels…)`) is the bug rationalised as intent.

**Why it breaks the paper.** The server advances the *global* model by a step `P×` larger than the reference, and `P×` larger than what every client and the rebuild path apply. With the default `P=10` (`decomfl_strategy.py:34`) this is a silent 10× learning-rate inflation on the global model only.

**Why it is worse than a tuning quirk.** It is **internally inconsistent**: a client that participates every round never sees the global model (it reverts and re-syncs via rebuild), so its view of "the model" is the *rebuild* trajectory (`1/P` step), while the server's `global_params_flat` follows the `P×` step. The moment any client misses a round and rebuilds, its reconstructed model and the server's model are on different trajectories. The seed/gradient-history replay — the paper's central correctness guarantee — is therefore violated by construction whenever P>1. Prior audit H4 noted this loop is slow; it did not notice it is also **wrong**.

**Fix.** Drop the `* self.P`: `x_current = x_current - self.eta * delta` (with `delta` already `/(N·P)`). Add a property test: for a fixed seed list and gradient scalars, a client that trains every round must produce a model bit-identical (within FP tolerance) to a client that misses every round and rebuilds. That test fails today.

**Verdict: rebuild** (the aggregation step). One-line: server update violates the paper's `1/P` averaging and contradicts its own rebuild path.

---

### B1-C2 — Server regenerates perturbations on `cuda` while clients regenerate on CPU/MPS — same seed, different `z`, silent divergence

**Evidence.**
- Server: `self.device = 'cuda' if torch.cuda.is_available() else 'cpu'` (`decomfl_strategy.py:77`); perturbation via `torch.Generator(device=self.device)` + `torch.randn(..., device=self.device)` (`:212-217`).
- Python client: device is whatever the client passes (`DeComFLClient(..., device=...)` — `decomfl_client.py:33`); Docker client is CPU, Mac is `mps`, a CUDA client is `cuda`.
- C++ client: `torch::Generator()` defaults to **CPU** (`ZerothOrderEstimator.cpp:14-19`).
- PyTorch fact: `torch.randn` with the same generator seed produces **different values on CPU vs CUDA** — confirmed upstream (https://github.com/pytorch/pytorch/issues/79496) and in the reproducibility note (https://docs.pytorch.org/docs/stable/notes/randomness.html). MPS is a third distinct stream.

**Why it breaks the paper.** DeComFL's entire premise is that server and clients **independently regenerate the identical `z`** from a shared seed, so only the scalar `g` is transmitted. If the server runs on a GPU box (the common deployment — FL servers are spawned on the EC2/training host) and a client runs on CPU, `z_server ≠ z_client`. The server then reconstructs the update direction `g·z_server` from a `g` that was measured along `z_client`. The dot-product structure that makes ZO work collapses; the model walks in an essentially random direction. **There is no error — it silently does not learn.** This is the single highest-risk fidelity defect because it only manifests in heterogeneous hardware (exactly the federated setting the platform targets: mobile + Jetson + server).

The reference sidesteps this because in their experiments the same `device` is used consistently; the platform mixes devices by design (M4 Max Mac, Jetson ARM64, Docker CPU clients, GPU server).

**Fix.** Pin perturbation generation to **CPU, float32** on *every* node (server and all clients), then `.to(device)` the resulting `z` if needed for the forward pass. Generation must never touch CUDA/MPS. Add a cross-device golden-vector test (CPU seed → fixed bytes; assert CUDA path matches after the CPU-generate-then-move fix).

**Verdict: rebuild** (server perturbation path + the device contract). One-line: cross-device RNG divergence silently nullifies learning in the heterogeneous case.

---

## High findings

### B1-H1 — C++ ↔ Python RNG parity is asserted in a comment, never tested — and is not guaranteed across libtorch versions

**Evidence.** `ZerothOrderEstimator.cpp:11-13`:
```cpp
// C++ torch::Generator uses the same Mersenne Twister as Python, producing identical outputs.
```
`generatePerturbation` uses `gen.set_current_seed(seed)` + `torch::randn` (`:15-19`). The header repeats "bit-identical vectors to Python's torch.randn" (`ZerothOrderEstimator.h:18-22`).

**Assessment.** The claim is *plausible* for a fixed libtorch build: PyTorch's CPU `torch.randn` uses the same MT19937 + normal-distribution implementation whether driven from Python or libtorch, and `manual_seed`/`set_current_seed` are the same underlying call. **But it is unverified, undocumented as to version, and brittle.** The normal-sampler implementation and the MT19937 box-muller/ziggurat path have changed between PyTorch releases; nothing pins the mobile libtorch version to the server's torch version. If they drift, B1-C2's failure mode (wrong `z`) reappears on the mobile path even with the device fix. I cannot confirm bit-identity without running both — **flagging as uncertain**.

**Fix.** Add a committed golden-vector fixture: a small JSON of `(seed, n) → first 16 float32 values` generated by the Python server's exact torch version, asserted by a C++ unit test and a Python unit test. Pin the mobile `libtorch` version to the server's `torch` version in build config. Treat any mismatch as a release blocker.

**Verdict: refactor** (C++ estimator). One-line: math is right but the cross-language determinism the protocol depends on is untested.

### B1-H2 — Forward difference is used everywhere; reference defaults to central, halving μ-bias

**Evidence.** Python `zeroth_order.py:105` and C++ `ZerothOrderEstimator.cpp:55` both compute forward `(f(x+μz) − f(x))/μ`. The reference defaults to `rge_central` (`random_gradient_estimator.py:21,130-176`): `(f(x+μz) − f(x−μz))/(2μ)`.

**Assessment.** Both are valid ZO estimators and the paper's *main-text* derivation is forward, so this is **not** a correctness break. But forward difference carries an `O(μ)` bias term; central difference is `O(μ²)` — strictly lower bias for the same μ, at the cost of one extra forward pass per perturbation. For LLM fine-tuning (the platform's headline use case) the bias matters. The platform offers no central option, so users cannot match the reference's accuracy/quality.

**Fix.** Add a `grad_estimate_method` flag (`forward` | `central`) threaded from strategy config through to both estimators. Default forward to preserve current behaviour; document the trade-off.

**Verdict: salvage** (with the central option added). One-line: forward is paper-valid but strictly higher-bias than the reference default.

### B1-H3 — README "Byzantine-robust" claim is unsupported by both the paper and the code

**Evidence.**
- `README.md:32` "Byzantine-robust aggregation for secure federated learning"; `:82`, `:387`; `framework/README.md:9,213` "Byzantine-robust optimization".
- The paper makes **no** Byzantine-robustness claim (full-text check of arXiv v4): the threat model is honest-but-curious at most.
- The DeComFL aggregation path (`decomfl_strategy.py:aggregate_fit`) is a **plain mean** of gradient scalars (`coordinator.py:_calculate_average_gradients:327-335` sums then divides by `num_clients`). No trimmed mean, no median, no Krum, no clipping. A single client can submit an arbitrarily large `g` scalar and move the global model unboundedly — *more* dangerous than in standard FL because one scalar scales a full `d`-dimensional `z`.
- The only "robustness" in the repo is the FedAvg `MAX_SAMPLES`/`MAX_NUM_EXAMPLES` count cap (`strategy.py:81`, `coordinator.py:55`) — that caps *weighting*, not the scalar magnitude, and it is on the **FedAvg** path, not DeComFL. The wiki's trimmed-mean example (`05_strategies.md:444`) is illustrative code, not wired in.

**Why it matters for a startup.** "Byzantine-robust … for secure federated learning" on the public README is a security claim a customer or investor will rely on. It is false for the shipped DeComFL path and is a liability.

**Fix.** Remove the Byzantine claims from both READMEs, or implement an actual robust aggregator on the DeComFL scalar path (e.g., coordinate-free here — robust mean / trimmed mean / clipping on the `g_{i,r}^k` scalars, which is cheap because they are scalars, not vectors) and gate the claim behind it. Cheapest honest fix: per-scalar magnitude clipping `|g| ≤ τ` plus client-count-robust mean — and *then* the claim is partially defensible.

**Verdict: kill** (the claim, as written). One-line: the paper doesn't claim it and the code doesn't do it.

---

## Medium findings

### B1-M1 — Per-perturbation revert: Python subtracts a running sum (FP-lossy), C++ restores a snapshot (exact)

- Python `decomfl_client.py:147,210,218`: accumulates `total_perturbation -= step_update` then `self.x_current -= total_perturbation` to "revert."
- C++ `DeComFLClient.cpp`: `auto x_initial = x_current_.clone();` … `x_current_ = x_initial;` (exact restore).
- The paper requires reverting to `x_{i,r}^1` exactly (Alg. 2 L17). The Python subtract-the-sum approach is algebraically a no-op but accumulates float32 rounding over `K` steps; over many rounds the client's working point drifts from the true pre-round state. The C++ snapshot is the correct, paper-faithful pattern.
- **Fix.** Python should `x_initial = self.x_current.clone()` and restore, matching C++. Removes drift and the `total_perturbation` bookkeeping. (`zeroth_order._set_flat_params` writes to `model`, but `x_current` is the source of truth — clone is cheap relative to the model.)

### B1-M2 — `model.eval()` called inside the hot estimator loop; disables nothing useful and forbids BN/dropout-trained federations from matching the paper's stochastic `ξ`

- `zeroth_order.py:73` calls `model.eval()` on every `compute_gradient_scalar`. The paper's `f_i(·; ξ)` is the stochastic mini-batch loss in **train** mode (dropout/BN active is part of the objective being optimised). Forcing `eval()` changes the objective for any model with dropout/BN and is also redundant work per call.
- **Fix.** Set train/eval once per round at the client, not per perturbation; default to `train()` to match the paper's stochastic objective unless the model has no stochastic layers.

### B1-M3 — `np.random.seed(seed)` mutates global numpy RNG (re-confirming prior M5) — but note it only seeds the *seed source*, not the perturbation

- `decomfl_strategy.py:82` `np.random.seed(seed)`; `generate_seeds` (`:107`) draws from the global numpy RNG. Prior audit M5 flagged the multi-server clobber. **Clarification for this report:** numpy here only generates the *integer seeds* that are then transmitted verbatim; the perturbation `z` is drawn from `torch.Generator`. So numpy/torch RNG-source divergence is **not** a fidelity risk (the seed integers are shared on the wire). The real RNG risk is B1-C2 (torch device), not the numpy/torch split. Still fix M5 (use `np.random.Generator(np.random.PCG64(seed))`) for multi-tenant isolation.

### B1-M4 — Gradient-history population is conditioned on a stringly-typed type check

- `coordinator.py:285` `if 'DeComFL' in str(type(self.strategy)) and hasattr(...)`. If a subclass renames or wraps the strategy, history silently stops being recorded → rebuild returns empty histories → late-joining clients silently train from a stale model. Fidelity-relevant because the rebuild guarantee depends on this dict being populated every round.
- **Fix.** `isinstance(self.strategy, DeComFL)` or a capability flag on the Strategy base.

### B1-M5 — Aggregation is O(K·P·N·d) Python-loop tensor ops (extends prior H4)

- `decomfl_strategy.py:185-194` regenerates `z` (a `d`-vector) and does `delta += g*z` inside a `K×P×N` Python loop. Prior H4 called this slow. **Fidelity angle:** it also regenerates each `z` **once per client** even though `z` for `(k,p)` is identical across clients — so it is `N×` redundant. Correct and faster: average the scalars over clients first (already computed in `_calculate_average_gradients`), regenerate each `z` **once**, accumulate `avg_g · z`. This is `O(K·P·d)`, drops the N factor, and is what the math actually is.
- **Fix.** Aggregate scalars → single regeneration per `(k,p)` → single `delta`. Same result, `N×` faster, fewer FP-summation-order artefacts.

---

## Low / quality

- **Wiki is not a trustworthy paper map.** `06_decomfl.md` invents "Algorithm 3/4" numbering, a fake BibTeX (`:595-602`), wrong acronym expansion, and rationalises the C1 bug ("P factor cancels in full derivation," `:99,:333`). Rewrite against the real paper; cite arXiv 2405.15861 and the reference repo.
- **`num_examples` is collected but unused on the DeComFL path.** `aggregate_fit` ignores it (`decomfl_strategy.py:171`); the mean is unweighted. The paper's server averages scalars uniformly over `|C_r|`, so unweighted is paper-correct — but then stop collecting `num_examples` on this path, or document that DeComFL intentionally ignores sample counts (unlike FedAvg).
- **Seed dtype ceiling.** `np.random.randint(0, 2**31-1)` (`:107`) caps seeds at 31 bits; C++ takes `int64_t`. Fine today (collision probability negligible) but document the bound; if seeds ever widen, the C++ path silently accepts values Python can't generate.
- **No determinism test exists at all** for the one property the entire protocol rests on (server `z` == client `z` for a given seed/device/dtype). The prior audit's M6 ("no tests for decomfl_client.fit gradient-scalar correctness") undersells this — the missing test is the *RNG parity* test, not just gradient correctness.

---

## Prioritized recommendations

1. **(C1) Remove the `* self.P` in `aggregate_fit`** (`decomfl_strategy.py:200`). One-line fix; restores the paper's `1/P` averaging and makes the global trajectory consistent with the rebuild path. Add the participate-vs-rebuild equivalence property test (it fails today).
2. **(C2) Force perturbation generation to CPU/float32 on every node**, server and client, then move `z` to the compute device. Add a cross-device golden-vector test. Without this, heterogeneous federations silently do not learn.
3. **(H1) Pin libtorch == server torch version and add a committed golden-vector fixture** asserted from both Python and C++. Block releases on mismatch.
4. **(H3) Delete the "Byzantine-robust" claims** from both READMEs now; if the claim is wanted, implement scalar-magnitude clipping + robust mean on the DeComFL `g` scalars (cheap — they're scalars) and re-add the claim behind it.
5. **(M5/H4) Rewrite aggregation** to average scalars first and regenerate each `z` once — `O(K·P·d)`, drops the redundant `N×`, more numerically stable, and is literally what the math is.
6. **(H2) Add `central` ZO option** threaded through strategy → both estimators; default forward, document bias trade-off for LLM fine-tuning.
7. **(M1/M2)** Python: snapshot-restore revert (match C++); set train/eval once per round, default `train()` to match the paper's stochastic objective.
8. **(M4)** Replace the `'DeComFL' in str(type(...))` guard with `isinstance` so rebuild histories never silently stop populating.
9. **Rewrite `06_decomfl.md`** against arXiv 2405.15861 + reference repo; fix the citation, acronym, and remove the C1 rationalisation.

---

## What I could not verify (flagged uncertainty)

- **Bit-identical CPU `torch.randn` between the mobile libtorch build and the server's Python torch** (B1-H1). Plausible for matched versions; unproven here. Treat as a release blocker until a golden-vector test exists.
- **Reference Appendix A.4 multi-perturbation equation verbatim** — the arXiv HTML truncated A.4. I confirmed the `1/P` averaging directly from the reference implementation (`random_gradient_estimator.py:176,93`) instead, which is authoritative for "what the authors actually do."

## Key files

- `framework/src/fedlearn/server/decomfl_strategy.py:77,82,180-208,212-217` — server update (C1), device (C2), numpy RNG (M3).
- `framework/src/fedlearn/client/decomfl_client.py:115,147,208-219` — client update + revert (M1), rebuild.
- `framework/src/fedlearn/estimators/zeroth_order.py:73,105` — forward diff (H2), eval() (M2).
- `framework/src/fedlearn/server/coordinator.py:285,304-336` — gradient-history guard (M4), unweighted mean.
- `origin/fed-mobile:mobile_client/shared/src/ZerothOrderEstimator.cpp:11-19,55` — RNG-parity claim (H1), forward diff.
- `origin/fed-mobile:mobile_client/shared/src/DeComFLClient.cpp` — snapshot revert (M1, correct), rebuild.
- `README.md:32,82,387` + `framework/README.md:9,213` — Byzantine claim (H3).
- `docs/wikis/framework/06_decomfl.md:99,333,595-602` — wiki errors (Low).

## Sources

- DeComFL paper: https://arxiv.org/abs/2405.15861 · https://arxiv.org/html/2405.15861v4 · https://openreview.net/forum?id=Vy9ltlTXXd
- Reference implementation: https://github.com/ZidongLiu/DeComFL (`cezo_fl/gradient_estimators/random_gradient_estimator.py`)
- PyTorch cross-device RNG divergence: https://github.com/pytorch/pytorch/issues/79496 · https://docs.pytorch.org/docs/stable/notes/randomness.html
