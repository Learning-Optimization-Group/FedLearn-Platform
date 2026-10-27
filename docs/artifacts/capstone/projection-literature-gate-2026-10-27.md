# Projection study: literature and feasibility gate

27 October 2026. Research extension authorized for projection only; DP-BREM, backbone/LoRA and FLARE/failure qualification remain queued. Gate opened at 19:38:44 UTC; maximum gate deadline 29 October 2026, 19:38:44 UTC. This deadline is an outer limit, not a requirement to spend two days on preparation.

## Decision supported by the current inspection

**Proceed to a bounded design; do not claim novelty established.** Projection can reduce perturbation dimension, but the broad privacy/utility trade-off and projection error are already studied. The useful extension is a controlled audit of the explanation for this project's result, including a prediction-preserving padding control, signal-retention intervention, crossed projection seeds and bounded attacker optimization. If a new optimization method or first discovery is required, this gate has not met that requirement.

## Closest primary sources inspected

| Source and inspection depth | Relevant overlap | Difference and claim boundary |
| --- | --- | --- |
| [DP-GRAPE, v2, 18 May 2026](https://arxiv.org/html/2506.15588v2), abstract and methodology §§3.1–3.2 | Projects gradients before privatization; examines projection choice and operation ordering. | Deep-model sample-gradient/Adam training, rather than this fixed-feature, server-clipped client mechanism. “Project before noise” is not a contribution here. No head-to-head result implied. |
| [D2P2-SGD, v2, 12 September 2025](https://arxiv.org/html/2509.09485v2), abstract, dimension/utility discussion and clipping-bias sections | Already discusses privacy error, projection approximation, clipping and dimension choice, including cases where projection hurts. | Different optimizer and time-varying mechanism. A synthetic example that merely restates the bias/noise trade-off is explanatory replication, not a discovery. |
| [FedRP, v2, 23 August 2026](https://arxiv.org/html/2509.10041v2), abstract and section inventory | Random projection already appears in DP federated optimization and communication. | Uses ADMM and projections hidden from the server. The proposed extension uses a public data-independent basis and a trusted coordinator. Not a matched baseline; proof assumptions not independently verified here. |
| [Wireless FL with Limited Communication and DP, 2021](https://arxiv.org/abs/2106.00564), abstract | JL projection, noise and convergence/communication trade-offs in FL predate this project. | Over-the-air channel model and local-message assumptions differ. No “first projected private FL” claim. |
| [Allouah et al., ICML 2023](https://proceedings.mlr.press/v202/allouah23a.html), abstract | Existing theoretical privacy/robustness/utility analysis and a dimension-aware robust method. | Different protocol and privacy scope; not reproduced. No new trilemma claimed. |
| [HiDRA, v3, 15 December 2024](https://arxiv.org/abs/2312.14461), abstract | Shows that defense evaluation can change substantially with attack construction and implementation assumptions. | Attacks a different class of strong aggregators. It motivates attack qualification; it is not a drop-in matched attack for this mean-only study. |

This is a targeted search and selected-section inspection, not a systematic review or certification of any paper's proofs. None of these sources establishes that this project's exact proposed audit is first; absence from this search is not evidence of priority.

## Source inspection

- `repro/capstone_pilot_mechanisms.py:public_projection` constructs public orthonormal columns, uses a common seeded basis across reduced widths and returns identity at full width. There is no JL rescaling or post-projection variance normalization.
- `repro/run_effusion_pilots.py` uses the single public basis seed for the medical candidates. `repro/finalist_design.py` varies training seeds, not projection seeds. Existing twelve-seed inference therefore does not characterize projection-choice variation.
- `framework/src/fedlearn/simulation/owned_data_attack.py:ascent_upload` accepts matching head/features and configurable ascent steps/restarts. It searches a norm ball using owned examples and previous releases. It retains the best surrogate candidate along each trajectory. It does not certify a global optimum or see current honest uploads.
- `framework/src/fedlearn/server/composed_strategy.py` already provides clipped uniform mean with Gaussian noise. The new study should reuse it rather than change historical reducers or depend on the optional ByzFL installation.

## Analytical feasibility evidence

For a two-output linear head with k features and two biases, isotropic coordinate noise of SD σ has expected squared parameter norm `(2k+2)σ²`. For a fixed feature x, the binary logit-difference noise variance is `2σ²(||x||²+1)`. Both statements follow directly from independent Gaussian coordinates; neither is a new theorem or an AUROC prediction.

Zero-padding x from 128 to 512 features preserves the score variance. It nevertheless increases total expected parameter-noise energy by `1026/258 = 3.976744`. A standalone 10,000-draw generated calculation observed a ratio of 3.975597 with exactly matching coupled scores. The [receipt](projection-gate-analytical-2026-10-27.json) binds the inspected source files and records numerical checks. No model was fit and no medical input was opened. Initial NumPy matrix multiplication emitted warnings despite finite, independently agreeing outputs; the recorded qualification uses elementwise sums with floating-point exceptions enabled. The backend warning cause remains unresolved.

This negative control is feasible and useful for checking the project's explanation. It is not evidence that projection helps prediction, that the training implementation passes, or that the proposed research is novel.

## What is still required

The [design](../../superpowers/specs/2026-10-27-projection-mechanism-design.md) is a proposed bounded protocol. Before main fits: review it, implement and test generated-only qualification, fix or report any failed invariance, run at most twelve qualification cells, measure runtime, then freeze the source/configuration and exact cell manifest. Retain every qualification outcome. Main outcomes cannot be used to alter seeds, sample sizes or the matrix.
