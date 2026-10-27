# Projection mechanism and failure-boundary study

27 October 2026. Proposed design for the first authorized research extension. User decisions: proceed in phases, start with projection, and consider DP-BREM, backbone/LoRA and FLARE/failure work only if time allows. All numerical study choices below are proposed implementation decisions, not additional user requirements.

## Purpose and contribution boundary

Explain which parts of the observed projection result survive controlled checks of predictive signal, noise geometry, projection choice and attacker optimization. The existing NIH findings remain frozen historical evidence. This extension uses generated data only and cannot establish external medical validity. No new algorithm, first discovery or state-of-the-art result is promised. The [literature gate](../../artifacts/capstone/projection-literature-gate-2026-10-27.md) identifies substantial prior overlap.

The central questions are: (1) can the same prediction function have different total parameter-noise energy, (2) where does loss of informative signal outweigh any private-utility advantage, (3) how variable is a fixed-width result across public projection choices, and (4) does its attacked ranking change when the permitted attacker gets more optimization effort?

Do not add a new defense, tune for an improved score, reopen medical bundles/holdout/test artifacts, install new dependencies, launch a clinical fit or start another queued project.

## Phases and limits

1. **Gate/design:** targeted literature/source inspection, analytical checks and a fresh builder review. At most 48 hours from 27 October 2026, 19:38:44 UTC. Feasibility and explanation are separate from novelty.
2. **Qualification:** build generated-data helpers and independent checks, then at most twelve qualification training cells. Stop after 30 minutes of measured qualification execution or after a correctness failure. Diagnose failures before any main fitting. No replacement fits to obtain favorable utility.
3. **Frozen measurement:** only after qualification and a source/configuration checkpoint, run the exact 136-cell matrix below once. A 12-hour main-run wall-time ceiling stops the batch with partial records. No new seeds/cells, runtime-based subsampling or retries under the same study ID. A defect requires a separately recorded amendment before any resumed fitting.
4. **Consolidation:** one bounded report with all outcomes, source/config hashes, independent metric checks and three figures: signal-retention effects, per-basis variability and attack-budget response. Preserve the existing manuscript and receipts. Review time and contribution before choosing a queued phase.

A one-week effort estimate is a planning bound, not a completion guarantee. If qualification projects more than 12 hours for main execution, stop and propose a smaller design before inspecting main outcomes. Negative or ambiguous findings are valid outcomes and do not justify enlarging the study.

## Generated workload and unchanged training rule

Use D=512 independent standard-normal features, a fixed public teacher β=e₀, and binary labels drawn from `Bernoulli(sigmoid(2*x[0]))`. Generate exactly 4,096 training and 8,192 evaluation examples once with independently named PCG64 streams at dataset seed 20261027. Evaluation labels cannot enter fitting, attack search or configuration selection. No normalization, whitening, JL rescaling or post-projection standardization. All client assignments use only training rows.

Use twenty equally weighted logical clients: sixteen own 205 rows and four own 204 rows, fixed by a public permutation per training seed. Use the existing two-output linear head with zero initialization, twenty full-participation rounds, one local SGD epoch, batch size 128, LR=0.03, no momentum/weight decay and outer upload clip C=0.05. Use canonical `ComposedStrategy(rule='mean')`; no optional ByzFL dependency is needed. New experiment orchestration lives under the capstone reproduction directory, not inside historical runners.

Noise settings are zero-noise controls and GDP calibration at ε=4, δ=1e-5 over twenty releases. For N=20 the per-release replace-one sensitivity is `2C/N`; obtain z through the existing accountant and set σ=z*2C/N. Independently verify the normal-CDF conversion. Fixed datasets/projections/assignments are public inputs to this abstract client-update mechanism. Generated public randomness and diagnostics mean this study is not a protected deployed release.

Four training seed IDs are 24001–24004. Partition, minibatch, attack and Gaussian-noise streams have distinct deterministic namespaces. Training-seed changes do not change the generated dataset or basis. Seed derivation must be independent of cell order and Python hash randomization. Common random numbers across comparisons are deliberate pairings, not independent draws.

## Projection controls

**Full512:** identity transform; one shared baseline per training seed/noise setting. Never recount reused baseline results as new fits.

**Controlled128:** four public orthonormal 512×128 bases with exact retained teacher energy η in `{0, 0.25, 0.75, 1}`. First column is `sqrt(η)*e₀ + sqrt(1−η)*e₁₂₈`; remaining columns are e₁,…,e₁₂₇. Verify `PᵀP=I` and `||Pᵀβ||²=η`. Use one unchanged full dataset and teacher across η: rotating both teacher and data with P would invalidate this intervention. η=0 deliberately removes all population predictive information, while η=1 retains the teacher. These oracle-designed diagnostic subspaces are not a deployable data-selected method.

**Random128:** five public orthonormal projection bases at seeds 20261101–20261105, using the existing QR construction. Cross every basis with all four training seeds. Log retained teacher energy and basis hash. Never pick the best basis or treat all twenty fits as twenty independent basis draws.

**Padded512:** take Random128 basis 20261101, transform to 128 coordinates, then append 384 exact zeros. Train a 512-feature head. This is the same predictive representation as that Random128 arm, not the raw Full512 representation. Couple active-coordinate and bias Gaussian noise to the matching Random128 run and draw independent noise for the inactive coordinates. Both marginal noise laws must remain isotropic at the declared σ. Keep inactive local deltas exactly zero: no decay/regularizer. Check scores, active updates, clipping decisions and final predictions against Random128 within 1e-5 absolute tolerance; do not demand bytewise equality. Added inactive noise is not a privacy improvement or claim that arbitrary lower-rank noise protects a full-dimensional update.

The existing canonical generator draws shape-dependent full tensors; equal seeds alone do not provide this coupling. Use a **study-only subclass** in the reproduction directory: call the parent `_pipeline(..., apply_noise=False)` once, preserve its clipping/reduction/cohort checks and reported sensitivity, then add the independently checked Gaussian draw before returning to canonical `aggregate_fit`. Do not change production strategy code or inject noise twice. Use distinct active-row, bias and inactive-row generators derived from training seed and round. Random128/20261101 and Padded512 share active-row/bias draws by explicit coordinate mapping; inactive entries have independent streams. Require per-coordinate variance, cross-coordinate covariance and coupled-coordinate checks, plus no-noise parity against the unmodified strategy. All study arms use this same adapter with their declared layouts; full-size native-noise distribution checks independently verify its marginal law.

## Exact proposed main budget

| Block | Crossings | New cells |
| --- | --- | ---: |
| Full512 clean baseline | 4 training seeds × 2 noise settings | 8 |
| Controlled128 clean | 4 η values × 4 training seeds × 2 noise settings | 32 |
| Random128 clean | 5 projection seeds × 4 training seeds × 2 noise settings | 40 |
| Padded512 clean | 4 training seeds × 2 noise settings | 8 |
| Attack effort | Full512 and Random128/20261101 × 4 training seeds × 2 noise settings × 3 effort budgets | 48 |
| **Total** | No reuse counted twice; 20 rounds/cell; 20 clients/round | **136** |

Total maximum is 2,720 main rounds and 54,400 main client fits, plus the separately labelled qualification budget. The synthetic recipe intentionally differs from the NIH workload; absolute scores are not directly comparable.

## Permitted attacker and budget qualification

Use a fixed coalition of four client IDs selected by the public training-seed permutation. It receives only those clients' training examples/labels and computed local updates, the matching head/projected features, the public transform and previous released global/history. It must not see current raw honest updates, evaluation examples/labels or future/private noise draws. Fixed coalition identities/data remain outside the protected honest-client replacement in the scoped accounting argument.

Use `ascent_upload` with steps `{20,80,160}`, three restarts and step ratio 0.25: respective per-round gradient budgets `{60,240,480}` and loss budgets `{63,243,483}`. Keep the existing constrained initialization from the negative coalition-mean delta; each of the four malicious clients uploads the chosen delta, and the server re-clips actual reconstructed uploads. Log represented norms and clipping.

At the same fixed state, long searches must retain the best observed surrogate candidate and reproduce the short-search prefixes under identical seeds. Independently recompute their objectives. Across full training trajectories, neither evaluation damage nor surrogate objective is required to be monotone: histories diverge. More steps are not a worst-case attack certificate and do not remove the surrogate's honest-update forecasting limitation.

Qualification must establish finite objectives/gradients and a generated damaging positive control for the attack path. Apply the main damaging-control decision **separately at each of the three effort budgets**: if that budget's Full512 no-noise attack fails to reduce mean evaluation AUROC by 0.02 versus its clean baseline across the four paired training seeds, mark robustness interpretation at that budget inconclusive. Publish its raw results regardless; never select a different budget retrospectively to qualify the failed one. Do not retune or expand until it causes damage. This descriptive gate is not a general robustness test.

## Analytical checks and measured outcomes

For two output rows with independent isotropic Gaussian coordinate noise, parameter-noise energy expectation is `(2k+2)σ²`. Conditional fixed-input logit-difference variance is `2σ²(||x||²+1)`. Verify both with generated Monte Carlo and an independent formula. Compute score effects directly using the realized noise and pre-release evaluation features; never infer AUROC from total noise norm alone.

Record per round: complete cohort count, raw and clipped upload norms, clipping fraction/shrinkage, mean clean/attacked aggregate, aggregate alignment with the known teacher after lifting, realized parameter-noise energy, predicted score-noise variance and measured score perturbations. Diagnostics are research outputs, not additional private releases. Zero norms yield a documented null alignment rather than NaN. Padding diagnostics distinguish active from inactive energy.

Primary descriptive estimand: each Controlled128 η arm minus Full512 final AUROC, reported separately by noise setting and training seed. Report signed points, including harms; η=0 is a deliberately hard boundary check, not an anticipated algorithmic win. Secondary: log loss, Brier score, clean-minus-attacked AUROC, private-minus-clean difference, retained teacher energy and padding discrepancies. Independently recompute rank-based AUROC with ties and finite-input checks from saved generated predictions; also retain theoretical teacher/observed-label metrics as reference, not fitted baselines.

No discovery p-values or clinical technical margin. Four training seeds and five bases support finite-grid descriptive comparisons, not precise population claims. For random bases report all 5×4 points, each basis's training-seed mean/range, and between-basis variation separately from within-basis variation. Do not pool reused Full512 baselines as independent samples. Do not claim joint data/training/projection uncertainty. The generated dataset remains a single fixed draw.

## Software shape, qualification and evidence

Proposed focused files under `docs/artifacts/capstone/repro/`:

- `projection_study_design.py`: typed resolved cells, separate RNG namespaces, exact counts and budget validation.
- `projection_study_workload.py`: generated splits/teacher, controlled/random/padded transforms and client packs.
- `projection_study_strategy.py`: study-only canonical pipeline adapter and explicitly coupled isotropic Gaussian draws; no changes to historical strategy code.
- `run_projection_study.py`: preflight, qualification and one-attempt execution; reuse canonical strategy, accountant and attack; no medical runner imports that load bundles.
- `analyze_projection_study.py`: independent metric/geometry checks, public JSON and plots.
- Corresponding `test_*.py` files containing meaningful invariant, admission, isolation and artifact-failure checks.

Use the existing clean CPU environment and Python 3.12/PyTorch/NumPy/SciPy. No dependency changes. Qualification cells use separately named generated seeds and outputs; their scores may reveal recipe weakness but cannot justify tuning the already frozen proposed main recipe without an explicit amendment. Up to twelve qualification cells cover local replay, native strategy parity, padded/active noise coupling and attack budgets; exact allocation belongs in the implementation plan before any qualification fit.

The clean CPU profile does not contain Matplotlib. Render only aggregate JSON with the already available research plotting environment, recording its package versions separately. Do not silently import that environment's dependencies into fitting or alter the clean environment locks. Numerical metric verification must remain executable in the clean CPU profile.

Before main measurement, freeze: public manifest with all 136 named cells, dataset/generator hashes, bases, seed derivations, resolved settings, inspected source hashes, output schema, source commit and qualification receipt. Refuse an existing output directory. Record failed/incomplete cells and preserve logs; never synthesize final models or mark success solely from a progress counter. The runner must not make network calls, accept medical data paths or access hardware services. Main execution can use one CPU host; physical placement is not this study's variable.

Minimum correctness checks: nonorthonormal/invalid transforms rejected; exact teacher retention; fixed dataset unaffected by basis/training seed; no global RNG mutation; complete client coverage without evaluation overlap; canonical clipping/noise sensitivity; analytical score variance; padding invariance; independent AUROC ties; allowed attack inputs and evaluation-budget logs; failure cannot produce a successful aggregate; main manifest tamper/source drift refusal. Run these before training. New scientific code requires red/green tests before implementation.

## Success and stop decisions

Success is a reproducible account of which proposed explanations survive, including null/negative results. A sign reversal between retention levels identifies only a bracket on this fixed grid; absence of a reversal means no measured crossover, not permission for a denser search. Repeated random bases quantify only this generated design. The zero-padding identity is an established linear-model property used as an implementation/explanation control, not novel theory.

Stop for source/configuration drift, invariance failure, invalid privacy calibration, forbidden attacker access, nonfinite metrics, exhausted budgets or a review finding that removes the meaningful empirical distinction. Preserve the old reports and record limitations. At completion, reassess whether the evidence adds explanatory value before discussing DP-BREM or any later phase.
