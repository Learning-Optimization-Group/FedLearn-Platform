# Prospective Effusion diagnostic pilots

25 October 2026. Exploratory development-selection study following successful Phase 2 qualification. This protocol and its resolved source/input manifest must be committed locally before fitting. The existing holdout remains closed. Raw models, diagnostics and selection-driven choices are research artifacts, not a whole-program DP release.

## Fixed workload and budget

Use the pinned R0 train/development-selection bundle, 80 nonempty whole-patient clients with full participation, zero-initialized binary linear head, 20 rounds, seven local epochs, batch256, and two development seeds20261211 and20261212. Unless explicitly varied below, LR0.03 and outer clip0.05. Private iterative cells use exact GDP epsilon4/delta1e-5 over20 releases, verifying actual noise. The scalar ERM comparator has one final calibrated Gaussian release at the same total epsilon/delta under its separately declared sensitivity.

The first screen has80 cells:

| Group | Design | Runs |
| --- | --- | ---: |
| Implementation anchors | Native/ByzFL mean, clean/private, two seeds | 8 |
| Center dynamics | tau/C0.25,1,4; clean/owned PGD f16; noise-free/private; two seeds | 24 |
| Attack qualification | ByzFL mean; own individual sign flip x-4, coordinated full-norm owned sign flip, constrained owned-data PGD; f16/80; clean/private; two seeds | 12 |
| Existing robust control | Trimmed mean k16; clean/owned PGD f16; clean/private; two seeds | 8 |
| Public projection | Fixed nested public widths64/128/256; ByzFL mean; clean/private; two seeds. Width512 anchors reused | 12 |
| Nonprivate recipe check | Unbucketed median; LR0.03/C0.05 and LR0.1/C1; two seeds | 4 |
| Scalar convex comparator | B1, lambda0.01/0.1/1; clean/private; two seeds | 12 |

The arithmetic allocates development effort rather than claiming confirmation. Every outcome, failure and unrun cell remains accounted for. Implementation anchors run first; an unexplained native/ByzFL parameter discrepancy above1e-6 stops later fitting. Saved-head and engine AUROC must agree within1e-10. The unperturbed comparator is a matched mechanism control; the separate nonprivate recipe check assesses practical utility under a bounded common tuning budget.

## Attack information, constraints and qualification

Malicious identities are seeded and persist across rounds. Owned attacks share information only among same-type malicious clients. Inputs are their own examples/labels, own locally trained updates, current released global model, previous successfully released delta, public cohort/recipe and the declared optimizer budget. They cannot read current honest updates or selection labels. Owned sign flip negates the coalition mean and uses the full joint weight/bias clip radius. PGD starts at that same candidate, uses20 projected steps with step ratio0.25 and three restarts, and retains the best owned-data loss of a uniform-mean surrogate. The maximum budget is63 loss and60 gradient evaluations per round. Different robust reducers can respond differently to this surrogate.

Floating reconstruction may slightly change raw upload norms. The server's mandatory outer clip enforces the reducer input bound; report intended and represented/pre-server norms without claiming certified floating-point privacy. Previously released history advances only after a successful aggregation. Research diagnostics include actual budget, center/history state and norms.

Use mean clean-to-attack AUROC loss as the positive-control endpoint. An attack is considered damaging for this pilot if the mean loss over its two seeds is at least0.02 in the nonprivate reference. Report each seed and private damage separately; no significance or equivalence claim follows from two seeds. Failure to meet this criterion is failed attack qualification, not evidence that defenses are robust. Improving on the original under-sized sign flip alone does not establish a better direction than full-norm sign flip. Keep both controls visible.

## Projection and convex-head boundaries

Projection uses public seed20261025, orthonormal nested columns and no data-fitted coordinate rescaling. Width512 is the identity control. Record clean utility cost alongside private utility, not only privacy gains. Existing shared normalization assumptions remain.

The convex comparator uses scalar logistic loss, equal client-average weighting, row-norm bound1 and all-coordinate regularization including intercept. T is fixed from public constants to make the solver residual sensitivity at most1% of the exact-minimizer component, with a5000-step cap. For n80, lambda0.01/0.1/1 gives454/50/9 steps. Sensitivity includes both exact and finite-solver components; raw fits and optimizer diagnostics are not private outputs. Its trusted coordinator and scalar parameterization differ from the historical two-logit iterative head. Report those changes explicitly; this is a utility comparator rather than a production federated deployment claim. See [derivation and preparation](phase-3-mechanism-preparation-2026-10-25.md) and [implementation boundaries](client-private-erm-implementation-2026-10-25.md).

## Conditional nonprivate bucket stage

After the screen, choose one shared nonprivate recipe using the larger two-seed mean clean AUROC of the unbucketed median; ties choose LR0.03/C0.05. Freeze the selected recipe and the screen summary hash before this stage. This is declared development selection, not an independent test.

The bucket stage uses median with bucket sizes1/2/4, IID/patient-whole Dirichlet alpha0.1 partitions, two seeds, and four conditions: clean, owned PGD f4/80, owned PGD f16/80, and raw-honest oracle Fang f16/80 with b2. All arms are nonprivate. The two IID unbucketed clean references are reused from the pinned recipe screen rather than rerun, leaving46 new cells. The oracle arm is a separate nonprivate stress test and carries no client-DP claim. Record actual contaminated-bucket fraction; do not infer failure solely from whether a bucket contains an attacker.

## Analysis, freezing and failures

Record AUROC, average precision, Brier score and sensitivity/specificity at probability0.5, every seed, actual privacy calibration and complete resolved configurations. That fixed threshold is descriptive and is not a newly selected clinical operating point. Patient identities, memberships, margins and raw records remain local. Public summaries contain aggregates and nonidentifying hashes only.

The center screen records inner versus outer clipping, released-center norm/alignment and mean update norm. A clean tau=C first step should agree with clipped mean; later private history may preserve signal or noise. No candidate wins by assumption.

Preserve partial round journals, failed returned records and available final heads. Refuse existing outputs or changed source/runtime/input/config/snapshot bytes. Each new stage has its own prospective public manifest and local source checkpoint. Do not relax tolerances or replace failed runs silently.

Candidate selection for the later finalist study is exploratory and will use these retained pilot results. Freeze its narrower scientific question, fresh training seeds, practical margins, precision target and failure rules before execution. Reused selection cohorts and two-seed pilots cannot establish clinical/external validity, equivalence, or universal method superiority.
