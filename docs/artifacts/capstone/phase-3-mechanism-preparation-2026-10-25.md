# Diagnostic pilot mechanism preparation

25 October 2026. Preparation only. No patient-data fitting, scoring or new study launch is authorized by this document. The four-cell Phase 2 qualification remains a prerequisite to medical-workload pilots, with its separate local checkpoint approval still pending.

## Decisions and boundaries

The accepted diagnostic phase needs a permitted-information attack and a fixed/public dimension-reduction comparator. These are prepared as standalone functions in `repro/capstone_pilot_mechanisms.py`, with generated-data tests in `repro/test_capstone_pilot_mechanisms.py`. They do not modify the frozen Phase 2 engine or runner, load a cohort, or provide an engine integration.

Ruling: keep this preparation separate from the Phase 2 source snapshot. Changing the engine before its reconciliation would invalidate the current preflight and enlarge the qualification question. The cost is a later explicit integration and checkpoint before medical pilot execution.

## Owned-data constrained attack

The candidate maximizes binary cross-entropy on coalition-owned examples using projected gradient ascent over one shared upload delta with L2 norm at most C. The fixed-budget surrogate model is:

`theta_surrogate = theta_released + (1-rho)*delta_previous_release + rho*delta_attack`.

Here rho is the declared coalition fraction and the previous released delta is a forecast, not knowledge of the current honest updates. Missing history is zero. The returned vector is flattened weight followed by bias. The attack API accepts public model parameters, the coalition's own examples/labels, an initial candidate and previous released history. It has no current-honest-update or selection-data argument. Integration must enforce that information boundary; an API name alone does not prove the caller used permitted inputs.

Each restart takes exactly `steps` gradient evaluations and `steps+1` loss evaluations. The first restart uses the projected supplied initial candidate; later restarts use a local seeded Gaussian direction projected into the same ball. The best observed owned-data objective is retained, including the initial candidate. This is a bounded heuristic with a uniform-mean surrogate, not a provably optimal attack, a reproduction of a named published algorithm, or an attack tailored to every robust reducer.

The pilot must include both the original sign flip and a full-norm sign-flip initialization under the same upload bound. Improving on an under-sized upload alone would not demonstrate a better attack direction. The one-dimensional synthetic positive control confirms that constrained uploads can damage a clipped mean; it does not establish superiority over a full-norm sign flip or damage on Effusion. Before ranking defenses, qualify this attack on the actual reference and record any failed qualification. Do not tune its search using selection labels.

Privacy argument boundary: when replacing one honest client's data, coalition-owned data and identities are fixed. Conditional on previous released history, the malicious uploads cannot depend on current raw honest updates. This avoids the A14 oracle dependency. It does not establish privacy for a coalition member whose raw data are shared to other attackers and may affect multiple uploads. Nor does it establish privacy of diagnostics, validation outcomes, preprocessing or the whole program. Research seeds are reproducibility controls, not production secret randomness.

## Fixed public projection

For input width d and reduced width k, generate a d-by-d Gaussian matrix using only the declared public seed, take a QR orthonormal basis, and use its first k columns. Reduced widths share nested columns. Transform rows by `x_projected = x @ Q_k`. No factor sqrt(d/k), training-derived PCA, coordinate standardization or label-based feature selection is applied by this helper. The full-width control is exactly the original features through the identity matrix.

The projection has operator norm one in exact arithmetic. It can lose useful signal; it does not guarantee improved AUROC. With a k-wide two-logit head, the number of released parameters becomes 2(k+1). At fixed aggregate sensitivity and Gaussian multiplier, reducing that count reduces RMS coordinate-noise norm, but optimization, clipping and lost signal still determine utility. Compare clean and private cells at each width, retain every width, and freeze one projection seed before inspecting patient outcomes.

Any existing shared/private preprocessing limitation remains. A public projection adds no data-fitted transformation, but does not repair protected normalization or turn subsequent data-dependent model selection into a private release.

## Private convex-head comparator: feasibility assessment

Stock sample-private logistic regression is not automatically a client-private comparator. A suitable candidate is output perturbation of a strongly convex, equally client-weighted logistic objective. The classical ERM work provides the sensitivity approach and conditions; the client-averaged objective and finite-iteration bound below are the proposed adaptation, not a claim that its stock implementation already exists here. [Chaudhuri, Monteleoni and Sarwate, 2011](https://jmlr.org/papers/volume12/chaudhuri11a/chaudhuri11a.pdf).

Let each client's loss be its mean binary logistic loss after a fixed row-norm bound B. Include an intercept in theta and regularize every coordinate, including that intercept. For a fixed nonempty cohort of n clients:

This derivation uses one scalar logit with d+1 parameters, not the attack helper's two-logit softmax parameterization. Reusing a two-logit head would require different gradient/Hessian constants (sqrt(2)*L and L^2/2) and recalibration. A scalar comparator also changes parameter count and clipping geometry relative to the historical head; include that in the interpretation rather than attributing all utility changes to the optimizer.

`F_D(theta) = (1/n)*sum_clients mean_owned_rows logistic_loss(theta; x,y) + (lambda/2)*||theta||^2`.

With augmented row norm at most L=sqrt(B^2+1), each client's mean loss gradient has norm at most L. F is lambda-strongly convex. Its exact minimizer has replace-one-client sensitivity at most `2L/(lambda*n)`: neighboring objectives differ by at most `2L/n` in gradient, and strong convexity converts that into a minimizer-distance bound.

A finite solver cannot simply inherit the exact-minimizer bound. One conservative route uses exactly T full-gradient steps from zero with step size `1/beta`, where `beta=lambda+L^2/4`. Writing `q=1-lambda/beta`, the deterministic optimization error is at most `q^T*L/lambda`; the approximate solver therefore has sensitivity at most:

`2L/(lambda*n) + 2*q^T*L/lambda`.

This uses bounded logistic gradients/Hessians, regularization of every parameter, a fixed cohort and fixed public T, B and lambda. It is an exact-arithmetic argument that needs numerical implementation checks. A private convergence test or data-dependent failure/release rule is not covered by it. One final Gaussian release could be calibrated against that sensitivity using the existing exact GDP helper with one release. Releasing intermediate raw optimizer states would invalidate that output-only analysis.

This comparator is feasible in principle and merits a small declared utility screen after qualification. A centralized simulator implementation has a trusted-coordinator/data-access assumption; it would be a utility comparator, not evidence of a distributed deployment. A federated implementation can compute client-average gradients without returning the raw examples, but its communication and transcript assumptions must be stated separately. Do not claim it has been implemented or tested on Effusion yet. Objective perturbation is another relevant candidate; its Hessian and group-privacy conditions need separate analysis before substituting it.

## Remaining gates

- Preserve the current Phase 2 manifest and finish its four real reconciliation cells after checkpoint approval.
- Integrate attacks with an explicit malicious-client-only data view and tests proving current honest records cannot affect its upload.
- Record the attack's actual budget, initialization, norms and permitted-input provenance per round; keep those research diagnostics outside released-model DP claims.
- Integrate projection before head construction, validate parameter dimensions and use matched clean/private controls.
- Freeze a small patient-data pilot matrix, source/input hashes, failure handling and selection rules before execution.
- Treat generated-data damage as mechanism evidence only; Phase 3 is incomplete until the declared actual-workload pilots and attack qualification are evaluated.
