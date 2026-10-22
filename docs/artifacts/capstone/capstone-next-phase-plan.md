# Capstone next phase: learning quality first, simulator delivered at the end

Approved direction, 20 October 2026. This priority update supersedes the previous
dashboard-first next step. It preserves the completed 360-cell study, its original
protocol, results and independent checks. It does not recast that study as evidence
of clinical usefulness or promise that further tuning will produce a clinical model.

## Working objective

Deliver a reproducible MSc capstone with a stronger, qualified prediction baseline;
an explicit comparison with federated learning; supported secure-aggregation,
central-DP and local-DP comparisons; controlled attacks and evidence-bound heatmaps;
practical conclusions; a final report/presentation; and a reliable Simulation Lab
integrated into the existing FedLearn dashboard. Preserve patient/content separation,
historical evidence, privacy-unit clarity and an untouched final evaluation.

**Target decision, 20 October:** the follow-up should detect one specific
condition, rather than predict any finding. **22 October:** the user selected the
NIH `Effusion` label, evaluated on a locked, patient-disjoint internal holdout carved
label-blind from validation; see the draft [protocol v1](effusion-baseline-protocol-v1.md).
The paragraph below records the pre-decision state. **Gate v1 (22 October)
failed criterion (a):** CheXpert features are not distinguishably better than the
existing ImageNet features (0.829 vs 0.822 selection AUROC), though both clear the
0.80 floor; see the [gate result](effusion-gate-v1-result-2026-10-22.md). Federated,
privacy and attack work wait for an explicit amendment choosing how to proceed.
Choose it using the intended use, label quality, positive-patient counts and an
independent evaluation plan before relabeling data or fitting candidates. The
completed No Finding/any-finding study retains its original target and results.
The [development-label feasibility audit](condition-feasibility-audit-2026-10-20.md)
compares positive image/patient counts and reference-standard access without
selecting a condition or scoring models.

The [medical-backbone candidate check](medical-backbone-candidate-2026-10-21.md)
now pins a CheXpert-specific DenseNet checkpoint and verifies synthetic CPU feature
extraction/replay. It establishes technical preparation only: no condition selection,
NIH fitting, utility gain or new clinical qualification. Its 1,024-wide features need
a separately versioned cache and qualification; the existing 512-wide workload stays
historical. A [development-only real-image probe](medical-input-probe-2026-10-22.md)
then bound eight train/validation X-rays to the original audit and confirmed the
preprocessing contract and frozen 1,024-wide features; it is still not a utility result. The learning sequence below remains binding.

## Required sequence and gates

1. **Learning baseline is the immediate priority.** Select the specific condition
   and define its intended prediction use, labels and evaluation cohort. Audit the
   current features/training/thresholds, then run a bounded centralized comparison
   using training and validation only. Compare a small, justified set of stronger
   feature/backbone or fine-tuning candidates where hardware and data permit.
   State success criteria before selection: meaningful discrimination and useful
   sensitivity/specificity/calibration, compared with simple controls and the older
   baseline. Do not substitute an arbitrary accuracy target for medical usefulness.
   If the existing task/data cannot support the purpose, document the evidence and
   justify a revised task/dataset; do not switch solely to obtain a higher score.
2. **Qualify the federated baseline without privacy noise or attacks.** Use matched
   data, model, evaluation and training controls to distinguish losses from the
   representation, optimization and federation. Freeze a viable learning recipe
   before expanding the mechanism/attack matrix. Record a failed gate honestly.
3. **Measure privacy/security costs on clean training first.** Compare no protection,
   central DP, local DP and secure aggregation only where actually implemented and
   qualified. Define what is protected (patient, record or client update), adjacency,
   trust assumptions, composition and release boundaries. Equal epsilon values in
   different threat models are not automatically equivalent comparisons. Standard
   secure summation does not provide ordinary coordinate trimming, and secure
   aggregation alone does not supply differential privacy.
   The [implementation-readiness audit](privacy-security-readiness-2026-10-20.md)
   identifies the required local-DP reporting/qualification work and the separate
   DeComFL secure channel; it is not a new utility study or an approved combined path.
4. **Add bounded attacks after clean mechanism controls.** Predefine attacker
   access, fractions, budgets and valid defense combinations. Repeat selected cells
   across declared seeds; retain failed, partial and incompatible outcomes. Compare
   each attacked run with its own matched clean control.
5. **Produce heatmaps and practical conclusions.** Show absolute performance and
   attack-induced drops separately, with uncertainty, sample sizes, privacy scope,
   communication costs and measured runtime. Mark incompatible/unrun cells explicitly,
   never as zero or successful protection. Avoid choosing a winner from a single seed.
6. **Deliver a good simulator as a required final product.** Complete the existing
   dashboard integration with real configuration, owned start/stop, actual progress,
   persistent history/reopen, comparisons, heatmaps and safe research-record export.
   Verify the real browser-to-API-to-worker path, cancellation/restart/ownership,
   metric/provenance consistency and a reproducible demo. Label replay separately
   from live execution. Research results and final simulator are both required.
7. **Finish assessed deliverables.** Reconcile report, presentation, figures and
   simulator with the evidence; retain reproducibility/backup and rehearsal time.

## Evaluation and schedule boundaries

The previous official NIH test results have already been inspected. They remain
historical reference results, not an untouched test for the new selection process.
Tune only on development data. Establish a genuinely untouched final evaluation,
with patient/content independence and a prospectively frozen follow-up protocol,
before making new confirmatory claims. A post-hoc subset of the already examined
test set does not restore an unseen evaluation.

The stated deadline remains 5 December. From 20 October this is 46 days; exact
advisor title/format requirements remain open. Use a bounded baseline diagnostic
first, then size the follow-up matrix from observed learning and runtime. Reserve
time for the required simulator, report and rehearsal. An exhaustive all-model,
all-mechanism, all-attack sweep is not the objective.

## Existing implementation checkpoint

The bounded Python worker is committed and verified. The persistent Spring API,
committed locally at `53d60fc`,
passes 32 focused checks after the two Important review findings were reproduced
and corrected. The full backend gate passed all 1,207 tests, no skips, and 84.09%
line coverage. An admin-login HTTP EOF in the initial full run did not recur in
isolation or the instrumented full run; its cause is unresolved, not fixed.
Preserve the tested API checkpoint and private ledger. The React workflow now
passes 179 tests, lint and production build, with 57.74% line coverage. The actual
browser-to-API-to-NIH-worker flow, ownership, reload/export/comparison, partial stop,
keyboard/responsive checks and disabled-workload refusal also pass. The
[aggregate receipt](simulation-lab-browser-verification-2026-10-21.json) and
[live/replay guide](simulation-lab-demo-guide.md) record their scope and limitations.
These are short any-finding validation demonstrations on one Mac; they do not qualify
a disease model, a privacy-effect comparison or a physical-device benchmark.
The working report PDF now includes the dashboard checkpoint in eight visually
checked pages, with a portable renderer and source/font/dependency receipt. It
remains a working historical-study report. Off-device preservation of the new
checkpoint and final presentation/rehearsal remain open. Further dashboard expansion must not displace the learning-baseline priority.

The simulator remains an explicit final acceptance gate. Deferring its expansion
now does not remove it from the project.
