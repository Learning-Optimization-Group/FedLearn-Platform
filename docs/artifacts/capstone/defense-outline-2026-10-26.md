# Defense outline and speaker notes

26 October 2026. Editable content outline: 12 main slides and 3 backup slides. Proposed main-talk timing is approximately 15 minutes; the advisor has not confirmed duration or format. This is not an exported slide deck and has not been rehearsed. Suggested figures are existing scientific evidence, not decorative illustrations.

## 01 — Measuring privacy and robustness trade-offs in federated learning

**On-slide content**

Anurag Lnu · MS Computer Science · Rochester Institute of Technology

An auditable platform, a conditional utility result, and verified distributed execution

**Speaker notes — 35 seconds**

The question is how representation, aggregation and privacy calibration interact when clients can send malicious updates. The deliverable combines an inspectable implementation with controlled measurements. I will separate the original simulated-client scientific comparison from the later physical-host execution study.

**Source:** [Report abstract/§1](capstone-final-report-v5-working.md).

## 02 — Research question and contribution

**On-slide content**

- Compare composite mechanisms under a declared workload and adversary.
- Preserve configuration, completion, failure and artifact identity.
- Use evidence to choose a research baseline within the tested scope.

**Speaker notes — 65 seconds**

The computing contribution is the execution path and its contracts, together with qualified integration of reducers, privacy calibration and scientific auditing. The empirical contribution is conditional guidance rather than a new theorem. Running many configurations does not by itself establish novelty. The conclusions apply to the tested recipe, cohort and adversary.

**Source:** [Report§1/§2](capstone-final-report-v5-working.md), [claim map C1–C4](consolidation-evidence-map-2026-10-26.md).

## 03 — Implemented execution paths

**On-slide content**

- Browser → authenticated API → supervised simulation worker.
- Canonical coordinator and strategy components shared with the library.
- Separate CLI path for training across physical hosts.

**Speaker notes — 70 seconds**

The browser workflow persists ownership and configuration, exposes actual progress and retains partial outcomes. Worker exit status and evidence reconciliation are required for completion. The research batch runners support the larger matrices. The physical-host extension uses the canonical gRPC client and coordinator, but it is not launched as a distributed job by the dashboard. Do not imply a unified hospital orchestration system.

**Suggested evidence:** the existing curated Simulation Lab screenshot, with its historical date visible if used.

**Source:** [Report§2](capstone-final-report-v5-working.md), [browser receipt](simulation-lab-effusion-browser-verification-2026-10-22.json).

## 04 — Workload and evaluation boundary

**On-slide content**

- NIH ChestX-ray14 Effusion; frozen ImageNet ResNet-18 features.
- 68,650 training images; 9,075 selection images from 2,800 patients.
- Patient separation plus known duplicate quarantine.
- No independent-cohort confirmation.

**Speaker notes — 70 seconds**

The positive class is the exact Effusion label token; labels originate from radiology reports. Known cross-group duplicate content motivated quarantine of 111 images from nine development patients. The final comparison reuses the development-selection cohort. Its fresh training seeds are not fresh patients. The old internal holdout was opened once historically and remains closed; no new external cohort was evaluated because approved access was unavailable.

**Source:** [Report§3/§6](capstone-final-report-v5-working.md), [freeze decision](research-freeze-2026-10-26.md).

## 05 — Frozen finalist comparison

**On-slide content**

4 policies × 2 noise settings × 2 attack settings × 12 training seeds

192 main cells  + 4 native anchors = 196 completed cells

80 logical clients · 20 rounds · full participation

**Speaker notes — 70 seconds**

The four policies are clipped mean512, projected128, trim16 and released-history center_quarter. All use zero heads, SGD learning rate 0.03, seven local epochs, batch 256 and outer clipping C=0.05. The common recipe compares composite mechanisms; it does not establish that every method is individually tuned to its optimum. The primary comparison is private-clean projected128 minus mean512. Secondary contrasts are descriptive and not multiplicity-adjusted.

**Source:** [Finalist accounting and arms](effusion-finalists-2026-10-25.json), [report§3](capstone-final-report-v5-working.md).

## 06 — Privacy and attacker assumptions

**On-slide content**

- Calibrated central Gaussian mechanism: ε=4, δ=10⁻⁵.
- Replacement of one honest logical client, with fixed preprocessing/assignment.
- Trusted coordinator sees individual updates.
- Owned-data attacker; no current raw honest updates.

**Speaker notes — 85 seconds**

Sixteen persistent malicious clients use their own examples/updates and released history. The attack uses a uniform-mean surrogate rather than a worst-case objective for every reducer. For the bounded update mechanism, full-participation GDP composes across 20 rounds. The cumulative research workflow, saved diagnostics and public research randomness are not qualified as a protected private release. Epsilon refers to one configured training run, not a composed privacy bound for the whole experimental matrix. Secure aggregation, local privacy and protection from a malicious coordinator are separate guarantees.

**Source:** [Report§3/§6](capstone-final-report-v5-working.md), [claim C8](consolidation-evidence-map-2026-10-26.md).

## 07 — Projection improves private utility within this cohort

**On-slide content**

Private-clean gain: **+0.019176 AUROC**

- Training-seed 95% interval: **[+0.011011,+0.027340]**.
- Fixed-head patient 95% interval: **[+0.013136,+0.025893]**.
- Clean no-noise cost: **0.014923 AUROC**.

**Speaker notes — 90 seconds**

The seed interval holds patients fixed; the patient-cluster bootstrap holds the fitted head pairs fixed. Neither is a joint uncertainty interval. The lower seed bound exceeds the prespecified +0.01 technical margin under this conditional target. The gain is less than half the two-seed pilot point of +0.0406. One final seed favors full width. These qualifications prevent turning a conditional finding into a universal recommendation.

**Suggested figure:** [Paired gain and separate intervals](figures/remediation-2026-10-25/primary-paired-gain.png).

**Source:** [Finalist primary result](effusion-finalists-2026-10-25.json), [report§4](capstone-final-report-v5-working.md).

## 08 — Absolute utility matters more than an interaction label

**On-slide content**

| Policy | Clean, GDP4 | Owned PGD, GDP4 |
| --- | ---: | ---: |
| mean512 | 0.740361 | 0.551304 |
| projected128 | 0.759536 | 0.683331 |
| trim16 | 0.678365 | 0.605114 |
| center_quarter | 0.666497 | 0.508300 |

**Speaker notes — 80 seconds**

A favorable interaction or lower noise scale does not establish the best absolute utility. Trimming improves private attacked utility relative to mean512 but loses private-clean utility. Center_quarter is not supported as a private default in these conditions. Stale noisy history is a possible explanation, not an established cause. Projection's attacked result is limited to this heuristic adversary and does not certify general Byzantine robustness.

**Source:** [Finalist arm means](effusion-finalists-2026-10-25.json), [report Table 1](capstone-final-report-v5-working.md).

## 09 — Training across three physical hosts

**On-slide content**

Mac-only versus Mac+G14+Orin

- 3/6/24/80 logical clients.
- 52 nonprivate CPU cells; 320 complete rounds.
- 13,180 client fits and acknowledged uploads.
- Maximum placement parameter difference: approximately 3×10⁻⁸.

**Speaker notes — 70 seconds**

Half the cells are Mac-only controls. The 48 systems cells use five rounds and three paired seeds; four twenty-round anchors replay a historical seed. Three physical hosts are three failure domains even when they multiplex 80 clients. One host owns 26 or 27 logical clients here, beyond the original sixteen-client attack budget if that whole host were compromised. Stable reduction order and independent client RNG preserve semantics across transport. Four historical anchor comparisons also pass their engineering tolerance. This extension contains no DP or attack qualification.

**Source:** [CPU verification](multihost-verification-2026-10-26.json), [protocol](multihost-protocol-2026-10-26.md).

## 10 — Three hosts cost more for these short jobs

**On-slide content**

- Every five-round pair is slower end to end on three hosts.
- Observed three-host time: **1.87–2.93× Mac-only time**.
- Steady rounds improve at 3 clients; worsen at 24/80.

**Speaker notes — 80 seconds**

Whole-cell time includes startup, per-cell data staging and shutdown. Steady-round timing uses adjacent completion intervals on the server's monotonic clock after the first round. Three-host placement permits two fits per host, adding resources; it does not hold total concurrency fixed. The study does not isolate polling, transport or host speed as the cause. The negative result is retained rather than optimizing until a favorable graph appears.

**Suggested figure:** [Paired timing ratios](figures/multihost-2026-10-26/cpu-placement-speedup.png). Explain that ratios above 1 favor three hosts and whiskers are observed minimum/maximum over three pairs, not confidence intervals.

**Source:** [CPU measurements](multihost-results-2026-10-26.json), [results interpretation](multihost-results-2026-10-26.md).

## 11 — Reproduction and failure evidence

**On-slide content**

- Three fresh CPU environments, with package-origin checks.
- 164 bounded profile checks passed.
- Generated transport success and failure paths.
- Tracked-files-only source export reproduces both methods.

**Speaker notes — 70 seconds**

The minimal CPU profile is independently installed from hashed dependencies and does not inherit the original shared environment. Generated three-host qualification completes four cases, twenty rounds and sixty client fits. A tracked-source export completes two local cases with exact simulator head agreement. The failure tests exercise unknown, stale, ahead, duplicate and missing-client behavior through real gRPC; they are generated local-transport tests, not physical hospital-outage qualification. This is not a fresh verification of every unrelated platform feature or an external-party replication.

**Source:** [Clean verification](clean-reproduction-verification-2026-10-26.json), [reproduction guide](clean-reproduction.md).

## 12 — Conclusion and limits

**On-slide content**

- Conditional private-utility evidence favors projected128 among four tested policies.
- The platform preserves and audits execution across simulation and physical transport.
- Independent-cohort, deep-model and protected-production claims remain open.

**Speaker notes — 75 seconds**

The research scope is frozen. Approved access to an unused cohort was unavailable, so no external evaluation was attempted and no substitute research project was started. The conclusions remain conditional on the workload, selected cohort and adversary. The practical recommendation is a research baseline with an explicit clean-utility cost, supported by an inspectable execution path. It is not a claim of clinical readiness, a universally best aggregator or production privacy.

**Source:** [Report§7](capstone-final-report-v5-working.md), [claim map](consolidation-evidence-map-2026-10-26.md), [freeze decision](research-freeze-2026-10-26.md).

## 13 — Backup: sensitivity and noise

**On-slide content**

Mean: Δ=2C/80

Trim16: Δ=2C/(80−32)

Released-history center: Δ=2min(C,τ)/80

σ=zΔ; μ=√T/z; no sampling amplification

**Speaker notes**

These formulas condition on the fixed-input/client model stated in the report. The centered sensitivity uses released history and the qualified one-step mechanism. Smaller noise alone does not ensure utility. If asked about patient privacy, a patient change within one fixed honest client is contained within whole-client replacement, but the full preprocessing/selection/research pipeline is not covered and no smaller epsilon follows automatically.

**Source:** [Report§3](capstone-final-report-v5-working.md).

## 14 — Backup: what the attacker sees

**On-slide content**

- Coalition-owned data and released history.
- Matching projected features/head in projected128 arms.
- Uniform-mean attack surrogate.
- Current raw honest updates are excluded.

**Speaker notes**

Do not explain projection's advantage as the attack accidentally remaining in the original feature space: that is not what the implementation does. The attack is weaker than a reducer-specific adaptive worst-case optimizer, which is a real limit. Oracle attacks that read current raw honest updates are separate nonprivate stresses because changing one honest client can also change malicious uploads.

**Source:** [Report§3/§6](capstone-final-report-v5-working.md).

## 15 — Backup: remaining questions

**On-slide content**

| Question | Honest answer |
| --- | --- |
| Best method overall? | Best private candidate among four scoped policies here. |
| Better than NVIDIA FLARE? | No matched framework comparison was run. |
| Clinically validated? | No; no independent cohort and no clinical-use qualification. |
| Production-private? | No; cumulative research release and malicious-host protection remain outside scope. |
| GPU/deep-model result? | No; frozen heads and CPU execution only. |

**Speaker notes**

These are boundaries of the completed evidence, not gaps to hide. Do not turn the access failure into a negative predictive result. A future study needs its own access, cohort, privacy/model assumptions and frozen evaluation. The capstone deliverable is the bounded implementation and evidence already shown.

**Source:** [Current claim boundaries](consolidation-evidence-map-2026-10-26.md).
