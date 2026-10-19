# NIH development-cohort quarantine design

Date: 19 October 2026. Status: selected under the continuous capstone mandate, before classifier performance inspection; implemented and independently reverified; final review pending.

## Reason for the extension

The [complete image audit](../../artifacts/capstone/nih-data-audit-2026-10-19.md) found three exact RGB groups crossing independent splits, eight cross-patient groups and six contradictory-label groups. These are overlapping categories. The original manifests remain valid metadata evidence but fail the image-content gate. Do not weaken that gate or overwrite the original cohort to fit the intended schedule.

## Choice and tradeoff

Use a separately named, deterministic whole-patient quarantine for development data. Merely deleting colliding image rows leaves the rest of an ambiguous patient's data in development. Changing official-test membership weakens comparability. Whole-patient quarantine costs 111 development images from nine declared patients on the present audit, while preserving the primary official test and all original evidence.

This is a data-quality policy chosen before model-performance inspection, not an assertion that these patients or labels are clinically wrong. No performance-based exclusions, identity corrections or relabeling are allowed. The cost of an over-conservative decision is loss of legitimate development data; retain exact excluded identities/reasons privately so that cost is measurable.

## Deterministic policy: `nih-development-quarantine-v1`

1. Require a fully reverified original source preparation and image audit. Audit completion can have `qualified: false`, but every image must have decoded successfully. Corruption remains a blocker; this policy cannot make unreadable images acceptable.
2. Use exact RGB/dimension groups from that full audit. A group is ambiguous when it contains more than one declared patient ID or more than one binary label.
3. Collect every **non-test** patient ID occurring in any ambiguous group. Quarantine all original training/validation rows for those patients, including rows outside the duplicate groups. All endpoints of an ambiguous cross-patient edge are covered by this rule; no arbitrary representative is chosen.
4. Keep all remaining training/validation rows in their original split and order. Never redraw validation to fill the removed rows. Keep all official-test rows, labels, patients and order exactly unchanged.
5. Recompute whole-patient training-client allocation with the original declared seed/client count. Record new assignments and actual counts; old client manifests must not be reused with new row indices.
6. Assert no exact RGB group crosses the selected independent splits and no ambiguous group retains any development row. Original patient disjointness still holds. Within-patient, same-label duplicates may remain; within-test ambiguities remain explicitly disclosed.
7. Write a new immutable output containing split/client CSVs, quarantined-row CSV and a reason ledger linking excluded patients to audit group digests. Bind original preparation/audit reports, source hashes, policy version, selected/quarantined rows, assignment hash and payload hashes. Publish a completion report last. Never emit a broad `data_clean` or clinical-validity claim.
8. A verifier independently reconstructs the same policy from the original verified preparation/audit, compares every output byte, and refuses stale settings, changed reasons, forged exclusions, altered official test, incomplete output or mismatched source identities.

Expected present counts, to be checked rather than hardcoded: training 68,650 images / 22,399 patients; validation 17,763 / 5,600; official test 25,596 / 2,797. Excluded development: 111 images / nine patients. Labels and feature extraction remain the explicit No Finding/any-finding task.

## Integration boundary

Implement in a separate `framework/benchmarks/nih_cohort.py` module with its own schema and verifier. Do not make `verify_preparation` accept a modified original report. Future feature preparation must distinguish the original audit and explicit curated cohort; it cannot accept a failed audit through an ignore flag. Selected-cohort verification must establish the content gate before publishing a feature cache.

The existing frozen-feature plan is deferred until this adapter is implemented and its interfaces are added explicitly. Patient-to-client grouping, explicit evaluation split selection, learning pilots and protocol freeze follow in order. Do not skip these gates to start the confirmatory matrix on October 20.

## Scientific disclosure

State that official-test label prevalence and duplicate/ambiguity membership were examined during preparation, before the prospective phase-2 protocol and without classifier performance inspection. Primary official-test membership stays unchanged. Predeclare any evaluation sensitivity analysis independently of the eventual measured effects. A small number of confirmed duplicates establishes a real preparation failure; it does not measure how much a model score would have been inflated.
