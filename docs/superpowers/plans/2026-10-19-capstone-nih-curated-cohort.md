# NIH curated development cohort implementation plan

**Goal:** Resolve the confirmed development/evaluation content overlap through a reproducible whole-patient quarantine while preserving official test and original evidence.

**Architecture:** Reverify the original manifests and complete image audit, derive implicated non-test patients solely from exact RGB groups, and reconstruct a separate cohort with retained row order and freshly allocated training clients. Publish immutable manifests, an excluded-row ledger and narrow content-gate provenance; the verifier reconstructs all decisions rather than accepting edited exclusion lists.

**Tech stack:** Python 3.12, CSV/JSON/hashlib/pathlib, NumPy, existing NIH preparation/audit modules, whole-patient partitioner and pytest.

**Spec:** [NIH development-cohort quarantine design](../specs/2026-10-19-capstone-nih-cohort-curation-design.md).

**Execution:** Direct sequential execution under the continuous capstone mandate, after closing the image-audit slice. Frozen feature preparation remains deferred until this gate is implemented and reverified.

## Global constraints

- Policy is exactly `nih-development-quarantine-v1`. No performance-based exclusions, relabeling, redraw of validation, source modifications or overwritten artifacts.
- A failed original content qualification is accepted only as **verified evidence** of known duplicates; any undecoded image blocks this policy.
- Quarantine all non-test rows for a patient implicated in any cross-patient or contradictory-label exact RGB group. Keep every original official-test row unchanged.
- Original patient-separated split membership/order survives for retained rows. Recompute training clients from retained patient IDs with the original seed/client count; never reuse old row indices.
- Preserve every quarantined row and its cause/group-digest ledger privately. Public docs contain aggregate counts and hashes, not patient/image dumps.
- Use a new output outside source/original preparation/audit directories. Refuse existing/symlink output, reserve exclusively, and publish the fsynced completion report last.
- Narrow claims: selected development rows have no ambiguous exact RGB group and independent splits have no exact RGB overlap. Within-test ambiguities remain retained and disclosed. No near-duplicate, clinical validity, learning or privacy claim.

## Review focus

1. An ambiguous group spans multiple original patients and splits: all implicated non-test patients are quarantined wholly, and official-test rows remain byte-identical.
2. A same-patient pair has contradictory labels: quarantine that development patient; same-patient same-label duplicates alone do not trigger quarantine.
3. Cross-patient groups occur only within training: quarantine both development patients so duplicate content cannot cross simulated clients.
4. A crafted reason list excludes an unrelated patient or deletes a positive row: reconstruction must refuse even if every declared file hash is recomputed.
5. Quarantine leaves too few patients for the original client count, or empties a split: refuse before publishing output rather than create empty clients or redraw splits.

## Files and interfaces

- Create `framework/benchmarks/nih_cohort.py` for curated manifests, provenance, verifier and CLI.
- Create `framework/tests/test_capstone_nih_cohort.py`; reuse the real synthetic NIH tree fixture.
- Update `wikis/capstone/README.md`, the NIH data-audit report and the deferred feature plan after real qualification.

```python
def prepare_cohort(root: Path, original_preparation: Path, image_audit: Path,
                   output: Path, *, audit_workers: int = 1) -> dict: ...
def verify_cohort(root: Path, original_preparation: Path, image_audit: Path,
                  output: Path, *, audit_workers: int = 1) -> dict: ...
def main(argv: list[str] | None = None) -> None: ...
```

Consume `verify_preparation`, `load_source`, `split_patients`, `verify_image_audit` and `patient_iid_partition`. Audit `images.csv` binds source identities; `duplicates.json` gives exact RGB groups already independently reconstructed by full verification. Derive actual ambiguous membership from group member rows, not from boolean summary flags alone. CSV format is the original five-column NIH schema. Output files are `train.csv`, `validation.csv`, `test.csv`, all `client-NNNN.csv`, `quarantined.csv`, `reasons.json` and `report.json`.

Report schema is independently named, with policy version, original preparation/audit SHA-256 values, original source hashes, original settings, original/selected/excluded split counts, reason categories, retained official-test ambiguity counts, precise content-gate assertions, assignment hash and every payload hash. `verify_preparation` remains unchanged and must not accept this new schema implicitly.

## Task 1: Implement, verify and qualify the curated cohort

- [x] Write failing tests with synthetic exact duplicates across train/test and validation/test, as well as within-development cross-patient and same-patient contradictory-label cases. Assert whole-patient exclusion, correct reason digests, unchanged test bytes/order/labels, retained training/validation row order, new client union/ownership and deterministic outputs.

```python
def test_quarantine_keeps_official_test_and_removes_whole_development_patient(case, tmp_path):
    root, original, audit, implicated_patient = case
    output = tmp_path / "curated"
    report = prepare_cohort(root, original, audit, output)
    assert (output / "test.csv").read_bytes() == (original / "test.csv").read_bytes()
    assert report["development_content_qualified"] is True
    assert implicated_patient not in read_patients(output / "train.csv")
```

- [x] Include clean-input/no-quarantine and within-patient same-label duplicate controls. Test undecoded original images, infeasible remaining clients, empty retained train/validation, altered originals/audit, unsafe/existing outputs and interrupted publication.
- [x] Test forged/rehashed exclusions, reasons, settings, official-test changes, client row indices and row/patient ordering. Verification reconstructs policy and every output byte; no edited list is trusted.
- [x] Run `PYTHONPATH=framework/src:framework <python> -m pytest -q -o addopts= framework/tests/test_capstone_nih_cohort.py`. Expected RED: missing module/functions before implementation.
- [x] Implement policy reconstruction, full-source audit verification, exact CSV payloads and report assertions. Derive non-test implicated patients from group members. Build selected splits as ordered filters, quarantine every original non-test row for those IDs, and refuse impossible splits/clients. Check every selected exact RGB group for remaining cross-split or development ambiguity before writing.
- [x] Implement immutable publication and full reconstruction verifier with clear refusal errors; CLI has `--data-root --original-preparation --image-audit --output --audit-workers --verify-only`, no ignore/force flag. Successful curation keeps the original failed audit unchanged.
- [x] Run targeted NIH/partition/workload/reproduction checks and full framework suite with coverage and thread exceptions treated as errors. Expected GREEN; keep logs.
- [x] Prepare the real cohort into a new ignored directory and separately reverify it. Independently check quarantine membership/reasons and exact retained rows against original metadata and full audit. Expected current counts: train 68,650/22,399; validation 17,763/5,600; official test 25,596/2,797; excluded 111 images/nine development patients. These are assertions against measured outputs, not hardcoded cohort selection.
- [x] Update the data-audit report and capstone plan, then revise frozen-feature inputs to require this explicitly verified cohort alongside original manifests/audit. Do not extract features until that interface revision is complete.
- [x] Commit as `feat(capstone): quarantine ambiguous NIH development patients`.
