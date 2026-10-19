# NIH image integrity audit implementation plan

**Goal:** Decode and fingerprint the complete prepared NIH cohort, retaining corruption and duplicate evidence before feature extraction.

**Architecture:** A standalone audit consumes verified patient manifests, reads each source image once, records raw-file and canonical RGB-pixel SHA-256 identities, and reconstructs duplicate groups across the three independent splits. It publishes a new audit directory with a completion report last. A completed audit can fail qualification; that is retained evidence, not permission to train.

**Tech stack:** Python 3.12, Pillow, standard-library CSV, hashlib, JSON and pathlib; pytest fixtures use real tiny PNG files.

**Spec:** [NIH workload design](../specs/2026-10-18-capstone-nih-workload-design.md), data-contract corruption/content-duplication gate.

**Execution:** Direct sequential execution under the continuous capstone instruction. Keep scientific decisions and deviations in the execution ledger. This bounded slice precedes feature preparation and engine integration.

## Global constraints

- Never modify source files or an existing output, including symlinks. Outputs must be outside the source dataset and preparation directory.
- Preserve original test membership and all patient manifests, even if a collision is found. No silent exclusions, relabeling or deduplication.
- Exact decoded RGB pixels and dimensions define content identity. Raw bytes are hashed separately so re-encoded copies can be detected.
- Accept single-frame PNG input; refuse decode failures, non-PNG or multi-frame images. Pillow decompression warnings become audit failures. Bound input bytes to 64 MiB and decoded pixels to 16,777,216 per image before RGB allocation.
- Process one image at a time per worker, with at most 16 workers and 128 queued identities; no full-resolution cohort in memory. Record every source row, including failed images; failures have no pixel digest.
- Cross-split raw or RGB duplicates and image failures make `qualified` false. Within-split duplicates are disclosed, including cross-patient and contradictory-label cases, rather than silently removed.
- A completed audit report states exact-identity limits: no near-duplicate or clinical-quality guarantee, no learning or DP claim. The manifest report remains unchanged with its original `image_content_verified: false` boundary.
- A verifier must reconstruct all image fingerprints from source files and compare every output byte; rehashing a forged audit must not pass.

## Review focus

1. Re-encoded PNGs with different bytes but identical RGB pixels must form the same content group, including grayscale/RGB encodings.
2. Corrupt, oversized, animated or mislabeled files must remain visible as failed rows, without publishing a qualified report.
3. A duplicate in training and its client CSV is not a cross-split leak; only train/validation/test form the independent cohort.
4. A write interruption or existing output must never overwrite evidence or yield a verifiable completion marker.
5. Source mutation and forged/rehashed payloads must fail full verification, including changes that preserve image dimensions and labels.

## File map and interfaces

- Create `framework/benchmarks/nih_image_audit.py`: audit, full re-verification and CLI.
- Create `framework/tests/test_capstone_nih_image_audit.py`: synthetic image and filesystem contracts.
- Reuse the `nih_tree` fixture from `test_capstone_nih_data` through an explicit import; do not change the original fixture or parser.
- Update `wikis/capstone/README.md` after real-data audit with counts, duplicate/failure status and remaining gates.

Public interfaces:

```python
def audit_images(root: Path, preparation: Path, output: Path, *, workers: int = 1) -> dict: ...
def verify_image_audit(root: Path, preparation: Path, output: Path, *, workers: int = 1) -> dict: ...
def main(argv: list[str] | None = None) -> None: ...
```

`images.csv` rows follow sorted source image IDs with fields `image_index,patient_id,split,label,image_path,file_bytes,file_sha256,width,height,rgb_sha256,error`. Errors are stable categories, not machine paths. `duplicates.json` contains raw/RGB groups with split, patient and label summaries. `report.json` records schema/audit version, preparation-report SHA-256, source hashes, Pillow version, canonical pixel policy, limits, counts, group counts, qualification and payload hashes. `report.json` is the final atomic completion marker. Audit-qualified input is still rechecked for source identity during subsequent feature reads.

## Task 1: Implement and qualify the image audit

- [ ] Write real-image tests first. The positive fixture has 24 unique RGB images and verified manifests. Assert all rows and source hashes, zero duplicates/failures, `qualified` true and identical rerun bytes.

```python
def test_complete_audit_decodes_every_prepared_image(nih_tree, tmp_path):
    prep = tmp_path / "manifests"
    prepare(nih_tree, prep, num_clients=2)
    report = audit_images(nih_tree, prep, tmp_path / "audit")
    assert report["images"] == 24
    assert report["failed_images"] == 0
    assert report["qualified"] is True
```

- [ ] Add duplicate tests by copying an image into a different independent split and by re-encoding it with a PNG text chunk. Assert both byte-identical and pixel-identical collision groups, preserved output rows and false qualification. Test within-split groups separately; client CSV membership must not inflate counts.
- [ ] Add corruption, PNG-format, animation, byte/pixel limit, refusal, source mutation, forged-payload and interrupted-write tests. Existing outputs contain a sentinel that must remain unchanged.
- [ ] Run `PYTHONPATH=framework/src:framework <python> -m pytest -q -o addopts= framework/tests/test_capstone_nih_image_audit.py`. Expected RED: missing module/public functions before implementation.
- [ ] Implement bounded PNG reading, SHA-256 over original bytes, RGB fingerprint over a versioned namespace plus dimensions and canonical RGB bytes. Use Pillow `verify` then reopen and `load`; refuse unexpected format/frames and oversized content before conversion. Build independent split identity from verified manifests. Do not scan client CSVs as independent splits.
- [ ] Publish all deterministic payloads exclusively, then fsync/replace the report. Implement verification by recomputing fingerprints, duplicate groups and the report against current source; compare all file bytes and refuse extra/missing files. Provide `--data-root --preparation --output --verify-only --workers`; return nonzero if audit is unqualified while preserving completed failure evidence.
- [ ] Run targeted NIH/partition/workload/reproduction checks and the full framework suite with coverage and unhandled thread exceptions treated as errors. Expected GREEN. Record counts and logs.
- [ ] Run the audit on all 112,120 original NIH files into a new ignored output. Record elapsed time, counts, failures and exact duplicates. Independently inspect any groups against source files; do not change cohort policy merely to pass the audit.
- [ ] Update the capstone plan with evidence and practical consequences. Commit as `feat(capstone): audit NIH image integrity before feature extraction`.

## Verification commands

Use the existing root `.venv/bin/python`; `PYTHONPATH=framework/src:framework` from the capstone worktree. Full suite runs from `framework` with `PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 <python> -m pytest -q -W error::pytest.PytestUnhandledThreadExceptionWarning`.

Real command: `<python> -m benchmarks.nih_image_audit --data-root "$NIH_DATA_ROOT" --preparation "$NIH_PREPARATION" --output "$NIH_IMAGE_AUDIT"`. A separate `--verify-only` invocation is a full reread/decode and is required before calling the saved audit independently reverified. No model is trained and no official-test model performance is observed in this slice.
