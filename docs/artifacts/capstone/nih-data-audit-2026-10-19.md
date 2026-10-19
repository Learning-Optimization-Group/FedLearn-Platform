# NIH content audit — 19 October 2026

## Verdict

The original patient-separated cohort is **not ready for feature extraction**. All 112,120 images decode successfully, but exact RGB content crosses independent splits despite disjoint patient IDs. Preserve this failed qualification as evidence; do not bypass it by changing the audit flag.

## Method and evidence

The audit consumes the verified seed-42 patient manifests: 68,758 training images, 17,766 validation images and the unchanged 25,596-image official test set. Every source image is read with bounded file/pixel limits, verified as a single-frame PNG and fully decoded. SHA-256 identifies original file bytes and canonical RGB pixels with dimensions. These are exact-identity checks, not near-duplicate detection or clinical quality review.

The completed audit records an identity row for every image and publishes its report last. It can complete successfully as an evidence artifact while failing qualification for training. Its original preparation report remains unchanged and still describes metadata-only verification.

| Check | Observed result |
| --- | ---: |
| Images examined / successfully decoded | 112,120 / 112,120 |
| Decode failures | 0 |
| Byte-identical duplicate groups | 0 |
| Exact RGB-content duplicate groups | 25 |
| RGB groups crossing independent splits | 3 |
| RGB groups entirely within one split | 22 |
| RGB groups crossing declared patient IDs | 8 |
| RGB groups with contradictory binary labels | 6 |

These categories overlap. The three cross-split groups include two training/test groups and one validation/test group; one has contradictory labels. File bytes differ within every duplicate group, so file SHA-256 alone would miss the overlaps. Different encodings with identical model-input pixels are sufficient to contaminate a split even when metadata patient IDs differ.

A separate implementation re-opened every member of all 25 reported groups, compared full RGB arrays directly, checked original file hashes, and checked patient IDs/labels against the original metadata. All groups were independently confirmed. A separate full second read/decode reconstructed every audit payload and report exactly. Both CLI invocations exit nonzero because cohort qualification fails, not because evidence verification failed.

Private evidence is retained with the original manifests, per-image fingerprints, duplicate membership and independent checks. No patient/image-level manifest is included in this report. `images.csv` SHA-256: `bb267d0a96d8b19b4da47113c8b76ec05d51e1f39b7a6753f475c2a555c48829`. `duplicates.json` SHA-256: `2e381b3cbc1396a4fadc2e674052280d7c20db4da1952da5b278766581ac7ee6`.

## Consequences for the capstone

Patient separation remains necessary, but this audit disproves the assumption that declared patient separation alone establishes content separation for this downloaded cohort. This does not establish which declared identity or label is correct, and it does not justify correcting labels manually.

The official test set should remain intact for the primary evaluation. The next implementation must derive a separate, explicitly versioned development cohort with a deterministic quarantine policy. Preserve both the failed original cohort and the reasons for every excluded row. Decide this before observing model performance, then disclose the metadata/content inspection in the phase-2 protocol.

A conservative candidate policy quarantines whole non-test patients implicated in either cross-patient exact RGB groups or contradictory labels for identical RGB content. On the audited manifests this would exclude eight training patients (108 images) and one validation patient (three images), retaining 68,650 training images and 17,763 validation images. Official test would retain all 25,596 images. These are policy calculations, not yet a prepared or qualified replacement cohort.

This policy also addresses ambiguous within-training identities that could otherwise place identical content in different simulated clients. It does not resolve label ambiguity inside official test, which remains disclosed. Primary official-test results and any prospectively declared ambiguity/patient-weighting sensitivity must be reported separately.

## Practical project guidance

Dataset preparation should report both patient ownership and decoded-content identity. A clean metadata report or unique filenames cannot replace that check. Preserve the evidence and explicit failure state so downstream training cannot silently proceed on a cohort known to overlap. The platform can reuse this pattern for future dataset preflight checks after the capstone workload is qualified.

No NIH learning, privacy benefit, robustness benefit, clinical efficacy or near-duplicate safety has been demonstrated by this audit.

## Implementation review and remaining limits

The final review independently ran all 32 focused tests and reopened all 51 members of the 25 RGB groups. The full framework check passed 1,664 tests with 14 deselected and 81.04% coverage. No correctness blocker was found for this audit or its failed-qualification outcome.

Two verification limitations remain deferred. Oversized refused files have no full content hash, so same-size changes to those failed inputs are not established by re-verification; they remain unqualified. The report verifier compares normalized JSON values rather than the original report byte representation, so formatting changes can pass. The real audit has no oversized/failed images, and its saved report and both invocation outputs were separately compared byte for byte and are identical. Neither limitation changes the confirmed collisions, but future evidence consumers must not interpret the verifier as unconditional byte integrity for every failed input and every report representation.
