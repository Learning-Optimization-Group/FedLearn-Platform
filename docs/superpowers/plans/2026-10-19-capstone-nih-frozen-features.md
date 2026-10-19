# NIH frozen feature cache implementation plan

Status: deferred after the complete content audit found cross-split duplicates. Implement the [separate curated development cohort](../specs/2026-10-19-capstone-nih-cohort-curation-design.md) first, then revise these interfaces and real counts explicitly. Do not bypass the failed original audit.

**Goal:** Create a source-bound, independently loadable frozen ResNet-18 cache for the verified patient-separated NIH cohort without examining test performance.

**Architecture:** A preparation command fully re-verifies the image audit before extraction, then checks each image's raw and canonical RGB identity again as it prepares bounded forward batches. Features are float32 NumPy arrays, labels are int64 arrays, and original split CSVs preserve aligned patient/image identities. A final report binds all payloads, transforms and actual frozen model state. A loader validates that contract independently of the original image tree.

**Tech stack:** Python 3.12, PyTorch, torchvision, NumPy, Pillow and pytest. Reuse `_build_backbone`, `_IMAGENET_MEAN` and `_IMAGENET_STD` from `benchmarks/dp_on_head_xray.py`; do not reuse its ImageFolder cache identity.

**Spec:** [NIH workload design](../specs/2026-10-18-capstone-nih-workload-design.md), frozen feature section. Engine integration and validation pilots remain separate slices.

**Execution:** Direct sequential execution under the continuous-work mandate. This plan may be prepared while the preceding full image audit runs, but real feature extraction requires a qualified completed audit.

## Global constraints

- Preserve official test and all original metadata/manifests. No balancing, subsetting, triggered features, clinical claims or model selection on official test.
- Use the same frozen ResNet-18 representation for all later head sizes. Default pretrained weights are identified by resolved torchvision weight enum and the SHA-256 of every frozen state tensor, including buffers.
- Preprocessing is fixed RGB conversion, bilinear antialiased resize to 224×224, float tensor in [0,1], and ImageNet mean/std normalization. Record versions, device, dtype and exact transform parameters.
- Explicitly label random-weight test caches as non-pretrained. Never reuse a random cache as a pretrained cache.
- Bounded forward batches, positive non-boolean integer batch size at most 256, device explicitly selected from CPU/MPS/CUDA and refused if unavailable. Do not enable silent device fallback.
- Output must be new and outside source, preparation and audit directories. Refuse existing outputs and symlinks. Reserve output exclusively; report is published last with fsync and atomic replace. Retain incomplete evidence and refuse it on reload.
- All three independent splits retain every row in their original order, labels and patient/image identities. Features must be finite float32 arrays of shape (rows,512); labels int64 and exactly equal to source CSVs. Do not silently turn validation into test.
- Cache loading uses `np.load(allow_pickle=False)` and a fixed file allowlist. Verify all payload hashes before accepting arrays. No arbitrary paths from JSON. Caller may supply the expected report SHA-256 to detect stale/replaced cache identities.
- The original ImageFolder cache/legacy workload behavior remains unchanged. No real cache or patient metadata enters git.

## Review focus

1. A source image changes after full audit verification but before extraction: refuse the mismatched raw/RGB identity before completion publication.
2. Feature rows, labels or identities are permuted or rehashed: compare against the bound manifest identity and reject stale report identity; require explicit expected report hash in future NIH engine configs.
3. Model state includes batch-normalization buffers: all frozen state entries, shapes and dtypes must contribute to the model identity; train mode or gradients must never be enabled during extraction.
4. Interrupted/partial output and NaN/Inf/wrong-shaped arrays must not reload as usable features.
5. Different devices may produce slightly different values: record execution device and library versions; verify against a same-device direct reference with stated tolerance, never claim cross-device bit equality.

## File map and interfaces

- Create `framework/benchmarks/prepare_nih_features.py`: source/audit orchestration, frozen model and bounded image batches, immutable cache creation and CLI.
- Create `framework/src/fedlearn/simulation/nih_cache.py`: safe cache verification/loading; no backbone construction or original-image access.
- Create `framework/tests/test_capstone_nih_features.py`: real tiny NIH PNGs, direct frozen reference and cache refusal cases.
- Update `wikis/capstone/README.md`: actual cache identity, dimensions, preparation cost and remaining integration/pilot gates after qualification.

```python
# benchmarks.prepare_nih_features
def prepare(root: Path, preparation: Path, image_audit: Path, output: Path, *,
            device: str = "cpu", batch_size: int = 64, pretrained: bool = True,
            backbone_seed: int = 1234, audit_workers: int = 1) -> dict: ...
def main(argv: list[str] | None = None) -> None: ...

# fedlearn.simulation.nih_cache
def load_nih_cache(output: Path, *, expected_report_sha256: str | None = None) -> dict: ...
```

Cache files: `train-x.npy`, `train-y.npy`, `train.csv`, corresponding `validation-*` and `test-*`, and `report.json`. Returned loader dictionary has `splits` mapping each split to `x`, `y`, `image_ids`, `patient_ids`, plus `provenance` and `report_sha256`. Arrays are NumPy arrays; the future workload converts them to tensors. The report records schema/version, dataset/classes, source/preparation/audit hashes, source split manifest hashes, transform, backbone identity/state digest, library/device/dtype information, counts/dimensions, and fixed payload hashes. No model-performance fields.

## Task 1: Implement cache creation and validation

- [ ] Write tests first using the existing `nih_tree` fixture with two clients and a completed audit. Compare all 24 feature rows with a separately built frozen random ResNet-18 on exactly the declared transforms; use CPU tolerance `rtol=1e-5, atol=1e-6`, and assert label/identity alignment and patient separation across all three splits.

```python
def test_cache_preserves_original_split_rows_and_frozen_reference(prepared, tmp_path):
    root, manifests, audit = prepared
    output = tmp_path / "features"
    report = prepare(root, manifests, audit, output, pretrained=False, batch_size=4)
    loaded = load_nih_cache(output)
    assert report["feat_dim"] == 512
    assert sum(len(split["y"]) for split in loaded["splits"].values()) == 24
    assert report["pretrained"] is False
```

- [ ] Add tests for unqualified audit, source mutation during extraction, all invalid batch/seed/device inputs, existing/unsafe output, batch bound and eval/no-gradient state, source/report/payload tampering, rehashed wrong labels/row identities, changed expected report digest, nonfinite and malformed arrays, extra/missing files and interruption before final report publication. A pretrained cache test may reuse cached weights offline; never require network for tests.
- [ ] Run `PYTHONPATH=framework/src:framework <python> -m pytest -q -o addopts= framework/tests/test_capstone_nih_features.py`. Expected RED: missing preparation/loader modules before implementation.
- [ ] Implement a fixed-schema loader: expected digest first, fixed payload allowlist and SHA-256 validation, array dtype/shape/finite checks, CSV schema/labels/canonical IDs and nonempty disjoint patient ownership. Bind CSVs to the original preparation manifest digests stored in provenance. Raise ValueError with the failing contract; do not guess/reorder/repair.
- [ ] Implement preparation: validate arguments and new output before expensive work; verify manifests and full image audit, refuse false qualification; construct the frozen backbone in a scoped RNG context and hash named state tensors. Use exact declared transforms; compare raw and RGB fingerprints during each read, collect bounded batches and write split arrays and exact source split CSVs. Publish the complete report last. Reload with its expected digest before reporting success. CLI arguments mirror public parameters; errors exit nonzero.
- [ ] Run targeted cache, NIH, workload and reproduction tests, then the full framework suite with coverage and thread exceptions treated as errors. Expected GREEN; retain logs and explicit same-device reference tolerance.
- [ ] After the preceding real NIH audit qualifies, run full audit verification and extraction as one separate preparation invocation into a fresh ignored output. Prefer available MPS on this laptop only after an explicit device check; record actual device/version and elapsed time. Extraction may read official-test images but must not evaluate a classifier on them.
- [ ] Independently reload the complete real cache with its expected report digest and confirm 68,758/17,766/25,596 rows, 512 finite features each, exact labels/IDs/patient ownership and unchanged official membership. Inspect no model metrics. Update the capstone plan with measured costs and limitations.
- [ ] Commit as `feat(capstone): prepare source-bound frozen NIH features`.

## Follow-on gate

Next implement explicit NIH validation/test selection and patient-preserving engine clients. A cache alone does not establish useful learning, patient-preserving downstream allocation, threshold transfer, robust aggregation benefit or privacy. Freeze the confirmatory protocol only after a bounded training/validation pilot.
