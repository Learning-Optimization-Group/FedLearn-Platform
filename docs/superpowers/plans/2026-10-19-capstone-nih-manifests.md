# NIH manifests and patient allocation implementation plan

**Goal:** Produce independently verifiable NIH No Finding/any-finding manifests and whole-patient client assignments without changing the existing simulation workload.

**Architecture:** A standalone preparation module validates NIH metadata, official lists and image paths, then writes patient-disjoint split/client manifests. A separate patient-grouped partition function returns row indices and preserves the existing row-based partitioners. The preparation report binds the label policy, source bytes, split algorithm, client assignments and manifest files.

**Tech stack:** Python 3.12, standard-library CSV/JSON/hashlib/pathlib, NumPy and pytest. Pillow is used only for synthetic image fixtures in this slice.

**Spec:** [NIH workload design](../specs/2026-10-18-capstone-nih-workload-design.md).

**Execution:** Sequential implementation with a failing regression before each behavior change and verification after each task. Direct execution authorized on 19 October; progress and verification are recorded in the worktree execution ledger.

## Scope and global constraints

- Implement the manifest/allocation foundation only. Frozen features, image-content auditing, explicit engine evaluation splits, learning pilots and confirmatory experiments require the next slice.
- Exactly `No Finding` maps to 0; one or more recognized findings maps to 1. These are report-derived labels, not verified clinical diagnoses.
- Preserve official test membership. Validation comes only from official train/validation patients, using seed 42 and fraction 0.2 by default.
- No patient crosses training/validation/test or two clients. Training is the union of client rows; do not treat training and its client subsets as mutually disjoint splits.
- Preserve every selected row and its label, patient ID and relative image path. Do not balance classes or infer patient-level DP.
- Refuse unknown/empty/duplicate/contradictory labels, duplicate IDs, inconsistent patients, missing/extra images, unsafe paths and incomplete/overlapping official lists.
- Never modify source images, overwrite preparation outputs, alter old Kermany partition behavior, or modify retained experiment records.
- Report `image_content_verified: false` explicitly. This slice proves metadata/path consistency, not image decoding or absence of duplicate image content.
- Keep paths in generated manifests relative to the supplied dataset root. Reports contain source identities and hashes, not machine-specific root paths.
- Preserve the original capstone branch and reproduction anchors. Work on `capstone`, which contains the original 18 commits and merged `main`.

## Review focus

1. Mixed-label patients: a patient with both labels must remain one group; stratify validation by whether any of that patient's images has a finding. Task 3 tests this.
2. Filesystem aliases: duplicate basenames, non-files and symlinks escaping the dataset root must refuse, including when their names otherwise match metadata. Task 1 tests this.
3. Invalid numerical settings: booleans, non-integer client/seed inputs, non-finite fractions and more clients than training patients must refuse before output publication. Tasks 2 and 3 test this.
4. Input ordering: reordering metadata/list lines must not change the selected identities or client assignment for the same settings, though source-byte hashes will change. Tasks 1–3 test this.
5. Interrupted/stale preparation: partial directories, missing reports, changed source bytes and modified CSV payloads must fail verification. Existing caller-owned outputs must remain untouched. Task 3 tests this.

## File map and interfaces

Create `framework/benchmarks/nih_data.py` with these immutable records and public functions:

```python
@dataclass(frozen=True)
class NihRow:
    image_index: str
    patient_id: str
    image_path: str
    label: int
    findings: tuple[str, ...]

@dataclass(frozen=True)
class NihSource:
    rows: tuple[NihRow, ...]  # sorted by image_index
    official_train: frozenset[str]
    official_test: frozenset[str]
    source_hashes: dict[str, str]

def load_source(root: Path) -> NihSource: ...
def split_patients(source: NihSource, *, validation_fraction: float, seed: int
                   ) -> dict[str, tuple[NihRow, ...]]: ...
def prepare(root: Path, output: Path, *, num_clients: int = 10,
            validation_fraction: float = 0.2, seed: int = 42) -> dict: ...
def verify_preparation(root: Path, output: Path) -> dict: ...
def main(argv: list[str] | None = None) -> None: ...
```

Add `patient_iid_partition(patient_ids: Sequence[str], num_clients: int, seed: int) -> List[np.ndarray]` to `framework/src/fedlearn/simulation/partition.py` and its `__all__`.

Create `framework/tests/test_capstone_nih_data.py`; extend `framework/tests/test_simulation_partition.py`. Update `wikis/capstone/README.md` with the command, actual counts and the unverified-content boundary after real-data preparation succeeds.

The source module may share fixtures/validation expectations with the mobile preparation, but must not call its private pneumonia-label parser or translate pneumonia absence into No Finding. It must not import the FL client.

## Verification environment

From the capstone worktree root, reuse the existing interpreter without modifying it:

```bash
export CAPSTONE_PYTHON="$(git rev-parse --git-common-dir)/../.venv/bin/python"
export PYTHONPATH=framework/src:framework
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Targeted pytest examples below disable coverage aggregation; the final full-suite run retains the framework's coverage gate.

## Task 1: Validate the NIH source and explicit label policy

**Files:** Create `framework/benchmarks/nih_data.py` and `framework/tests/test_capstone_nih_data.py`.

**Consumes:** An existing NIH directory containing `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt` and `images_*/images/*.png`.

**Produces:** `NihSource` and `NihRow` as defined above. Patient IDs are canonical positive decimal strings; filenames must match ASCII `[0-9]{8}_[0-9]{3}\.png` and agree with the patient column.

- [x] Write a synthetic NIH fixture helper using `tmp_path`, `csv.DictWriter` and small PNGs. Include negative, positive and mixed-label patients; reserve separate patients for official test. No real patient records enter tests.
- [x] Add the first failing label-policy test:

```python
def test_any_finding_is_positive_even_when_pneumonia_is_absent(nih_tree):
    source = load_source(nih_tree)
    by_image = {r.image_index: r for r in source.rows}
    assert by_image["00000001_000.png"].label == 0  # No Finding
    assert by_image["00000002_000.png"].label == 1  # Mass, not pneumonia
    assert by_image["00000003_000.png"].label == 1  # Effusion|Pneumonia
```

- [x] Run `"$CAPSTONE_PYTHON" -m pytest -q -o addopts='' framework/tests/test_capstone_nih_data.py`; confirm the missing source implementation causes the expected failure.
- [x] Implement parsing with the 14 recognized findings from the NIH metadata. Read/hash the exact metadata/list bytes being parsed; accept a UTF-8 BOM in CSV headers. Reject duplicate required headers and malformed/missing fields.
- [x] Index relative PNG paths; reject duplicate basenames, non-files, resolved paths outside root and metadata/image mismatches. Preserve source data and do not decode images in this task.
- [x] Validate exact official list coverage and disjoint image/patient membership. Sort returned rows independently of input order.
- [x] Add parameterized refusals for `No Finding|Mass`, unknown labels, repeated findings, empty labels, duplicate metadata/list entries, bad patient IDs, missing/extra images, incomplete lists and official patient overlap. Add escaping-symlink and duplicate-directory cases.
- [x] Run the source tests, inspect the refusal reasons and confirm hashes match independently computed SHA-256 values. Commit only the module and tests with `feat(capstone): validate NIH finding labels and source identities`.

## Task 2: Allocate complete patient groups to clients

**Files:** Modify `framework/src/fedlearn/simulation/partition.py` and `framework/tests/test_simulation_partition.py`.

**Consumes:** A one-dimensional sequence of nonempty canonical patient strings aligned with training rows, plus integer client count/seed. Reject booleans masquerading as integers, negative seeds, missing IDs and mixed identifier types.

**Produces:** Sorted int64 row-index arrays, one nonempty array per client. Groups are uniformly shuffled and divided by patient count; image counts and label distributions need not be balanced.

- [x] Add a failing test with uneven group sizes and repeated patients:

```python
def test_patient_iid_never_splits_groups_and_covers_every_row():
    ids = ["1", "1", "1", "2", "3", "3", "4", "5", "5", "6"]
    parts = patient_iid_partition(ids, num_clients=3, seed=42)
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(ids)))
    owner = {}
    for client, indices in enumerate(parts):
        assert len(indices) > 0
        for i in indices:
            assert owner.setdefault(ids[i], client) == client
```

- [x] Run the new test and confirm it fails because the patient partitioner is absent.
- [x] Implement a sorted unique-patient list, a local `np.random.default_rng(seed)` permutation, `np.array_split` over whole patients, and a mapping back to original row indices. Require `num_clients <= number_of_patients`; do not repair empty clients by splitting a patient.
- [x] Test repeatability, seed dependence on a sufficiently large fixture, arbitrary row order, single-client behavior, exact-one-patient-per-client behavior, empty inputs and too many clients. Assert patient counts differ by at most one while uneven image counts are permitted.
- [x] Run `"$CAPSTONE_PYTHON" -m pytest -q -o addopts='' framework/tests/test_simulation_partition.py framework/tests/test_simulation_workloads.py framework/tests/test_engine_reproduces_capstone.py`. Existing row-based behavior and engine/harness parity must stay unchanged.
- [x] Commit with `feat(capstone): partition simulation rows by whole patients`.

## Task 3: Prepare, verify and exercise split/client manifests

**Files:** Extend `framework/benchmarks/nih_data.py` and `framework/tests/test_capstone_nih_data.py`; update `wikis/capstone/README.md`.

**Consumes:** `load_source`, `patient_iid_partition`, fraction/seed/client settings and a new output directory.

**Produces:** `train.csv`, `validation.csv`, `test.csv`, `client-0000.csv` through the requested client count, and `report.json`. CSV columns are `image_index,patient_id,image_path,label,findings`; labels remain numeric 0/1 and findings use the original validated pipe-delimited spelling. Rows are ordered by image ID.

- [x] Add a failing mixed-patient split test:

```python
def test_split_membership_is_patient_disjoint_and_keeps_official_test(nih_tree):
    source = load_source(nih_tree)
    splits = split_patients(source, validation_fraction=0.2, seed=42)
    patients = {k: {r.patient_id for r in rows} for k, rows in splits.items()}
    assert not patients["train"] & patients["validation"]
    assert not patients["train"] & patients["test"]
    assert not patients["validation"] & patients["test"]
    assert {r.image_index for r in splits["test"]} == source.official_test
    assert {r.image_index for r in splits["train"] + splits["validation"]} == source.official_train
```

- [x] Stratify official train/validation patients by any positive image. Within each nonempty stratum, order by `(SHA256("capstone-nih-split-v1:{seed}:{patient_id}"), patient_id)`. Select `max(1, floor(n * fraction))` validation patients; refuse strata too small to retain both training and validation. Empty strata are allowed; report realized class counts rather than asserting image-level stratification.
- [x] Validate finite fraction strictly between zero and one, integer non-boolean seed/client count, nonempty train/validation and feasible whole-patient client allocation before creating output.
- [x] Reserve output with exclusive directory creation. Write CSV files, then publish a temporary report with `os.replace` as the completion marker. Existing outputs, including symlinks, must never be replaced; incomplete output without a complete report must never verify.
- [x] Record schema/preparation version, dataset/label definition, seed/fraction, split and allocation algorithm IDs, source hashes, manifest hashes, a canonical assignment digest, and actual image/patient/positive counts per split/client. Set `image_content_verified` to false and state the missing decoding/content-duplication checks.
- [x] Implement `verify_preparation` to require the report, verify hashes against source and CSV payloads, and compare reconstructed identities/labels/splits/client ownership against the declared deterministic settings. Unknown schema/algorithm IDs refuse.
- [x] Test same-settings byte repeatability in two fresh outputs, reordering invariance of membership, source/payload tampering, missing report, interruption before completion, existing output preservation and every invalid numerical setting. Independently assert the disjoint client union equals training exactly.
- [x] Add a CLI with `--data-root`, `--output`, `--clients` (default 10), `--validation-fraction` (default 0.2), `--seed` (default 42) and `--verify-only`. Verification does not create or modify output. CLI failures return nonzero with a useful error.
- [x] Run targeted tests, then the full framework suite with coverage and thread exceptions as errors:

```bash
cd framework
PYTHONPATH=src "$CAPSTONE_PYTHON" -m pytest -q -W error::pytest.PytestUnhandledThreadExceptionWarning
```

- [x] Prepare the real NIH metadata into a fresh ignored directory under `.temp/`, using the existing repository dataset root. Run verification in a second invocation. Record counts, hashes, timing and content-audit limitations without opening test-set model performance or copying images.
- [x] Update the capstone guide with the verified preparation command/results and what remains. Keep generated manifests private/ignored. Commit source, tests and guide with `feat(capstone): prepare verifiable patient-separated NIH manifests`.

## Completion and next boundary

This slice is complete when the real manifest verifies independently, synthetic refusal tests pass, every training patient has one client owner, existing reproduction checks pass and the full framework coverage gate is green. It does not complete the full NIH workload design.

The next plan must cover streamed image decoding/content-hash checks, frozen ResNet-18 features, cache provenance, explicit validation/test evaluation selection and a clean validation pilot. Keep the proposed head-size comparison and confirmatory protocol decisions separate; no head-size sweep or official-test tuning is authorized by this manifest plan.
