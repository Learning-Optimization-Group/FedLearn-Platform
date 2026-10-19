"""Patient-preserving NIH No Finding/any-finding data preparation.

Labels describe the source reports. A finding is not a verified clinical diagnosis.
"""
from __future__ import annotations

import csv
import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re


FINDINGS = frozenset({
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
})
SOURCE_FILES = ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")
_IMAGE_NAME = re.compile(r"[0-9]{8}_[0-9]{3}\.png")
SPLIT_ALGORITHM = "capstone-nih-split-v1:sha256-patient-any-finding"
PARTITION_ALGORITHM = "patient-iid-v1:sorted-patients-pcg64-array-split"
CSV_FIELDS = ("image_index", "patient_id", "image_path", "label", "findings")


@dataclass(frozen=True)
class NihRow:
    image_index: str
    patient_id: str
    image_path: str
    label: int
    findings: tuple[str, ...]


@dataclass(frozen=True)
class NihSource:
    rows: tuple[NihRow, ...]
    official_train: frozenset[str]
    official_test: frozenset[str]
    source_hashes: dict[str, str]


def _official_split(raw: bytes) -> frozenset[str]:
    names = raw.decode("utf-8-sig").splitlines()
    if not names or any(not _IMAGE_NAME.fullmatch(name) for name in names):
        raise ValueError("invalid or empty official split")
    if len(set(names)) != len(names):
        raise ValueError("duplicate official split entry")
    return frozenset(names)


def _metadata_rows(raw: bytes):
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        for field in ("Image Index", "Finding Labels", "Patient ID"):
            if (reader.fieldnames or []).count(field) != 1:
                raise ValueError("missing or duplicate required NIH metadata column")
        yield from reader
    except csv.Error as error:
        raise ValueError("malformed NIH metadata CSV") from error


def load_source(root: Path) -> NihSource:
    """Validate source identities and paths without reading image content.

    Hash the exact metadata/list bytes being parsed. Returned rows are sorted by
    image ID; their paths are relative to the dataset root.
    """
    root = Path(root).resolve(strict=True)
    raw = {name: (root / name).read_bytes() for name in SOURCE_FILES}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    images = {}
    for path in root.glob("images_*/images/*.png"):
        if (path.name in images or not path.is_file()
                or not path.resolve().is_relative_to(root)):
            raise ValueError("duplicate image filename, non-file image or image outside dataset root")
        images[path.name] = path.relative_to(root).as_posix()

    rows = {}
    for row in _metadata_rows(raw[SOURCE_FILES[0]]):
        if None in row:
            raise ValueError("metadata row has more values than columns")
        name, patient, finding = row["Image Index"], row["Patient ID"], row["Finding Labels"]
        if (not isinstance(name, str) or not _IMAGE_NAME.fullmatch(name)
                or name in rows or name not in images):
            raise ValueError("invalid, duplicate or missing metadata image")
        if (not isinstance(patient, str) or not re.fullmatch(r"[0-9]+", patient)
                or int(patient) <= 0 or int(name[:8]) != int(patient)):
            raise ValueError("invalid or inconsistent patient ID")
        labels = tuple(finding.split("|")) if isinstance(finding, str) else ()
        if (not labels or len(set(labels)) != len(labels)
                or (labels != ("No Finding",) and not set(labels) <= FINDINGS)):
            raise ValueError("unknown, empty, duplicate or contradictory finding labels")
        rows[name] = NihRow(name, str(int(patient)), images[name],
                            int(labels != ("No Finding",)), labels)
    if not rows or set(images) != set(rows):
        raise ValueError("image inventory and nonempty metadata must match exactly")

    train = _official_split(raw[SOURCE_FILES[1]])
    test = _official_split(raw[SOURCE_FILES[2]])
    if train & test or train | test != rows.keys():
        raise ValueError("official splits overlap or do not cover metadata")
    if {rows[n].patient_id for n in train} & {rows[n].patient_id for n in test}:
        raise ValueError("official splits have patient overlap")
    return NihSource(tuple(rows[name] for name in sorted(rows)), train, test, source_hashes)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _fraction(value) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value < 1):
        raise ValueError("validation_fraction must be finite and strictly between 0 and 1")
    return float(value)


def split_patients(source: NihSource, *, validation_fraction: float, seed: int
                   ) -> dict[str, tuple[NihRow, ...]]:
    """Reserve whole validation patients within patient-level any-finding strata."""
    fraction, seed = _fraction(validation_fraction), _integer(seed, "seed", 0)
    positive = {}
    for row in source.rows:
        if row.image_index in source.official_train:
            positive[row.patient_id] = positive.get(row.patient_id, False) or row.label == 1
    validation = set()
    for stratum in (False, True):
        patients = [patient for patient, value in positive.items() if value == stratum]
        if not patients:
            continue
        patients.sort(key=lambda patient: (
            hashlib.sha256(f"capstone-nih-split-v1:{seed}:{patient}".encode()).digest(), patient))
        count = max(1, math.floor(len(patients) * fraction))
        if count >= len(patients):
            raise ValueError("not enough patients in a stratum for nonempty training and validation")
        validation.update(patients[:count])
    return {
        "train": tuple(r for r in source.rows
                       if r.image_index in source.official_train and r.patient_id not in validation),
        "validation": tuple(r for r in source.rows
                            if r.image_index in source.official_train and r.patient_id in validation),
        "test": tuple(r for r in source.rows if r.image_index in source.official_test),
    }


def _csv_bytes(rows: tuple[NihRow, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    writer.writerows((r.image_index, r.patient_id, r.image_path, r.label, "|".join(r.findings)) for r in rows)
    return output.getvalue().encode("utf-8")


def _json_bytes(value: dict | list) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _build_preparation(source: NihSource, *, num_clients: int, validation_fraction: float, seed: int):
    import numpy as np
    from fedlearn.simulation.partition import patient_iid_partition

    num_clients = _integer(num_clients, "num_clients", 1)
    seed, fraction = _integer(seed, "seed", 0), _fraction(validation_fraction)
    splits = split_patients(source, validation_fraction=fraction, seed=seed)
    training = splits["train"]
    parts = patient_iid_partition([r.patient_id for r in training], num_clients, seed)
    for client, indices in enumerate(parts):
        splits[f"client-{client:04d}"] = tuple(training[i] for i in indices)
    payloads = {f"{name}.csv": _csv_bytes(rows) for name, rows in splits.items()}
    assignment = [[client, [training[i].image_index for i in indices]] for client, indices in enumerate(parts)]
    report = {
        "schema_version": 1,
        "preparation_version": "capstone-nih-manifests-v1",
        "dataset": "NIH_ChestXray14",
        "classes": ["NO_FINDING", "FINDING"],
        "label_definition": "No Finding = 0; any recognized finding = 1; report-derived labels, not diagnoses",
        "seed": seed, "num_clients": num_clients, "validation_fraction": fraction,
        "split_algorithm": SPLIT_ALGORITHM,
        "partition_algorithm": PARTITION_ALGORITHM,
        "allocation_numpy_version": np.__version__,
        "source_hashes": source.source_hashes,
        "image_content_verified": False,
        "content_limitations": ["Images have not been decoded for corruption checks.",
                                "Cross-split image-content duplicates have not been checked."],
        "independent_split_patient_overlap": 0,
        "client_patient_overlap": 0,
        "assignment_sha256": hashlib.sha256(_json_bytes(assignment)).hexdigest(),
        "manifest_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        "splits": {
            name: {"images": len(rows), "patients": len({r.patient_id for r in rows}),
                   "positive_images": sum(r.label == 1 for r in rows),
                   "negative_images": sum(r.label == 0 for r in rows)}
            for name, rows in splits.items()
        },
    }
    return payloads, report


def prepare(root: Path, output: Path, *, num_clients: int = 10,
            validation_fraction: float = 0.2, seed: int = 42) -> dict:
    """Create new manifests; report.json is published last as the completion marker.

    A failed write leaves incomplete output for inspection. It cannot verify and
    must not be reused; choose a new output path rather than overwriting it.
    """
    root, output = Path(root), Path(output)
    num_clients, seed = _integer(num_clients, "num_clients", 1), _integer(seed, "seed", 0)
    fraction = _fraction(validation_fraction)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite preparation output: {output}")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("output must be outside the original source dataset")
    payloads, report = _build_preparation(load_source(root), num_clients=num_clients,
                                         validation_fraction=fraction, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation; concurrent creation must also refuse.
    for name, data in payloads.items():
        with (output / name).open("xb") as file:
            file.write(data)
    temporary = output / ".report.json.tmp"
    with temporary.open("xb") as file:
        file.write(_json_bytes(report))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, output / "report.json")
    return report


def verify_preparation(root: Path, output: Path) -> dict:
    """Reconstruct declared patient ownership and compare every manifest byte."""
    output = Path(output)
    try:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("missing, incomplete or invalid preparation report") from error
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int:
        raise ValueError("invalid preparation report schema")
    if (report["schema_version"] != 1 or report.get("split_algorithm") != SPLIT_ALGORITHM
            or report.get("partition_algorithm") != PARTITION_ALGORITHM):
        raise ValueError("unsupported preparation schema or algorithm")
    source = load_source(root)
    if report.get("source_hashes") != source.source_hashes:
        raise ValueError("preparation source hashes no longer match")
    try:
        payloads, expected = _build_preparation(source, num_clients=report["num_clients"],
                                               validation_fraction=report["validation_fraction"],
                                               seed=report["seed"])
    except KeyError as error:
        raise ValueError("preparation report is missing settings") from error
    if {p.name for p in output.iterdir()} != {*payloads, "report.json"}:
        raise ValueError("missing or unexpected preparation files")
    for name, data in payloads.items():
        try:
            actual = (output / name).read_bytes()
        except OSError as error:
            raise ValueError(f"unreadable manifest: {name}") from error
        if actual != data:
            raise ValueError(f"manifest identity, label or patient assignment mismatch: {name}")
    if _json_bytes(report) != _json_bytes(expected):
        raise ValueError("preparation provenance, hash or count mismatch")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = (verify_preparation(args.data_root, args.output) if args.verify_only else
                  prepare(args.data_root, args.output, num_clients=args.clients,
                          validation_fraction=args.validation_fraction, seed=args.seed))
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(_json_bytes(report).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
