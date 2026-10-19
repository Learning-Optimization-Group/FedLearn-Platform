"""Patient-preserving NIH No Finding/any-finding data preparation.

Labels describe the source reports. A finding is not a verified clinical diagnosis.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re


FINDINGS = frozenset({
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
})
SOURCE_FILES = ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")
_IMAGE_NAME = re.compile(r"[0-9]{8}_[0-9]{3}\.png")


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

    reader = csv.DictReader(io.StringIO(raw[SOURCE_FILES[0]].decode("utf-8-sig"), newline=""))
    for field in ("Image Index", "Finding Labels", "Patient ID"):
        if (reader.fieldnames or []).count(field) != 1:
            raise ValueError("missing or duplicate required NIH metadata column")
    rows = {}
    for row in reader:
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
