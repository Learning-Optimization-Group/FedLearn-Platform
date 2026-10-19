"""Decode and fingerprint NIH images before frozen feature preparation."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import io
import json
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import sys

from PIL import __version__ as pillow_version

from fedlearn.simulation.image_identity import read_image_identity

from benchmarks.nih_data import load_source, split_patients, verify_preparation


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16_777_216
FIELDS = ("image_index", "patient_id", "split", "label", "image_path", "file_bytes",
          "file_sha256", "width", "height", "rgb_sha256", "error")
AUDIT_VERSION = "capstone-nih-image-audit-v1"
PIXEL_POLICY = "capstone-nih-rgb-v1:PNG-to-RGB-uint8;SHA256(namespace-NUL,widthxheight-NUL,pixels)"


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _duplicate_groups(rows, field):
    identities = {}
    for row in rows:
        if row[field]:
            identities.setdefault(row[field], []).append(row)
    groups = []
    for digest, members in sorted(identities.items()):
        if len(members) < 2:
            continue
        patients = sorted({r["patient_id"] for r in members})
        labels = sorted({r["label"] for r in members})
        splits = sorted({r["split"] for r in members})
        groups.append({"sha256": digest, "image_indices": [r["image_index"] for r in members],
                       "splits": splits, "patient_ids": patients, "labels": labels,
                       "cross_split": len(splits) > 1, "cross_patient": len(patients) > 1,
                       "contradictory_labels": len(labels) > 1})
    return groups


def _worker_count(workers):
    if isinstance(workers, bool) or not isinstance(workers, Integral) or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 to 16")
    return int(workers)


def _read_records(root, source, ownership, workers):
    items = [{"image_index": row.image_index, "patient_id": row.patient_id,
              "split": ownership[row.image_index], "label": row.label, "image_path": row.image_path}
             for row in source.rows]
    read = partial(read_image_identity, root, max_file_bytes=MAX_FILE_BYTES, max_pixels=MAX_PIXELS)
    if workers == 1:
        for index, item in enumerate(items, 1):
            yield read(item)
            if index % 1000 == 0:
                print(f"Audited {index}/{len(items)} images", file=sys.stderr, flush=True)
        return
    # Spawn isolates Pillow warning policies. Bound queued work to 128 image
    # identities; only workers hold full-resolution images, never the parent.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(items), 128):
            batch = items[start:start + 128]
            yield from pool.map(read, batch, chunksize=8)
            if start // 1000 != (start + len(batch)) // 1000:
                print(f"Audited {start + len(batch)}/{len(items)} images", file=sys.stderr, flush=True)


def _build_audit(root: Path, preparation: Path, workers: int):
    prepared = verify_preparation(root, preparation)
    preparation_raw = (preparation / "report.json").read_bytes()
    source = load_source(root)
    if source.source_hashes != prepared["source_hashes"]:
        raise ValueError("source changed after preparation verification")
    splits = split_patients(source, validation_fraction=prepared["validation_fraction"],
                            seed=prepared["seed"])
    ownership = {row.image_index: split for split, rows in splits.items() for row in rows}
    records = list(_read_records(root, source, ownership, workers))
    duplicates = {"file": _duplicate_groups(records, "file_sha256"),
                  "rgb": _duplicate_groups(records, "rgb_sha256")}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    payloads = {"images.csv": stream.getvalue().encode(), "duplicates.json": _json_bytes(duplicates)}
    groups = [group for kind in duplicates.values() for group in kind]
    failures = sum(bool(r["error"]) for r in records)
    cross_split = sum(group["cross_split"] for group in groups)
    report = {
        "schema_version": 1, "audit_version": AUDIT_VERSION,
        "preparation_sha256": hashlib.sha256(preparation_raw).hexdigest(),
        "source_hashes": source.source_hashes,
        "pillow_version": pillow_version, "pixel_policy": PIXEL_POLICY,
        "max_file_bytes": MAX_FILE_BYTES, "max_pixels": MAX_PIXELS,
        "images": len(records), "decoded_images": len(records) - failures,
        "failed_images": failures, "cross_split_duplicate_groups": cross_split,
        "within_split_duplicate_groups": len(groups) - cross_split,
        "duplicate_group_counting": "file and RGB groups counted separately; one pair may appear in both",
        "qualified": failures == 0 and cross_split == 0,
        "qualification_rule": "all images decode as bounded single-frame PNG; no exact file/RGB duplicates across independent splits",
        "limitations": ["Exact byte and decoded RGB identity only; near-duplicates are not detected.",
                        "Within-split duplicate groups are retained and disclosed, not removed.",
                        "No image-quality, diagnostic accuracy, learning or privacy guarantee."],
        "payload_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    return payloads, report


def audit_images(root: Path, preparation: Path, output: Path, *, workers: int = 1) -> dict:
    """Publish immutable audit evidence, including completed failed qualifications."""
    root, preparation, output = Path(root), Path(preparation), Path(output)
    workers = _worker_count(workers)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    if any(output.resolve().is_relative_to(parent.resolve()) for parent in (root, preparation)):
        raise ValueError("audit output must be outside source and preparation directories")
    payloads, report = _build_audit(root, preparation, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
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


def verify_image_audit(root: Path, preparation: Path, output: Path, *, workers: int = 1) -> dict:
    """Re-read and decode original images; reconstruct every audit payload byte."""
    root, preparation, output = Path(root), Path(preparation), Path(output)
    workers = _worker_count(workers)
    try:
        saved = json.loads((output / "report.json").read_text())
        if not isinstance(saved, dict) or saved.get("audit_version") != AUDIT_VERSION:
            raise ValueError("unsupported audit schema")
        if {p.name for p in output.iterdir()} != {"images.csv", "duplicates.json", "report.json"}:
            raise ValueError("incomplete or unexpected audit files")
        payloads, expected = _build_audit(root, preparation, workers)
        if any((output / name).read_bytes() != data for name, data in payloads.items()):
            raise ValueError("audit payload differs from current image content")
        if _json_bytes(saved) != _json_bytes(expected):
            raise ValueError("audit provenance or summary mismatch")
    except (OSError, ValueError) as error:
        raise ValueError(f"audit verification failed: {error}") from error
    return expected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        operation = verify_image_audit if args.verify_only else audit_images
        report = operation(args.data_root, args.preparation, args.output, workers=args.workers)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(_json_bytes(report).decode(), end="")
    if not report["qualified"]:
        parser.exit(2, "Image audit completed but did not qualify; inspect retained evidence.\n")


if __name__ == "__main__":
    main()
