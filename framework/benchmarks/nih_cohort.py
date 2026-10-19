"""Derive an explicit NIH development cohort without changing official test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.nih_data import _csv_bytes, _json_bytes, load_source, split_patients, verify_preparation
from benchmarks.nih_image_audit import verify_image_audit
from fedlearn.simulation.partition import patient_iid_partition


POLICY_VERSION = "nih-development-quarantine-v1"
COHORT_VERSION = "capstone-nih-curated-cohort-v1"
SPLITS = ("train", "validation", "test")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _counts(rows):
    return {"images": len(rows), "patients": len({r.patient_id for r in rows}),
            "positive_images": sum(r.label == 1 for r in rows),
            "negative_images": sum(r.label == 0 for r in rows)}


def _causes(members):
    causes = set()
    if len({r.patient_id for r in members}) > 1:
        causes.add("cross_patient_rgb_identity")
    if len({r.label for r in members}) > 1:
        causes.add("contradictory_rgb_labels")
    return causes


def _build_cohort(root, original_preparation, image_audit, audit_workers):
    original = verify_preparation(root, original_preparation)
    audit = verify_image_audit(root, original_preparation, image_audit, workers=audit_workers)
    if audit["failed_images"] or audit["decoded_images"] != audit["images"]:
        raise ValueError("all source images must decode before cohort curation")
    source = load_source(root)
    if source.source_hashes != audit["source_hashes"]:
        raise ValueError("source changed after image audit verification")
    splits = split_patients(source, validation_fraction=original["validation_fraction"], seed=original["seed"])
    by_image = {r.image_index: r for r in source.rows}
    test_patients = {r.patient_id for r in splits["test"]}
    duplicate_bytes = (image_audit / "duplicates.json").read_bytes()
    if _sha256(duplicate_bytes) != audit["payload_hashes"]["duplicates.json"]:
        raise ValueError("image audit groups changed after verification")
    groups = json.loads(duplicate_bytes)["rgb"]
    reasons = {}
    for group in groups:
        members = [by_image[name] for name in group["image_indices"]]
        causes = _causes(members)  # Derive from source rows, not group summary flags.
        if not causes:
            continue
        for patient in {r.patient_id for r in members} - test_patients:
            entry = reasons.setdefault(patient, {"causes": set(), "groups": set()})
            entry["causes"].update(causes)
            entry["groups"].add(group["sha256"])
    selected = {split: tuple(r for r in rows if split == "test" or r.patient_id not in reasons)
                for split, rows in splits.items()}
    if any(not selected[split] for split in SPLITS):
        raise ValueError("quarantine leaves an empty independent split; refusing cohort")
    quarantined = tuple(r for r in source.rows if r.patient_id in reasons)
    retained = {r.image_index: (split, r) for split, rows in selected.items() for r in rows}
    retained_test_ambiguous = 0
    for group in groups:
        members = [retained[name] for name in group["image_indices"] if name in retained]
        if len(members) < 2:
            continue
        member_splits = {split for split, _ in members}
        if len(member_splits) > 1:
            raise ValueError("selected cohort retains cross-split RGB content")
        ambiguous = bool(_causes([row for _, row in members]))
        if ambiguous and member_splits != {"test"}:
            raise ValueError("selected development cohort retains ambiguous RGB content")
        retained_test_ambiguous += ambiguous and member_splits == {"test"}
    training = selected["train"]
    parts = patient_iid_partition([r.patient_id for r in training], original["num_clients"], original["seed"])
    client_rows = {f"client-{cid:04d}": tuple(training[i] for i in indices)
                   for cid, indices in enumerate(parts)}
    payloads = {f"{split}.csv": _csv_bytes(rows)
                for split, rows in {**selected, **client_rows, "quarantined": quarantined}.items()}
    if payloads["test.csv"] != (original_preparation / "test.csv").read_bytes():
        raise ValueError("curation must preserve exact official-test manifest bytes")
    reason_ledger = {"policy_version": POLICY_VERSION, "patients": [
        {"patient_id": patient, "causes": sorted(entry["causes"]),
         "rgb_group_sha256s": sorted(entry["groups"])} for patient, entry in sorted(reasons.items())]}
    payloads["reasons.json"] = _json_bytes(reason_ledger)
    assignment = [[cid, [training[i].image_index for i in indices]] for cid, indices in enumerate(parts)]
    report = {
        "schema_version": 1, "cohort_version": COHORT_VERSION, "policy_version": POLICY_VERSION,
        "dataset": original["dataset"], "classes": original["classes"],
        "label_definition": original["label_definition"],
        "source_hashes": source.source_hashes,
        "original_preparation_sha256": _sha256((original_preparation / "report.json").read_bytes()),
        "image_audit_sha256": _sha256((image_audit / "report.json").read_bytes()),
        "original_audit_qualified": audit["qualified"],
        "seed": original["seed"], "num_clients": original["num_clients"],
        "validation_fraction": original["validation_fraction"],
        "split_algorithm": "retain-original-split-order-after-whole-patient-quarantine-v1",
        "partition_algorithm": original["partition_algorithm"], "allocation_numpy_version": np.__version__,
        "original_splits": {split: _counts(rows) for split, rows in splits.items()},
        "selected_splits": {split: _counts(rows) for split, rows in selected.items()},
        "excluded_splits": {split: _counts(tuple(r for r in rows if r.patient_id in reasons))
                            for split, rows in splits.items()},
        "clients": {client: _counts(rows) for client, rows in client_rows.items()},
        "quarantined": _counts(quarantined),
        "reason_patient_counts": {cause: sum(cause in entry["causes"] for entry in reasons.values())
                                  for cause in ("cross_patient_rgb_identity", "contradictory_rgb_labels")},
        "development_content_qualified": True, "independent_split_rgb_overlap_groups": 0,
        "retained_official_test_ambiguous_rgb_groups": int(retained_test_ambiguous),
        "official_test_unchanged": True, "assignment_sha256": _sha256(_json_bytes(assignment)),
        "payload_hashes": {name: _sha256(data) for name, data in payloads.items()},
        "limitations": ["Exact decoded RGB identity only; near-duplicates are not detected.",
                        "Official-test ambiguities are retained; labels and true patient identities are not clinically verified.",
                        "Whole-patient quarantine is a conservative data-quality policy, not a performance-based exclusion.",
                        "No useful-learning, clinical, privacy or downstream engine ownership guarantee."],
    }
    return payloads, report


def prepare_cohort(root: Path, original_preparation: Path, image_audit: Path,
                   output: Path, *, audit_workers: int = 1) -> dict:
    """Create new, explicitly curated manifests; retain the failed original audit."""
    root, original_preparation, image_audit, output = map(Path, (root, original_preparation, image_audit, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite cohort output: {output}")
    if any(output.resolve().is_relative_to(parent.resolve()) for parent in (root, original_preparation, image_audit)):
        raise ValueError("cohort output must be outside source and original evidence directories")
    payloads, report = _build_cohort(root, original_preparation, image_audit, audit_workers)
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


def verify_cohort(root: Path, original_preparation: Path, image_audit: Path,
                  output: Path, *, audit_workers: int = 1) -> dict:
    """Reconstruct policy and compare every saved file, including exact report bytes."""
    root, original_preparation, image_audit, output = map(Path, (root, original_preparation, image_audit, output))
    try:
        raw_report = (output / "report.json").read_bytes()
        saved = json.loads(raw_report)
        if (not isinstance(saved, dict) or type(saved.get("schema_version")) is not int
                or saved["schema_version"] != 1 or saved.get("cohort_version") != COHORT_VERSION
                or raw_report != _json_bytes(saved)):
            raise ValueError("unsupported or noncanonical cohort report")
        payloads, expected = _build_cohort(root, original_preparation, image_audit, audit_workers)
        if {p.name for p in output.iterdir()} != {*payloads, "report.json"}:
            raise ValueError("missing or unexpected cohort files")
        if any((output / name).read_bytes() != data for name, data in payloads.items()):
            raise ValueError("cohort rows, reasons or assignments differ from policy")
        if raw_report != _json_bytes(expected):
            raise ValueError("cohort provenance or policy summary mismatch")
    except (OSError, ValueError) as error:
        raise ValueError(f"cohort verification failed: {error}") from error
    return expected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--original-preparation", type=Path, required=True)
    parser.add_argument("--image-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-workers", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        operation = verify_cohort if args.verify_only else prepare_cohort
        report = operation(args.data_root, args.original_preparation, args.image_audit,
                           args.output, audit_workers=args.audit_workers)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(_json_bytes(report).decode(), end="")


if __name__ == "__main__":
    main()
