"""Effusion labels for the train and selection partitions, plus an aggregate label audit.

Protocol v1: positive iff the exact token `Effusion` is in the pipe-split findings;
every other finding (and `No Finding`) is negative. Holdout rows are reduced to
identity while parsing, so their findings never enter memory as labels. The audit
output carries aggregate counts only, no patient or image identifiers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_effusion_holdout as holdout  # noqa: E402

Refusal = holdout.Refusal
TARGET = 'Effusion'
VOCABULARY = frozenset({'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                        'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'})
PINS = {'report_sha256': holdout.REPORT_SHA256, 'audit_report_sha256': holdout.AUDIT_REPORT_SHA256,
        'lock_sha256': '083aa7ce95e40af7bdc00d27d396b0ac5954e7f6ebd444200ce97651baccc794'}


def _label(findings):
    tokens = findings.split('|')
    if tokens == ['No Finding']:
        return 0
    if not tokens or len(set(tokens)) != len(tokens) or not set(tokens) <= VOCABULARY:
        raise Refusal('unknown_finding')
    return int(TARGET in tokens)


def load_labels(cache, lock_dir, *, pins=PINS):
    """Return ({'train'|'selection': {'rows': identity rows, 'y': int64 array}}, provenance)."""
    plan, provenance = holdout.verify_lock(cache, lock_dir, report_sha256=pins['report_sha256'],
                                           lock_sha256=pins['lock_sha256'], extra=('findings',))
    labels = {}
    for split, rows in plan.items():
        y = np.array([_label(row['findings']) for row in rows], dtype=np.int64)
        labels[split] = {'rows': [{key: row[key] for key in holdout.KEEP} for row in rows], 'y': y}
    return labels, provenance


def holdout_labels(cache, lock_dir, *, pins=PINS):
    """Labels of the locked holdout rows. For the one-time protocol section 8 evaluation ONLY.

    Verifies every pin and the partition first, then re-hashes the exact bytes it parses against the
    lock, so the rows labelled are the rows that were locked.
    """
    _, provenance = holdout.verify_lock(cache, lock_dir, report_sha256=pins['report_sha256'],
                                        lock_sha256=pins['lock_sha256'])
    lock_dir, cache = Path(lock_dir), Path(cache)
    raw_holdout = (lock_dir / 'holdout.csv').read_bytes()
    raw_validation = (cache / 'validation.csv').read_bytes()
    if (hashlib.sha256(raw_holdout).hexdigest() != provenance['lockManifestSha256']['holdout']
            or hashlib.sha256(raw_validation).hexdigest() != provenance['inputManifestSha256']['validation']):
        raise Refusal('holdout_bytes_changed_after_verification')
    rows = holdout._manifest(raw_holdout, 'holdout', list(holdout.KEEP))
    wanted = {row['image_index'] for row in rows}
    validation = {row['image_index']: row for row in
                  holdout._manifest(raw_validation, 'validation', extra=('findings',), keep_extra=wanted)}
    if any({key: validation[row['image_index']][key] for key in holdout.KEEP} != row for row in rows):
        raise Refusal('holdout_identity_mismatch')
    y = np.array([_label(validation[row['image_index']]['findings']) for row in rows], dtype=np.int64)
    return {'rows': rows, 'y': y, 'provenance': provenance}


def effusion_labels(cache, lock_dir, *, pins=PINS):
    return load_labels(cache, lock_dir, pins=pins)[0]


def audit_labels(cache, lock_dir, audit_dir, *, pins=PINS):
    """Aggregate prevalence and exact-duplicate label consistency for train and selection."""
    labels, provenance = load_labels(cache, lock_dir, pins=pins)
    raw_audit = (Path(audit_dir) / 'report.json').read_bytes()
    if hashlib.sha256(raw_audit).hexdigest() != pins['audit_report_sha256']:
        raise Refusal('audit_report_hash_mismatch')
    raw_duplicates = (Path(audit_dir) / 'duplicates.json').read_bytes()
    if hashlib.sha256(raw_duplicates).hexdigest() != json.loads(raw_audit)['payload_hashes']['duplicates.json']:
        raise Refusal('audit_duplicates_hash_mismatch')
    groups = [group['image_indices'] for kind in ('file', 'rgb') for group in json.loads(raw_duplicates)[kind]]
    report = {'schema': 'capstone.effusion.label_audit.v1', 'target': TARGET,
              'labelRule': 'positive iff exact token Effusion in pipe-split findings; all else negative',
              'inputs': provenance, 'splits': {}, 'duplicateGroups': {},
              'holdoutLabelsRead': False, 'testManifestRead': False}
    for split, part in labels.items():
        y, patients = part['y'], [row['patient_id'] for row in part['rows']]
        by_patient = {}
        for patient, value in zip(patients, y):
            by_patient.setdefault(patient, set()).add(int(value))
        report['splits'][split] = {
            'images': len(y), 'positiveImages': int(y.sum()), 'imagePrevalence': float(y.mean()),
            'patients': len(by_patient), 'positivePatients': sum(1 in v for v in by_patient.values()),
            'patientsWithMixedLabels': sum(len(v) == 2 for v in by_patient.values()),
        }
        index = {row['image_index']: int(value) for row, value in zip(part['rows'], y)}
        inside = [group for group in groups if index.keys() & set(group)]
        if any(not set(group) <= index.keys() for group in inside):
            raise Refusal('duplicate_group_crosses_partition')
        report['duplicateGroups'][split] = {
            'groups': len(inside), 'effusionContradictory': sum(len({index[i] for i in group}) > 1 for group in inside)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--lock-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise Refusal('output_exists')
        result = audit_labels(args.cache_dir, args.lock_dir, args.audit_dir)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'status': 'audited', 'splits': result['splits'], 'duplicateGroups': result['duplicateGroups']}))
