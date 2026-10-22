"""Carve the curated validation split into a selection half and a locked holdout half.

Label-blind: label and findings columns are dropped on read and never influence the
assignment. Patients stay whole. Train and the official test manifest are not split;
test.csv is never opened. Outputs are written once and never replaced.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

REPORT_SHA256 = '6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'
AUDIT_REPORT_SHA256 = '7851361e9fa6f72d24745b781f10d5453800a16bcb3148abed775a3cd2bb2b8d'
NAMESPACE = 'capstone-effusion-holdout-v1'
KEEP = ('image_index', 'patient_id', 'image_path')
MANIFEST_FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']
IMAGE_PATH = re.compile(r'images_\d{3}/images/(\d{8})_\d{3}\.png')


class Refusal(ValueError):
    pass


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _manifest(data, split):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    if reader.fieldnames != MANIFEST_FIELDS:
        raise Refusal(f'manifest_schema:{split}')
    rows = [{key: row[key] for key in KEEP} for row in reader]
    for row in rows:
        match = IMAGE_PATH.fullmatch(row['image_path'])
        if not match or row['image_path'].rsplit('/', 1)[1] != row['image_index'] or row['patient_id'] != str(int(match.group(1))):
            raise Refusal(f'image_identity:{split}')
    if not rows or len({row['image_index'] for row in rows}) != len(rows):
        raise Refusal(f'manifest_rows:{split}')
    return rows


def _write(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(KEEP), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    data = stream.getvalue().encode()
    path.write_bytes(data)
    return _sha256(data)


def lock_holdout(cache, audit_dir, output, *, report_sha256=REPORT_SHA256, audit_report_sha256=AUDIT_REPORT_SHA256):
    cache, audit_dir, output = Path(cache), Path(audit_dir), Path(output)
    if output.exists():
        raise Refusal('output_exists')
    raw_report = (cache / 'report.json').read_bytes()
    if _sha256(raw_report) != report_sha256:
        raise Refusal('report_hash_mismatch')
    hashes = json.loads(raw_report)['payload_hashes']
    manifests, manifest_sha = {}, {}
    for split in ('train', 'validation'):
        data = (cache / f'{split}.csv').read_bytes()
        manifest_sha[split] = _sha256(data)
        if manifest_sha[split] != hashes.get(f'{split}.csv'):
            raise Refusal(f'manifest_hash_mismatch:{split}')
        manifests[split] = _manifest(data, split)
    if {r['patient_id'] for r in manifests['train']} & {r['patient_id'] for r in manifests['validation']}:
        raise Refusal('patient_overlap')
    raw_audit = (audit_dir / 'report.json').read_bytes()
    if _sha256(raw_audit) != audit_report_sha256:
        raise Refusal('audit_report_hash_mismatch')
    raw_duplicates = (audit_dir / 'duplicates.json').read_bytes()
    duplicates_sha = _sha256(raw_duplicates)
    if duplicates_sha != json.loads(raw_audit)['payload_hashes']['duplicates.json']:
        raise Refusal('audit_duplicates_hash_mismatch')
    validation_images = {r['image_index'] for r in manifests['validation']}
    duplicate_groups = json.loads(raw_duplicates)
    for kind in ('file', 'rgb'):
        for group in duplicate_groups[kind]:
            if validation_images & set(group['image_indices']) and len(set(group['patient_ids'])) != 1:
                raise Refusal('cross_patient_duplicate_content')
    by_patient = {}
    for row in manifests['validation']:
        by_patient.setdefault(row['patient_id'], []).append(row)
    ranked = sorted(by_patient, key=lambda p: (_sha256(f'{NAMESPACE}\0{p}'.encode()), int(p)))
    holdout_patients = set(ranked[:len(ranked) // 2])
    parts = {'holdout': [], 'selection': []}
    for row in manifests['validation']:
        parts['holdout' if row['patient_id'] in holdout_patients else 'selection'].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.lock-', dir=output.parent))
    lock = {
        'schema': 'capstone.effusion.holdout_lock.v1', 'namespace': NAMESPACE,
        'rule': 'validation patients ordered by SHA256(namespace NUL patient_id); first floor(n/2) to holdout; '
                'patient-whole; label-blind (target columns dropped on read)',
        'cacheReportSha256': report_sha256, 'inputManifestSha256': manifest_sha,
        'auditReportSha256': audit_report_sha256, 'auditDuplicatesSha256': duplicates_sha,
        'validationImages': len(manifests['validation']),
        'images': {name: len(rows) for name, rows in parts.items()},
        'patients': {name: len({r['patient_id'] for r in rows}) for name, rows in parts.items()},
        'manifestSha256': {name: _write(staging / f'{name}.csv', rows) for name, rows in parts.items()},
        'testManifestRead': False,
    }
    (staging / 'lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    staging.rename(output)
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = lock_holdout(args.cache_dir, args.audit_dir, args.output)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'locked', 'images': result['images'], 'patients': result['patients']}))
