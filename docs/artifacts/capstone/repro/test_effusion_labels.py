"""Effusion labels for train/selection only: exact token, closed vocabulary, holdout discarded."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).parent
FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(directory, findings_for, *, groups=()):
    root = Path(directory); cache = root / 'cache'; audit = root / 'audit'
    cache.mkdir(); audit.mkdir()
    for split, patients in [('train', range(1, 9)), ('validation', range(11, 31))]:
        with (cache / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
            for patient in patients:
                for image in range(2):
                    name = f'{patient:08d}_{image:03d}.png'
                    writer.writerow(dict(image_index=name, patient_id=str(patient), image_path=f'images_001/images/{name}',
                                         label='1', findings=findings_for(patient, image)))
    report = json.dumps({'payload_hashes': {f'{s}.csv': sha((cache / f'{s}.csv').read_bytes()) for s in ['train', 'validation']}}).encode()
    (cache / 'report.json').write_bytes(report)
    duplicates = json.dumps({'file': [], 'rgb': [dict(image_indices=list(g), patient_ids=[str(int(g[0][:8]))] * len(g),
                                                     splits=['train']) for g in groups]}).encode()
    (audit / 'duplicates.json').write_bytes(duplicates)
    audit_report = json.dumps({'payload_hashes': {'duplicates.json': sha(duplicates)}}).encode()
    (audit / 'report.json').write_bytes(audit_report)
    load('lock_effusion_holdout').lock_holdout(cache, audit, root / 'lock', report_sha256=sha(report), audit_report_sha256=sha(audit_report))
    pins = dict(report_sha256=sha(report), lock_sha256=sha((root / 'lock' / 'lock.json').read_bytes()),
                audit_report_sha256=sha(audit_report))
    return cache, root / 'lock', audit, pins


class EffusionLabelTests(unittest.TestCase):
    def setUp(self):
        self.module = load('effusion_labels')

    def test_exact_token_labels_for_train_and_selection_only(self):
        findings = lambda p, i: ['No Finding', 'Effusion', 'Pleural_Thickening|Effusion', 'Infiltration'][(p + i) % 4]
        with tempfile.TemporaryDirectory() as directory:
            cache, lock, _, pins = fixture(directory, findings)
            result = self.module.effusion_labels(cache, lock, pins=pins)
            self.assertEqual(sorted(result), ['selection', 'train'])
            for split, part in result.items():
                expected = [int(findings(int(r['patient_id']), int(r['image_index'][9:12])) in ('Effusion', 'Pleural_Thickening|Effusion'))
                            for r in part['rows']]
                self.assertEqual(part['y'].tolist(), expected)
            holdout = {line.split(',')[0] for line in (lock / 'holdout.csv').read_text().splitlines()[1:]}
            self.assertFalse(holdout & {r['image_index'] for r in result['selection']['rows']})
            self.assertEqual(len(result['selection']['rows']), 20)

    def test_unknown_finding_token_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, lock, _, pins = fixture(directory, lambda p, i: 'Effusions' if p == 3 else 'No Finding')
            with self.assertRaisesRegex(ValueError, 'unknown_finding'):
                self.module.effusion_labels(cache, lock, pins=pins)

    def test_audit_counts_contradictory_duplicates_without_identifiers(self):
        findings = lambda p, i: 'Effusion' if (p, i) == (2, 1) else 'No Finding'
        with tempfile.TemporaryDirectory() as directory:
            cache, lock, audit, pins = fixture(directory, findings, groups=[('00000002_000.png', '00000002_001.png'),
                                                                          ('00000003_000.png', '00000003_001.png')])
            report = self.module.audit_labels(cache, lock, audit, pins=pins)
            self.assertEqual(report['duplicateGroups']['train'], {'groups': 2, 'effusionContradictory': 1})
            self.assertEqual(report['splits']['train']['positiveImages'], 1)
            self.assertNotIn('00000002', json.dumps(report))


if __name__ == '__main__':
    unittest.main()
