"""Label-blind, patient-whole carve of the validation split into selection and locked holdout."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('lock_effusion_holdout.py')
FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(directory, *, findings='Effusion', duplicate_patients=None):
    root = Path(directory); cache = root / 'cache'; audit = root / 'audit'
    cache.mkdir(); audit.mkdir()
    for split, patients in [('train', range(1, 11)), ('validation', range(11, 31))]:
        with (cache / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
            for patient in patients:
                for image in range(1 + patient % 3):
                    name = f'{patient:08d}_{image:03d}.png'
                    writer.writerow(dict(image_index=name, patient_id=str(patient), image_path=f'images_001/images/{name}',
                                         label=str(patient % 2), findings=findings if patient % 2 else 'No Finding'))
    (cache / 'test.csv').write_text('poisoned held-out fixture: must not be read')
    report = {'payload_hashes': {f'{s}.csv': sha((cache / f'{s}.csv').read_bytes()) for s in ['train', 'validation']}}
    raw = json.dumps(report).encode(); (cache / 'report.json').write_bytes(raw)
    groups = []
    if duplicate_patients:
        groups.append({'image_indices': [f'{p:08d}_000.png' for p in duplicate_patients],
                       'patient_ids': [str(p) for p in duplicate_patients], 'splits': ['validation']})
    duplicates = json.dumps({'file': [], 'rgb': groups}).encode(); (audit / 'duplicates.json').write_bytes(duplicates)
    audit_raw = json.dumps({'payload_hashes': {'duplicates.json': sha(duplicates)}}).encode()
    (audit / 'report.json').write_bytes(audit_raw)
    return cache, audit, sha(raw), sha(audit_raw)


class HoldoutLockTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('lock_effusion_holdout', SCRIPT)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)

    def lock(self, directory, **kwargs):
        cache, audit, report, audit_report = fixture(directory, **kwargs)
        output = Path(directory) / 'lock'
        return self.module.lock_holdout(cache, audit, output, report_sha256=report, audit_report_sha256=audit_report), output

    def read(self, path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_carve_is_patient_whole_disjoint_complete_and_label_free(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self.lock(directory)
            holdout, selection = self.read(output / 'holdout.csv'), self.read(output / 'selection.csv')
            self.assertEqual(result['patients'], {'holdout': 10, 'selection': 10})
            self.assertFalse({r['patient_id'] for r in holdout} & {r['patient_id'] for r in selection})
            self.assertEqual(len(holdout) + len(selection), result['validationImages'])
            for rows in (holdout, selection):
                self.assertTrue(all(set(r) == {'image_index', 'patient_id', 'image_path'} for r in rows))
            lock = json.loads((output / 'lock.json').read_text())
            self.assertEqual(lock['manifestSha256']['holdout'], sha((output / 'holdout.csv').read_bytes()))
            self.assertNotIn('label', json.dumps(lock).lower().replace('label-blind', ''))

    def test_assignment_does_not_depend_on_labels_or_findings(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.lock(first, findings='Effusion'); self.lock(second, findings='Cardiomegaly|Mass')
            for name in ('holdout.csv', 'selection.csv'):
                self.assertEqual((Path(first) / 'lock' / name).read_bytes(), (Path(second) / 'lock' / name).read_bytes())

    def test_wrong_report_pin_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, audit, _, audit_report = fixture(directory); output = Path(directory) / 'lock'
            with self.assertRaisesRegex(ValueError, 'report_hash'):
                self.module.lock_holdout(cache, audit, output, report_sha256='f' * 64, audit_report_sha256=audit_report)
            self.assertFalse(output.exists())

    def test_cross_patient_duplicate_content_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'cross_patient_duplicate'):
                self.lock(directory, duplicate_patients=[11, 12])

    def test_existing_lock_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, audit, report, audit_report = fixture(directory)
            output = Path(directory) / 'lock'; output.mkdir(); (output / 'lock.json').write_text('original lock')
            with self.assertRaisesRegex(ValueError, 'output_exists'):
                self.module.lock_holdout(cache, audit, output, report_sha256=report, audit_report_sha256=audit_report)
            self.assertEqual((output / 'lock.json').read_text(), 'original lock')


if __name__ == '__main__':
    unittest.main()
