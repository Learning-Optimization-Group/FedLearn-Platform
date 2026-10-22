"""Versioned XRV feature cache: holdout exclusion, label-free payloads and refusal paths."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).parent
FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']
AUDIT_FIELDS = ['image_index', 'patient_id', 'split', 'label', 'image_path', 'file_bytes', 'file_sha256',
                'width', 'height', 'rgb_sha256', 'error']
DIM = 16


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png(value, mode='L'):
    from PIL import Image
    image = Image.new(mode, (4, 4), value if mode == 'L' else (value, value + 1, value + 2))
    stream = io.BytesIO(); image.save(stream, format='PNG')
    rgb = image.convert('RGB'); digest = hashlib.sha256(b'capstone-nih-rgb-v1\0')
    digest.update(b'4x4\0'); digest.update(rgb.tobytes())
    return stream.getvalue(), digest.hexdigest()


class FakeExtractor:
    def __init__(self, shift=0.0):
        self.shift, self.seen = shift, 0

    def features(self, batch):
        self.seen += len(batch)
        flat = batch.reshape(len(batch), -1).astype(np.float32)
        return flat[:, :DIM] + np.float32(self.shift)

    def identity(self):
        return {'modelId': 'fake', 'parameterStateSha256': '0' * 64}

    def state_unchanged(self):
        return True


def fake_preprocess(gray):
    return (gray.astype(np.float32)[None, :, :] / 255.0)


def fixture(directory, *, colour_patient=None):
    root = Path(directory); cache = root / 'cache'; audit = root / 'audit'; images = root / 'images'
    for path in (cache, audit, images / 'images_001' / 'images'):
        path.mkdir(parents=True)
    audit_rows = []
    for split, patients in [('train', range(1, 9)), ('validation', range(11, 31))]:
        with (cache / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
            for patient in patients:
                name = f'{patient:08d}_000.png'; path = f'images_001/images/{name}'
                writer.writerow(dict(image_index=name, patient_id=str(patient), image_path=path,
                                     label=str(patient % 2), findings='Effusion' if patient % 2 else 'No Finding'))
                data, rgb = png(patient * 5, 'RGB' if patient == colour_patient else 'L')
                (images / path).write_bytes(data)
                audit_rows.append(dict(image_index=name, patient_id=str(patient), split=split, label='0', image_path=path,
                                       file_bytes=str(len(data)), file_sha256=sha(data), width='4', height='4',
                                       rgb_sha256=rgb, error=''))
    (cache / 'test.csv').write_text('poisoned held-out fixture: must not be read')
    report = json.dumps({'payload_hashes': {f'{s}.csv': sha((cache / f'{s}.csv').read_bytes()) for s in ['train', 'validation']}}).encode()
    (cache / 'report.json').write_bytes(report)
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator='\n')
    writer.writeheader(); writer.writerows(audit_rows); images_csv = stream.getvalue().encode()
    (audit / 'images.csv').write_bytes(images_csv)
    duplicates = json.dumps({'file': [], 'rgb': []}).encode(); (audit / 'duplicates.json').write_bytes(duplicates)
    audit_report = json.dumps({'payload_hashes': {'images.csv': sha(images_csv), 'duplicates.json': sha(duplicates)}}).encode()
    (audit / 'report.json').write_bytes(audit_report)
    lock = load('lock_effusion_holdout').lock_holdout(cache, audit, root / 'lock', report_sha256=sha(report),
                                                      audit_report_sha256=sha(audit_report))
    holdout = (root / 'lock' / 'holdout.csv').read_text().splitlines()[1:]
    for line in holdout:  # holdout images must never be needed: remove them
        (images / line.split(',')[2]).unlink()
    pins = dict(report_sha256=sha(report), lock_sha256=sha((root / 'lock' / 'lock.json').read_bytes()),
                audit_report_sha256=sha(audit_report), audit_images_sha256=sha(images_csv))
    return dict(cache=cache, lock_dir=root / 'lock', audit_dir=audit, image_root=images), pins, lock


class CacheBuildTests(unittest.TestCase):
    def setUp(self):
        self.module = load('build_xrv_cache')

    def build(self, inputs, pins, output, extractor=None):
        return self.module.build_cache(**inputs, output=output, extractor=extractor or FakeExtractor(), pins=pins,
                                       preprocess=fake_preprocess, feat_dim=DIM, workers=0, batch_size=3)

    def test_builds_train_and_selection_only_without_labels_or_holdout_access(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, lock = fixture(directory); output = Path(directory) / 'out'
            report = self.build(inputs, pins, output)
            self.assertEqual(sorted(p.name for p in output.iterdir()),
                             ['report.json', 'selection-x.npy', 'selection.csv', 'train-x.npy', 'train.csv'])
            self.assertEqual(report['splits']['train']['images'], 8)
            self.assertEqual(report['splits']['selection']['images'], lock['images']['selection'])
            for split in ('train', 'selection'):
                header = (output / f'{split}.csv').read_text().splitlines()[0]
                self.assertEqual(header, 'image_index,patient_id,image_path')
                x = np.load(output / f'{split}-x.npy')
                self.assertEqual((x.dtype, x.shape[1]), (np.dtype('float32'), DIM))
                self.assertEqual(report['payload_hashes'][f'{split}-x.npy'], sha((output / f'{split}-x.npy').read_bytes()))
            self.assertFalse(report['labelsStored']); self.assertTrue(report['holdoutExcluded'])
            self.assertEqual(report['feat_dim'], DIM)

    def test_wrong_lock_pin_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, _ = fixture(directory); output = Path(directory) / 'out'
            with self.assertRaisesRegex(ValueError, 'lock_hash'):
                self.build(inputs, {**pins, 'lock_sha256': 'f' * 64}, output)
            self.assertFalse(output.exists())

    def test_altered_selection_manifest_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, _ = fixture(directory); path = inputs['lock_dir'] / 'selection.csv'
            rows = path.read_text().splitlines(); path.write_text('\n'.join(rows[:-1]) + '\n')
            with self.assertRaisesRegex(ValueError, 'selection_manifest_hash'):
                self.build(inputs, pins, Path(directory) / 'out')

    def test_refused_image_fails_whole_build_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, _ = fixture(directory, colour_patient=3); output = Path(directory) / 'out'
            with self.assertRaisesRegex(ValueError, 'image_refused'):
                self.build(inputs, pins, output)
            self.assertFalse(output.exists())
            failure = json.loads(Path(f'{output}.failed.json').read_text())
            self.assertEqual(failure['refused'][0]['reason'], 'decode_colour_channels_differ')

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, _ = fixture(directory); output = Path(directory) / 'out'
            output.mkdir(); (output / 'report.json').write_text('original')
            with self.assertRaisesRegex(ValueError, 'output_exists'):
                self.build(inputs, pins, output)
            self.assertEqual((output / 'report.json').read_text(), 'original')

    def test_verify_replays_sample_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, pins, _ = fixture(directory); output = Path(directory) / 'out'
            self.build(inputs, pins, output)
            kwargs = dict(**inputs, output=output, pins=pins, preprocess=fake_preprocess, feat_dim=DIM, sample=3)
            result = self.module.verify_cache(extractor=FakeExtractor(), **kwargs)
            self.assertTrue(result['allSampledRowsExactlyEqual'])
            self.assertEqual(result['sampledRows'], {'train': 3, 'selection': 3})
            with self.assertRaisesRegex(ValueError, 'replay_mismatch'):
                self.module.verify_cache(extractor=FakeExtractor(shift=1.0), **kwargs)
            path = output / 'train-x.npy'; path.chmod(0o600); path.write_bytes(path.read_bytes()[:-4] + b'\0\0\0\1')
            with self.assertRaisesRegex(ValueError, 'payload_hash'):
                self.module.verify_cache(extractor=FakeExtractor(), **kwargs)


if __name__ == '__main__':
    unittest.main()
