"""Development-only sampling and provenance controls without model dependencies."""
import csv
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('probe_xray_development.py')
FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fixture(directory, *, overlap=False, traversal=False):
    root = Path(directory); root.mkdir(exist_ok=True)
    for split, patients in [('train', range(1, 6)), ('validation', range(5 if overlap else 6, 10 if overlap else 11))]:
        with (root / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
            for patient in patients:
                for image in range(2):
                    name = f'{patient:08d}_{image:03d}.png'
                    writer.writerow(dict(image_index=name, patient_id=str(patient), image_path=f'images_001/images/{name}' if not traversal else f'../{name}', label='0', findings='No Finding'))
    (root / 'test.csv').write_text('poisoned held-out fixture: must not be read')
    report = {'cache_version':'capstone-nih-frozen-features-v1', 'pretrained':True,
              'payload_hashes':{f'{s}.csv':sha((root/f'{s}.csv').read_bytes()) for s in ['train','validation']},
              'image_audit_sha256':'0'*64}
    report['payload_hashes']['test.csv'] = 'f'*64
    raw = json.dumps(report).encode(); (root/'report.json').write_bytes(raw)
    return sha(raw)

class DevelopmentSelectionTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('probe_xray_development', SCRIPT)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)

    def test_selects_four_distinct_patients_per_development_split_without_reading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory)
            selected = self.module.select_development(Path(directory), expected)
            self.assertEqual(len(selected['rows']), 8)
            self.assertEqual([sum(r['split']==s for r in selected['rows']) for s in ['train','validation']], [4,4])
            self.assertEqual(len({r['patient_id'] for r in selected['rows']}), 8)
            self.assertTrue(all(set(r)=={'split','image_index','patient_id','image_path'} for r in selected['rows']))

    def test_patient_overlap_refuses_even_when_csv_hashes_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory, overlap=True)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.module.select_development(Path(directory), expected)

    def test_mutated_manifest_refuses_before_image_access(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory); path=Path(directory)/'train.csv'; path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'manifest_hash'):
                self.module.select_development(Path(directory), expected)

    def test_wrong_report_pin_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture(directory)
            with self.assertRaisesRegex(ValueError, 'report_hash'):
                self.module.select_development(Path(directory), 'f'*64)

    def test_noncanonical_raw_image_path_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory, traversal=True)
            with self.assertRaisesRegex(ValueError, 'image_identity'):
                self.module.select_development(Path(directory), expected)

    def test_changed_raw_image_is_refused_before_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root/'images_001/images/00000001_000.png'; path.parent.mkdir(parents=True); path.write_bytes(b'changed pixels')
            row = {'image_index':'00000001_000.png', 'image_path':'images_001/images/00000001_000.png'}
            audit = {'file_sha256':sha(b'original image bytes')}
            with self.assertRaisesRegex(ValueError, 'raw_image_hash'):
                self.module.read_bound_image(root, row, audit)

    def test_unreadable_test_manifest_does_not_block_development_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = fixture(directory); held_out = Path(directory)/'test.csv'; held_out.chmod(0)
            try:
                self.assertEqual(len(self.module.select_development(Path(directory), expected)['rows']), 8)
            finally:
                held_out.chmod(0o600)

    def test_colour_image_and_wrong_pixel_identity_are_refused_not_converted(self):
        from PIL import Image
        def png(image):
            stream = io.BytesIO(); image.save(stream, format='PNG'); return stream.getvalue()
        def identity(image):
            rgb = image.convert('RGB'); digest = hashlib.sha256(b'capstone-nih-rgb-v1\0')
            digest.update(f'{rgb.width}x{rgb.height}\0'.encode()); digest.update(rgb.tobytes())
            return {'rgb_sha256': digest.hexdigest(), 'width': str(rgb.width), 'height': str(rgb.height)}
        colour = Image.new('RGB', (4, 4), (10, 20, 30))
        with self.assertRaisesRegex(ValueError, 'colour_channels_differ'):
            self.module.decode_grayscale(png(colour), identity(colour))
        gray = Image.new('L', (4, 4), 7)
        with self.assertRaisesRegex(ValueError, 'decoded_pixel_identity'):
            self.module.decode_grayscale(png(gray), identity(Image.new('L', (4, 4), 8)))
        pixels, mode, agrees = self.module.decode_grayscale(png(gray), identity(gray))
        self.assertEqual((pixels.shape, mode, agrees, int(pixels.max())), ((4, 4), 'L', True, 7))

if __name__ == '__main__':
    unittest.main()
