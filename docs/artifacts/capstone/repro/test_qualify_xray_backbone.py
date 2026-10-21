"""Check source-bound refusal before the optional model runtime is imported."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('qualify_xray_backbone.py')
WEIGHT_NAME = 'chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'

class BackboneRefusalTests(unittest.TestCase):
    def invoke(self, cache, output):
        return subprocess.run([sys.executable, str(SCRIPT), '--cache-dir', str(cache), '--output', str(output)], capture_output=True, text=True, timeout=10)

    def test_wrong_checkpoint_is_refused_before_optional_import_or_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory); (cache / WEIGHT_NAME).write_bytes(b'foreign checkpoint')
            output = cache / 'receipt.json'; result = self.invoke(cache, output)
            self.assertEqual(result.returncode, 2)
            self.assertIn('checkpoint_sha256_mismatch', result.stdout)
            self.assertEqual(json.loads(result.stdout)['status'], 'refused')
            self.assertFalse(output.exists())
            self.assertNotIn('Traceback', result.stderr)

    def test_existing_evidence_is_preserved_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory); output = cache / 'receipt.json'; output.write_text('keep original evidence')
            result = self.invoke(cache, output)
            self.assertEqual(result.returncode, 2)
            self.assertIn('output_exists', result.stdout)
            self.assertEqual(output.read_text(), 'keep original evidence')
            self.assertNotIn('Traceback', result.stderr)

if __name__ == '__main__':
    unittest.main()
