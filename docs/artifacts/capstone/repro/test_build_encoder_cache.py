"""A22 encoder caches: preprocessing contract and pinned, deterministic extractors."""
from pathlib import Path
import unittest

import numpy as np

import build_encoder_cache as enc

WEIGHTS = Path(__file__).resolve().parents[4] / '.temp' / 'capstone-effusion-2026-10-22' / 'weights-mimic_ch'


class EncoderCacheTests(unittest.TestCase):
    def test_dinov2_preprocess_contract(self):
        gray = np.zeros((1024, 1024), np.uint8); gray[:, 512:] = 255
        out = enc.dinov2_preprocess(gray)
        self.assertEqual(out.shape, (3, 224, 224)); self.assertEqual(out.dtype, np.float32)
        white = (1.0 - enc.IMAGENET_MEAN[:, 0, 0]) / enc.IMAGENET_STD[:, 0, 0]
        np.testing.assert_allclose(out[:, 0, -1], white, rtol=1e-6)
        with self.assertRaises(enc.Refusal):
            enc.dinov2_preprocess(gray.astype(np.float32))

    def test_dinov2_extractor_is_pinned_deterministic_and_768_wide(self):
        extractor = enc.DinoV2Extractor()
        batch = np.stack([enc.dinov2_preprocess(np.random.default_rng(i).integers(0, 255, (512, 512), dtype=np.uint8))
                          for i in range(4)])
        a, b = extractor.features(batch), extractor.features(batch[:2])
        self.assertEqual(a.shape, (4, 768)); self.assertTrue(np.array_equal(a[:2], b))
        self.assertTrue(extractor.state_unchanged())

    @unittest.skipUnless((WEIGHTS / enc.ENCODERS['R3']['weight_name']).is_file(), 'mimic_ch weights not present')
    def test_xrv_mimic_extractor_is_pinned_and_1024_wide(self):
        extractor = enc.XrvWeightsExtractor(WEIGHTS)
        batch = np.stack([enc.base.xrv_preprocess(np.random.default_rng(i).integers(0, 255, (512, 512)).astype(np.uint8))
                          for i in range(2)])
        self.assertEqual(extractor.features(batch).shape, (2, 1024))
        self.assertTrue(extractor.state_unchanged())


if __name__ == '__main__':
    unittest.main()
