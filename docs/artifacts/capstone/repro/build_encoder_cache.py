"""Build and verify A22 frozen-encoder feature caches (R2 DINOv2 ViT-B/14, R3 XRV DenseNet MIMIC-CXR).

Reuses build_xrv_cache's pinned image, decode and partition contract: train and the locked selection half
only; holdout images, labels and features are never touched and test.csv is never opened. Each encoder is
bound to a SHA-256-pinned checkpoint and verified by exact recomputation of a hash-chosen sample.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_xrv_cache as base  # noqa: E402

Refusal = base.Refusal
SOURCES = ('build_encoder_cache.py', 'build_xrv_cache.py', 'probe_xray_development.py')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
ENCODERS = {
    'R2': {'cache_version': 'capstone-nih-dinov2-base-cls-features-v1', 'feat_dim': 768,
           'model_id': 'facebook/dinov2-base', 'revision': 'f9e44c814b77203eaa57a6bdbbd535f21ede1415',
           'weight_name': 'model.safetensors',
           'weight_sha256': 'd73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'},
    'R3': {'cache_version': 'capstone-nih-xrv-mimic_ch-features-v1', 'feat_dim': 1024,
           'model_id': 'densenet121-res224-mimic_ch',
           'weight_name': 'mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt',
           'weight_sha256': '23b13a04459684ffa41247d068207a7657b4e1b4dec2b02431f5026bd75b1189'},
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dinov2_preprocess(gray):
    """Grayscale uint8 -> /255 -> bilinear (antialiased) 224x224 -> 3 identical channels -> ImageNet mean/std."""
    from PIL import Image
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise Refusal('preprocessing_input')
    resized = np.asarray(Image.fromarray(gray).resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    array = (np.repeat(resized[None, :, :], 3, axis=0) - IMAGENET_MEAN) / IMAGENET_STD
    if array.shape != (3, 224, 224):
        raise Refusal('preprocessing_shape')
    return array.astype(np.float32)


class _Pinned:
    def _digest(self):
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def identity(self):
        return dict(self._identity)

    def state_unchanged(self):
        return self._digest() == self.before


class DinoV2Extractor(_Pinned):
    """Frozen DINOv2 ViT-B/14 final-layer CLS token on MPS (bit-identical across repeats and batch sizes)."""

    def __init__(self, spec=ENCODERS['R2'], device='mps'):
        from importlib.metadata import version
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        folder = Path(snapshot_download(spec['model_id'], revision=spec['revision'], local_files_only=True))
        if _file_sha256(folder / spec['weight_name']) != spec['weight_sha256']:
            raise Refusal('checkpoint_sha256_mismatch')
        torch.use_deterministic_algorithms(True)
        self.torch, self.device = torch, device
        self.model = AutoModel.from_pretrained(folder, local_files_only=True).eval().requires_grad_(False)
        self.before = self._digest()
        self.model = self.model.to(device)
        self._identity = {'modelId': spec['model_id'], 'revision': spec['revision'], 'checkpointSha256': spec['weight_sha256'],
                          'parameterStateSha256': self.before, 'device': device, 'dtype': 'float32',
                          'interface': 'last_hidden_state[:, 0] (final-layer CLS token)',
                          'packages': {n: version(n) for n in ['torch', 'transformers', 'numpy', 'pillow']}}

    def features(self, batch):
        with self.torch.inference_mode():
            out = self.model(pixel_values=self.torch.from_numpy(np.ascontiguousarray(batch)).to(self.device))
            return out.last_hidden_state[:, 0].float().cpu().numpy()



class XrvWeightsExtractor(_Pinned):
    """Frozen CPU XRV DenseNet `features2()` bound to a pinned checkpoint (R1's interface, other weights)."""

    def __init__(self, weight_dir, spec=ENCODERS['R3'], threads=4):
        from importlib.metadata import version
        import torch
        import torchxrayvision as xrv
        checkpoint = Path(weight_dir).resolve() / spec['weight_name']
        if not checkpoint.is_file() or _file_sha256(checkpoint) != spec['weight_sha256']:
            raise Refusal('checkpoint_sha256_mismatch')
        if version('torchxrayvision') != '1.5.7':
            raise Refusal('unsupported_library_version')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.model = xrv.models.DenseNet(weights=spec['model_id'], cache_dir=str(checkpoint.parent))
        if not Path(self.model.weights_filename_local).samefile(checkpoint):
            raise Refusal('loaded_checkpoint_identity_mismatch')
        self.model.eval().requires_grad_(False)
        self.before = self._digest()
        self._identity = {'modelId': spec['model_id'], 'checkpointSha256': spec['weight_sha256'],
                          'parameterStateSha256': self.before, 'device': 'cpu', 'torchThreads': torch.get_num_threads(),
                          'interface': 'features2 (ReLU + global average pool)',
                          'packages': {n: version(n) for n in ['torch', 'torchvision', 'torchxrayvision', 'numpy']}}

    def features(self, batch):
        with self.torch.inference_mode():
            return self.model.features2(self.torch.from_numpy(np.ascontiguousarray(batch))).numpy()


def encoder(name, weight_dir=None, threads=4):
    """(extractor, preprocess, spec) for an A22 representation."""
    spec = ENCODERS[name]
    if name == 'R2':
        return DinoV2Extractor(spec), dinov2_preprocess, spec
    return XrvWeightsExtractor(weight_dir, spec, threads), base.xrv_preprocess, spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'verify'])
    parser.add_argument('--representation', choices=sorted(ENCODERS), required=True)
    for flag in ('--cache-dir', '--lock-dir', '--audit-dir', '--image-root', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument('--weight-dir', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sample', type=int, default=32)
    args = parser.parse_args()
    inputs = dict(cache=args.cache_dir, lock_dir=args.lock_dir, audit_dir=args.audit_dir, image_root=args.image_root,
                  output=args.output)
    try:
        extractor, preprocess, spec = encoder(args.representation, args.weight_dir, args.threads)
        common = dict(extractor=extractor, preprocess=preprocess, feat_dim=spec['feat_dim'],
                      cache_version=spec['cache_version'])
        if args.command == 'build':
            result = base.build_cache(**inputs, **common, workers=args.workers, batch_size=args.batch_size,
                                      source_names=SOURCES)
            print(json.dumps({'status': 'built', 'splits': result['splits'], 'elapsedSeconds': result['elapsedSeconds']}))
        else:
            if args.receipt is None or args.receipt.exists():
                raise Refusal('receipt_missing_or_exists')
            result = base.verify_cache(**inputs, **common, sample=args.sample)
            with args.receipt.open('x') as stream:
                json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
            print(json.dumps({'status': 'verified', 'sampledRows': result['sampledRows']}))
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
