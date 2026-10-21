"""Source-bound synthetic execution check for one CheXpert feature candidate."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = 'densenet121-res224-chex'
WEIGHT_NAME = 'chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
WEIGHT_SHA256 = '1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4'


class Refusal(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def qualify(cache_dir, output):
    output = Path(output)
    if output.exists():
        raise Refusal('output_exists')
    checkpoint = Path(cache_dir).resolve() / WEIGHT_NAME
    if not checkpoint.is_file():
        raise Refusal('checkpoint_missing')
    if sha256(checkpoint) != WEIGHT_SHA256:
        raise Refusal('checkpoint_sha256_mismatch')

    from importlib.metadata import version
    import inspect
    import time
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchxrayvision as xrv

    if version('torchxrayvision') != '1.5.7':
        raise Refusal('unsupported_library_version')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261021)
    model = xrv.models.DenseNet(weights=MODEL_ID, cache_dir=str(checkpoint.parent))
    if not Path(model.weights_filename_local).samefile(checkpoint):
        raise Refusal('loaded_checkpoint_identity_mismatch')
    model.eval().requires_grad_(False)

    def parameter_digest():
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    before = parameter_digest()
    raw = np.stack([
        np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (288, 1)),
        np.tile(np.linspace(255, 0, 288, dtype=np.uint8)[:, None], (1, 320)),
    ])
    endpoints = xrv.utils.normalize(np.array([0, 255], dtype=np.uint8), 255)
    if endpoints.tolist() != [-1024.0, 1024.0]:
        raise Refusal('normalization_contract_changed')
    crop = xrv.datasets.XRayCenterCrop()
    resize = xrv.datasets.XRayResizer(224, engine='skimage')
    inputs = np.stack([resize(crop(xrv.utils.normalize(image, 255)[None, :, :])) for image in raw])
    batch = torch.from_numpy(inputs)
    started = time.perf_counter()
    with torch.inference_mode():
        first = model.features2(batch)
        second = model.features2(batch)
        feature_map = model.features(batch)
        pooled = functional.adaptive_avg_pool2d(functional.relu(feature_map), (1, 1)).flatten(1)
    elapsed = time.perf_counter() - started
    finite = bool(torch.isfinite(first).all())
    repeat = torch.equal(first, second)
    pooling = torch.equal(first, pooled)
    stable = before == parameter_digest()
    if first.shape != (2, 1024) or not all([finite, repeat, pooling, stable]):
        raise Refusal('synthetic_feature_contract_failed')
    labels = xrv.models.model_urls[MODEL_ID]['labels']
    result = {
        'schema': 'capstone.medical_backbone.synthetic.v1',
        'status': 'synthetic_execution_pass', 'modelId': MODEL_ID,
        'scope': 'Synthetic images only; no NIH access, classifier fitting, utility estimate or clinical qualification.',
        'checkpointSha256': WEIGHT_SHA256,
        'packages': {name: version(name) for name in ['torch', 'torchvision', 'torchxrayvision', 'numpy', 'scikit-image', 'imageio', 'tifffile', 'lazy-loader']},
        'moduleSha256': {name: sha256(inspect.getsourcefile(module)) for name, module in [('models', xrv.models), ('datasets', xrv.datasets), ('utils', xrv.utils)]},
        'device': str(first.device), 'cpuThreads': torch.get_num_threads(),
        'parameterCount': sum(p.numel() for p in model.parameters()),
        'syntheticImageCount': 2, 'rawInputShape': list(raw.shape),
        'syntheticRawSha256': hashlib.sha256(raw.tobytes()).hexdigest(),
        'preprocessing': {'normalization8BitEndpoints': endpoints.tolist(), 'channels': 1, 'centerCrop': True, 'resize': [224, 224], 'resizeEngine': 'skimage', 'dtype': str(batch.dtype)},
        'preprocessedSha256': hashlib.sha256(inputs.tobytes()).hexdigest(),
        'featureMapShape': list(feature_map.shape), 'featureVectorShape': list(first.shape),
        'featureSha256': hashlib.sha256(first.contiguous().numpy().tobytes()).hexdigest(),
        'featuresFinite': finite, 'repeatedFeaturesExactlyEqual': repeat,
        'manualReluAveragePoolExactlyEqual': pooling, 'parametersAndBuffersUnchanged': stable,
        'parameterStateSha256': before, 'publishedTrainedTargets': [name for name in labels if name],
        'untrainedOutputSlots': len([name for name in labels if not name]),
        'threeSyntheticFeatureForwardsWallSeconds': elapsed,
        'timingScope': 'Only three two-image CPU feature forwards; no dataset throughput or device-scaling estimate.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = qualify(args.cache_dir, args.output)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': result['status'], 'featureVectorShape': result['featureVectorShape']}))
