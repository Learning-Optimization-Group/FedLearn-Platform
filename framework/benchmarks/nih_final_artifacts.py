"""Bound final checkpoints and aligned NIH scores for private research verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.nih_validation_pilot import _checkpoint
from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors
from fedlearn.simulation.config import config_hash, resolve
from fedlearn.simulation.core_qualification import qualify_core_record
from fedlearn.simulation.metrics import state_digest
from fedlearn.simulation.nih_cache import json_bytes, load_nih_cache, sha256_file
from fedlearn.simulation.rng import torch_rng_scope
from fedlearn.simulation.workloads import LinearHead, MLPHead

PAYLOADS = ('model.safetensors', 'validation.npz', 'test.npz')


def _context(cfg, record, pin):
    cfg = resolve(cfg)
    bare = {**record, 'meta': {k: v for k, v in record['meta'].items() if k != 'core_qualification'}}
    marker = qualify_core_record(bare, cfg, feature_report_sha256=pin,
        clients=cfg['federation']['num_clients'], evaluation_split=cfg['workload']['evaluation_split'])
    if record['meta'].get('core_qualification') != marker:
        raise ValueError('final artifacts require a matching qualified execution record')
    cache = load_nih_cache(Path(cfg['workload']['features_path']), expected_report_sha256=pin)
    if marker['feature_control'] != (not cache['provenance']['pretrained']):
        raise ValueError('feature control identity differs from qualified execution')
    provenance = cache['provenance']
    head = cfg['workload']['head']
    expected = {
        'source_dataset': provenance['dataset'], 'backbone': provenance['backbone'],
        'curated_cohort_sha256': provenance['curated_cohort_sha256'],
        'model_state_sha256': provenance['model_state_sha256'], 'source_hashes': provenance['source_hashes'],
        'train_manifest_sha256': provenance['selected_manifest_hashes']['train.csv'],
        'evaluation_manifest_sha256': provenance['selected_manifest_hashes'][f"{cfg['workload']['evaluation_split']}.csv"],
        'split_counts': provenance['splits'], 'n_train': len(cache['splits']['train']['y']),
        'n_test': len(cache['splits'][cfg['workload']['evaluation_split']]['y']),
        'parameters': 1026 if head['type'] == 'linear' else 515 * head['hidden'] + 2,
    }
    actual = {key: record['meta']['workload'].get(key) for key in expected}
    if json_bytes(actual) != json_bytes(expected):
        raise ValueError('record workload identity/count differs from verified cache/model')
    return cfg, cache, marker


def _model(cfg, params, final_digest):
    with torch_rng_scope(0):
        head = cfg['workload']['head']
        model = LinearHead(512, 2) if head['type'] == 'linear' else MLPHead(512, head['hidden'], 2)
    expected = model.state_dict()
    if set(params) != set(expected):
        raise ValueError('final parameter names differ from declared model')
    for key, value in params.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu'
                or value.shape != expected[key].shape or not torch.isfinite(value).all()):
            raise ValueError('invalid final parameter dtype/device/shape/value')
    if state_digest([params]) != final_digest:
        raise ValueError('final model digest differs from qualified record')
    model.load_state_dict(params, strict=True)
    return model.eval()


def _binding(cfg, record, cache, marker):
    return {'artifact_version': 'nih-final-research-artifacts-v1',
            'config_hash': config_hash(cfg), 'record_sha256': hashlib.sha256(json_bytes(record)).hexdigest(),
            'source_commit': record['meta']['provenance']['git_commit'],
            'features_report_sha256': cache['report_sha256'],
            'final_digest': record['meta']['final_digest'],
            'feature_control': marker['feature_control'], 'research_record_dp_safe': False,
            'prediction_recipe': {'device': 'cpu', 'batch': 'complete_split', 'logits_dtype': 'float32',
                'margin_dtype': 'float64_subtraction', 'torch_num_threads': torch.get_num_threads(),
                'torch': torch.__version__, 'numpy': np.__version__},
            'splits': {s: {'rows': len(cache['splits'][s]['y']),
                'manifest_sha256': cache['provenance']['selected_manifest_hashes'][f'{s}.csv']}
                for s in ('validation', 'test')}}


def _predict(model, split):
    # Use the same whole-split forward shape as the engine, avoiding batch-dependent
    # floating-point differences when independently reconciling final-round ranks.
    with torch.no_grad():
        logits = model(torch.from_numpy(np.array(split['x'], copy=True))).numpy()
    if logits.dtype != np.float32 or logits.shape != (len(split['y']), 2) or not np.isfinite(logits).all():
        raise ValueError('invalid final logits')
    double = logits.astype(np.float64)
    return {'logits': logits, 'margin': double[:, 1] - double[:, 0],
            'labels': np.array(split['y'], copy=True),
            'image_ids': np.asarray(split['image_ids']), 'patient_ids': np.asarray(split['patient_ids'])}


def write_final_artifacts(cfg, record, params, output, *, feature_report_sha256):
    """Persist research evidence; the last atomic manifest alone marks completion.

    Actual test scoring is restricted by the caller's prospectively frozen study
    protocol. Tests use explicitly labelled synthetic feature controls.
    """
    output = Path(output)
    cache_path = Path(cfg['workload']['features_path']).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('refusing existing final artifact output')
    if output.resolve().is_relative_to(cache_path):
        raise ValueError('final artifact output must be outside the immutable cache')
    cfg, cache, marker = _context(cfg, record, feature_report_sha256)
    model = _model(cfg, params, record['meta']['final_digest'])
    proof = _binding(cfg, record, cache, marker)
    output.mkdir(parents=True)
    blob = save_safetensors([(k, v.detach().numpy()) for k, v in params.items()])
    (output/'model.safetensors').write_bytes(blob)
    # Check codec round-trip before marking a checkpoint complete.
    restored, _ = load_safetensors(blob)
    for key, value in restored:
        if not np.array_equal(value, params[key].numpy()):
            raise ValueError('checkpoint roundtrip differs from final parameters')
    for split in ('validation', 'test'):
        np.savez(output/f'{split}.npz', **_predict(model, cache['splits'][split]))
    proof['parameter_shapes'] = {k: list(v.shape) for k, v in params.items()}
    proof['payloads'] = {name: {'sha256': sha256_file(output/name), 'bytes': (output/name).stat().st_size}
                         for name in PAYLOADS}
    _checkpoint(output/'manifest.json', proof)
    return proof


def verify_final_artifacts(cfg, record, output, *, feature_report_sha256):
    """Reload checkpoint and recompute every score/identity against the bound cache."""
    try:
        output = Path(output)
        if output.is_symlink() or not output.is_dir() or {p.name for p in output.iterdir()} != set(PAYLOADS)|{'manifest.json'}:
            raise ValueError('missing or unexpected final artifact files')
        if any((output/name).is_symlink() or not (output/name).is_file() for name in (*PAYLOADS, 'manifest.json')):
            raise ValueError('final artifact payload is not a regular file')
        cfg, cache, marker = _context(cfg, record, feature_report_sha256)
        proof = json.loads((output/'manifest.json').read_text())
        if (set(proof) != set(_binding(cfg, record, cache, marker))|{'parameter_shapes','payloads'}
                or any(proof[k] != v for k, v in _binding(cfg, record, cache, marker).items())):
            raise ValueError('final artifact execution binding differs')
        if proof['payloads'] != {name: {'sha256': sha256_file(output/name), 'bytes': (output/name).stat().st_size} for name in PAYLOADS}:
            raise ValueError('final artifact payload hash/size differs')
        tensors, metadata = load_safetensors((output/'model.safetensors').read_bytes())
        if metadata or len({k for k, _ in tensors}) != len(tensors):
            raise ValueError('invalid checkpoint metadata/keys')
        params = {k: torch.from_numpy(v.copy()) for k, v in tensors}
        model = _model(cfg, params, record['meta']['final_digest'])
        if proof['parameter_shapes'] != {k: list(v.shape) for k, v in params.items()}:
            raise ValueError('checkpoint parameter shape binding differs')
        splits = {}
        for split in ('validation', 'test'):
            expected = _predict(model, cache['splits'][split])
            with np.load(output/f'{split}.npz', allow_pickle=False) as archive:
                if set(archive.files) != set(expected) or len(archive.files) != len(expected):
                    raise ValueError('final score array names differ')
                arrays = {key: archive[key] for key in archive.files}
            if any(arrays[k].dtype != v.dtype or not np.array_equal(arrays[k], v) for k, v in expected.items()):
                raise ValueError('final scores/labels/identities differ from checkpoint and cache')
            splits[split] = arrays
        return {'manifest': proof, 'splits': splits, 'parameters': {k: v.numpy() for k, v in params.items()}}
    except (OSError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        raise ValueError(f'incomplete or invalid final artifacts: {error}') from error
