"""Build and verify the versioned 1,024-wide CheXpert-DenseNet feature cache.

Covers train and the locked-protocol selection half only. The holdout manifest is
read solely to prove the partition; holdout images, labels and features are never
touched, and test.csv is never opened. Payloads store no labels. Every image passes
the development probe's byte, pixel and decode contract or the whole build fails.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_xray_development as probe  # noqa: E402

Refusal = probe.Refusal
CACHE_VERSION = 'capstone-nih-xrv-chex-features-v1'
VERIFY_NAMESPACE = 'capstone-xrv-cache-replay-v1'
KEEP = ('image_index', 'patient_id', 'image_path')
PINS = {
    'report_sha256': probe.REPORT_SHA256,
    'lock_sha256': '083aa7ce95e40af7bdc00d27d396b0ac5954e7f6ebd444200ce97651baccc794',
    'audit_report_sha256': probe.AUDIT_REPORT_SHA256,
    'audit_images_sha256': probe.AUDIT_IMAGES_SHA256,
}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _rows(data, fields, name):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    if reader.fieldnames != fields:
        raise Refusal(f'manifest_schema:{name}')
    rows = [{key: row[key] for key in KEEP} for row in reader]
    for row in rows:
        probe._check_identity(row)
    if not rows or len({row['image_index'] for row in rows}) != len(rows):
        raise Refusal(f'manifest_rows:{name}')
    return rows


def plan_inputs(cache, lock_dir, audit_dir, pins):
    """Verify every pin and the lock partition; return the train/selection plan."""
    cache, lock_dir = Path(cache), Path(lock_dir)
    raw_report = (cache / 'report.json').read_bytes()
    report_sha = _sha256(raw_report)
    if report_sha != pins['report_sha256']:
        raise Refusal('report_hash_mismatch')
    hashes = json.loads(raw_report)['payload_hashes']
    manifests, manifest_sha = {}, {}
    for split in ('train', 'validation'):
        data = (cache / f'{split}.csv').read_bytes()
        manifest_sha[split] = _sha256(data)
        if manifest_sha[split] != hashes.get(f'{split}.csv'):
            raise Refusal(f'manifest_hash_mismatch:{split}')
        manifests[split] = _rows(data, probe.MANIFEST_FIELDS, split)
    if {r['patient_id'] for r in manifests['train']} & {r['patient_id'] for r in manifests['validation']}:
        raise Refusal('patient_overlap')
    raw_lock = (lock_dir / 'lock.json').read_bytes()
    if _sha256(raw_lock) != pins['lock_sha256']:
        raise Refusal('lock_hash_mismatch')
    lock = json.loads(raw_lock)
    if lock['cacheReportSha256'] != report_sha or lock['inputManifestSha256'] != manifest_sha:
        raise Refusal('lock_input_mismatch')
    parts = {}
    for name in ('selection', 'holdout'):
        data = (lock_dir / f'{name}.csv').read_bytes()
        if _sha256(data) != lock['manifestSha256'][name]:
            raise Refusal(f'{name}_manifest_hash_mismatch')
        parts[name] = _rows(data, list(KEEP), name)
    validation = {row['image_index']: row for row in manifests['validation']}
    selection_ids = [row['image_index'] for row in parts['selection']]
    holdout_ids = {row['image_index'] for row in parts['holdout']}
    if (set(selection_ids) & holdout_ids or set(selection_ids) | holdout_ids != set(validation)
            or any(validation[row['image_index']] != row for row in parts['selection'])):
        raise Refusal('lock_partition_mismatch')
    plan = {'train': manifests['train'], 'selection': [validation[i] for i in selection_ids]}
    samples = [{'split': 'train', **row} for row in plan['train']]
    samples += [{'split': 'validation', **row} for row in plan['selection']]
    audit = probe.load_audit_rows(audit_dir, samples, report_sha256=pins['audit_report_sha256'],
                                  images_sha256=pins['audit_images_sha256'])
    provenance = {
        'cacheReportSha256': report_sha, 'inputManifestSha256': manifest_sha,
        'lockSha256': pins['lock_sha256'], 'lockManifestSha256': lock['manifestSha256'],
        'auditReportSha256': pins['audit_report_sha256'], 'auditImagesSha256': pins['audit_images_sha256'],
    }
    return plan, audit, provenance


def xrv_preprocess(gray):
    """Pinned XRV contract: [-1024, 1024] scaling, center crop, skimage resize to 224."""
    import torchxrayvision as xrv
    array = xrv.datasets.XRayResizer(224, engine='skimage')(xrv.datasets.XRayCenterCrop()(xrv.utils.normalize(gray, 255)[None, :, :]))
    if array.shape != (1, 224, 224):
        raise Refusal('preprocessing_shape')
    return array


def _prepare(job):
    root, row, audit_row, preprocess = job
    try:
        data = probe.read_bound_image(root, row, audit_row)
        gray, mode, agrees = probe.decode_grayscale(data, audit_row)
        if not agrees:
            raise Refusal('decoder_disagreement')
        array = np.asarray(preprocess(gray), dtype=np.float32)
        if not np.isfinite(array).all():
            raise Refusal('preprocessing_nonfinite')
        return array, mode, None
    except Refusal as error:
        return None, None, str(error)


class XrvExtractor:
    """Frozen CPU `features2()` extractor bound to the pinned checkpoint and library."""

    def __init__(self, weight_dir, threads=4):
        from importlib.metadata import version
        import inspect
        from qualify_xray_backbone import MODEL_ID, WEIGHT_NAME, WEIGHT_SHA256, sha256
        checkpoint = Path(weight_dir).resolve() / WEIGHT_NAME
        if not checkpoint.is_file() or sha256(checkpoint) != WEIGHT_SHA256:
            raise Refusal('checkpoint_sha256_mismatch')
        import torch
        import torchxrayvision as xrv
        if version('torchxrayvision') != '1.5.7':
            raise Refusal('unsupported_library_version')
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.model = xrv.models.DenseNet(weights=MODEL_ID, cache_dir=str(checkpoint.parent))
        if not Path(self.model.weights_filename_local).samefile(checkpoint):
            raise Refusal('loaded_checkpoint_identity_mismatch')
        self.model.eval().requires_grad_(False)
        self.before = self._digest()
        self._identity = {
            'modelId': MODEL_ID, 'checkpointSha256': WEIGHT_SHA256, 'parameterStateSha256': self.before,
            'device': 'cpu', 'torchThreads': torch.get_num_threads(), 'interface': 'features2 (ReLU + global average pool)',
            'packages': {name: version(name) for name in ['torch', 'torchvision', 'torchxrayvision', 'numpy', 'pillow', 'scikit-image', 'imageio']},
            'moduleSha256': {name: sha256(inspect.getsourcefile(module)) for name, module in [('models', xrv.models), ('datasets', xrv.datasets), ('utils', xrv.utils)]},
        }

    def _digest(self):
        digest = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def features(self, batch):
        with self.torch.inference_mode():
            return self.model.features2(self.torch.from_numpy(np.ascontiguousarray(batch))).numpy()

    def identity(self):
        return dict(self._identity)

    def state_unchanged(self):
        return self._digest() == self.before


def _fail(failure, staging, reason, details):
    shutil.rmtree(staging, ignore_errors=True)
    with failure.open('x') as stream:
        json.dump({'schema': 'capstone.xrv_cache.failure.v1', 'reason': reason, **details}, stream, indent=2, sort_keys=True)
        stream.write('\n')
    raise Refusal(reason)


def _write_manifest(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(KEEP), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())


def build_cache(cache, lock_dir, audit_dir, image_root, output, *, extractor, pins=PINS, preprocess=xrv_preprocess,
                feat_dim=1024, workers=8, batch_size=64, cache_version=CACHE_VERSION,
                source_names=('build_xrv_cache.py', 'probe_xray_development.py', 'qualify_xray_backbone.py')):
    output = Path(output)
    failure = Path(f'{output}.failed.json')
    if output.exists() or failure.exists():
        raise Refusal('output_exists')
    plan, audit, provenance = plan_inputs(cache, lock_dir, audit_dir, pins)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.cache-', dir=output.parent))
    started = time.perf_counter()
    pool = ProcessPoolExecutor(workers) if workers else None
    splits, payload_hashes = {}, {}
    try:
        for split, rows in plan.items():
            features = np.empty((len(rows), feat_dim), dtype=np.float32)
            modes = collections.Counter()
            batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
            jobs = lambda chunk: [(image_root, row, audit[row['image_index']], preprocess) for row in chunk]  # noqa: E731
            pending = [pool.submit(_prepare, job) for job in jobs(batches[0])] if pool else None
            for index, chunk in enumerate(batches):
                results = [future.result() for future in pending] if pool else [_prepare(job) for job in jobs(chunk)]
                if pool and index + 1 < len(batches):
                    pending = [pool.submit(_prepare, job) for job in jobs(batches[index + 1])]
                refused = [{'split': split, 'image_index': row['image_index'], 'reason': reason}
                           for row, (_, _, reason) in zip(chunk, results) if reason]
                if refused:
                    _fail(failure, staging, f'image_refused:{len(refused)}',
                          {'refused': refused, 'inputs': provenance, 'processedBeforeFailure': {split: index * batch_size}})
                values = extractor.features(np.stack([array for array, _, _ in results]))
                if values.shape != (len(chunk), feat_dim) or values.dtype != np.float32 or not np.isfinite(values).all():
                    _fail(failure, staging, 'feature_contract_failed', {'split': split, 'batch': index, 'inputs': provenance})
                start = index * batch_size
                features[start:start + len(chunk)] = values
                modes.update(mode for _, mode, _ in results)
                if index % 50 == 0 or index + 1 == len(batches):
                    print(json.dumps({'split': split, 'done': start + len(chunk), 'total': len(rows),
                                      'elapsedSeconds': round(time.perf_counter() - started, 1)}), flush=True)
            np.save(staging / f'{split}-x.npy', features)
            _write_manifest(staging / f'{split}.csv', rows)
            for name in (f'{split}-x.npy', f'{split}.csv'):
                payload_hashes[name] = _sha256((staging / name).read_bytes())
            splits[split] = {'images': len(rows), 'patients': len({row['patient_id'] for row in rows}), 'pngModes': dict(modes)}
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    if not extractor.state_unchanged():
        _fail(failure, staging, 'parameter_state_changed', {'inputs': provenance})
    sources = {name: _sha256((Path(__file__).resolve().parent / name).read_bytes()) for name in source_names}
    report = {
        'schema_version': 1, 'cache_version': cache_version, 'dataset': 'NIH_ChestXray14',
        'feat_dim': feat_dim, 'feature_dtype': 'float32', 'splits': splits, 'payload_hashes': payload_hashes,
        'inputs': provenance, 'extractor': extractor.identity(), 'preprocess': getattr(preprocess, '__name__', str(preprocess)),
        'decodeContract': 'Pillow PNG; L or channel-identical opaque RGB/RGBA -> channel 0; audit raw-byte and RGB identity; imageio cross-decode equal',
        'labelsStored': False, 'holdoutExcluded': True, 'testManifestRead': False,
        'workers': workers, 'batchSize': batch_size, 'elapsedSeconds': round(time.perf_counter() - started, 1),
        'sourceSha256': sources,
        'limitations': ['Frozen features only; no classifier, labels or utility estimate.',
                        'Holdout features are extracted only at the protocol final evaluation.',
                        'The 512-wide historical cache is separate and unchanged.'],
    }
    (staging / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    staging.rename(output)
    return report


def verify_cache(cache, lock_dir, audit_dir, image_root, output, *, extractor, pins=PINS, preprocess=xrv_preprocess,
                 feat_dim=1024, sample=32, cache_version=CACHE_VERSION):
    """Recheck pins and payload hashes, then recompute a hash-chosen sample exactly."""
    output = Path(output)
    plan, audit, provenance = plan_inputs(cache, lock_dir, audit_dir, pins)
    raw_report = (output / 'report.json').read_bytes()
    report = json.loads(raw_report)
    for name, digest in report['payload_hashes'].items():
        if _sha256((output / name).read_bytes()) != digest:
            raise Refusal(f'payload_hash_mismatch:{name}')
    if report['inputs'] != provenance or report['feat_dim'] != feat_dim or report['cache_version'] != cache_version:
        raise Refusal('report_provenance_mismatch')
    sampled = {}
    for split, rows in plan.items():
        if _rows((output / f'{split}.csv').read_bytes(), list(KEEP), split) != rows:
            raise Refusal(f'manifest_mismatch:{split}')
        stored = np.load(output / f'{split}-x.npy')
        if stored.shape != (len(rows), feat_dim) or stored.dtype != np.float32:
            raise Refusal(f'payload_shape_mismatch:{split}')
        picks = sorted(range(len(rows)), key=lambda i: _sha256(f'{VERIFY_NAMESPACE}\0{split}\0{rows[i]["image_index"]}'.encode()))[:sample]
        results = [_prepare((image_root, rows[i], audit[rows[i]['image_index']], preprocess)) for i in picks]
        if any(reason for _, _, reason in results):
            raise Refusal(f'replay_image_refused:{split}')
        if not np.array_equal(extractor.features(np.stack([array for array, _, _ in results])), stored[picks]):
            raise Refusal(f'replay_mismatch:{split}')
        sampled[split] = len(picks)
    return {'schema': 'capstone.xrv_cache.verification.v1', 'cacheReportSha256': _sha256(raw_report),
            'payloadHashesVerified': sorted(report['payload_hashes']), 'sampledRows': sampled,
            'allSampledRowsExactlyEqual': True, 'parametersUnchanged': extractor.state_unchanged(),
            'extractor': extractor.identity(), 'samplingRule': f'ascending SHA256("{VERIFY_NAMESPACE}" NUL split NUL image_index)'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'verify'])
    for flag in ('--cache-dir', '--lock-dir', '--audit-dir', '--image-root', '--weight-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--sample', type=int, default=32)
    args = parser.parse_args()
    inputs = dict(cache=args.cache_dir, lock_dir=args.lock_dir, audit_dir=args.audit_dir, image_root=args.image_root, output=args.output)
    try:
        if args.command == 'build':
            result = build_cache(**inputs, extractor=XrvExtractor(args.weight_dir, args.threads), workers=args.workers, batch_size=args.batch_size)
            print(json.dumps({'status': 'built', 'splits': result['splits'], 'elapsedSeconds': result['elapsedSeconds']}))
        else:
            if args.receipt is None or args.receipt.exists():
                raise Refusal('receipt_missing_or_exists')
            result = verify_cache(**inputs, extractor=XrvExtractor(args.weight_dir, args.threads), sample=args.sample)
            with args.receipt.open('x') as stream:
                json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
            print(json.dumps({'status': 'verified', 'sampledRows': result['sampledRows']}))
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
