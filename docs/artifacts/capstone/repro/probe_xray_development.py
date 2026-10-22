"""Check medical-model inputs on a source-bound development sample.

Development-only: reads the pinned cache report and the train/validation manifests,
never the test manifest or any label column; fits nothing and scores nothing.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import warnings

REPORT_SHA256 = '6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'
AUDIT_REPORT_SHA256 = '7851361e9fa6f72d24745b781f10d5453800a16bcb3148abed775a3cd2bb2b8d'
AUDIT_IMAGES_SHA256 = 'bb267d0a96d8b19b4da47113c8b76ec05d51e1f39b7a6753f475c2a555c48829'
DEVELOPMENT_SPLITS = ('train', 'validation')
PER_SPLIT = 4
SELECTION_NAMESPACE = 'capstone-xray-development-probe-v1'
MANIFEST_FIELDS = ['image_index', 'patient_id', 'image_path', 'label', 'findings']
SAMPLE_FIELDS = ('split', 'image_index', 'patient_id', 'image_path')
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16 * 1024 * 1024
IMAGE_INDEX = re.compile(r'(\d{8})_\d{3}\.png')
IMAGE_DIRECTORY = re.compile(r'images_\d{3}')
HEX64 = re.compile(r'[0-9a-f]{64}')


class Refusal(ValueError):
    pass


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _check_path(row):
    match = IMAGE_INDEX.fullmatch(row.get('image_index', ''))
    parts = row.get('image_path', '').split('/')
    if (not match or len(parts) != 3 or not IMAGE_DIRECTORY.fullmatch(parts[0])
            or parts[1] != 'images' or parts[2] != row['image_index']):
        raise Refusal('image_identity_noncanonical')
    return match.group(1)


def _check_identity(row):
    prefix = _check_path(row)
    if row.get('patient_id') != str(int(prefix)):
        raise Refusal('image_identity_patient_mismatch')


def _rank(split, patient_id):
    return _sha256(f'{SELECTION_NAMESPACE}\0{split}\0{patient_id}'.encode()), int(patient_id)


def select_development(cache, expected_report):
    """Verify pins and whole-set separation, then pick PER_SPLIT patients per split.

    Selection depends only on a namespaced hash of the patient identifier, never on
    labels, findings, image content or any model output.
    """
    cache = Path(cache)
    raw_report = (cache / 'report.json').read_bytes()
    report_sha = _sha256(raw_report)
    if report_sha != expected_report:
        raise Refusal('report_hash_mismatch')
    hashes = json.loads(raw_report)['payload_hashes']
    manifests, manifest_sha = {}, {}
    for split in DEVELOPMENT_SPLITS:
        data = (cache / f'{split}.csv').read_bytes()
        manifest_sha[split] = _sha256(data)
        if manifest_sha[split] != hashes.get(f'{split}.csv'):
            raise Refusal(f'manifest_hash_mismatch:{split}')
        reader = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
        rows = list(reader)
        if reader.fieldnames != MANIFEST_FIELDS or not rows:
            raise Refusal(f'manifest_schema:{split}')
        for row in rows:
            _check_identity(row)
        if len({row['image_index'] for row in rows}) != len(rows):
            raise Refusal(f'duplicate_image:{split}')
        manifests[split] = rows
    patients = {split: {row['patient_id'] for row in rows} for split, rows in manifests.items()}
    images = {split: {row['image_index'] for row in rows} for split, rows in manifests.items()}
    if patients['train'] & patients['validation']:
        raise Refusal('patient_overlap')
    if images['train'] & images['validation']:
        raise Refusal('image_overlap')
    selected = []
    for split, rows in manifests.items():
        by_patient = {}
        for row in rows:
            by_patient.setdefault(row['patient_id'], []).append(row)
        if len(by_patient) < PER_SPLIT:
            raise Refusal(f'too_few_patients:{split}')
        for patient in sorted(by_patient, key=lambda p: _rank(split, p))[:PER_SPLIT]:
            row = min(by_patient[patient], key=lambda r: r['image_index'])
            selected.append({'split': split, **{key: row[key] for key in SAMPLE_FIELDS[1:]}})
    return {
        'rows': selected, 'reportSha256': report_sha, 'manifestSha256': manifest_sha,
        'developmentImages': {split: len(rows) for split, rows in manifests.items()},
        'developmentPatients': {split: len(ids) for split, ids in patients.items()},
        'selectionRule': f'per split, ascending SHA256("{SELECTION_NAMESPACE}" NUL split NUL patient_id); '
                         f'first {PER_SPLIT} patients; lowest image_index per patient',
    }


def load_audit_rows(audit_dir, selected, *, report_sha256=AUDIT_REPORT_SHA256, images_sha256=AUDIT_IMAGES_SHA256):
    """Return pinned audit identities for the selected images only (no label column)."""
    audit_dir = Path(audit_dir)
    raw_report = (audit_dir / 'report.json').read_bytes()
    if _sha256(raw_report) != report_sha256:
        raise Refusal('audit_report_hash_mismatch')
    raw_images = (audit_dir / 'images.csv').read_bytes()
    images_sha = _sha256(raw_images)
    if images_sha != images_sha256 or images_sha != json.loads(raw_report)['payload_hashes']['images.csv']:
        raise Refusal('audit_images_hash_mismatch')
    wanted = {row['image_index']: row for row in selected}
    found = {}
    for row in csv.DictReader(io.StringIO(raw_images.decode('utf-8'), newline='')):
        sample = wanted.get(row['image_index'])
        if sample is None:
            continue
        if row['image_index'] in found:
            raise Refusal('audit_duplicate_row')
        if any(row[key] != sample[key] for key in SAMPLE_FIELDS):
            raise Refusal('audit_binding_mismatch')
        if row['error'] or not HEX64.fullmatch(row['file_sha256']):
            raise Refusal('audit_row_unqualified')
        found[row['image_index']] = {key: row[key] for key in ('image_index', 'image_path', 'file_bytes', 'file_sha256', 'width', 'height', 'rgb_sha256')}
    if set(found) != set(wanted):
        raise Refusal('audit_row_missing')
    return found


def read_bound_image(root, row, audit):
    """Read raw bytes only after path checks, and refuse them unless they match the audit."""
    _check_path(row)
    for key in ('image_index', 'image_path'):
        if key in audit and audit[key] != row[key]:
            raise Refusal('audit_binding_mismatch')
    expected = audit.get('file_sha256', '')
    if not HEX64.fullmatch(expected):
        raise Refusal('audit_hash_missing')
    root = Path(root).resolve(strict=True)
    path = root / row['image_path']
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise Refusal('image_identity_escape')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise Refusal('raw_image_too_large')
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if _sha256(data) != expected or ('file_bytes' in audit and str(len(data)) != str(audit['file_bytes'])):
        raise Refusal('raw_image_hash_mismatch')
    return data


def decode_grayscale(data, audit):
    """Decode verified PNG bytes to one 8-bit channel under an explicit contract.

    Accepts 8-bit grayscale ('L'), or RGB/RGBA whose colour channels are identical
    and whose alpha (if any) is fully opaque; then keeps channel 0, matching the
    pinned XRV reshape rule. Anything else is refused, not converted.
    """
    import numpy as np
    from PIL import Image
    import imageio.v3 as iio

    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
                raise Refusal('decode_not_single_frame_png')
            if image.width * image.height > MAX_PIXELS:
                raise Refusal('decode_pixel_limit')
            image.load()
            mode, size = image.mode, image.size
            rgb = image.convert('RGB')
            pixels = np.asarray(image)
    digest = hashlib.sha256(b'capstone-nih-rgb-v1\0')
    digest.update(f'{rgb.width}x{rgb.height}\0'.encode('ascii'))
    digest.update(rgb.tobytes())
    if digest.hexdigest() != audit['rgb_sha256'] or [str(v) for v in size] != [audit['width'], audit['height']]:
        raise Refusal('decoded_pixel_identity_mismatch')
    if pixels.dtype != np.uint8:
        raise Refusal(f'decode_unsupported_mode:{mode}')
    if mode == 'L':
        gray = pixels
    elif mode in ('RGB', 'RGBA'):
        if not (np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 0], pixels[..., 2])):
            raise Refusal('decode_colour_channels_differ')
        if mode == 'RGBA' and not bool((pixels[..., 3] == 255).all()):
            raise Refusal('decode_alpha_not_opaque')
        gray = pixels[..., 0]
    else:
        raise Refusal(f'decode_unsupported_mode:{mode}')
    reference = np.asarray(iio.imread(data, extension='.png'))
    if reference.ndim > 2:
        reference = reference[:, :, 0]
    return np.ascontiguousarray(gray), mode, bool(np.array_equal(reference, gray))


def probe(cache, audit_dir, image_root, weight_dir, output):
    output = Path(output)
    if output.exists():
        raise Refusal('output_exists')
    selection = select_development(cache, REPORT_SHA256)
    audit = load_audit_rows(audit_dir, selection['rows'])
    from qualify_xray_backbone import MODEL_ID, WEIGHT_NAME, WEIGHT_SHA256, sha256
    checkpoint = Path(weight_dir).resolve() / WEIGHT_NAME
    if not checkpoint.is_file() or sha256(checkpoint) != WEIGHT_SHA256:
        raise Refusal('checkpoint_sha256_mismatch')

    from importlib.metadata import version
    import inspect
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import torchxrayvision as xrv

    if version('torchxrayvision') != '1.5.7':
        raise Refusal('unsupported_library_version')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
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
    crop = xrv.datasets.XRayCenterCrop()
    resize = xrv.datasets.XRayResizer(224, engine='skimage')
    samples, inputs = [], []
    for row in selection['rows']:
        record = {key: row[key] for key in ('split', 'image_index')}
        try:
            data = read_bound_image(image_root, row, audit[row['image_index']])
            gray, mode, agrees = decode_grayscale(data, audit[row['image_index']])
            if not agrees:
                raise Refusal('decoder_disagreement')
            array = resize(crop(xrv.utils.normalize(gray, 255)[None, :, :]))
            if array.shape != (1, 224, 224) or not np.isfinite(array).all():
                raise Refusal('preprocessing_contract_failed')
            record.update(status='decoded', mode=mode, decodedShape=list(gray.shape),
                          rawSha256Verified=True, rgbIdentityVerified=True, decodersAgree=True,
                          preprocessedSha256=_sha256(array.tobytes()),
                          preprocessedRange=[float(array.min()), float(array.max())])
            inputs.append(array)
        except Refusal as error:
            record.update(status='refused', reason=str(error))
        samples.append(record)
    refused = [s for s in samples if s['status'] == 'refused']
    result = {
        'schema': 'capstone.medical_backbone.development_probe.v1',
        'scope': 'Development split images only; test manifest and label columns are never read. '
                 'No classifier fitting, diagnostic scoring, utility estimate or clinical qualification.',
        'modelId': MODEL_ID, 'checkpointSha256': WEIGHT_SHA256,
        'cacheReportSha256': selection['reportSha256'], 'manifestSha256': selection['manifestSha256'],
        'auditReportSha256': AUDIT_REPORT_SHA256, 'auditImagesSha256': AUDIT_IMAGES_SHA256,
        'developmentImages': selection['developmentImages'],
        'developmentPatients': selection['developmentPatients'],
        'patientOverlap': 0, 'selectionRule': selection['selectionRule'],
        'preprocessing': {'decode': 'Pillow PNG; L or channel-identical RGB/RGBA (opaque) -> channel 0; imageio cross-decode must agree',
                          'normalize': 'xrv.utils.normalize(maxval=255) -> [-1024, 1024]',
                          'centerCrop': True, 'resize': [224, 224], 'resizeEngine': 'skimage', 'channels': 1},
        'packages': {name: version(name) for name in ['torch', 'torchvision', 'torchxrayvision', 'numpy', 'pillow', 'scikit-image', 'imageio']},
        'moduleSha256': {name: sha256(inspect.getsourcefile(module)) for name, module in [('models', xrv.models), ('datasets', xrv.datasets), ('utils', xrv.utils)]},
        'samples': samples,
    }
    if refused:
        result['status'] = 'development_probe_failed'
        result['failure'] = 'One or more selected images were refused; samples were not substituted.'
    else:
        batch = torch.from_numpy(np.stack(inputs))
        with torch.inference_mode():
            first = model.features2(batch)
            second = model.features2(batch)
            pooled = functional.adaptive_avg_pool2d(functional.relu(model.features(batch)), (1, 1)).flatten(1)
            single = torch.cat([model.features2(batch[i:i + 1]) for i in range(len(batch))])
        stable = before == parameter_digest()
        finite = bool(torch.isfinite(first).all())
        distinct = len({_sha256(row.contiguous().numpy().tobytes()) for row in first}) == len(first)
        checks = {
            'featureShape': list(first.shape), 'featuresFinite': finite,
            'repeatedFeaturesExactlyEqual': torch.equal(first, second),
            'manualReluAveragePoolExactlyEqual': torch.equal(first, pooled),
            'allFeatureRowsDistinct': distinct, 'parametersAndBuffersUnchanged': stable,
            'singleVersusBatchedMaxAbsDifference': float((single - first).abs().max()),
            'featureSha256': _sha256(first.contiguous().numpy().tobytes()),
            'parameterStateSha256': before,
        }
        for sample, row in zip(samples, first):
            sample['featureSummary'] = {'mean': float(row.mean()), 'std': float(row.std()), 'min': float(row.min()),
                                        'max': float(row.max()), 'zeroFraction': float((row == 0).float().mean())}
        result['featureChecks'] = checks
        passed = (first.shape == (len(samples), 1024) and finite and checks['repeatedFeaturesExactlyEqual']
                  and checks['manualReluAveragePoolExactlyEqual'] and distinct and stable)
        result['status'] = 'development_probe_pass' if passed else 'development_probe_failed'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--weight-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = probe(args.cache_dir, args.audit_dir, args.image_root, args.weight_dir, args.output)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': result['status'], 'samples': len(result['samples'])}))
    raise SystemExit(0 if result['status'] == 'development_probe_pass' else 2)
