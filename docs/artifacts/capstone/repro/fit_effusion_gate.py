"""Effusion protocol v1, sections 4, 5 and 7: the 20 fixed centralised fits and the learning gate.

Fits on train, selects and scores on the selection half only. The holdout is never
indexed and the official test arrays are never opened. Every fit is retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import effusion_labels  # noqa: E402

Refusal = effusion_labels.Refusal
GRID = [(c, w) for c in (0.001, 0.01, 0.1, 1.0, 10.0) for w in (None, 'balanced')]
MAX_ITER = 2000
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261022
AUROC_FLOOR = 0.80
TARGET_SENSITIVITY = 0.90
HISTORICAL_REPORT_SHA256 = effusion_labels.PINS['report_sha256']


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def standardiser(train):
    """Train-only mean/std (float64); zero-variance columns get std 1."""
    mean = train.astype(np.float64).mean(axis=0)
    std = train.astype(np.float64).std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def select(fits):
    """Index of the converged, finite fit with the highest selection AUROC; ties -> smaller C, then unweighted."""
    eligible = [i for i, fit in enumerate(fits) if fit['converged'] and fit['finite']]
    if not eligible:
        raise Refusal('no_converged_fit')
    return min(eligible, key=lambda i: (-fits[i]['selectionAuroc'], fits[i]['C'], fits[i]['classWeight'] is not None))


def operating_point(y, scores, target):
    """Highest threshold whose sensitivity (score >= threshold) reaches `target`."""
    positives = np.sort(scores[y == 1])[::-1]
    k = math.ceil(target * len(positives) - 1e-9)
    threshold = float(positives[k - 1])
    return {'threshold': threshold, **confusion(y, scores >= threshold)}


def confusion(y, predicted):
    tp = int((predicted & (y == 1)).sum()); fn = int((~predicted & (y == 1)).sum())
    tn = int((~predicted & (y == 0)).sum()); fp = int((predicted & (y == 0)).sum())
    ratio = lambda a, b: a / (a + b) if a + b else None  # noqa: E731
    return {'sensitivity': ratio(tp, fn), 'specificity': ratio(tn, fp), 'ppv': ratio(tp, fp), 'npv': ratio(tn, fn),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def cluster_bootstrap(patients, *, reps, seed):
    """Yield image-index arrays from patients resampled with replacement (whole patients)."""
    groups = {}
    for index, patient in enumerate(patients):
        groups.setdefault(patient, []).append(index)
    members = [np.array(groups[p]) for p in sorted(groups)]
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        yield np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])


def _patient_max(y, scores, patients):
    order = {}
    for value, score, patient in zip(y, scores, patients):
        label, best = order.get(patient, (0, -np.inf))
        order[patient] = (max(label, int(value)), max(best, float(score)))
    labels, maxima = zip(*order.values())
    return np.array(labels), np.array(maxima)


def run_gate(features, y_train, y_selection, patients, *, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED,
             max_iter=MAX_ITER, reference=None):
    """Fit the fixed grid for R0 and R1, select, bootstrap and apply the pre-registered gate."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    fits, heads, scores, cross_check = [], {}, {}, 0.0
    for name in ('R0', 'R1'):
        train, selection = features[name]
        mean, std = standardiser(train)
        z_train = ((train - mean) / std).astype(np.float64)
        z_selection = ((selection - mean) / std).astype(np.float64)
        block = []
        for c, weight in GRID:
            model = LogisticRegression(C=c, class_weight=weight, solver='lbfgs', max_iter=max_iter)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(z_train, y_train)
            margin = model.decision_function(z_selection)
            finite = bool(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all() and np.isfinite(margin).all())
            if not finite:
                margin = np.nan_to_num(margin)
            block.append({'representation': name, 'C': c, 'classWeight': weight,
                          'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'finite': finite,
                          'iterations': int(model.n_iter_[0]), 'fitSeconds': round(time.perf_counter() - started, 2),
                          'trainAuroc': float(roc_auc_score(y_train, model.decision_function(z_train))),
                          'selectionAuroc': float(roc_auc_score(y_selection, margin)),
                          'coefficientL2': float(np.linalg.norm(model.coef_)),
                          '_model': model, '_margin': margin})
        chosen = block[select(block)]
        exact = np.einsum('ij,j->i', z_selection, chosen['_model'].coef_[0]) + chosen['_model'].intercept_[0]
        cross_check = max(cross_check, float(np.abs(exact - chosen['_margin']).max()))
        heads[name] = {'mean': mean, 'std': std, 'coef': chosen['_model'].coef_[0].copy(),
                       'intercept': float(chosen['_model'].intercept_[0]), 'C': chosen['C'], 'classWeight': chosen['classWeight']}
        scores[name] = chosen['_margin']
        for fit in block:
            fit['selected'] = fit is chosen
            fits.append({k: v for k, v in fit.items() if not k.startswith('_')})
    controls = {'C0': np.zeros(len(y_selection))}
    if reference is not None:
        controls['X'] = reference

    def summary(name, margin):
        platt = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(margin[:, None], y_selection)
        calibrated = platt.predict_proba(margin[:, None])[:, 1]
        fpr, tpr, thresholds = roc_curve(y_selection, margin)
        youden = int(np.argmax(tpr - fpr))
        patient_y, patient_score = _patient_max(y_selection, margin, patients)
        return {'auroc': float(roc_auc_score(y_selection, margin)),
                'averagePrecision': float(average_precision_score(y_selection, margin)),
                'prevalence': float(y_selection.mean()),
                'patientLevelAuroc': float(roc_auc_score(patient_y, patient_score)),
                'platt': {'slope': float(platt.coef_[0][0]), 'intercept': float(platt.intercept_[0])},
                'operatingPoint': operating_point(y_selection, calibrated, TARGET_SENSITIVITY),
                'youden': {'marginThreshold': float(thresholds[youden]), **confusion(y_selection, margin >= thresholds[youden])},
                'inSampleBrierAfterPlatt': float(np.mean((calibrated - y_selection) ** 2))}

    named = {**scores, **{k: v for k, v in controls.items() if k != 'C0'}}
    results = {name: summary(name, margin) for name, margin in named.items()}
    results['C0'] = {'auroc': 0.5, 'prevalence': float(y_selection.mean()),
                     'operatingPoint': {'threshold': 0.0, **confusion(y_selection, np.ones(len(y_selection), bool))},
                     'alwaysNegative': confusion(y_selection, np.zeros(len(y_selection), bool))}
    draws = {name: [] for name in named}
    draws['delta_R1_R0'] = []
    if 'X' in named:
        draws['delta_R1_X'] = []
    skipped = 0
    for index in cluster_bootstrap(patients, reps=reps, seed=seed):
        y = y_selection[index]
        if y.min() == y.max():
            skipped += 1
            continue
        values = {name: roc_auc_score(y, margin[index]) for name, margin in named.items()}
        for name, value in values.items():
            draws[name].append(value)
        draws['delta_R1_R0'].append(values['R1'] - values['R0'])
        if 'X' in named:
            draws['delta_R1_X'].append(values['R1'] - values['X'])
    intervals = {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for name, v in draws.items()}
    criteria = {'deltaAurocIntervalAboveZero': intervals['delta_R1_R0'][0] > 0,
                'r1AurocAtLeastFloor': results['R1']['auroc'] >= AUROC_FLOOR}
    gate = {'passed': all(criteria.values()), 'criteria': criteria, 'aurocFloor': AUROC_FLOOR,
            'deltaAuroc': results['R1']['auroc'] - results['R0']['auroc'], 'deltaAurocInterval': intervals['delta_R1_R0']}
    bootstrap = {'reps': reps, 'seed': seed, 'skippedSingleClassDraws': skipped, 'intervals95': intervals,
                 'method': 'patient-cluster percentile bootstrap on the selection half'}
    return {'fits': fits, 'selected': results, 'gate': gate, 'bootstrap': bootstrap, 'blasCrossCheckMaxAbsDiff': cross_check,
            '_heads': heads, '_scores': named}


def _load_representations(historical, xrv, xrv_report_sha256, labels):
    """R0 rows from the historical 512-wide cache and R1 rows from the new cache, aligned to the label plan."""
    raw = (Path(historical) / 'report.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != HISTORICAL_REPORT_SHA256:
        raise Refusal('historical_report_hash_mismatch')
    hashes = json.loads(raw)['payload_hashes']
    for name in ('train-x.npy', 'validation-x.npy'):
        if _sha256_file(Path(historical) / name) != hashes[name]:
            raise Refusal(f'historical_payload_hash_mismatch:{name}')
    import csv
    order = {}
    for split in ('train', 'validation'):  # image_index column only; no label fields are parsed
        with (Path(historical) / f'{split}.csv').open(newline='') as stream:
            reader = csv.reader(stream)
            if next(reader)[0] != 'image_index':
                raise Refusal(f'historical_manifest_schema:{split}')
            order[split] = {row[0]: i for i, row in enumerate(reader)}
    train_index = np.array([order['train'][r['image_index']] for r in labels['train']['rows']])
    selection_index = np.array([order['validation'][r['image_index']] for r in labels['selection']['rows']])
    r0_train = np.load(Path(historical) / 'train-x.npy')[train_index]
    r0_selection = np.asarray(np.load(Path(historical) / 'validation-x.npy', mmap_mode='r')[np.sort(selection_index)])
    r0_selection = r0_selection[np.argsort(np.argsort(selection_index))]
    raw = (Path(xrv) / 'report.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != xrv_report_sha256:
        raise Refusal('xrv_report_hash_mismatch')
    report = json.loads(raw)
    r1 = {}
    for split in ('train', 'selection'):
        for name in (f'{split}-x.npy', f'{split}.csv'):
            if _sha256_file(Path(xrv) / name) != report['payload_hashes'][name]:
                raise Refusal(f'xrv_payload_hash_mismatch:{name}')
        with (Path(xrv) / f'{split}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if rows != labels[split]['rows']:
            raise Refusal(f'xrv_manifest_mismatch:{split}')
        r1[split] = np.load(Path(xrv) / f'{split}-x.npy')
    if r0_train.shape[1] != 512 or r1['train'].shape[1] != 1024:
        raise Refusal('feature_width_mismatch')
    return {'R0': (r0_train, r0_selection), 'R1': (r1['train'], r1['selection'])}, report


def reference_scores(weight_dir, selection_features):
    """Control X: the checkpoint's own Effusion classifier row on the cached features2 output."""
    from build_xrv_cache import XrvExtractor
    extractor = XrvExtractor(weight_dir)
    model = extractor.model
    index = list(model.pathologies).index('Effusion')
    weight = model.classifier.weight.detach().numpy()[index].astype(np.float64)
    bias = float(model.classifier.bias.detach().numpy()[index])
    import torch
    with torch.inference_mode():
        check = model.classifier(torch.from_numpy(selection_features[:256]))[:, index].numpy()
    margin = selection_features.astype(np.float64) @ weight + bias
    if not np.allclose(check, margin[:256], rtol=0, atol=1e-4):
        raise Refusal('reference_head_mismatch')
    return margin, {'pathologyIndex': index, 'checkpointSha256': extractor.identity()['checkpointSha256'],
                    'note': 'Pre-sigmoid classifier output; forward() applies a strictly increasing sigmoid/op_norm, so AUROC is unchanged.'}


def main(args):
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    started = time.perf_counter()
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, xrv_report = _load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    reference, reference_meta = reference_scores(args.weight_dir, features['R1'][1])
    patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    result = run_gate(features, labels['train']['y'], labels['selection']['y'], patients, reference=reference)
    from importlib.metadata import version
    record = {
        'schema': 'capstone.effusion.gate.v1', 'protocol': 'effusion-baseline-protocol-v1 (frozen 22 October 2026)',
        'scope': 'Train fit, selection-half selection/scoring only. No holdout, official test, federation, DP or attacks.',
        'inputs': {**provenance, 'xrvCacheReportSha256': args.xrv_report_sha256,
                   'xrvCacheVersion': xrv_report['cache_version']},
        'counts': {split: {'images': len(labels[split]['y']), 'positives': int(labels[split]['y'].sum())} for split in ('train', 'selection')},
        'grid': [{'C': c, 'classWeight': w} for c, w in GRID], 'maxIter': MAX_ITER, 'solver': 'lbfgs',
        'standardisation': 'train-only mean/std per representation', 'selectionRule': 'highest selection AUROC among converged fits; ties smaller C, then unweighted',
        'operatingPointRule': f'highest Platt-probability threshold with selection sensitivity >= {TARGET_SENSITIVITY}',
        'referenceControl': reference_meta,
        'numericalNote': ('numpy 2.1.2 with Apple Accelerate raises spurious divide/overflow/invalid RuntimeWarnings on '
                          'ordinary matmul (reproduced on random normal data; result equals einsum to ~1e-15). Those matmul '
                          'warnings are suppressed; instead every fit is checked finite and selected margins are recomputed '
                          'without BLAS (blasCrossCheckMaxAbsDiff).'),
        'packages': {name: version(name) for name in ('numpy', 'scikit-learn', 'scipy', 'torch', 'torchxrayvision')},
        'sourceSha256': {name: _sha256_file(Path(__file__).resolve().parent / name)
                         for name in ('fit_effusion_gate.py', 'effusion_labels.py', 'lock_effusion_holdout.py', 'build_xrv_cache.py')},
        **{k: v for k, v in result.items() if not k.startswith('_')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gate-', dir=output.parent))
    try:
        record['elapsedSeconds'] = round(time.perf_counter() - started, 1)
        (staging / 'gate.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        np.savez(staging / 'selected-heads.npz', **{f'{name}_{key}': np.asarray(value if value is not None else 'none')
                                                    for name, head in result['_heads'].items() for key, value in head.items()})
        np.savez(staging / 'selection-scores.npz', y=labels['selection']['y'], **result['_scores'])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--weight-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument('--xrv-report-sha256', required=True)
    args = parser.parse_args()
    try:
        record = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'gate_passed' if record['gate']['passed'] else 'gate_failed', 'gate': record['gate'],
                      'elapsedSeconds': record['elapsedSeconds']}))
