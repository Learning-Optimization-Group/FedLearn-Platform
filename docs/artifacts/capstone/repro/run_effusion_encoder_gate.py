"""Effusion protocol v1 amendment A22 gate: centralised R0 vs R1 (CheXpert DenseNet), R2 (DINOv2 ViT-B/14)
and R3 (MIMIC DenseNet) on the selection half.

Same L2 logistic grid, selection rule and standardisation as the original gate, and a patient-cluster
bootstrap of every representation's AUROC and its difference from R0. A representation passes if its
AUROC - R0 interval is above 0 and its AUROC is at least 0.80. Holdout untouched.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import effusion_labels  # noqa: E402
import fit_effusion_gate as gate  # noqa: E402

Refusal = effusion_labels.Refusal
CANDIDATES = ('R1', 'R2', 'R3')
WIDTHS = {'R0': 512, 'R1': 1024, 'R2': 768, 'R3': 1024}
BOOTSTRAP_SEED = 20261042


def load_cache(directory, report_sha256, labels, width):
    """Train and selection rows of a build_cache output, hash-checked and aligned to the label plan."""
    raw = (Path(directory) / 'report.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_sha256:
        raise Refusal(f'report_hash_mismatch:{directory}')
    report = json.loads(raw)
    out = {}
    for split in ('train', 'selection'):
        for name in (f'{split}-x.npy', f'{split}.csv'):
            if gate._sha256_file(Path(directory) / name) != report['payload_hashes'][name]:
                raise Refusal(f'payload_hash_mismatch:{name}')
        with (Path(directory) / f'{split}.csv').open(newline='') as stream:
            if list(csv.DictReader(stream)) != labels[split]['rows']:
                raise Refusal(f'manifest_mismatch:{split}')
        out[split] = np.load(Path(directory) / f'{split}-x.npy')
        if out[split].shape[1] != width:
            raise Refusal(f'feature_width_mismatch:{directory}')
    return (out['train'], out['selection']), report


def fit_representation(name, train, selection, y_train, y_selection, *, max_iter=gate.MAX_ITER):
    """The original gate's grid for one representation; returns (fits, selected margin)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mean, std = gate.standardiser(train)
    z_train = ((train - mean) / std).astype(np.float64)
    z_selection = ((selection - mean) / std).astype(np.float64)
    block = []
    for c, weight in gate.GRID:
        model = LogisticRegression(C=c, class_weight=weight, solver='lbfgs', max_iter=max_iter)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(z_train, y_train)
        margin = model.decision_function(z_selection)
        finite = bool(np.isfinite(model.coef_).all() and np.isfinite(margin).all())
        block.append({'representation': name, 'C': c, 'classWeight': weight,
                      'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'finite': finite,
                      'iterations': int(model.n_iter_[0]), 'fitSeconds': round(time.perf_counter() - started, 2),
                      'trainAuroc': float(roc_auc_score(y_train, model.decision_function(z_train))),
                      'selectionAuroc': float(roc_auc_score(y_selection, np.nan_to_num(margin))),
                      '_margin': np.nan_to_num(margin)})
    chosen = gate.select(block)
    for i, fit in enumerate(block):
        fit['selected'] = i == chosen
    return [{k: v for k, v in f.items() if not k.startswith('_')} for f in block], block[chosen]['_margin']


def bootstrap(y, patients, margins, *, reps=gate.BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    from sklearn.metrics import roc_auc_score
    draws = {name: [] for name in margins} | {f'delta_{n}_R0': [] for n in margins if n != 'R0'}
    skipped = 0
    for index in gate.cluster_bootstrap(patients, reps=reps, seed=seed):
        yy = y[index]
        if yy.min() == yy.max():
            skipped += 1
            continue
        values = {name: roc_auc_score(yy, m[index]) for name, m in margins.items()}
        for name, value in values.items():
            draws[name].append(value)
            if name != 'R0':
                draws[f'delta_{name}_R0'].append(value - values['R0'])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}, skipped


def main(args):
    from sklearn.metrics import roc_auc_score
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, r1_report = gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    reports = {}
    features['R2'], reports['R2'] = load_cache(args.r2_cache, args.r2_report_sha256, labels, WIDTHS['R2'])
    features['R3'], reports['R3'] = load_cache(args.r3_cache, args.r3_report_sha256, labels, WIDTHS['R3'])
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    patients = [r['patient_id'] for r in labels['selection']['rows']]
    fits, margins = [], {}
    for name in ('R0', *CANDIDATES):
        block, margins[name] = fit_representation(name, *features[name], y_train, y_selection)
        fits += block
        print(json.dumps({'representation': name, 'selectedAuroc': max(f['selectionAuroc'] for f in block if f['selected'])}),
              flush=True)
    intervals, skipped = bootstrap(y_selection, patients, margins)
    auroc = {name: float(roc_auc_score(y_selection, m)) for name, m in margins.items()}
    results = {name: {'auroc': auroc[name], 'interval95': intervals[name], 'width': WIDTHS[name],
                      **({'deltaVsR0': auroc[name] - auroc['R0'], 'deltaInterval95': intervals[f'delta_{name}_R0'],
                          'passed': intervals[f'delta_{name}_R0'][0] > 0 and auroc[name] >= gate.AUROC_FLOOR}
                         if name != 'R0' else {})}
               for name in margins}
    summary = {
        'schema': 'capstone.effusion.encoder_gate.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A22',
        'scope': 'Centralised gate on the selection half; holdout untouched.', 'results': results, 'fits': fits,
        'bootstrap': {'reps': gate.BOOTSTRAP_REPS, 'seed': BOOTSTRAP_SEED, 'skippedSingleClassDraws': skipped},
        'aurocFloor': gate.AUROC_FLOOR, 'grid': [{'C': c, 'classWeight': w} for c, w in gate.GRID],
        'inputs': {'labelInputs': provenance, 'r1ReportSha256': args.xrv_report_sha256,
                   'r2ReportSha256': args.r2_report_sha256, 'r3ReportSha256': args.r3_report_sha256},
        'encoders': {'R1': r1_report.get('extractor'), 'R2': reports['R2']['extractor'], 'R3': reports['R3']['extractor']},
        'sourceSha256': {n: gate._sha256_file(HERE / n) for n in ('run_effusion_encoder_gate.py', 'fit_effusion_gate.py')},
    }
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    np.savez(output / 'selection-margins.npz', **margins)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--r2-cache', '--r3-cache', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--r2-report-sha256', '--r3-report-sha256'):
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'results': {k: {kk: v[kk] for kk in ('auroc', 'deltaVsR0', 'passed') if kk in v}
                                                        for k, v in summary['results'].items()}}))
