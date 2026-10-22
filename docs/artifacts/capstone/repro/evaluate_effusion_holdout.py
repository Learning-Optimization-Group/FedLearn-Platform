"""Effusion protocol v1 section 8 / amendment A10: the one-time holdout evaluation of the frozen 22 R0 models.

Phase 1 (selection half only): recompute each model's selection margins from its saved head, require
them to equal the stored margins, fit Platt recalibration and the sensitivity-0.90 operating point, and
write the frozen calibration (hashed) BEFORE any holdout label is loaded. Phase 2: score the holdout once.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_effusion_federated as federated  # noqa: E402  (also pins this worktree's fedlearn)
import torch  # noqa: E402

from fit_effusion_gate import cluster_bootstrap, confusion, operating_point  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
EPSILONS = (8.0, 4.0, 1.0)
TARGET_SENSITIVITY = 0.90
BOOTSTRAP_SEED = 20261029
GROUPS = {'fed_clean': [f'fed_clean_{s}' for s in SEEDS],
          **{f'dp_eps{e:g}': [f'dp_eps{e:g}_{s}' for s in SEEDS] for e in EPSILONS}}


def federated_margin(z32, head):
    """float64 logit margin of a saved two-output head, computed exactly as the engine's forward."""
    from fedlearn.simulation.workloads import LinearHead
    model = LinearHead(z32.shape[1], 2)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in head.items()})
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(z32))).to(torch.float64)
    return (logits[:, 1] - logits[:, 0]).numpy()


def _logistic_fit(x, y):
    from sklearn.linear_model import LogisticRegression
    fit = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(x[:, None], y)
    return float(fit.intercept_[0]), float(fit.coef_[0][0])


def freeze(y, margin):
    """Platt recalibration and the highest threshold with selection sensitivity >= 0.90."""
    intercept, slope = _logistic_fit(margin, y)
    platt = {'slope': slope, 'intercept': intercept}
    point = operating_point(y, probabilities(margin, platt), TARGET_SENSITIVITY)
    return {'platt': platt, 'threshold': point['threshold'], 'selectionOperatingPoint': point}


def probabilities(margin, platt):
    return 1.0 / (1.0 + np.exp(-(platt['slope'] * margin + platt['intercept'])))


def ece(y, p, bins=10):
    """Expected calibration error over equal-mass bins."""
    order = np.argsort(p, kind='stable')
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in np.array_split(order, bins) if len(b)) / len(y))


def calibration_fit(y, p):
    """(intercept, slope) of logit(P(y=1)) regressed on logit(p)."""
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return _logistic_fit(np.log(q / (1 - q)), y)


def holdout_metrics(y, margin, frozen, patients, *, with_calibration=True):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if frozen.get('constant'):
        p = np.full(len(y), frozen['probability'])
        out = {'auroc': 0.5, 'averagePrecision': float(y.mean()), 'threshold': frozen['threshold'],
               **confusion(y, np.ones(len(y), bool)), 'brier': float(np.mean((p - y) ** 2)), 'ece': ece(y, p)}
        return out
    p = probabilities(margin, frozen['platt'])
    by_patient = {}
    for value, score, patient in zip(y, margin, patients):
        label, best = by_patient.get(patient, (0, -np.inf))
        by_patient[patient] = (max(label, int(value)), max(best, float(score)))
    patient_y, patient_score = (np.array(v) for v in zip(*by_patient.values()))
    out = {'auroc': float(roc_auc_score(y, margin)), 'averagePrecision': float(average_precision_score(y, margin)),
           'prevalence': float(y.mean()), 'threshold': frozen['threshold'],
           **confusion(y, p >= frozen['threshold']), 'brier': float(np.mean((p - y) ** 2)), 'ece': ece(y, p),
           'patientLevelAuroc': float(roc_auc_score(patient_y, patient_score))}
    if with_calibration:
        out['calibrationIntercept'], out['calibrationSlope'] = calibration_fit(y, p)
    return out


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import effusion_labels
    import fit_effusion_gate
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists: the holdout is evaluated once; any re-run is post-hoc')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    pinned = {'gate': (Path(args.gate_dir) / 'gate.json', args.gate_sha256),
              'recipe': (Path(args.recipe_dir) / 'summary.json', args.recipe_sha256),
              'dpRecipe': (Path(args.dp_recipe_dir) / 'summary.json', args.dp_recipe_sha256),
              'historicalReport': (Path(args.historical_cache) / 'report.json', fit_effusion_gate.HISTORICAL_REPORT_SHA256)}
    for name, (path, expected) in pinned.items():
        if _sha256(path) != expected:
            raise Refusal(f'{name}_hash_mismatch')
    authority = federated.check_gate(json.loads((Path(args.gate_dir) / 'gate.json').read_bytes()), args.amendment)
    report = json.loads((Path(args.historical_cache) / 'report.json').read_bytes())
    if _sha256(Path(args.historical_cache) / 'validation-x.npy') != report['payload_hashes']['validation-x.npy']:
        raise Refusal('validation_features_hash_mismatch')
    order = {}
    with (Path(args.historical_cache) / 'validation.csv').open(newline='') as stream:
        import csv
        reader = csv.reader(stream)
        if next(reader)[0] != 'image_index':
            raise Refusal('validation_manifest_schema')
        order = {row[0]: i for i, row in enumerate(reader)}
    validation_x = np.load(Path(args.historical_cache) / 'validation-x.npy', mmap_mode='r')
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = heads['R0_mean'], heads['R0_std']

    def z_rows(rows):
        index = np.array([order[r['image_index']] for r in rows])
        x = np.asarray(validation_x[np.sort(index)])[np.argsort(np.argsort(index))]
        return ((x - mean) / std).astype(np.float32), ((x - mean) / std).astype(np.float64)

    # ---- Phase 1: selection half only --------------------------------------------------------
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    y_sel = labels['selection']['y']
    z32_sel, z64_sel = z_rows(labels['selection']['rows'])
    stored_recipe = np.load(Path(args.recipe_dir) / 'selection-margins.npz')
    stored_dp = np.load(Path(args.dp_recipe_dir) / 'selection-margins.npz')
    stored_gate = np.load(Path(args.gate_dir) / 'selection-scores.npz')
    models = {'convex': ('convex', None)}
    for s in SEEDS:
        models[f'fed_clean_{s}'] = ('federated', Path(args.recipe_dir) / f'fed_{s}.json')
        for e in EPSILONS:
            models[f'dp_eps{e:g}_{s}'] = ('federated', Path(args.dp_recipe_dir) / f'dp_eps{e:g}_{s}.json')
    heads_by_model, frozen, integrity = {}, {'C0': {'constant': True, 'threshold': 0.0, 'probability': float(y_sel.mean())}}, {}
    for name, (kind, record) in models.items():
        if kind == 'convex':
            margin = z64_sel @ heads['R0_coef'].astype(np.float64) + float(heads['R0_intercept'])
            integrity[name] = float(np.abs(margin - stored_gate['R0']).max())
            if integrity[name] > 1e-9:
                raise Refusal('convex_selection_margin_mismatch')
        else:
            head = json.loads(record.read_text())['meta']['finalHead']
            heads_by_model[name] = head
            margin = federated_margin(z32_sel, head)
            stored = (stored_recipe[name.replace('fed_clean_', 'fed_')] if name.startswith('fed_clean_') else stored_dp[name])
            integrity[name] = float(np.abs(margin - stored).max())
            if not np.array_equal(margin, stored):
                raise Refusal(f'selection_margin_mismatch:{name}')
        frozen[name] = freeze(y_sel, margin)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.holdout-', dir=output.parent))
    try:
        calibration_bytes = (json.dumps({'protocol': 'A10', 'frozen': frozen, 'selectionMarginIntegrity': integrity},
                                        indent=2, sort_keys=True) + '\n').encode()
        (staging / 'frozen-calibration.json').write_bytes(calibration_bytes)
        frozen_sha = hashlib.sha256(calibration_bytes).hexdigest()
        print(json.dumps({'phase': 'frozen', 'models': len(frozen), 'sha256': frozen_sha}), flush=True)

        # ---- Phase 2: the holdout, once --------------------------------------------------------
        started = time.perf_counter()
        hold = effusion_labels.holdout_labels(args.historical_cache, args.lock_dir)
        y, patients = hold['y'], np.array([r['patient_id'] for r in hold['rows']])
        if {r['patient_id'] for r in hold['rows']} & {r['patient_id'] for r in labels['selection']['rows']}:
            raise Refusal('holdout_selection_patient_overlap')
        z32, z64 = z_rows(hold['rows'])
        margins = {'C0': np.zeros(len(y)),
                   'convex': z64 @ heads['R0_coef'].astype(np.float64) + float(heads['R0_intercept'])}
        margins.update({name: federated_margin(z32, head) for name, head in heads_by_model.items()})
        point = {name: holdout_metrics(y, margins[name], frozen[name], patients) for name in frozen}
        keys = ('auroc', 'averagePrecision', 'sensitivity', 'specificity', 'ppv', 'npv', 'brier', 'ece',
                'calibrationIntercept', 'calibrationSlope', 'patientLevelAuroc')
        draws = {name: {k: [] for k in keys} for name in frozen}
        group_draws = {g: {k: [] for k in keys} for g in GROUPS}
        contrast_names = ['fed_clean-convex'] + [f'dp_eps{e:g}-fed_clean' for e in EPSILONS]
        contrast_draws = {c: [] for c in contrast_names}
        skipped = 0
        for index in cluster_bootstrap(patients, reps=args.reps, seed=BOOTSTRAP_SEED):
            if y[index].min() == y[index].max():
                skipped += 1
                continue
            replicate = {name: holdout_metrics(y[index], margins[name][index], frozen[name], patients[index])
                         for name in frozen if name != 'C0'}
            for name, metrics in replicate.items():
                for k in keys:
                    if metrics.get(k) is not None:
                        draws[name][k].append(metrics[k])
            for group, members in GROUPS.items():
                for k in keys:
                    values = [replicate[m][k] for m in members]
                    if all(v is not None for v in values):
                        group_draws[group][k].append(float(np.mean(values)))
            contrast_draws['fed_clean-convex'].append(float(np.mean([replicate[m]['auroc'] for m in GROUPS['fed_clean']]))
                                                      - replicate['convex']['auroc'])
            for e in EPSILONS:
                contrast_draws[f'dp_eps{e:g}-fed_clean'].append(float(np.mean(
                    [replicate[f'dp_eps{e:g}_{s}']['auroc'] - replicate[f'fed_clean_{s}']['auroc'] for s in SEEDS])))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
        group_point = {g: {k: float(np.mean([point[m][k] for m in members])) for k in keys if all(point[m].get(k) is not None for m in members)}
                       for g, members in GROUPS.items()}
        evaluation = {
            'schema': 'capstone.effusion.holdout_evaluation.v1', 'protocol': 'effusion-baseline-protocol-v1 section 8, amendment A10',
            'gateAuthority': authority, 'frozenCalibrationSha256': frozen_sha,
            'holdout': {'images': int(len(y)), 'positives': int(y.sum()), 'patients': int(len(set(patients))),
                        'prevalence': float(y.mean()), 'labelProvenance': hold['provenance']},
            'models': {name: {'point': point[name], 'interval95': {k: ci(draws[name][k]) for k in keys}} for name in frozen},
            'groups': {g: {'members': GROUPS[g], 'seedMean': group_point[g],
                           'interval95': {k: ci(group_draws[g][k]) for k in keys}} for g in GROUPS},
            'contrasts': {c: {'point': (float(np.mean([point[m]['auroc'] for m in GROUPS['fed_clean']])) - point['convex']['auroc'])
                              if c == 'fed_clean-convex' else
                              float(np.mean([point[f'{c.split("-")[0]}_{s}']['auroc'] - point[f'fed_clean_{s}']['auroc'] for s in SEEDS])),
                              'interval95': ci(contrast_draws[c])} for c in contrast_names},
            'bootstrap': {'reps': args.reps, 'seed': BOOTSTRAP_SEED, 'skippedSingleClassDraws': skipped,
                          'method': 'patient-cluster percentile; seed means taken inside each replicate'},
            'inputs': {'labelInputs': provenance, **{f'{k}Sha256': v[1] for k, v in pinned.items()}},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('evaluate_effusion_holdout.py', 'effusion_labels.py', 'run_effusion_federated.py', 'fit_effusion_gate.py')},
            'phase2Seconds': round(time.perf_counter() - started, 1),
        }
        (staging / 'holdout-evaluation.json').write_text(json.dumps(evaluation, indent=2, sort_keys=True) + '\n')
        np.savez(staging / 'holdout-margins.npz', y=y, **margins)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--gate-dir', '--recipe-dir', '--dp-recipe-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--gate-sha256', '--recipe-sha256', '--dp-recipe-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    parser.add_argument('--reps', type=int, default=2000)
    args = parser.parse_args()
    try:
        result = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'evaluated', 'holdout': result['holdout']['images']}))
