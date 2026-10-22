"""Effusion protocol v1 amendment A16: one-run canary audit of central DP on R0.

8 of 80 clients are canaries uploading a fixed random direction of norm C every round. Each final
head is scored against the inserted directions and 1,000 never-inserted ones; the separation gives an
empirical mu and epsilon (Andrew et al., ICLR'24). Clean (positive control) and claimed eps 8/4/1 with
exact-GDP z; 10 seeds. Selection half only; holdout untouched.
"""
import argparse
from collections import OrderedDict
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_effusion_federated as federated  # noqa: E402  (also pins this worktree's fedlearn)
from run_effusion_gdp_recipe import DELTA, RECIPE, calibration, check_calibration  # noqa: E402

Refusal = federated.Refusal
SEEDS = tuple(range(20261101, 20261111))
PRIVACY = (None, 8.0, 4.0, 1.0)
CANARIES = 8
UNSEEN = 1000
UNSEEN_ID_BASE = 10_000  # direction ids for never-inserted canaries; real client ids are < 80
AUDIT_SEED = 20261035


def privacy_label(epsilon):
    return 'clean' if epsilon is None else f'eps{epsilon:g}'


def cells():
    return [{'epsilon': eps, 'seed': s, 'name': f'canary_{privacy_label(eps)}_{s}'} for eps in PRIVACY for s in SEEDS]


def behaviors():
    return [{'type': 'canary', 'fraction': CANARIES / federated.CLIENTS, 'norm': RECIPE['clip_norm']}]


def scores(final_head, attack_seed, ids):
    """<final head, u_id> for each direction id, with u_id from the engine's own canary_direction."""
    import torch
    from fedlearn.simulation.behaviors import canary_direction
    final = OrderedDict((k, torch.tensor(v, dtype=torch.float32)) for k, v in final_head.items())
    out = []
    for i in ids:
        u = canary_direction(attack_seed, i, final)
        out.append(float(sum(float((final[k].double() * u[k].double()).sum()) for k in final)))
    return out


def theory_mu(z, rounds=federated.ROUNDS):
    """mu of "canary inserted vs not": a C/n shift per round against noise std z * 2C/n."""
    return math.inf if z == 0 else math.sqrt(rounds) / (2.0 * z)


def main(args):
    import effusion_labels
    import fit_effusion_gate
    from sklearn.metrics import roc_auc_score
    from fedlearn.privacy.audit import canary_audit
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    authority = federated.check_gate(json.loads(raw_gate), args.amendment)
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R0']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R0_mean']) and np.array_equal(std, heads['R0_std'])):
        raise Refusal('standardiser_mismatch')
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    calibrations = {eps: calibration(eps) for eps in PRIVACY if eps is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.canary-audit-', dir=output.parent))
    started = time.perf_counter()
    runs, pooled = [], {privacy_label(eps): {'inserted': [], 'unseen': []} for eps in PRIVACY}
    try:
        for cell in cells():
            cal = calibrations.get(cell['epsilon'])
            privacy = None if cal is None else {'mode': 'central', 'epsilon': cal['rdpEquivalentEpsilon'], 'delta': DELTA}
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                info={'representation': 'R0', 'amendment': 'A16'}, privacy=privacy, recipe=RECIPE,
                behaviors=behaviors())
            z = float(record['meta']['noise_multiplier'])
            exact = None if cal is None else check_calibration(cal, z)
            canaries = sorted(int(c) for c, kinds in record['meta']['behavior_assignment'].items() if 'canary' in kinds)
            if len(canaries) != CANARIES:
                raise Refusal(f"canary_count:{cell['name']}:{len(canaries)}")
            attack_seed = int(record['meta']['seeds']['attack'])
            inserted = scores(record['meta']['finalHead'], attack_seed, canaries)
            unseen = scores(record['meta']['finalHead'], attack_seed, range(UNSEEN_ID_BASE, UNSEEN_ID_BASE + UNSEEN))
            label = privacy_label(cell['epsilon'])
            pooled[label]['inserted'] += inserted
            pooled[label]['unseen'] += unseen
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'noiseMultiplier': z, 'exactEpsilon': exact, 'canaries': canaries,
                         'insertedScores': inserted, 'unseenMean': float(np.mean(unseen)),
                         'unseenSd': float(np.std(unseen, ddof=1))})
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        audit = {}
        for eps in PRIVACY:
            label = privacy_label(eps)
            result = canary_audit(pooled[label]['inserted'], pooled[label]['unseen'], DELTA, seed=AUDIT_SEED)
            z = None if eps is None else calibrations[eps]['gdpNoiseMultiplier']
            result['theoryMuInsertedVsNot'] = None if z is None else theory_mu(z)
            result['claimedMuReplaceOne'] = None if z is None else math.sqrt(federated.ROUNDS) / z
            result['claimedEpsilon'] = eps
            result['leakBeyondClaim'] = (None if z is None else result['mu_interval95'][0] > theory_mu(z))
            result['meanAuroc'] = float(np.mean([r['auroc'] for r in runs if r['epsilon'] == eps]))
            audit[label] = result
        summary = {
            'schema': 'capstone.effusion.canary_audit.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A16',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE,
            'canaries': CANARIES, 'unseenPerRun': UNSEEN, 'seeds': list(SEEDS),
            'calibration': {f'{eps:g}': c for eps, c in calibrations.items()},
            'audit': audit, 'runs': runs,
            'unusableCells': sorted(r['name'] for r in runs if not r['finite'] or r['refusals']),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_canary_audit.py', 'run_effusion_gdp_recipe.py', 'run_effusion_federated.py')},
            'frameworkSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / name) for name in
                                      ('simulation/engine.py', 'simulation/behaviors.py', 'simulation/config.py',
                                       'privacy/audit.py', 'privacy/gdp_accountant.py')},
            'elapsedSeconds': round(time.perf_counter() - started, 1),
        }
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'audit': {k: {'mu': v['mu'], 'theory': v['theoryMuInsertedVsNot'],
                                                          'leak': v['leakBeyondClaim']} for k, v in summary['audit'].items()},
                      'elapsedSeconds': summary['elapsedSeconds']}))
