"""Effusion protocol v1 amendment A19: non-IID (patient-whole Dirichlet label skew) clients on R0.

alpha in {1.0, 0.1}. No DP: {mean, trimmed k=16, median} clean, and {mean, median} under Fang's attack
at 16/80. DP eps 4: mean clean. 5 seeds, 60 runs, each paired with its A14 IID counterpart (same seeds,
only the partition differs). Selection half only; holdout untouched.
"""
import argparse
from collections import OrderedDict
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
from run_effusion_recipe import linear_contrasts  # noqa: E402
from run_effusion_strong_attacks import ATTACKS, DP, RECIPES, trim_ratio  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
CLIENTS = federated.CLIENTS
ALPHAS = (1.0, 0.1)
FANG16 = [{**ATTACKS['fang_trim'], 'fraction': 16 / CLIENTS}]
# (tag, rule label, engine rule, k, condition, behaviors)
GROUPS = [('nodp', label, rule, k, 'clean', []) for label, rule, k in
          (('mean', 'mean', None), ('trim16', 'trimmed_mean', 16), ('median', 'median', None))] + \
         [('nodp', label, rule, None, 'fang_trim_m16', FANG16) for label, rule in (('mean', 'mean'), ('median', 'median'))] + \
         [('dp4', 'mean', 'mean', None, 'clean', [])]
CONTRAST_SEED = 20261039


def alpha_label(alpha):
    return f'a{alpha:g}'.replace('.', 'p')


def cells():
    return [{'alpha': a, 'tag': tag, 'ruleLabel': label, 'rule': rule, 'k': k, 'condition': condition,
             'behaviors': behaviors, 'seed': s, 'name': f'{alpha_label(a)}_{tag}_{label}_{condition}_{s}',
             'iid': f'{tag}_{label}_{condition}_{s}'}
            for a in ALPHAS for tag, label, rule, k, condition, behaviors in GROUPS for s in SEEDS]


class DirichletEffusionWorkload(federated.EffusionWorkload):
    """The Effusion workload with patient-whole Dirichlet label-skewed clients."""

    def __init__(self, train_x, train_y, train_patients, selection_x, selection_y, mean, std, *,
                 clients, partition_seed, info, alpha):
        import torch
        from fedlearn.simulation.partition import patient_dirichlet_partition
        super().__init__(train_x, train_y, train_patients, selection_x, selection_y, mean, std,
                         clients=clients, partition_seed=partition_seed, info=info)
        self.parts = patient_dirichlet_partition(list(train_patients), self.train_y.numpy(), int(clients),
                                                 float(alpha), int(partition_seed))
        positives = [int(self.train_y[torch.as_tensor(p)].sum()) for p in self.parts]
        self.info = {**self.info, 'partition_algorithm': f'patient-dirichlet-v1(alpha={alpha:g})',
                     'partitionOverride': 'config federation.partition reads patient_iid; the injected workload '
                                          'replaces it with this partition',
                     'clients_without_positive_images': sum(p == 0 for p in positives),
                     'client_images': [len(p) for p in self.parts], 'client_positive_images': positives}


def heterogeneity(workload):
    images = np.array(workload.info['client_images'], dtype=float)
    share = np.array(workload.info['client_positive_images'], dtype=float) / images
    return {'clientsWithoutPositiveImages': int(workload.info['clients_without_positive_images']),
            'imageCountCv': float(images.std() / images.mean()),
            'positiveShare': {'min': float(share.min()), 'median': float(np.median(share)), 'max': float(share.max())}}


def run_cell(cell, train_x, train_y, patients, selection_x, selection_y, mean, std, *, features_path, report_sha256):
    import torch
    private = cell['tag'] == 'dp4'
    rule_params = None if cell['k'] is None else {'trim_ratio': trim_ratio(cell['k'])}
    cfg = federated.config(cell['name'], cell['seed'], clients=CLIENTS, features_path=features_path,
                           report_sha256=report_sha256, privacy=DP if private else None, recipe=RECIPES[private],
                           rule=cell['rule'], rule_params=rule_params, behaviors=cell['behaviors'])
    workload = DirichletEffusionWorkload(train_x, train_y, patients, selection_x, selection_y, mean, std,
                                         clients=CLIENTS, partition_seed=cfg['seeds']['partition'],
                                         info={'representation': 'R0', 'amendment': 'A19'}, alpha=cell['alpha'])
    final = {}
    record = federated.run(cfg, workload=workload, on_final_parameters=final.update)
    model = workload.build_model()
    federated.apply_trainable_subset(model, OrderedDict(final))
    model.eval()
    with torch.no_grad():
        logits = model(workload.test_x).to(torch.float64)
    return record, (logits[:, 1] - logits[:, 0]).numpy(), heterogeneity(workload)


def main(args):
    import effusion_labels
    import fit_effusion_gate
    from sklearn.metrics import roc_auc_score
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    authority = federated.check_gate(json.loads(raw_gate), args.amendment)
    raw_a14 = (Path(args.a14_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a14).hexdigest() != args.a14_sha256:
        raise Refusal('a14_hash_mismatch')
    a14 = np.load(Path(args.a14_dir) / 'selection-margins.npz')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R0']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R0_mean']) and np.array_equal(std, heads['R0_std'])):
        raise Refusal('standardiser_mismatch')
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.noniid-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            record, margin, het = run_cell(cell, train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                                           features_path=args.historical_cache,
                                           report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256)
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            notes = sorted({str(line['attack_note']) for line in record['rounds'] if line.get('attack_note')})
            runs.append({k: v for k, v in cell.items() if k != 'behaviors'} | {
                'finite': finite, 'refusals': len(record['meta']['refusals']),
                'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                'attackers': sum(1 for kinds in record['meta']['behavior_assignment'].values() if 'fang_trim' in kinds),
                'heterogeneity': het, 'attackNotes': notes,
                'honestCosineLastRound': (record['rounds'][-1].get('updates') or {}).get('honest_cosine'),
                'clipRateLastRound': record['rounds'][-1]['privacy']['clip_rate']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        scores = {**margins, **{f'iid_{k}': a14[k] for k in a14.files}}
        contrasts = {}
        for a in ALPHAS:
            al = alpha_label(a)
            for tag, label, _, _, condition, _ in GROUPS:
                group = [c for c in cells() if c['alpha'] == a and c['tag'] == tag and c['ruleLabel'] == label
                         and c['condition'] == condition]
                if not all(c['name'] in usable for c in group):
                    continue
                key = f'{al}|{tag}|{label}|{condition}'
                contrasts[f'auroc|{key}'] = [{c['name']: 1} for c in group]
                contrasts[f'nonIidMinusIid|{key}'] = [{c['name']: 1, f"iid_{c['iid']}": -1} for c in group]
                if label != 'mean':
                    contrasts[f'ruleVsMean|{key}'] = [{c['name']: 1, f"{al}_{tag}_mean_{condition}_{c['seed']}": -1}
                                                      for c in group]
                if condition != 'clean':
                    contrasts[f'drop|{key}'] = [{f"{al}_{tag}_{label}_clean_{c['seed']}": 1, c['name']: -1}
                                                for c in group]
        summary = {
            'schema': 'capstone.effusion.noniid.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A19',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'alphas': list(ALPHAS),
            'recipes': {str(k): v for k, v in RECIPES.items()}, 'dp': DP, 'runs': runs,
            'heterogeneityBySeed': {alpha_label(a): {str(r['seed']): r['heterogeneity'] for r in runs if r['alpha'] == a}
                                    for a in ALPHAS},
            'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a14SummarySha256': args.a14_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_noniid.py', 'run_effusion_strong_attacks.py', 'run_effusion_federated.py',
                              'run_effusion_recipe.py')},
            'frameworkSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / 'simulation' / name)
                                      for name in ('engine.py', 'behaviors.py', 'config.py', 'partition.py')},
            'elapsedSeconds': round(time.perf_counter() - started, 1),
        }
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        np.savez(staging / 'selection-margins.npz', **margins)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a14-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a14-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
