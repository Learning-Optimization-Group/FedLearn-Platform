"""Effusion protocol v1 section 6: matched clean federated baseline and central-SGD control.

Runs the production simulation engine (ComposedStrategy + FLCoordinator via
SimulatedFederation) on the gate-selected R1 features with the frozen train
standardiser. Federated: 80 whole-patient IID clients. Control: one client holding
all of train, identical optimiser, epochs, batch and clipping. Scored on the
selection half only; no DP, attacks, holdout or official test.
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
sys.path.insert(1, str(HERE.parents[2] / 'framework' / 'src'))
import torch  # noqa: E402

from fedlearn.server.subset_federation import apply_trainable_subset  # noqa: E402
from fedlearn.simulation.config import default_seeds  # noqa: E402
from fedlearn.simulation.engine import run  # noqa: E402
from fedlearn.simulation.metrics import classification_metrics  # noqa: E402
from fedlearn.simulation.partition import patient_iid_partition  # noqa: E402
from fedlearn.simulation.workloads import FeatureHeadWorkload  # noqa: E402

SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
CLIENTS = 80
ROUNDS = 20
COMPARISON_SEED = 20261023


class Refusal(ValueError):
    pass


def auroc(y, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


class EffusionWorkload(FeatureHeadWorkload):
    """Linear two-output head over standardised R1 features; whole-patient IID clients."""

    def __init__(self, train_x, train_y, train_patients, selection_x, selection_y, mean, std, *,
                 clients, partition_seed, info):
        scale = lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))  # noqa: E731
        self.train_x, self.train_y = scale(train_x), torch.from_numpy(np.asarray(train_y, dtype=np.int64))
        self.test_x, self.test_y = scale(selection_x), torch.from_numpy(np.asarray(selection_y, dtype=np.int64))
        self.trigger_train_x = self.trigger_test_x = None
        self.has_trigger = False
        self.feat_dim, self.n_classes = int(train_x.shape[1]), 2
        self.head = {'type': 'linear', 'hidden': 128}
        self.parts = patient_iid_partition(list(train_patients), int(clients), int(partition_seed))
        positives = [int(self.train_y[torch.as_tensor(p)].sum()) for p in self.parts]
        self.info = {'dataset': 'nih', **info, 'auc_score': 'margin_v2', 'evaluation': 'selection half (protocol v1)',
                     'partition_algorithm': 'patient-iid-v1:sorted-patients-pcg64-array-split',
                     'clients_without_positive_images': sum(p == 0 for p in positives),
                     'client_images': [len(p) for p in self.parts], 'client_positive_images': positives}

    def _metrics(self, logits, labels):
        return classification_metrics(logits, labels, auc_score='margin_v2')


def config(name, seed, *, clients, features_path, report_sha256, rounds=ROUNDS):
    return {'name': name, 'seeds': default_seeds(seed),
            'workload': {'dataset': 'nih', 'features_path': str(features_path), 'features_report_sha256': report_sha256,
                         'evaluation_split': 'validation', 'balance': False, 'head': {'type': 'linear'}},
            'federation': {'num_clients': clients, 'rounds': rounds, 'partition': {'type': 'patient_iid'}},
            'client': {'algorithm': 'fedavg', 'local_epochs': 7, 'lr': 0.003, 'batch_size': 256},
            'server': {'clip_norm': 0.05, 'rule': 'mean'}, 'privacy': {'mode': 'none'}}


def run_one(name, seed, train_x, train_y, patients, selection_x, selection_y, mean, std, *, clients,
            features_path, report_sha256, rounds=ROUNDS, info=None):
    """One engine run; returns (record, float64 selection margins of the final head)."""
    cfg = config(name, seed, clients=clients, features_path=features_path, report_sha256=report_sha256, rounds=rounds)
    workload = EffusionWorkload(train_x, train_y, patients, selection_x, selection_y, mean, std, clients=clients,
                                partition_seed=cfg['seeds']['partition'], info=dict(info or {}))
    final = {}
    record = run(cfg, workload=workload, on_final_parameters=final.update)
    model = workload.build_model()
    apply_trainable_subset(model, OrderedDict(final))
    model.eval()
    with torch.no_grad():
        logits = model(workload.test_x).to(torch.float64)
    margin = (logits[:, 1] - logits[:, 0]).numpy()
    record['meta']['finalHead'] = {k: v.tolist() for k, v in final.items()}
    return record, margin


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def compare(y, patients, margins, reference, *, reps=2000, seed=COMPARISON_SEED):
    """Per-seed AUROC and paired patient-bootstrap differences fed-central, fed-convex, central-convex."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from fit_effusion_gate import cluster_bootstrap
    names = sorted(margins)
    pairs = [(f'fed_{s}', f'central_{s}') for s in SEEDS if f'fed_{s}' in margins]
    contrasts = {f'{a}-{b}': (a, b) for a, b in pairs}
    contrasts.update({f'{n}-convex': (n, 'convex') for n in names})
    scores = {**margins, 'convex': reference}
    draws = {key: [] for key in contrasts}
    for index in cluster_bootstrap(patients, reps=reps, seed=seed):
        yy = y[index]
        if yy.min() == yy.max():
            continue
        values = {n: roc_auc_score(yy, s[index]) for n, s in scores.items()}
        for key, (a, b) in contrasts.items():
            draws[key].append(values[a] - values[b])
    point = {n: {'auroc': auroc(y, s), 'averagePrecision': float(average_precision_score(y, s))} for n, s in scores.items()}
    return {'point': point,
            'differences': {key: {'point': point[a]['auroc'] - point[b]['auroc'],
                                  'interval95': [float(np.percentile(draws[key], 2.5)), float(np.percentile(draws[key], 97.5))]}
                            for key, (a, b) in contrasts.items()},
            'seedSummary': {kind: {'mean': float(np.mean([point[f'{kind}_{s}']['auroc'] for s in SEEDS])),
                                   'min': float(np.min([point[f'{kind}_{s}']['auroc'] for s in SEEDS])),
                                   'max': float(np.max([point[f'{kind}_{s}']['auroc'] for s in SEEDS]))}
                            for kind in ('fed', 'central') if all(f'{kind}_{s}' in point for s in SEEDS)},
            'bootstrap': {'reps': reps, 'seed': seed, 'method': 'patient-cluster percentile, paired on shared resamples'}}


def main(args):
    import effusion_labels
    import fit_effusion_gate
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    gate = json.loads(raw_gate)
    if not gate['gate']['passed']:
        raise Refusal('gate_not_passed')
    if gate['inputs']['xrvCacheReportSha256'] != args.xrv_report_sha256:
        raise Refusal('gate_cache_mismatch')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R1']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R1_mean']) and np.array_equal(std, heads['R1_std'])):
        raise Refusal('standardiser_mismatch')
    reference = ((selection_x - mean) / std).astype(np.float64) @ heads['R1_coef'].astype(np.float64) + float(heads['R1_intercept'])
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.federated-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    info = {'representation': 'R1 xrv-chex features2', 'standardiser': 'gate R1 train mean/std',
            'gateSha256': args.gate_sha256, 'xrvCacheReportSha256': args.xrv_report_sha256, 'labelInputs': provenance}
    try:
        for seed in SEEDS:
            for kind, clients in (('fed', CLIENTS), ('central', 1)):
                name = f'{kind}_{seed}'
                record, margin = run_one(name, seed, train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                                         clients=clients, features_path=args.xrv_cache, report_sha256=args.xrv_report_sha256,
                                         info=info)
                if not np.isfinite(margin).all():
                    record['meta']['nonfiniteMargins'] = True
                (staging / f'{name}.json').write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
                margins[name] = margin
                last = record['rounds'][-1]
                runs.append({'name': name, 'auroc': last['auc'], 'refusals': len(record['meta']['refusals']),
                             'clipRateLastRound': last['privacy']['clip_rate'], 'wallSeconds': record['meta']['wall_seconds']})
                print(json.dumps(runs[-1]), flush=True)
        summary = {
            'schema': 'capstone.effusion.federated_baseline.v1',
            'protocol': 'effusion-baseline-protocol-v1 section 6 with amendment A1',
            'scope': 'Clean federated and central-SGD heads on R1, scored on the selection half only. No DP, attacks, holdout or official test.',
            'inputs': {**info, 'historicalCache': str(args.historical_cache)},
            'recipe': config('fed', SEEDS[0], clients=CLIENTS, features_path=args.xrv_cache, report_sha256=args.xrv_report_sha256),
            'seeds': list(SEEDS), 'runs': runs,
            'comparison': compare(y_selection, selection_patients, margins, reference),
            'sourceSha256': {name: _sha256_file(HERE / name) for name in ('run_effusion_federated.py', 'fit_effusion_gate.py', 'effusion_labels.py')},
            'engineSourceSha256': _sha256_file(HERE.parents[2] / 'framework' / 'src' / 'fedlearn' / 'simulation' / 'engine.py'),
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument('--xrv-report-sha256', required=True)
    parser.add_argument('--gate-sha256', required=True)
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'seedSummary': summary['comparison']['seedSummary'], 'elapsedSeconds': summary['elapsedSeconds']}))
