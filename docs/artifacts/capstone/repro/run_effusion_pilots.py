"""Freeze and run declared development-only diagnostic pilots."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'framework/src'))
from fedlearn.estimators.params import trainable_state
from fedlearn.simulation.config import resolve
from fedlearn.simulation.effusion_bundle import load_effusion_bundle
from fedlearn.simulation.engine import run
from fedlearn.simulation.lab_contract import write_research_record
from fedlearn.simulation.partition import patient_dirichlet_partition
from capstone_pilot_mechanisms import public_projection
from client_private_erm import bound_rows, choose_solver_steps, fit_client_private_erm, release_gaussian
from run_effusion_byzfl_qualification import (BUNDLE_PIN, DEFAULTS, ZeroBundleWorkload,
    _checkpoint, _configuration, _configuration_hash, _read, _runtime, _source_paths,
    _validate_bundle_classification, sha)
from run_effusion_gdp_recipe import calibration, check_calibration

PROTOCOL = ROOT/'docs/artifacts/capstone/effusion-diagnostic-pilot-protocol-2026-10-25.md'
QUALIFICATION = ROOT/'docs/artifacts/capstone/phase-2-qualification-verification-2026-10-25.json'
PHASE3 = ROOT/'docs/artifacts/capstone/phase-3-verification-2026-10-25.json'
PROJECTION_SEED = 20261025
RECIPES = {'tight': {'clip_norm': .05, 'lr': .03}, 'loose': {'clip_norm': 1., 'lr': .1}}


def _cell(tag, seed, private, **overrides):
    cell = {'tag': tag, 'seed': seed, 'private': private, 'kind': 'fl', 'backend': 'byzfl',
            'rule': 'mean', 'width': 512, 'clip_norm': .05, 'lr': .03, 'attack': 'clean',
            'fraction': 0., 'bucket': 1, 'tau': .05, 'alpha': None,
            'regularization': None, 'row_bound': None, **overrides}
    cell['name'] = f"{tag}_{cell['backend']}_{'gdp4' if private else 'clean'}_s{seed}"
    return cell


def screen_cells(seeds=(20261211, 20261212)):
    cells = []
    for seed in seeds:
        for private in (False, True):
            cells.extend(_cell('mean', seed, private, backend=b) for b in ('native', 'byzfl'))
    for seed in seeds:
        for private in (False, True):
            for ratio in (.25, 1., 4.):
                for attack in ('clean', 'owned_pgd'):
                    cells.append(_cell(f'center_r{ratio:g}_{attack}', seed, private, rule='centered_clip', tau=.05*ratio,
                                       attack=attack, fraction=.2 if attack != 'clean' else 0.))
            for attack in ('sign_flip_scale', 'owned_sign_flip', 'owned_pgd'):
                cells.append(_cell(f'mean_{attack}_f16', seed, private, attack=attack, fraction=.2))
            for attack in ('clean', 'owned_pgd'):
                cells.append(_cell(f'trim16_{attack}', seed, private, rule='trimmed_mean', attack=attack,
                                   fraction=.2 if attack != 'clean' else 0.))
            for width in (64, 128, 256):
                cells.append(_cell(f'mean_k{width}', seed, private, width=width))
            for regularization in (.01, .1, 1.):
                cells.append(_cell(f'erm_l{regularization:g}', seed, private, kind='erm', backend='scalar',
                                   regularization=regularization, row_bound=1., clip_norm=None, lr=None))
        for recipe, values in RECIPES.items():
            cells.append(_cell(f'median_recipe_{recipe}', seed, False, rule='median', **values))
    return cells


def bucket_cells(recipe, seeds=(20261211, 20261212)):
    cells = []
    for seed in seeds:
        for alpha in (None, .1):
            for bucket in (1, 2, 4):
                for attack, fraction, label in (('clean', 0., 'clean'), ('owned_pgd', .05, 'pgd_f4'),
                                                ('owned_pgd', .2, 'pgd_f16'), ('fang_trim', .2, 'oracle_fang16')):
                    if alpha is None and bucket == 1 and attack == 'clean':
                        continue  # Reuse the pinned recipe-screen reference, not a replacement run.
                    tag = f"median_b{bucket}_{'iid' if alpha is None else 'a0p1'}_{label}"
                    cells.append(_cell(tag, seed, False, rule='median', bucket=bucket, alpha=alpha,
                                       attack=attack, fraction=fraction, **RECIPES[recipe]))
    return cells


def select_recipe(summary):
    if summary.get('schema') != 'capstone.effusion_pilot_result.v1' or summary.get('stage') != 'screen' \
            or summary.get('synthetic') is not False or summary.get('status') != 'complete':
        raise ValueError('complete declared real screen required')
    expected = {c['name']: c for c in screen_cells()}
    rows = summary.get('runs', [])
    if len(rows) != len(expected) or {r.get('name') for r in rows} != set(expected) \
            or any(r.get('status') != 'completed' or any(r.get(k) != v for k, v in expected[r['name']].items()) for r in rows):
        raise ValueError('incomplete or changed declared screen')
    points = {}
    for recipe in RECIPES:
        rows = [r for r in summary['runs'] if r['tag'] == f'median_recipe_{recipe}']
        if len(rows) != 2 or len({r['seed'] for r in rows}) != 2 or any(r['status'] != 'completed' for r in rows):
            raise ValueError('recipe screen is incomplete')
        points[recipe] = float(np.mean([r['metrics']['auroc'] for r in rows]))
    return 'loose' if points['loose'] > points['tight'] else 'tight'


def pinned_screen_reference(path, expected_sha256):
    path = Path(path)
    summary = _read(path, expected_sha256)
    recipe = select_recipe(summary)
    frozen = _read(path.parent/'frozen-plan.json', summary['plan_sha256'])
    public = _read(path.parent/'public-manifest.json', frozen['public_manifest_sha256'])
    if {k: frozen[k] for k in public} != public or frozen['cells'] != screen_cells() \
            or frozen['synthetic'] is not False or frozen['stage'] != 'screen':
        raise ValueError('recipe reference plan does not identify the declared real screen')
    if not summary.get('checkpoint_manifest'):
        raise ValueError('recipe reference lacks the committed manifest identity')
    _checkpoint(frozen, summary['checkpoint_commit'], ROOT/summary['checkpoint_manifest'])
    for row in summary['runs']:
        relative = f"runs/{row['name']}.json"
        if row['record'] != relative or sha(path.parent/relative) != row['sha256']:
            raise ValueError('recipe reference run hash differs')
        if row['tag'].startswith('median_recipe_'):
            raw = json.loads((path.parent/relative).read_text())
            if raw.get('pilot_metrics') != row['metrics']:
                raise ValueError('recipe decision metrics differ from saved record')
    return {'summary_sha256': expected_sha256, 'plan_sha256': summary['plan_sha256'],
            'public_manifest_sha256': frozen['public_manifest_sha256'],
            'checkpoint_commit': summary['checkpoint_commit'],
            'checkpoint_manifest': summary['checkpoint_manifest'], 'selected_recipe': recipe}


def validate_design(cells, *, stage, reference):
    if stage == 'screen':
        expected = screen_cells()
    elif stage == 'buckets' and reference and reference.get('selected_recipe') in RECIPES:
        expected = bucket_cells(reference['selected_recipe'])
    elif stage == 'finalists' and reference:
        from finalist_design import cells as finalist_cells
        receipt = json.loads(PHASE3.read_text())
        if receipt.get('status') != 'complete' or receipt.get('total_new_cells_completed') != 126 \
                or reference.get('screen_summary_sha256') != receipt.get('screen_summary_sha256') \
                or reference.get('bucket_summary_sha256') != receipt.get('bucket_summary_sha256') \
                or reference.get('phase3_verification_sha256') != sha(PHASE3) \
                or reference.get('selected_width') != 128 or reference.get('selected_center_tau') != .0125:
            raise ValueError('finalist comparison requires verified completed diagnostics and declared policies')
        expected = finalist_cells()
    else:
        raise ValueError('unknown or incomplete declared pilot stage')
    if cells != expected:
        raise ValueError('pilot cells differ from the declared stage budget')


def require_qualification():
    receipt = json.loads(QUALIFICATION.read_text())
    if receipt.get('status') != 'complete' or receipt.get('runs_completed') != 4:
        raise ValueError('actual-workload qualification must be complete first')


def configuration(cell, settings, bundle, bundle_sha256):
    if cell['kind'] == 'erm':
        bound = choose_solver_steps(row_bound=cell['row_bound'], regularization=cell['regularization'],
                                    num_clients=settings['clients'], residual_ratio=.01, max_steps=5000)
        return {'kind': 'erm', 'public_solver_bound': bound, 'width': cell['width'],
                'seed': cell['seed'], 'private': cell['private'], 'epsilon': settings['epsilon'],
                'delta': settings['delta'], 'alpha': cell['alpha']}
    cfg_settings = {**settings, 'seed': cell['seed'], 'clip_norm': cell['clip_norm'], 'lr': cell['lr']}
    cal = calibration(settings['epsilon'], rounds=settings['rounds'], delta=settings['delta'])
    cfg = _configuration(cell, cfg_settings, bundle, bundle_sha256, cal)
    cfg['server']['rule'] = cell['rule']
    cfg['server']['rule_params'] = {'trim_ratio': .2, 'tau': cell['tau'], 'iterations': 1}
    if cell['backend'] == 'byzfl':
        cfg['server']['external_aggregation']['bucket_size'] = cell['bucket']
    if cell['attack'] != 'clean':
        behavior = {'type': cell['attack'], 'fraction': cell['fraction']}
        if cell['attack'] == 'sign_flip_scale':
            behavior['scale'] = -4.
        elif cell['attack'] == 'owned_pgd':
            behavior.update(steps=20, restarts=3, step_ratio=.25)
        elif cell['attack'] == 'fang_trim':
            behavior['b'] = 2.
        cfg['behaviors'] = [behavior]
    return resolve(cfg)


class PilotWorkload(ZeroBundleWorkload):
    def __init__(self, resolved, cell):
        super().__init__(resolved)
        if cell['alpha'] is not None:
            self.parts = patient_dirichlet_partition(self.train_patient_ids, self.train_y.numpy(),
                resolved['federation']['num_clients'], cell['alpha'], resolved['seeds']['partition'])
        width = cell['width']
        matrix = public_projection(self.feat_dim, width, seed=PROJECTION_SEED)
        if width != self.feat_dim:
            self.train_x = (self.train_x.double() @ matrix).float()
            self.test_x = (self.test_x.double() @ matrix).float()
        self.feat_dim = width
        if any(not len(part) for part in self.parts):
            raise ValueError('pilot requires nonempty fixed clients')
        assignments, clients = [], {}
        for cid, part in enumerate(self.parts):
            assignments.append({'client': cid, 'images': [self.train_image_ids[i] for i in part]})
            positives = int(self.train_y[torch.as_tensor(part)].sum())
            clients[str(cid)] = {'images': len(part), 'positive_images': positives,
                'negative_images': len(part)-positives, 'patients': len({self.train_patient_ids[i] for i in part})}
        self.info.update(clients=clients, partition_algorithm='patient-iid-v1' if cell['alpha'] is None else 'patient-dirichlet-v1',
            assignment_sha256=_configuration_hash(assignments), assignment_version='capstone-pilot-client-assignment-v1',
            pilot={'alpha': cell['alpha'], 'width': width, 'projection_seed': PROJECTION_SEED,
                   'projection_sha256': hashlib.sha256(matrix.numpy().tobytes()).hexdigest(),
                   'partition_override': 'actual injected patient-whole partition is bound by cell and assignment hash'})


def _paths():
    paths = _source_paths()
    for path in (Path(__file__).resolve(), HERE/'capstone_pilot_mechanisms.py', HERE/'client_private_erm.py',
                 HERE/'test_run_effusion_pilots.py', HERE/'test_client_private_erm.py',
                 HERE/'analyze_effusion_pilots.py', HERE/'test_analyze_effusion_pilots.py', PROTOCOL, QUALIFICATION,
                 HERE/'finalist_design.py', HERE/'test_finalist_design.py', HERE/'finalist_inference.py',
                 HERE/'test_finalist_inference.py', HERE/'run_effusion_finalists.py', HERE/'test_run_effusion_finalists.py',
                 HERE/'analyze_effusion_finalists.py', HERE/'test_analyze_effusion_finalists.py',
                 ROOT/'docs/artifacts/capstone/effusion-finalist-protocol-2026-10-25.md', PHASE3):
        paths[path.relative_to(ROOT).as_posix()] = path
    return paths


def runtime():
    return {**_runtime(), 'sklearn': importlib.metadata.version('scikit-learn')}


def preflight(bundle, bundle_sha256, output, *, cells, settings=None, synthetic=False, stage='screen', reference=None):
    bundle, output = Path(bundle).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('pilot plan already exists')
    settings = dict(DEFAULTS if settings is None else settings)
    if set(settings) != set(DEFAULTS) or torch.get_num_threads() != 1:
        raise ValueError('pilot settings schema or CPU thread count differs')
    data = load_effusion_bundle(bundle, expected_report_sha256=bundle_sha256)
    _validate_bundle_classification(data, bundle_sha256, synthetic)
    if not synthetic and (bundle_sha256 != BUNDLE_PIN or settings != DEFAULTS):
        raise ValueError('medical pilot must use fixed bundle/settings')
    if not synthetic:
        require_qualification()
        validate_design(cells, stage=stage, reference=reference)
    if len(cells) != len({c['name'] for c in cells}) or not cells:
        raise ValueError('missing or duplicate pilot cells')
    configs = {c['name']: configuration(c, settings, bundle, bundle_sha256) for c in cells}
    for cell in cells:
        base_cell = {**cell, 'kind': 'fl', 'backend': 'byzfl', 'private': False,
                     'rule': 'mean', 'attack': 'clean', 'fraction': 0., 'bucket': 1, 'clip_norm': .05, 'lr': .03}
        workload = PilotWorkload(configuration(base_cell, settings, bundle, bundle_sha256), cell)
        if any(torch.count_nonzero(v) for v in trainable_state(workload.build_model(cell['seed'])).values()):
            raise ValueError('initial head is not zero')
    paths = _paths()
    public = {'schema': 'capstone.effusion_pilot_plan.v1', 'stage': stage, 'synthetic': synthetic,
              'settings': settings, 'cells': cells, 'bundle_report_sha256': bundle_sha256,
              'source_sha256': {name: sha(path) for name, path in paths.items()},
              'configuration_sha256': {name: _configuration_hash(cfg) for name, cfg in configs.items()},
              'runtime': runtime(), 'reference': reference,
              'scope': ('prespecified fresh-seed comparison conditional on reused development cohort and selected policies; '
                        'raw research diagnostics outside DP release claims' if stage == 'finalists' else
                        'fixed train/development selection; exploratory pilots; raw research diagnostics outside DP release claims')}
    output.mkdir(parents=True, exist_ok=False)
    write_research_record(output/'public-manifest.json', public)
    plan = {**public, 'bundle_path': str(bundle), 'configurations': configs,
            'public_manifest_sha256': sha(output/'public-manifest.json')}
    write_research_record(output/'plan.json', plan)
    for name, path in paths.items():
        target = output/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return plan


def _metrics(y, margins):
    y, margins = np.asarray(y), np.asarray(margins, dtype=np.float64)
    if not np.isfinite(margins).all():
        raise ValueError('nonfinite pilot margins')
    probability = expit(margins)
    positive = margins >= 0
    return {'auroc': float(roc_auc_score(y, margins)), 'average_precision': float(average_precision_score(y, margins)),
            'brier': float(np.mean((probability-y)**2)), 'sensitivity_at_half': float(positive[y == 1].mean()),
            'specificity_at_half': float((~positive[y == 0]).mean())}


def execute(plan_dir, expected_plan_sha256, output, *, checkpoint_commit=None, checkpoint_manifest=None):
    plan_dir, output = Path(plan_dir), Path(output)
    if output.exists():
        raise FileExistsError('pilot output already exists')
    plan = _read(plan_dir/'plan.json', expected_plan_sha256)
    public = _read(plan_dir/'public-manifest.json', plan['public_manifest_sha256'])
    if {key: plan[key] for key in public} != public:
        raise ValueError('private pilot plan differs from public manifest')
    if {name: sha(path) for name, path in _paths().items()} != plan['source_sha256'] or runtime() != plan['runtime']:
        raise ValueError('source/dependency/runtime changed')
    snapshot = plan_dir/'sources'
    if any(p.is_symlink() for p in snapshot.rglob('*')) or {p.relative_to(snapshot).as_posix(): sha(p)
        for p in snapshot.rglob('*') if p.is_file()} != plan['source_sha256']:
        raise ValueError('source snapshot changed')
    bundle = Path(plan['bundle_path'])
    data = load_effusion_bundle(bundle, expected_report_sha256=plan['bundle_report_sha256'])
    _validate_bundle_classification(data, plan['bundle_report_sha256'], plan['synthetic'])
    if not plan['synthetic'] and (plan['bundle_report_sha256'] != BUNDLE_PIN or plan['settings'] != DEFAULTS):
        raise ValueError('medical pilot must use fixed inputs/settings')
    if not plan['synthetic']:
        require_qualification()
        validate_design(plan['cells'], stage=plan['stage'], reference=plan['reference'])
    _checkpoint(plan, checkpoint_commit, checkpoint_manifest)
    configs = {c['name']: configuration(c, plan['settings'], bundle, plan['bundle_report_sha256']) for c in plan['cells']}
    if configs != plan['configurations'] or {n: _configuration_hash(c) for n, c in configs.items()} != plan['configuration_sha256']:
        raise ValueError('resolved pilot configurations changed')
    output.mkdir(parents=True, exist_ok=False)
    (output/'runs').mkdir()
    shutil.copytree(snapshot, output/'sources')
    shutil.copyfile(plan_dir/'plan.json', output/'frozen-plan.json')
    shutil.copyfile(plan_dir/'public-manifest.json', output/'public-manifest.json')
    outcomes, heads, comparisons, parity_failed = [], {}, [], False
    settings = plan['settings']
    for cell in plan['cells']:
        final, observed, record = {}, [], None
        metrics, status = None, 'not_run' if parity_failed else 'error'
        try:
            if parity_failed:
                raise RuntimeError('not run after baseline backend mismatch')
            cfg = configs[cell['name']]
            base_cell = {**cell, 'kind': 'fl', 'backend': 'byzfl', 'private': False, 'rule': 'mean',
                         'attack': 'clean', 'fraction': 0., 'bucket': 1, 'clip_norm': .05, 'lr': .03}
            workload = PilotWorkload(configuration(base_cell, settings, bundle, plan['bundle_report_sha256']), cell)
            if cell['kind'] == 'fl':
                journal = output/'rounds'/cell['name']
                journal.mkdir(parents=True, exist_ok=False)
                def save_round(line):
                    write_research_record(journal/f"round-{line['round']:04d}.json", line)
                    observed.append(line)
                record = run(cfg, workload=workload, on_round=save_round, on_final_parameters=final.update)
                if len(record['rounds']) != settings['rounds'] or record['meta']['refusals'] or not final:
                    raise ValueError('incomplete/refused pilot run')
                if any(r['participants']['aggregated'] != settings['clients'] for r in record['rounds']):
                    raise ValueError('pilot cohort changed')
                if cell['private']:
                    cal = calibration(settings['epsilon'], rounds=settings['rounds'], delta=settings['delta'])
                    exact = check_calibration(cal, record['meta']['noise_multiplier'], settings['rounds'], settings['delta'])
                    record['pilot_accounting'] = {**cal, 'exact_epsilon': exact, 'releases': settings['rounds']}
                model = workload.build_model()
                model.load_state_dict(final)
                with torch.no_grad():
                    logits = model(workload.test_x).double()
                margins = (logits[:, 1]-logits[:, 0]).numpy()
                metrics = _metrics(workload.test_y.numpy(), margins)
                if abs(metrics['auroc']-record['rounds'][-1]['auc']) > 1e-10:
                    raise ValueError('saved-head metric does not match engine')
                if cell['tag'] == 'mean':
                    key = (cell['seed'], cell['private'])
                    heads.setdefault(key, {})[cell['backend']] = final
                    if len(heads[key]) == 2:
                        native, external = heads[key]['native'], heads[key]['byzfl']
                        difference = max(float((native[k]-external[k]).abs().max()) for k in native)
                        comparisons.append({'seed': cell['seed'], 'private': cell['private'], 'max_parameter_difference': difference})
                        if difference > 1e-6:
                            parity_failed = True
                            raise ValueError('baseline backend mismatch')
            else:
                bound = cfg['public_solver_bound']
                fitted = fit_client_private_erm(workload.train_x.numpy(), workload.train_y.numpy(), workload.parts,
                    row_bound=cell['row_bound'], regularization=cell['regularization'], steps=bound['steps'], max_steps=5000)
                cal = calibration(settings['epsilon'], rounds=1, delta=settings['delta'])
                released = release_gaussian(fitted['theta'], sensitivity=bound['sensitivity'],
                    noise_multiplier=cal['gdpNoiseMultiplier'] if cell['private'] else 0., seed=cell['seed']+2000)
                record = {'public_solver_bound': bound, 'research_diagnostics': fitted['research_diagnostics'],
                    'research_fit_head': fitted['theta'].tolist(), 'released_head': released['theta'].tolist(),
                    'accounting': {**released, 'theta': None, 'releases': 1,
                        'exact_epsilon': check_calibration(cal, released['noise_multiplier'], 1, settings['delta']) if cell['private'] else None},
                    'scope': 'trusted coordinator; only final noised head is a candidate DP output; raw fit/diagnostics are research-only'}
                margins = np.einsum('ij,j->i', bound_rows(workload.test_x.numpy(), cell['row_bound']),
                                    released['theta'][:-1], optimize=False) + released['theta'][-1]
                metrics = _metrics(workload.test_y.numpy(), margins)
            status = 'completed'
        except Exception as exc:
            if record is None:
                record = {'rounds': observed}
            record['pilot_failure'] = {'error_type': type(exc).__name__, 'error': str(exc), 'status': status}
            if cell['tag'] == 'mean':
                parity_failed = True
        if final:
            record.setdefault('meta', {})['finalHead'] = {k: v.tolist() for k, v in final.items()}
        if metrics is not None:
            record['pilot_metrics'] = metrics
        relative = f"runs/{cell['name']}.json"
        write_research_record(output/relative, record)
        outcome = {**cell, 'status': status, 'metrics': metrics, 'record': relative, 'sha256': sha(output/relative)}
        outcomes.append(outcome)
        write_research_record(output/f"cell-receipt-{cell['name']}.json", outcome)
    result = {'schema': 'capstone.effusion_pilot_result.v1', 'stage': plan['stage'],
              'status': 'complete' if all(r['status'] == 'completed' for r in outcomes) else 'incomplete',
              'plan_sha256': expected_plan_sha256, 'checkpoint_commit': checkpoint_commit,
              'checkpoint_manifest': Path(checkpoint_manifest).resolve().relative_to(ROOT).as_posix() if checkpoint_manifest else None,
              'synthetic': plan['synthetic'], 'reference': plan['reference'], 'runs': outcomes,
              'backend_comparisons': comparisons, 'scope': public['scope']}
    write_research_record(output/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('preflight')
    prepare.add_argument('--bundle', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--stage', choices=('screen', 'buckets'), default='screen')
    prepare.add_argument('--reference', type=Path)
    prepare.add_argument('--reference-sha256')
    launch = sub.add_parser('execute')
    launch.add_argument('--plan-dir', type=Path, required=True)
    launch.add_argument('--plan-sha256', required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--checkpoint-commit', required=True)
    launch.add_argument('--checkpoint-manifest', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'preflight':
        reference = None
        cells = screen_cells()
        if args.stage == 'buckets':
            if args.reference is None or args.reference_sha256 is None:
                parser.error('bucket stage requires pinned screen summary')
            reference = pinned_screen_reference(args.reference, args.reference_sha256)
            cells = bucket_cells(reference['selected_recipe'])
        preflight(args.bundle, BUNDLE_PIN, args.output, cells=cells, stage=args.stage, reference=reference)
        print(json.dumps({'status': 'preflight_complete', 'cells': len(cells), 'model_fits': 0,
                          'plan_sha256': sha(args.output/'plan.json'), 'public_manifest_sha256': sha(args.output/'public-manifest.json')}))
    else:
        result = execute(args.plan_dir, args.plan_sha256, args.output, checkpoint_commit=args.checkpoint_commit,
                         checkpoint_manifest=args.checkpoint_manifest)
        print(json.dumps({'status': result['status'], 'cells': len(result['runs']),
                          'completed': sum(r['status'] == 'completed' for r in result['runs'])}))
        return 0 if result['status'] == 'complete' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
