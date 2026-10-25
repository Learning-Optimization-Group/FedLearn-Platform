"""Synthetic checks; these do not establish any actual finalist result."""
import hashlib
import json

import numpy as np
import pytest

import analyze_effusion_finalists as analysis
from finalist_design import SEEDS, cells
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import config_hash
from fedlearn.simulation.partition import patient_iid_partition


def rows():
    result = []
    for cell in cells():
        value = .6 + (cell['seed'] - SEEDS[0]) * .001
        value += .04 if cell['method'] == 'projected128' else 0
        value -= .02 if cell['private'] else 0
        value -= .08 if cell['attack'] != 'clean' else 0
        metrics = dict.fromkeys(analysis.METRICS, value)
        result.append(cell | {'status': 'completed', 'metrics': metrics})
    return result


def test_primary_point_exact_seeds_and_prespecified_contrast_signs():
    result = analysis.summarize(rows())
    assert len(result['arms']) == 16
    primary = result['primary']
    assert primary['seed_interval']['point'] == pytest.approx(.04)
    assert set(primary['seed_interval']['seed_effects']) == set(SEEDS)
    assert primary['technical_success'] is True
    contrasts = {c['id']: c for c in result['contrasts']}
    assert contrasts['attack_damage:mean512:private']['seed_interval']['point'] == pytest.approx(.08)
    assert contrasts['privacy_cost:mean512:clean']['seed_interval']['point'] == pytest.approx(.02)
    assert contrasts['interaction:projected128']['seed_interval']['point'] == pytest.approx(0)
    assert len(result['contrasts']) == 31


def test_interaction_is_private_advantage_change_minus_nonprivate_change():
    data = rows()
    for row in data:
        if row['method'] == 'projected128' and row['attack'] == 'owned_pgd':
            row['metrics']['auroc'] += .03 if row['private'] else -.02
    result = analysis.summarize(data)
    interaction = next(c for c in result['contrasts'] if c['id'] == 'interaction:projected128')
    assert interaction['seed_interval']['point'] == pytest.approx(.05)


def test_wholly_negative_primary_seed_interval_reports_negative_gain():
    data = rows()
    for row in data:
        if row['method'] == 'projected128' and row['private'] and row['attack'] == 'clean':
            row['metrics']['auroc'] -= .08
    primary = analysis.summarize(data)['primary']
    assert primary['seed_interval']['ci_high'] < 0
    assert primary['technical_success'] is False
    assert primary['conclusion'] == 'negative_seed_mean_gain'


@pytest.mark.parametrize('mutation', ['duplicate', 'unknown', 'seed'])
def test_refuses_changed_row_identity(mutation):
    data = rows()
    if mutation == 'duplicate':
        data[-1] = data[-2]
    elif mutation == 'unknown':
        data[-1]['name'] = 'unknown'
    else:
        data[-1]['seed'] = 99
    with pytest.raises(ValueError):
        analysis.summarize(data)


def test_incomplete_retains_all_outcomes_and_has_no_pair_inference():
    data = rows()
    data[-1].update(status='error', metrics=None)
    result = analysis.summarize(data)
    assert result['status'] == 'incomplete'
    assert len(result['outcomes']) == 196
    assert result['primary'] is None
    assert all(c['seed_interval'] is None for c in result['contrasts'])
    assert result['failures'] == [{'name': data[-1]['name'], 'status': 'error'}]


def test_missing_rows_are_explicit_without_partial_inference():
    result = analysis.summarize(rows()[:-1])
    assert len(result['outcomes']) == 196
    assert result['outcomes'][-1]['status'] == 'missing'
    assert result['primary'] is None


def test_saved_head_scoring_uses_float32_linear_then_double_difference():
    x = np.zeros((4, 512), dtype=np.float32)
    x[:, 0] = [-2, -1, 1, 2]
    head = {'head.weight': [[0.] * 512, [1.] + [0.] * 511], 'head.bias': [0., .25]}
    margins = analysis.saved_head_margins(x, head, 512)
    assert np.array_equal(margins, [-1.75, -.75, 1.25, 2.25])
    head['head.weight'][1].pop()
    with pytest.raises(ValueError, match='head'):
        analysis.saved_head_margins(x, head, 512)


def test_reports_never_export_raw_record_arrays_ids_or_paths(tmp_path):
    result = analysis.summarize(rows())
    analysis.write_outputs(tmp_path / 'export', result, {'schema': 'audit', 'runs': []})
    assert {p.name for p in (tmp_path / 'export').iterdir()} == {
        'summary.json', 'arms.csv', 'contrasts.csv', 'report.md', 'scoring-audit.json'}
    for path in (tmp_path / 'export').iterdir():
        assert b'\r\n' not in path.read_bytes()
        for forbidden in ('patient_ids', 'finalHead', 'head.weight', 'bundle_path', 'margins', '/Users/'):
            assert forbidden not in path.read_text()
    with pytest.raises(FileExistsError):
        analysis.write_outputs(tmp_path / 'export', result, {})


def test_nonfinite_metrics_refused():
    data = rows()
    data[-1]['metrics']['auroc'] = float('nan')
    with pytest.raises(ValueError):
        analysis.summarize(data)


@pytest.fixture
def artifact_fixture(tmp_path, monkeypatch):
    """Generated arrays/records only; mock the unavailable real bundle/checkpoint.

    Configurations, records, head scoring, hashes, budgets and exports are real
    code paths. This is not a test of committed checkpoint or medical data.
    """
    directory = tmp_path / 'run'
    (directory / 'runs').mkdir(parents=True)
    x = np.zeros((8, 512), dtype=np.float32)
    y = np.array([0, 1] * 4, dtype=np.int64)
    evaluation = {'x': x, 'y': y, 'patient_ids': [str(100 + i // 2) for i in range(8)]}
    train = {'x': np.zeros((80, 512), dtype=np.float32), 'y': np.array([0, 1] * 40),
             'patient_ids': [str(i + 1) for i in range(80)], 'image_ids': [f'image{i}' for i in range(80)]}
    report = {'dataset': 'NIH_ChestXray14', 'target': 'Effusion',
              'bundle_version': 'capstone-nih-effusion-r0-selection-v1', 'pretrained': True,
              'counts': {'train': {'images': 80, 'patients': 80, 'positives': 40},
                         'selection': {'images': 8, 'patients': 4, 'positives': 4}},
              'provenance': {'synthetic_fixture': True}}
    seen = []

    def bundle_loader(path, *, expected_report_sha256):
        assert str(path) == 'generated_fixture_bundle'
        assert expected_report_sha256 == analysis.pilot.BUNDLE_PIN
        seen.append('bundle')
        return {'evaluation': evaluation, 'train': train, 'report': report,
                'report_sha256': analysis.pilot.BUNDLE_PIN}

    def checkpoint(plan, commit, manifest):
        assert commit == 'a' * 40
        assert manifest == analysis.pilot.ROOT / 'declared-fixture-manifest.json'
        seen.append('checkpoint')

    monkeypatch.setattr(analysis.pilot, 'load_effusion_bundle', bundle_loader)
    monkeypatch.setattr(analysis.pilot, '_checkpoint', checkpoint)

    def save(path, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    receipt = {'status': 'complete', 'total_new_cells_completed': 126,
               'screen_summary_sha256': 'c' * 64, 'bucket_summary_sha256': 'd' * 64}
    phase3 = tmp_path / 'generated-phase3.json'
    phase3_hash = save(phase3, receipt)
    monkeypatch.setattr(analysis.pilot, 'PHASE3', phase3)
    reference = {'screen_summary_sha256': 'c' * 64, 'bucket_summary_sha256': 'd' * 64,
                 'phase3_verification_sha256': phase3_hash,
                 'selected_width': 128, 'selected_center_tau': .0125}

    configs = {c['name']: analysis.pilot.configuration(c, analysis.pilot.DEFAULTS,
               'generated_fixture_bundle', analysis.pilot.BUNDLE_PIN) for c in cells()}
    public = {'schema': 'capstone.effusion_pilot_plan.v1', 'stage': 'finalists', 'synthetic': False,
              'settings': analysis.pilot.DEFAULTS, 'cells': cells(),
              'bundle_report_sha256': analysis.pilot.BUNDLE_PIN,
              'source_sha256': {'generated_fixture.py': 'b' * 64}, 'reference': reference,
              'configuration_sha256': {n: analysis.pilot._configuration_hash(v) for n, v in configs.items()}}
    public_hash = save(directory / 'public-manifest.json', public)
    plan = public | {'public_manifest_sha256': public_hash, 'bundle_path': 'generated_fixture_bundle',
                     'configurations': configs}
    plan_hash = save(directory / 'frozen-plan.json', plan)
    metrics = analysis.pilot._metrics(y, np.zeros(len(y)))
    outcomes = []
    for cell in cells():
        sens = (2 * .0125 / 80 if cell['method'] == 'center_quarter' else
                2 * .05 / 48 if cell['method'] == 'trim16' else 2 * .05 / 80)
        z = analysis._calibration()['gdpNoiseMultiplier'] if cell['private'] else 0.
        head = {'head.weight': [[0.] * cell['width']] * 2, 'head.bias': [0., 0.]}
        cfg = configs[cell['name']]
        assignments, clients = [], {}
        parts = patient_iid_partition(train['patient_ids'], 80, cfg['seeds']['partition'])
        for cid, part in enumerate(parts):
            assignments.append({'client': cid, 'images': [train['image_ids'][i] for i in part]})
            positive = int(train['y'][part].sum())
            clients[str(cid)] = {'images': len(part), 'positive_images': positive,
                                 'negative_images': len(part) - positive,
                                 'patients': len({train['patient_ids'][i] for i in part})}
        projection = analysis.pilot.public_projection(512, cell['width'], seed=20261025)
        workload = {'feat_dim': cell['width'], 'n_classes': 2, 'n_train': 80, 'n_test': 8,
                    'parameters': 2 * cell['width'] + 2,
                    'auc_score': 'margin_v2', 'auc_score_definition': 'float64(logit_1)-float64(logit_0)',
                    'dataset': 'nih', 'source_dataset': report['dataset'], 'target': report['target'],
                    'bundle_version': report['bundle_version'],
                    'backbone': 'resnet18 ImageNet (R0), train-standardised',
                    'training_device': 'cpu', 'torch_num_threads': 1, 'pretrained': True,
                    'allow_random_features': False, 'evaluation_split': 'validation',
                    'evaluation': 'development selection half; no holdout or official test',
                    'features_report_sha256': analysis.pilot.BUNDLE_PIN,
                    'bundle_provenance': report['provenance'], 'split_counts': report['counts'],
                    'clients': clients, 'partition_algorithm': 'patient-iid-v1',
                    'assignment_sha256': analysis.pilot._configuration_hash(assignments),
                    'assignment_version': 'capstone-pilot-client-assignment-v1',
                    'pilot': {'alpha': None, 'width': cell['width'], 'projection_seed': 20261025,
                              'projection_sha256': hashlib.sha256(projection.numpy().tobytes()).hexdigest(),
                              'partition_override': 'actual injected patient-whole partition is bound by cell and assignment hash'}}
        raw = {'meta': {'noise_multiplier': z, 'refusals': [], 'finalHead': head,
                        'config': cfg, 'config_hash': config_hash(cfg), 'seeds': cfg['seeds'],
                        'workload': workload,
                        'behavior_assignment': BehaviorPlan(cfg['behaviors'], 80, cfg['seeds']['attack'], 2).assignment()},
               'pilot_metrics': metrics,
               'rounds': [{'round': t, 'auc': metrics['auroc'], 'refusal': None,
                           'participants': {'sampled': 80, 'submitted': 80, 'admitted': 80,
                                            'accepted': 80, 'aggregated': 80, 'dropped': 0, 'straggled': 0},
                           'privacy': {'sensitivity': sens, 'noise_std': z * sens,
                                       'server_noise_std': z * sens, 'client_noise_std': 0.}}
                          for t in range(1, 21)]}
        if cell['private']:
            raw['pilot_accounting'] = analysis._calibration() | {'releases': 20,
                'exact_epsilon': analysis.pilot.check_calibration(analysis._calibration(), z, 20, 1e-5)}
        relative = f"runs/{cell['name']}.json"
        digest = save(directory / relative, raw)
        outcomes.append(cell | {'status': 'completed', 'metrics': metrics,
                                'record': relative, 'sha256': digest})
    summary = {'schema': 'capstone.effusion_pilot_result.v1', 'stage': 'finalists',
               'synthetic': False, 'status': 'complete', 'plan_sha256': plan_hash,
               'checkpoint_commit': 'a' * 40, 'checkpoint_manifest': 'declared-fixture-manifest.json',
               'reference': reference, 'runs': outcomes,
               'backend_comparisons': [{'seed': s, 'private': p, 'max_parameter_difference': 0.}
                                       for s in SEEDS[:2] for p in (False, True)]}
    pin = save(directory / 'summary.json', summary)
    return directory, pin, summary, save, seen


def test_generated_artifact_integration_all_196_heads_and_private_audits(artifact_fixture, tmp_path):
    directory, pin, _, _, seen = artifact_fixture
    result = analysis.analyze(directory, pin, tmp_path / 'analysis')
    assert result['accounting']['completed'] == 196
    assert result['primary']['seed_interval']['point'] == 0
    assert result['primary']['patient_interval']['point'] == 0
    assert result['primary']['patient_interval']['draws'] == 2000
    assert seen == ['checkpoint', 'bundle']
    audit = json.loads((tmp_path / 'analysis' / 'scoring-audit.json').read_text())
    assert len(audit['runs']) == 196
    assert len(audit['backend_comparisons']) == 4
    assert sum('delta_at_epsilon4' in r for r in audit['runs']) == 98
    for path in (tmp_path / 'analysis').iterdir():
        text = path.read_text()
        assert 'generated_fixture_bundle' not in text
        assert 'patient_ids' not in text
        assert 'head.weight' not in text


@pytest.mark.parametrize('mutation', ['hash', 'head', 'metrics', 'rounds', 'noise', 'parity', 'sampled'])
def test_tampered_artifacts_refused_even_when_summary_is_repinned(artifact_fixture, tmp_path, mutation):
    directory, pin, summary, save, _ = artifact_fixture
    row = summary['runs'][-1]
    path = directory / row['record']
    raw = json.loads(path.read_text())
    if mutation == 'head':
        raw['meta']['finalHead']['head.bias'] = [10., -10.]
    elif mutation == 'metrics':
        row['metrics']['auroc'] = .8
    elif mutation == 'rounds':
        raw['rounds'][0]['participants']['aggregated'] = 79
    elif mutation == 'noise':
        raw['rounds'][0]['privacy']['noise_std'] *= 2
    elif mutation == 'parity':
        summary['backend_comparisons'].pop()
    elif mutation == 'sampled':
        raw['rounds'][0]['participants']['sampled'] = 79
    else:
        raw['meta']['finalHead']['head.bias'] = [1., 2.]
    digest = save(path, raw)
    if mutation != 'hash':
        row['sha256'] = digest
    pin = save(directory / 'summary.json', summary)
    with pytest.raises(ValueError):
        analysis.analyze(directory, pin, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()


def test_changed_selection_reference_refused(artifact_fixture, tmp_path):
    directory, _, summary, save, _ = artifact_fixture
    plan = json.loads((directory / 'frozen-plan.json').read_text())
    public = json.loads((directory / 'public-manifest.json').read_text())
    public['reference']['selected_width'] = 64
    plan['reference'] = public['reference']
    summary['reference'] = public['reference']
    plan['public_manifest_sha256'] = save(directory / 'public-manifest.json', public)
    summary['plan_sha256'] = save(directory / 'frozen-plan.json', plan)
    pin = save(directory / 'summary.json', summary)
    with pytest.raises(ValueError, match='finalist'):
        analysis.analyze(directory, pin, tmp_path / 'rejected')


def test_incomplete_backend_failure_exports_all_outcomes_without_inference(artifact_fixture, tmp_path):
    directory, _, summary, save, _ = artifact_fixture
    row = summary['runs'][0]
    raw = json.loads((directory / row['record']).read_text())
    raw['meta']['finalHead']['head.bias'][1] = .01
    raw['pilot_failure'] = {'error_type': 'ValueError', 'error': 'backend mismatch'}
    row['status'] = 'error'
    row['sha256'] = save(directory / row['record'], raw)
    summary['status'] = 'incomplete'
    summary['backend_comparisons'][0]['max_parameter_difference'] = float(np.float32(.01))
    pin = save(directory / 'summary.json', summary)
    result = analysis.analyze(directory, pin, tmp_path / 'incomplete')
    assert len(result['outcomes']) == 196
    assert result['primary'] is None
    assert all(c['seed_interval'] is None for c in result['contrasts'])
    audit = json.loads((tmp_path / 'incomplete' / 'scoring-audit.json').read_text())
    assert audit['backend_comparisons'][0]['status'] == 'failed'


@pytest.mark.parametrize('kwargs', [{'bootstrap_draws': 20}, {'bootstrap_seed': 1},
                                   {'bootstrap_draws': 2000.}, {'bootstrap_seed': True}])
def test_frozen_bootstrap_settings_required_before_loading(tmp_path, monkeypatch, kwargs):
    def forbidden_read(*args):
        raise AssertionError('must refuse settings before reading artifacts')
    monkeypatch.setattr(analysis, '_load_run', forbidden_read)
    with pytest.raises(ValueError, match='bootstrap'):
        analysis.analyze(tmp_path / 'run', 'a' * 64, tmp_path / 'output', **kwargs)
    assert not (tmp_path / 'output').exists()


def test_cli_rejects_alternate_bootstrap_settings_before_loading(monkeypatch):
    monkeypatch.setattr('sys.argv', ['analyze', '--run-dir', 'unused', '--summary-sha256', 'a' * 64,
                                    '--output', 'unused', '--bootstrap-draws', '20'])
    monkeypatch.setattr(analysis, 'analyze', lambda *args, **kwargs: pytest.fail('analysis must not run'))
    with pytest.raises(SystemExit) as error:
        analysis.main()
    assert error.value.code == 2


def test_same_width_swapped_records_refused_after_consistent_rehash(artifact_fixture, tmp_path):
    directory, _, summary, save, _ = artifact_fixture
    left = next(r for r in summary['runs'] if r['method'] == 'mean512' and r['role'] == 'main'
                and r['private'] is False and r['attack'] == 'clean' and r['seed'] == SEEDS[1])
    right = next(r for r in summary['runs'] if r['method'] == 'mean512' and r['role'] == 'main'
                 and r['private'] is False and r['attack'] == 'clean' and r['seed'] == SEEDS[0])
    a = json.loads((directory / left['record']).read_text())
    b = json.loads((directory / right['record']).read_text())
    left['sha256'] = save(directory / left['record'], b)
    right['sha256'] = save(directory / right['record'], a)
    left['metrics'], right['metrics'] = b['pilot_metrics'], a['pilot_metrics']
    pin = save(directory / 'summary.json', summary)
    with pytest.raises(ValueError, match='config'):
        analysis.analyze(directory, pin, tmp_path / 'rejected')


@pytest.mark.parametrize('field', ['config_hash', 'seeds', 'workload', 'behavior_assignment'])
def test_repinned_raw_metadata_must_match_exact_cell_identity(artifact_fixture, tmp_path, field):
    directory, _, summary, save, _ = artifact_fixture
    row = summary['runs'][-1]
    raw = json.loads((directory / row['record']).read_text())
    if field == 'config_hash':
        raw['meta'][field] = '0' * 64
    elif field == 'seeds':
        raw['meta'][field]['partition'] += 1
    elif field == 'workload':
        raw['meta'][field]['assignment_sha256'] = '0' * 64
    else:
        raw['meta'][field] = {}
    row['sha256'] = save(directory / row['record'], raw)
    pin = save(directory / 'summary.json', summary)
    with pytest.raises(ValueError):
        analysis.analyze(directory, pin, tmp_path / 'rejected')
