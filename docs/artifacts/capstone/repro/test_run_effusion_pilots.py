"""Prospective pilot accounting and true-engine checks on generated bundles."""
import hashlib
import json

import pytest
import torch

from test_run_effusion_byzfl_qualification import inputs
import run_effusion_pilots as pilot


def test_screen_has_a_complete_declared_budget_without_duplicate_cells():
    cells = pilot.screen_cells()
    assert len(cells) == 80
    assert len({c['name'] for c in cells}) == 80
    assert {c['width'] for c in cells} == {64, 128, 256, 512}
    assert {c['regularization'] for c in cells if c['kind'] == 'erm'} == {.01, .1, 1.}
    assert all(c['bucket'] == 1 for c in cells if c['private'])
    assert all(c['attack'] not in ('fang_trim', 'ipm', 'alie', 'min_max') for c in cells if c['private'])


def test_centered_screen_has_attacked_and_clean_controls_for_every_radius():
    centers = [c for c in pilot.screen_cells() if c['rule'] == 'centered_clip']
    for seed in (20261211, 20261212):
        for private in (False, True):
            for tau in (.0125, .05, .2):
                rows = [c for c in centers if c['seed'] == seed and c['private'] == private and c['tau'] == tau]
                assert {c['attack'] for c in rows} == {'clean', 'owned_pgd'}


def test_bucket_stage_uses_the_shared_recipe_and_retains_low_high_attackers():
    cells = pilot.bucket_cells(recipe='loose')
    assert len(cells) == 46
    assert {c['bucket'] for c in cells} == {1, 2, 4}
    assert {c['fraction'] for c in cells} == {0., .05, .2}
    assert {c['alpha'] for c in cells} == {None, .1}
    assert all(not c['private'] and c['clip_norm'] == 1. and c['lr'] == .1 for c in cells)


def test_recipe_selection_uses_both_complete_seeds_and_declared_tie_rule():
    summary = screen_fixture()
    assert pilot.select_recipe(summary) == 'tight'
    with pytest.raises(ValueError, match='incomplete'):
        pilot.select_recipe(summary | {'runs': summary['runs'][:-1]})


def screen_fixture():
    return {'schema': 'capstone.effusion_pilot_result.v1', 'stage': 'screen', 'synthetic': False, 'status': 'complete',
            'runs': [{**c, 'status': 'completed', 'metrics': {'auroc': .8}} for c in pilot.screen_cells()]}


@pytest.mark.parametrize('changed', [{'stage': 'buckets'}, {'synthetic': True}, {'status': 'incomplete'}])
def test_recipe_selection_refuses_other_studies_or_partial_results(changed):
    with pytest.raises(ValueError):
        pilot.select_recipe(screen_fixture() | changed)


def test_real_design_refuses_an_undeclared_cell_budget():
    with pytest.raises(ValueError, match='declared'):
        pilot.validate_design(pilot.screen_cells()[:-1], stage='screen', reference=None)


def test_pinned_recipe_reference_binds_records_and_rejects_changed_metrics(tmp_path, monkeypatch):
    summary = screen_fixture() | {'checkpoint_commit': '0123456', 'checkpoint_manifest': 'docs/example-manifest.json'}
    public = {'cells': pilot.screen_cells(), 'synthetic': False, 'stage': 'screen'}
    manifest = tmp_path/'public-manifest.json'
    manifest.write_text(json.dumps(public))
    frozen = public | {'public_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    plan = tmp_path/'frozen-plan.json'
    plan.write_text(json.dumps(frozen))
    summary['plan_sha256'] = hashlib.sha256(plan.read_bytes()).hexdigest()
    (tmp_path/'runs').mkdir()
    for row in summary['runs']:
        row['record'] = f"runs/{row['name']}.json"
        raw = tmp_path/row['record']
        raw.write_text(json.dumps({'pilot_metrics': row['metrics']}))
        row['sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    source = tmp_path/'summary.json'
    source.write_text(json.dumps(summary))
    called = []
    monkeypatch.setattr(pilot, '_checkpoint', lambda *args: called.append(args))
    pin = hashlib.sha256(source.read_bytes()).hexdigest()
    assert pilot.pinned_screen_reference(source, pin)['selected_recipe'] == 'tight'
    assert len(called) == 1
    row = next(r for r in summary['runs'] if r['tag'].startswith('median_recipe_'))
    (tmp_path/row['record']).write_text(json.dumps({'pilot_metrics': {'auroc': .99}}))
    with pytest.raises(ValueError, match='hash'):
        pilot.pinned_screen_reference(source, pin)


def small_cells():
    return [c for c in pilot.screen_cells(seeds=(71,))
            if c['tag'] == 'mean' or c['kind'] == 'erm' and c['regularization'] == .1 and c['private']]


def freeze(inputs, tmp_path):
    path = tmp_path / 'pilot-plan'
    plan = pilot.preflight(inputs['bundle'], inputs['bundle_sha256'], path,
                           cells=small_cells(), settings=inputs['settings'], synthetic=True)
    return path, hashlib.sha256((path/'plan.json').read_bytes()).hexdigest(), plan


def test_preflight_freezes_configs_without_fitting(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'run', lambda *a, **kw: pytest.fail('preflight fitted'))
    path, _, plan = freeze(inputs, tmp_path)
    assert plan['synthetic'] is True
    assert len(plan['cells']) == 5
    assert set(plan['configurations']) == {c['name'] for c in plan['cells']}
    assert (path/'public-manifest.json').is_file()


def test_projection_and_patient_whole_skew_bind_actual_workload(inputs):
    cell = small_cells()[0] | {'width': 8, 'alpha': .1}
    cfg = pilot.configuration(cell, inputs['settings'], inputs['bundle'], inputs['bundle_sha256'])
    workload = pilot.PilotWorkload(pilot.resolve(cfg), cell)
    assert workload.feat_dim == 8
    assert workload.train_x.shape == (8, 8)
    assert all(len(part) for part in workload.parts)
    owners = {}
    for cid, part in enumerate(workload.parts):
        for index in part:
            patient = workload.train_patient_ids[index]
            assert patient not in owners or owners[patient] == cid
            owners[patient] = cid
    assert workload.info['pilot']['alpha'] == .1
    assert not any(torch.count_nonzero(p) for p in workload.build_model(71).parameters())


def test_generated_pilot_runs_true_fl_and_scalar_private_erm(inputs, tmp_path):
    path, pin, _ = freeze(inputs, tmp_path)
    output = tmp_path/'pilot-output'
    result = pilot.execute(path, pin, output)
    assert result['status'] == 'complete'
    assert len(result['runs']) == 5
    assert all(r['status'] == 'completed' for r in result['runs'])
    assert all(c['max_parameter_difference'] == 0. for c in result['backend_comparisons'])
    erm = next(r for r in result['runs'] if r['kind'] == 'erm')
    record = json.loads((output/erm['record']).read_text())
    assert record['accounting']['releases'] == 1
    assert record['accounting']['exact_epsilon'] <= 4. + 1e-6
    assert len(record['released_head']) == 513
    for r in result['runs']:
        assert hashlib.sha256((output/r['record']).read_bytes()).hexdigest() == r['sha256']


def test_changed_snapshot_is_rejected_before_training(inputs, tmp_path, monkeypatch):
    path, pin, plan = freeze(inputs, tmp_path)
    name = next(iter(plan['source_sha256']))
    (path/'sources'/name).write_text('changed')
    monkeypatch.setattr(pilot, 'run', lambda *a, **kw: pytest.fail('tampered plan fitted'))
    with pytest.raises(ValueError, match='snapshot'):
        pilot.execute(path, pin, tmp_path/'refused')


def test_failed_cell_preserves_rounds_and_accounts_for_all_planned_cells(inputs, tmp_path, monkeypatch):
    path, pin, _ = freeze(inputs, tmp_path)
    real, count = pilot.run, []
    def refused(cfg, **kwargs):
        record = real(cfg, **kwargs)
        count.append(cfg['name'])
        if len(count) == 1:
            record['meta']['refusals'] = ['generated refusal']
        return record
    monkeypatch.setattr(pilot, 'run', refused)
    output = tmp_path/'failed'
    result = pilot.execute(path, pin, output)
    assert result['status'] == 'incomplete'
    assert len(result['runs']) == 5
    failed = next(r for r in result['runs'] if r['status'] == 'error')
    raw = json.loads((output/failed['record']).read_text())
    assert raw['rounds'] and raw['meta']['finalHead']
    assert raw['pilot_failure']['error_type'] == 'ValueError'
    assert len(count) == 1
    assert all(row['status'] == 'not_run' for row in result['runs'][1:])
