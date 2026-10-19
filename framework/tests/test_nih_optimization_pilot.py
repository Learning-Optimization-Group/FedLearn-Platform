"""Prospective v2 budget, multiple-metric gate and fixed execution identity."""
import copy
import subprocess

import pytest

from benchmarks.nih_optimization_pilot import (
    initial_configs, followup_configs, select_learning_rate, eligible, summarize_record, source_identity,
)
from tests.test_nih_validation_pilot import PIN, _record


def records():
    values = [_record(c) for c in initial_configs('cache', PIN)]
    for value in values:
        value['meta']['workload']['auc_score'] = 'margin_v2'
        for line in value['rounds']:
            line['auc_probability'] = line['auc']
    return values


def test_fixed_ten_cell_v2_budget():
    initial = initial_configs('cache', PIN)
    assert [c['federation']['num_clients'] for c in initial] == [10]*3 + [80]*3
    assert [c['client']['lr'] for c in initial] == [0.001, 0.003, 0.01]*2
    follow = followup_configs('cache', PIN, 0.003)
    assert [c['seed'] for c in follow] == [20261020, 20261021, 20261019, 20261019]
    assert [c['server']['clip_norm'] for c in follow] == [None, None, 0.4, 0.4]
    assert follow[-1]['server']['rule'] == 'trimmed_mean'
    for c in initial + follow:
        assert c['federation']['rounds'] == 20 and c['client']['local_epochs'] == 1
        assert c['workload']['auc_score'] == 'margin_v2'
        assert c['workload']['evaluation_split'] == 'validation' and not c['workload']['balance']
        assert c['privacy']['mode'] == 'none' and not c['behaviors']
    with pytest.raises(ValueError):
        followup_configs('cache', PIN, 0.1)


@pytest.mark.parametrize('key,value', [('auc',0.6799),('balanced_accuracy',0.5999),('loss',0.6501),('ece',0.1001)])
def test_each_eligibility_gate_is_necessary(key, value):
    metrics = dict(auc=0.68, balanced_accuracy=0.60, loss=0.65, ece=0.10)
    assert eligible(metrics)
    metrics[key] = value
    assert not eligible(metrics)


def test_selects_final_loss_only_among_eligible_eighty_client_cases():
    values = records()
    values[0]['rounds'][-1]['loss'] = 0.01
    values[3]['rounds'][0]['loss'] = 0.01
    values[3]['rounds'][-1]['auc'] = 0.99
    values[3]['rounds'][-1]['ece'] = 0.11
    values[4]['rounds'][-1]['loss'] = 0.59
    assert select_learning_rate(values) == 0.003
    values[5]['rounds'][-1]['loss'] = 0.59
    values[5]['rounds'][-1]['auc'] = 0.72
    assert select_learning_rate(values) == 0.01
    values[4]['rounds'][-1]['auc'] = 0.72
    assert select_learning_rate(values) == 0.003
    for value in values[3:]: value['rounds'][-1]['ece'] = 0.11
    assert select_learning_rate(values) is None


def test_no_survivor_selection_or_silent_recipe_changes():
    values = records()
    for invalid in (values[:-1], values[:-1]+[values[0]]):
        with pytest.raises(ValueError): select_learning_rate(invalid)
    values[0]['meta']['config']['client']['local_epochs'] = 2
    with pytest.raises(ValueError): select_learning_rate(values)


@pytest.mark.parametrize('mutation', ['version', 'provenance', 'diagnostic'])
def test_qualification_checks_score_identity_and_diagnostic(mutation):
    value = copy.deepcopy(records()[0])
    if mutation == 'version': value['meta']['config']['workload']['auc_score'] = 'probability_v1'
    elif mutation == 'provenance': value['meta']['workload']['auc_score'] = 'probability_v1'
    else: value['rounds'][-1]['auc_probability'] = float('nan')
    with pytest.raises(ValueError): summarize_record(value)


def test_source_identity_refuses_dirty_or_untracked_source(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)], check=True)
    src = tmp_path/'framework/src'
    src.mkdir(parents=True)
    file = src/'example.py'
    file.write_text('x=1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','.'], check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Test','-c','user.email=test@example.com',
                    'commit','-qm','Initial'], check=True)
    first = source_identity(tmp_path)
    assert first['files']['framework/src/example.py'] and first == source_identity(tmp_path)
    file.write_text('x=2\n')
    with pytest.raises(ValueError): source_identity(tmp_path)
    file.write_text('x=1\n')
    (src/'extra.py').write_text('x=3\n')
    with pytest.raises(ValueError): source_identity(tmp_path)


def test_real_runner_refuses_nonprotocol_pin_before_loading_or_creating_output(tmp_path, monkeypatch):
    from benchmarks import nih_optimization_pilot as pilot
    called = []
    monkeypatch.setattr(pilot, 'load_nih_cache', lambda *a, **k: called.append(True))
    with pytest.raises(ValueError, match='fixed protocol cache'):
        pilot.run_pilot(tmp_path/'cache', '1'*64, tmp_path/'output')
    assert not called and not (tmp_path/'output').exists()


def test_parent_rejects_a_helper_imported_from_another_checkout(tmp_path, monkeypatch):
    from types import ModuleType
    import sys
    from pathlib import Path
    from benchmarks import nih_optimization_pilot as pilot
    foreign = ModuleType('fedlearn.foreign')
    foreign.__file__ = str(tmp_path/'foreign.py')
    monkeypatch.setitem(sys.modules, 'fedlearn.foreign', foreign)
    with pytest.raises(ValueError, match='source location'):
        pilot.verify_imports(Path(pilot.__file__).resolve().parents[2], {})


def test_child_uses_pinned_checkout_despite_competing_pythonpath(tmp_path):
    import hashlib
    import json
    import os
    from pathlib import Path
    from benchmarks import nih_optimization_pilot as pilot
    from tests.test_simulation_engine_cli import _features
    root = Path(pilot.__file__).resolve().parents[2]
    competitor = tmp_path/'competitor'
    (competitor/'fedlearn').mkdir(parents=True)
    marker = tmp_path/'wrong-checkout'
    (competitor/'fedlearn/__init__.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError("wrong checkout")\n')
    manifest = {'source_identity': {'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ('framework/src', 'framework/benchmarks') for p in (root/folder).rglob('*.py')}}}
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(manifest))
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'workload': {'features_path': _features(tmp_path)},
        'federation': {'num_clients': 2, 'rounds': 1}, 'client': {'local_epochs': 1}}))
    out = tmp_path/'out'
    completed = subprocess.run(pilot.child_command(root, config, out, protocol), cwd=competitor,
        env=dict(os.environ, PYTHONPATH=str(competitor), OMP_NUM_THREADS='1'), capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert all(r['status']=='ok' for r in json.loads((out/'index.json').read_text())['runs'])
    proof = json.loads((out/'executed-source.json').read_text())
    assert proof['files']['framework/src/fedlearn/simulation/engine.py'] == manifest['source_identity']['files']['framework/src/fedlearn/simulation/engine.py']


@pytest.mark.parametrize('failure', ['child_error', 'timeout', 'mutation', 'budget'])
def test_orchestration_retains_ten_entries_and_never_selects_survivors(tmp_path, monkeypatch, failure):
    import json
    from pathlib import Path
    from benchmarks import nih_optimization_pilot as pilot
    root = tmp_path/'repo'
    document = root/pilot.DOCUMENT
    document.parent.mkdir(parents=True)
    document.write_text('fixed test protocol')
    monkeypatch.setattr(pilot, '__file__', str(root/'framework/benchmarks/nih_optimization_pilot.py'))
    identity = {'git_commit': 'test-commit', 'files': {}}
    state = {'mutated': False, 'calls': 0, 'clock_calls': 0}
    monkeypatch.setattr(pilot, 'source_identity', lambda _: ({**identity, 'git_commit':'changed'} if state['mutated'] else identity))
    monkeypatch.setattr(pilot, 'verify_imports', lambda *a: {'files': {}})
    monkeypatch.setattr(pilot, 'load_nih_cache', lambda *a, **k: {'provenance': {'pretrained': True}})
    if failure == 'budget':
        def clock():
            state['clock_calls'] += 1
            return 0 if state['clock_calls'] == 1 else 1801
        monkeypatch.setattr(pilot.time, 'monotonic', clock)
    def child(command, **kwargs):
        state['calls'] += 1
        if state['calls'] == 1 and failure == 'child_error':
            raise subprocess.CalledProcessError(1, command)
        if state['calls'] == 1 and failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 300)
        if state['calls'] == 1 and failure == 'mutation':
            state['mutated'] = True
            return
        config = json.loads(Path(command[-3]).read_text())
        record = _record(config)
        record['meta']['config_hash'] = pilot.config_hash(config)
        record['meta']['provenance'] = {'git_commit':'test-commit'}
        record['meta']['workload'].update(auc_score='margin_v2', features_report_sha256=pilot.FIXED_REPORT_SHA256)
        for line in record['rounds']: line['auc_probability'] = line['auc']
        directory = Path(command[-2])
        directory.mkdir()
        (directory/'record.json').write_text(json.dumps(record))
        (directory/'index.json').write_text(json.dumps({'runs':[{'status':'ok','file':'record.json'}]}))
    monkeypatch.setattr(pilot.subprocess, 'run', child)
    result = pilot.run_pilot(tmp_path/'cache', pilot.FIXED_REPORT_SHA256, tmp_path/'out')
    assert len(result['cases']) == 10 and result['selected_lr'] is None
    assert not result['complete'] and not result['core_optimization_gate_passed']
    assert all(c['status']=='not_run' for c in result['cases'][6:])
    assert json.loads((tmp_path/'out/index.json').read_text())['cases'] == result['cases']
    if failure == 'mutation':
        assert state['calls'] == 1 and result['source_integrity_failed']
        assert result['cases'][0]['status']=='failed'
    elif failure == 'budget':
        assert state['calls']==0 and all(c['status']=='not_run' for c in result['cases'])
    else:
        assert state['calls']==6 and result['cases'][0]['status']==('timeout' if failure=='timeout' else 'failed')
