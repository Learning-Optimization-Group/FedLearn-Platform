"""Full-schedule matched mechanism recipes and failure-preserving execution."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks import nih_mechanism_qualification as probe
from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import config_hash, resolve
from tests.test_capstone_nih_features import template
from tests.test_nih_head_cost_probe import _record as cost_record

PIN = '6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'


def _record(cfg):
    rec = cost_record({**cfg, 'name': 'linear'})
    meta = rec['meta']; meta['config'] = cfg; meta['config_hash'] = config_hash(cfg)
    meta['server_clip_norm'] = .05
    central = cfg['privacy']['mode'] == 'central'
    z = required_noise_multiplier(4., 1., 20, 1e-5) if central else 0.
    meta.update(privacy_mode=cfg['privacy']['mode'], noise_multiplier=z,
                accounted_epsilon=get_epsilon(compute_rdp(1., z, 20), 1e-5)[0] if central else None,
                behavior_assignment=BehaviorPlan(cfg['behaviors'], 80, 20262022, 2).assignment())
    meta['core_qualification']['privacy_mode'] = cfg['privacy']['mode']
    base = rec['rounds'][0]; rec['rounds'] = []
    sensitivity = .1 / (48 if cfg['server']['rule'] == 'trimmed_mean' else 80)
    for rnd in range(1, 21):
        line = copy.deepcopy(base); line['round'] = rnd
        line['privacy'] = {'sensitivity': sensitivity, 'noise_std': z*sensitivity,
                           'epsilon_spent': get_epsilon(compute_rdp(1., z, rnd), 1e-5)[0] if central else None}
        rec['rounds'].append(line)
    return rec


def test_twelve_recipes_enforce_matched_full_schedule_and_permitted_attacks():
    recipes = probe.configs('cache', PIN)
    assert [c['name'] for c in recipes] == [
        f'{condition}_{rule}_{mode}' for condition in ('clean', 'sign_flip', 'label_flip')
        for rule in ('mean', 'trimmed_mean') for mode in ('none', 'central')]
    for i, c in enumerate(recipes):
        assert c['seed'] == 20261022
        assert c['seeds'] == {'partition': 20261022, 'init': 20261022, 'client': 20261022,
                              'attack': 20262022, 'noise': 20263022}
        assert c['federation']['num_clients'] == c['federation']['clients_per_round'] == 80
        assert c['federation']['rounds'] == 20 and c['federation']['partition']['type'] == 'patient_iid'
        assert c['client']['local_epochs'] == 7 and c['client']['lr'] == .003 and c['client']['batch_size'] == 256
        assert c['server']['clip_norm'] == .05 and c['server']['rule_params']['trim_ratio'] == .2
        assert c['privacy'] == {'mode': ('none', 'central')[i % 2], 'epsilon': 4., 'delta': 1e-5, 'adjacency': 'replace_one'}
        assert c['workload']['evaluation_split'] == 'validation' and c['workload']['auc_score'] == 'margin_v2'
        assert not c['workload']['balance'] and not c['workload']['allow_random_features']
        assert c['behaviors'] == ([] if i < 4 else
            [{'type': 'sign_flip_scale', 'fraction': .2, 'scale': -4.}] if i < 8 else
            [{'type': 'label_flip', 'fraction': .2}])
    plans = [BehaviorPlan(c['behaviors'], 80, c['seeds']['attack'], 2) for c in recipes]
    assert all(p.assignment() == plans[4].assignment() for p in plans[4:8])
    assert all(p.assignment() == plans[8].assignment() for p in plans[8:])
    assert len(plans[4].malicious_ids()) == len(plans[8].malicious_ids()) == 16


def test_parent_requalifies_accounting_and_retains_poor_finite_results():
    for cfg in probe.configs('cache', PIN):
        summary = probe.summarize_record(_record(cfg), cfg, 'test-commit')
        assert summary['final']['auc'] == .4 and summary['final']['loss'] == 8.
        assert summary['parameters'] == 1026 and summary['federation_loop_seconds'] == 2.


@pytest.mark.parametrize('mutation', ['noise', 'assignment', 'partial', 'marker', 'source', 'rss', 'threads', 'device'])
def test_parent_refuses_unqualified_mechanisms(mutation):
    cfg = probe.configs('cache', PIN)[7]; rec = _record(cfg)
    if mutation == 'noise': rec['rounds'][0]['privacy']['noise_std'] *= 2
    elif mutation == 'assignment': rec['meta']['behavior_assignment'] = {}
    elif mutation == 'partial': rec['rounds'][0]['participants']['accepted'] = 79
    elif mutation == 'marker': rec['meta']['core_qualification']['research_record_dp_safe'] = True
    elif mutation == 'source': rec['meta']['provenance']['git_commit'] = 'other'
    elif mutation == 'rss': rec['meta']['peak_rss_bytes'] = 2**31+1
    elif mutation == 'threads': rec['meta']['workload']['torch_num_threads'] = 8
    elif mutation == 'device': rec['meta']['workload']['training_device'] = 'mps'
    with pytest.raises(ValueError): probe.summarize_record(rec, cfg, 'test-commit')


@pytest.mark.parametrize('change', ['test', 'oracle', 'clip'])
def test_child_refuses_nonprotocol_recipe_before_core(tmp_path, monkeypatch, change):
    cfg = probe.configs('cache', PIN)[0]
    if change == 'test': cfg['workload']['evaluation_split'] = 'test'
    elif change == 'oracle': cfg['behaviors'] = [{'type': 'ipm', 'fraction': .2, 'epsilon': 4.}]
    else: cfg['server']['clip_norm'] = .4
    monkeypatch.setattr(probe, 'verify_imports', lambda *a: {'files': {}})
    monkeypatch.setattr(probe, 'run_core', lambda *a, **k: pytest.fail('core reached'))
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    protocol = tmp_path/'protocol.json'; protocol.write_text(json.dumps({'source_identity': {'files': {}}}))
    out = tmp_path/'child'
    assert probe.run_child(Path(probe.__file__).resolve().parents[2], config, out, protocol) == 2
    assert not (out/'record.json').exists()
    assert json.loads((out/'execution.json').read_text())['status'] == 'failed'


def _orchestrate(tmp_path, monkeypatch, outcome):
    root = tmp_path/'repo'; doc = root/probe.DOCUMENT; doc.parent.mkdir(parents=True); doc.write_text('frozen')
    monkeypatch.setattr(probe, '__file__', str(root/'framework/benchmarks/nih_mechanism_qualification.py'))
    identity = {'git_commit': 'test-commit', 'files': {k: k+'-hash' for k in probe.PROOF_REQUIRED}}
    state = {'now': 0., 'calls': 0, 'mutated': False}
    monkeypatch.setattr(probe, 'source_identity', lambda _: {**identity, 'git_commit': 'changed'} if state['mutated'] else identity)
    monkeypatch.setattr(probe, 'verify_imports', lambda *a: {'files': identity['files']})
    monkeypatch.setattr(probe, 'load_nih_cache', lambda *a, **k: {'provenance': {'pretrained': True}})
    monkeypatch.setattr(probe.time, 'monotonic', lambda: state['now'])
    if outcome == 'budget': monkeypatch.setattr(probe, 'TOTAL_SECONDS', 0)
    def launch(command, **kw):
        state['calls'] += 1
        protocol = json.loads((tmp_path/'out/protocol.json').read_text())
        assert len(json.loads((tmp_path/'out/index.json').read_text())['cases']) == 12
        assert len(protocol['configs']) == len(protocol['config_hashes']) == 12
        for c in protocol['configs']:
            assert protocol['config_hashes'][c['name']] == hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        cfg = json.loads(Path(command[-3]).read_text()); directory = Path(command[-2]); directory.mkdir()
        state['now'] += 180. if outcome == 'exhaustion' else 3.
        for name, phase in [('executed-source-start.json', 'before_engine'), ('executed-source.json', 'after_engine_return')]:
            if outcome == 'proof' and state['calls'] == 1 and phase == 'after_engine_return': continue
            (directory/name).write_text(json.dumps({'phase': phase, 'files': identity['files']}))
        rec = _record(cfg)
        if outcome == 'partial' and state['calls'] == 1: rec['rounds'][0]['participants']['accepted'] = 79
        (directory/'record.json').write_text(json.dumps(rec))
        (directory/'execution.json').write_text(json.dumps({'status': 'ok', 'record_encoding': 'json', 'core_execution_seconds': 2.5}))
        if state['calls'] == 1:
            if outcome == 'mutation': state['mutated'] = True
            if outcome == 'error': raise subprocess.CalledProcessError(2, command)
            if outcome == 'timeout': raise subprocess.TimeoutExpired(command, kw['timeout'])
        if outcome == 'exhaustion': raise subprocess.TimeoutExpired(command, kw['timeout'])
    monkeypatch.setattr(probe.subprocess, 'run', launch)
    result = probe.run_qualification(tmp_path/'cache', PIN, tmp_path/'out')
    assert len(result['cases']) == 12
    assert json.loads((tmp_path/'out/index.json').read_text())['cases'] == result['cases']
    if outcome == 'success':
        assert result['complete'] and state['calls'] == 12 and result['child_execution_seconds'] == 36.
    elif outcome == 'budget':
        assert state['calls'] == 0 and all(c['status'] == 'not_run' for c in result['cases'])
    elif outcome == 'exhaustion':
        assert state['calls'] == 10 and result['child_execution_seconds'] == 1800.
        assert [c['status'] for c in result['cases']] == ['timeout']*10+['not_run']*2
    elif outcome in ('mutation', 'proof'):
        assert state['calls'] == 1 and result['source_integrity_failed']
        assert all(c['status'] == 'not_run' for c in result['cases'][1:])
    else:
        assert state['calls'] == 12 and not result['complete']
        assert result['cases'][0]['status'] == ('timeout' if outcome == 'timeout' else 'failed')
        assert 'record.json' in result['cases'][0]['artifacts']
        assert all(c['status'] == 'qualified' for c in result['cases'][1:])


@pytest.mark.parametrize('outcome', ['success', 'partial', 'error', 'timeout', 'budget', 'exhaustion', 'mutation', 'proof'])
def test_persistent_twelve_case_execution_and_failure_contract(tmp_path, monkeypatch, outcome):
    _orchestrate(tmp_path, monkeypatch, outcome)


def test_invalid_pin_and_existing_output_preserve_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'load_nih_cache', lambda *a, **k: pytest.fail('cache reached'))
    with pytest.raises(ValueError): probe.run_qualification(tmp_path/'cache', '0'*64, tmp_path/'out')
    (tmp_path/'out').mkdir(); (tmp_path/'out/sentinel').write_text('preserve')
    with pytest.raises(FileExistsError): probe.run_qualification(tmp_path/'cache', PIN, tmp_path/'out')
    assert (tmp_path/'out/sentinel').read_text() == 'preserve'


def test_real_small_core_dp_child_persists_complete_qualified_record(template, tmp_path, monkeypatch):
    cache = template[1]; pin = hashlib.sha256((cache/'report.json').read_bytes()).hexdigest()
    cfg = resolve({'seed': 42, 'workload': {'dataset': 'nih', 'features_path': str(cache), 'balance': False,
        'features_report_sha256': pin, 'evaluation_split': 'validation', 'allow_random_features': True},
        'federation': {'num_clients': 3, 'rounds': 2, 'partition': {'type': 'patient_iid'}},
        'client': {'local_epochs': 1, 'lr': .001}, 'server': {'clip_norm': .05, 'rule': 'trimmed_mean'},
        'privacy': {'mode': 'central', 'epsilon': 4.}, 'behaviors': [{'type': 'label_flip', 'fraction': .34}]})
    # Explicit synthetic feature control; production fixed-recipe refusals are tested separately.
    monkeypatch.setattr(probe, 'configs', lambda *a: [cfg])
    monkeypatch.setattr(probe, 'FIXED_REPORT_SHA256', pin)
    monkeypatch.setattr(probe, 'verify_imports', lambda *a: {'files': {}})
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    protocol = tmp_path/'protocol.json'; protocol.write_text(json.dumps({'source_identity': {'files': {}}}))
    out = tmp_path/'child'
    assert probe.run_child(Path(probe.__file__).resolve().parents[2], config, out, protocol) == 0
    rec = json.loads((out/'record.json').read_text())
    assert rec['meta']['core_qualification']['feature_control'] is True
    assert rec['meta']['core_qualification']['privacy_mode'] == 'central'
    assert len(rec['rounds']) == 2 and rec['rounds'][-1]['privacy']['epsilon_spent'] <= 4.000001
    assert json.loads((out/'execution.json').read_text())['core_execution_seconds'] >= rec['meta']['wall_seconds']


def test_actual_interrupted_child_keeps_start_proof_without_completion(tmp_path):
    root = Path(probe.__file__).resolve().parents[2]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ("framework/src", "framework/benchmarks") for p in (root/folder).rglob("*.py")}
    protocol = tmp_path/"protocol.json"; protocol.write_text(json.dumps({"source_identity": {"files": files}}))
    cfg = probe.configs("missing-cache", PIN)[0]
    config = tmp_path/"config.json"; config.write_text(json.dumps(cfg)); out = tmp_path/"child"
    # Only the slow engine boundary is replaced; actual run_child/proof/checkpoint code runs.
    bootstrap = """
import sys,time
from pathlib import Path
root=Path(sys.argv[1]);sys.path[:0]=[str(root/'framework/src'),str(root/'framework')]
from benchmarks import nih_mechanism_qualification as probe
probe.run_core=lambda *a,**k: time.sleep(30)
raise SystemExit(probe.run_child(root,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4])))
"""
    with (tmp_path/"child.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-I", "-c", bootstrap, str(root), str(config), str(out), str(protocol)],
                                 env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"), stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while not (out/"executed-source-start.json").exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert (out/"executed-source-start.json").exists(), (tmp_path/"child.log").read_text()
        finally:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
    assert not (out/"executed-source.json").exists() and not (out/"execution.json").exists()


def test_child_retains_complete_nonfinite_failed_record(tmp_path, monkeypatch):
    from fedlearn.simulation.core_qualification import CoreQualificationError
    cfg = probe.configs('cache', PIN)[1]; record = _record(cfg)
    record['rounds'][0]['loss'] = float('nan')
    monkeypatch.setattr(probe, 'verify_imports', lambda *a: {'files': {}})
    def refuse(*a, **k): raise CoreQualificationError('invalid utility', record=record)
    monkeypatch.setattr(probe, 'run_core', refuse)
    config = tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    protocol = tmp_path/'protocol.json'; protocol.write_text(json.dumps({'source_identity': {'files': {}}}))
    out = tmp_path/'child'
    assert probe.run_child(Path(probe.__file__).resolve().parents[2], config, out, protocol) == 2
    saved = json.loads((out/'record.json').read_text())
    assert saved['encoding'] == 'nonfinite_tagged_v1'
    assert saved['record']['rounds'][0]['loss'] == {'__nonfinite_float__': 'nan'}
    assert len(saved['record']['rounds']) == 20
    assert json.loads((out/'executed-source.json').read_text())['phase'] == 'after_engine_error'


def test_actual_cli_invalid_pin_refuses_without_output(tmp_path):
    root = Path(probe.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-m', 'benchmarks.nih_mechanism_qualification',
                            '--cache', str(tmp_path/'cache'), '--expected-report-sha256', '0'*64,
                            '--output', str(tmp_path/'out')], cwd=root,
                            env=dict(os.environ, PYTHONPATH='framework/src:framework'), capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and not (tmp_path/'out').exists()
    assert 'fixed protocol cache pin' in result.stderr
