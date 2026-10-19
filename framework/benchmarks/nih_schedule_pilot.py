"""Fixed matched-exposure NIH schedule probe; historical v2 remains immutable."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fedlearn.simulation.config import config_hash, resolve
from fedlearn.simulation.nih_cache import json_bytes, load_nih_cache
from benchmarks.nih_validation_pilot import _checkpoint
from benchmarks.nih_optimization_pilot import (
    FIXED_REPORT_SHA256, PER_CELL_SECONDS, TOTAL_SECONDS, _config as _base_config,
    source_identity as _source_identity, summarize_record, verify_imports,
)

DOCUMENT = 'docs/artifacts/capstone/nih-schedule-pilot-v3-protocol.md'
SCHEDULES = (('local7_round20', 7, 20), ('local1_round140', 1, 140))

CHILD_BOOTSTRAP = '''
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root/'framework/src'), str(root/'framework')]
from benchmarks.nih_optimization_pilot import verify_imports
from benchmarks.nih_validation_pilot import _checkpoint
import fedlearn.simulation.engine as engine
identity = json.loads(Path(sys.argv[4]).read_text())['source_identity']
output = Path(sys.argv[3])
output.mkdir(parents=True)
proof = verify_imports(root, identity['files'])
_checkpoint(output/'executed-source-start.json', {**proof, 'phase': 'before_engine'})
engine.main([sys.argv[2], '--out', str(output)])
proof = verify_imports(root, identity['files'])
_checkpoint(output/'executed-source.json', {**proof, 'phase': 'after_engine_return'})
'''


def child_command(root, config, output, protocol):
    return [sys.executable, '-I', '-c', CHILD_BOOTSTRAP, str(root), str(config), str(output), str(protocol)]


def _config(cache, pin, schedule, *, name=None, seed=20261019, clip=None, rule='mean'):
    choices = {name: (epochs, rounds) for name, epochs, rounds in SCHEDULES}
    if schedule not in choices:
        raise ValueError('schedule outside fixed prospective budget')
    epochs, rounds = choices[schedule]
    config = _base_config(cache, pin, 80, .003, name=name or schedule, seed=seed, clip=clip, rule=rule)
    config['client']['local_epochs'], config['federation']['rounds'] = epochs, rounds
    return resolve(config)


def initial_configs(cache, pin):
    return [_config(cache, pin, name) for name, _, _ in SCHEDULES]


def followup_configs(cache, pin, schedule):
    return [*[_config(cache, pin, schedule, name=f'selected_repeat_{seed}', seed=seed)
              for seed in (20261020, 20261021)],
            _config(cache, pin, schedule, name='selected_clip_mean', clip=.4),
            _config(cache, pin, schedule, name='selected_clip_trimmed', clip=.4, rule='trimmed_mean')]


def select_schedule(records):
    if len(records) != 2:
        raise ValueError('both initial cases must qualify before selection')
    summaries = {r['meta']['config']['name']: summarize_record(r) for r in records}
    wl = records[0]['meta']['config']['workload']
    actual = {r['meta']['config']['name']: r['meta']['config'] for r in records}
    expected = {c['name']: c for c in initial_configs(wl['features_path'], wl['features_report_sha256'])}
    if len(actual) != 2 or actual != expected:
        raise ValueError('initial recipes differ from fixed prospective schedules')
    return next((name for name, _, _ in SCHEDULES if summaries[name]['eligible']), None)


def source_identity(root):
    identity = _source_identity(root)
    identity['files'][DOCUMENT] = hashlib.sha256((Path(root)/DOCUMENT).read_bytes()).hexdigest()
    return identity


def run_probe(cache, pin, output):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    cache, output = Path(cache).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('refusing existing schedule-probe output')
    if output.resolve().is_relative_to(cache):
        raise ValueError('output must be outside immutable cache')
    if pin != FIXED_REPORT_SHA256:
        raise ValueError('real v3 probe requires fixed protocol cache digest')
    identity = source_identity(root)
    verify_imports(root, identity['files'])
    if not load_nih_cache(cache, expected_report_sha256=pin)['provenance']['pretrained']:
        raise ValueError('real probe requires pretrained features')
    output = output.resolve()
    output.mkdir(parents=True)
    for name in ('configs', 'logs', 'runs'):
        (output/name).mkdir()
    protocol = {'pilot_version': 'nih-schedule-pilot-v3', 'source_identity': identity,
                'protocol_document_sha256': hashlib.sha256((root/DOCUMENT).read_bytes()).hexdigest(),
                'initial_configs': initial_configs(cache, pin),
                'conditional_recipes_by_schedule': {name: followup_configs(cache, pin, name) for name, _, _ in SCHEDULES},
                'per_cell_seconds_cap': PER_CELL_SECONDS, 'total_seconds_cap': TOTAL_SECONDS,
                'official_test_evaluated': False}
    (output/'protocol.json').write_bytes(json_bytes(protocol))
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    cases, records = [], []
    integrity_failed = False
    spent_child_seconds = 0.0

    def proofs(directory, validate=False):
        found = {}
        for filename, phase in (('executed-source-start.json', 'before_engine'),
                                ('executed-source.json', 'after_engine_return')):
            path = directory/filename
            if path.exists():
                raw = path.read_bytes()
                found[phase] = {'file': path.relative_to(output).as_posix(),
                                'sha256': hashlib.sha256(raw).hexdigest()}
                if validate:
                    proof = json.loads(raw)
                    engine = 'framework/src/fedlearn/simulation/engine.py'
                    if (proof.get('phase') != phase or proof['files'].get(engine) != identity['files'][engine]
                            or any(identity['files'].get(k) != v for k, v in proof['files'].items())):
                        raise ValueError('child executed-source proof mismatch')
        if validate and len(found) != 2:
            raise ValueError('child did not retain both source-verification phases')
        return found

    def execute(config):
        nonlocal integrity_failed, spent_child_seconds
        name = config['name']
        config_path = output/'configs'/f'{name}.json'
        config_path.write_bytes(json_bytes(config))
        if integrity_failed:
            return {'name': name, 'status': 'not_run', 'reason': 'source_integrity'}, None
        before = time.monotonic()
        directory = output/'runs'/name
        child_seconds = 0.0
        try:
            if source_identity(root) != identity:
                integrity_failed = True
                raise ValueError('source identity changed')
            remaining = TOTAL_SECONDS-spent_child_seconds
            if remaining <= 0:
                return {'name': name, 'status': 'not_run', 'reason': 'global_budget_exhausted'}, None
            with (output/'logs'/f'{name}.log').open('xb') as log:
                child_started = time.monotonic()
                try:
                    subprocess.run(child_command(root, config_path, directory, output/'protocol.json'),
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(PER_CELL_SECONDS, remaining), check=True)
                finally:
                    child_seconds = time.monotonic()-child_started
                    spent_child_seconds += child_seconds
            if source_identity(root) != identity:
                integrity_failed = True
                raise ValueError('source identity changed')
            entries = json.loads((directory/'index.json').read_text())['runs']
            if len(entries) != 1 or entries[0]['status'] != 'ok' or Path(entries[0]['file']).name != entries[0]['file']:
                raise ValueError('child did not complete exactly one safe record')
            path = directory/entries[0]['file']
            raw = path.read_bytes()
            record = json.loads(raw)
            meta = record['meta']
            if (meta['config'] != config or meta['config_hash'] != config_hash(config)
                    or meta['provenance']['git_commit'] != identity['git_commit']):
                raise ValueError('config/source identity mismatch')
            source_proofs = proofs(directory, validate=True)
            elapsed = time.monotonic()-before
            return {'name': name, 'status': 'ok', 'elapsed_seconds': elapsed,
                    'child_execution_seconds': child_seconds, 'parent_control_seconds': elapsed-child_seconds,
                    'source_proofs': source_proofs,
                    'record_file': path.relative_to(output).as_posix(), 'record_sha256': hashlib.sha256(raw).hexdigest(),
                    'summary': summarize_record(record)}, record
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            try:
                integrity_failed = integrity_failed or source_identity(root) != identity
            except (OSError, ValueError, subprocess.SubprocessError):
                integrity_failed = True
            elapsed = time.monotonic()-before
            return {'name': name, 'status': 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'failed',
                    'error': str(error), 'elapsed_seconds': elapsed, 'child_execution_seconds': child_seconds,
                    'parent_control_seconds': elapsed-child_seconds, 'source_proofs': proofs(directory)}, None

    def retain(config, skip=None):
        case, record = execute(config) if skip is None else ({'name': config['name'], 'status': 'not_run', 'reason': skip}, None)
        cases.append(case)
        _checkpoint(output/'index.json', {'cases': cases})
        print(json.dumps({'name': case['name'], 'status': case['status']}), flush=True)
        return record

    for config in initial_configs(cache, pin):
        record = retain(config)
        if record is not None:
            records.append(record)
    selected = select_schedule(records) if len(records) == 2 and not integrity_failed else None
    for config in followup_configs(cache, pin, selected or SCHEDULES[0][0]):
        retain(config, None if selected is not None else 'initial_qualification_or_eligibility_failed')
    core = [c['summary']['eligible'] for c in cases if c['status'] == 'ok'
            and (c['name'] == selected or c['name'].startswith('selected_repeat_'))]
    elapsed = time.monotonic()-started
    result = {'pilot_version': protocol['pilot_version'], 'source_identity': identity,
              'protocol_document_sha256': protocol['protocol_document_sha256'], 'features_report_sha256': pin,
              'selected_schedule': selected, 'complete': all(c['status'] == 'ok' for c in cases),
              'source_integrity_failed': integrity_failed,
              'core_optimization_gate_passed': len(core) == 3 and all(core) and not integrity_failed,
              'elapsed_seconds': elapsed, 'child_execution_seconds': spent_child_seconds,
              'parent_control_seconds': elapsed-spent_child_seconds,
              'timing_scope': 'isolated child execution includes startup/cache/guards; parent control measured separately; engine timing in records',
              'cases': cases, 'official_test_evaluated': False}
    (output/'summary.json').write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--expected-report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_probe(args.cache, args.expected_report_sha256, args.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'source_identity')}))
    if not result['complete']:
        parser.exit(2, 'probe incomplete; retain failed/unrun cases\n')


if __name__ == '__main__':
    main()
