"""Fixed validation-only v2 optimization budget with multiple-metric eligibility."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from fedlearn.simulation.config import config_hash, resolve
from fedlearn.simulation.nih_cache import json_bytes, load_nih_cache
from benchmarks.nih_validation_pilot import _checkpoint, summarize_record as _summarize

LEARNING_RATES = (0.001, 0.003, 0.01)
DOCUMENT = 'docs/artifacts/capstone/nih-optimization-pilot-v2-protocol.md'
PER_CELL_SECONDS, TOTAL_SECONDS = 300, 1800
FIXED_REPORT_SHA256 = '6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd'

CHILD_BOOTSTRAP = '''
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root/'framework/src'), str(root/'framework')]
from benchmarks.nih_optimization_pilot import verify_imports
import fedlearn.simulation.engine as engine
identity = json.loads(Path(sys.argv[4]).read_text())['source_identity']
verify_imports(root, identity['files'])
engine.main([sys.argv[2], '--out', sys.argv[3]])
proof = verify_imports(root, identity['files'])
Path(sys.argv[3], 'executed-source.json').write_text(json.dumps(proof, sort_keys=True, indent=2)+'\\n')
'''


def child_command(root, config, output, protocol):
    return [sys.executable, '-I', '-c', CHILD_BOOTSTRAP, str(root), str(config), str(output), str(protocol)]


def verify_imports(root, fingerprints):
    """Reject foreign helper origins and bind loaded local modules to pinned bytes."""
    root = Path(root).resolve()
    files = {}
    for name, module in list(sys.modules.items()):
        if name not in ('fedlearn', 'benchmarks') and not name.startswith(('fedlearn.', 'benchmarks.')):
            continue
        location = getattr(module, '__file__', None)
        if location is None:
            continue
        path = Path(location).resolve()
        if not path.is_relative_to(root/'framework'):
            raise ValueError(f'imported source location outside pinned checkout: {name}')
        key = path.relative_to(root).as_posix()
        files[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(fingerprints.get(key) != digest for key, digest in files.items()):
        raise ValueError('imported source bytes differ from pinned fingerprints')
    return {'files': files}


def _config(cache, pin, clients, lr, *, name=None, seed=20261019, clip=None, rule='mean'):
    return resolve({'name': name or f'clients{clients}_lr_{lr:g}', 'seed': seed,
        'workload': {'dataset': 'nih', 'features_path': str(cache), 'features_report_sha256': pin,
                     'evaluation_split': 'validation', 'balance': False, 'allow_random_features': False,
                     'auc_score': 'margin_v2', 'head': {'type': 'linear'}},
        'federation': {'num_clients': clients, 'rounds': 20, 'partition': {'type': 'patient_iid'}},
        'client': {'local_epochs': 1, 'lr': lr, 'batch_size': 256},
        'server': {'rule': rule, 'clip_norm': clip, 'rule_params': {'trim_ratio': 0.2}},
        'privacy': {'mode': 'none'}, 'behaviors': [], 'eval': {'per_client_every': 0}})


def initial_configs(cache, pin):
    return [_config(cache, pin, clients, lr) for clients in (10, 80) for lr in LEARNING_RATES]


def followup_configs(cache, pin, lr):
    if lr not in LEARNING_RATES:
        raise ValueError('LR outside prospective budget')
    return [*[_config(cache, pin, 80, lr, name=f'clients80_repeat_{seed}', seed=seed)
              for seed in (20261020, 20261021)],
            _config(cache, pin, 80, lr, name='clients80_clip_mean', clip=0.4),
            _config(cache, pin, 80, lr, name='clients80_clip_trimmed', clip=0.4, rule='trimmed_mean')]


def eligible(metrics):
    return (metrics['auc'] >= 0.68 and metrics['balanced_accuracy'] >= 0.60
            and metrics['loss'] <= 0.65 and metrics['ece'] <= 0.10)


def summarize_record(record):
    summary = _summarize(record)
    if (record['meta']['config']['workload']['auc_score'] != 'margin_v2'
            or summary['workload'].get('auc_score') != 'margin_v2'):
        raise ValueError('v2 requires versioned margin scoring')
    for line in record['rounds']:
        value = line.get('auc_probability')
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('missing/invalid probability diagnostic')
    summary['final']['auc_probability'] = record['rounds'][-1]['auc_probability']
    summary['eligible'] = eligible(summary['final'])
    return summary


def select_learning_rate(records):
    if len(records) != 6:
        raise ValueError('all six initial cases must qualify')
    summaries = [summarize_record(r) for r in records]
    wl = records[0]['meta']['config']['workload']
    expected = {c['name']: c for c in initial_configs(wl['features_path'], wl['features_report_sha256'])}
    actual = {r['meta']['config']['name']: r['meta']['config'] for r in records}
    if len(actual) != 6 or actual != expected:
        raise ValueError('initial configs differ from fixed recipes')
    candidates = [s for s in summaries if s['clients'] == 80 and s['eligible']]
    return min(candidates, key=lambda s: (s['final']['loss'], -s['final']['auc'], s['lr']))['lr'] if candidates else None


def source_identity(root):
    """Bind committed code and protocol; refuse tracked edits and new source files."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    if git('status', '--porcelain', '--untracked-files=normal').strip():
        raise ValueError('pilot requires a clean tracked/source tree')
    paths = git('ls-files', '-z', '--', 'framework/src', 'framework/benchmarks', DOCUMENT).split(b'\0')
    return {'git_commit': git('rev-parse', 'HEAD').decode().strip(),
            'files': {os.fsdecode(p): hashlib.sha256((Path(root)/os.fsdecode(p)).read_bytes()).hexdigest()
                      for p in paths if p}}


def run_pilot(cache, pin, output):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    cache, output = Path(cache).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('refusing existing pilot output')
    if output.resolve().is_relative_to(cache):
        raise ValueError('output must be outside the immutable cache')
    if pin != FIXED_REPORT_SHA256:
        raise ValueError('real v2 pilot requires the fixed protocol cache digest')
    identity = source_identity(root)
    verify_imports(root, identity['files'])
    if not load_nih_cache(cache, expected_report_sha256=pin)['provenance']['pretrained']:
        raise ValueError('real pilot requires pretrained features')
    output = output.resolve()
    output.mkdir(parents=True)
    for directory in ('configs', 'logs', 'runs'):
        (output/directory).mkdir()
    protocol = {'pilot_version': 'nih-optimization-pilot-v2', 'source_identity': identity,
                'protocol_document_sha256': hashlib.sha256((root/DOCUMENT).read_bytes()).hexdigest(),
                'initial_configs': initial_configs(cache, pin), 'official_test_evaluated': False,
                'per_cell_seconds_cap': PER_CELL_SECONDS, 'total_seconds_cap': TOTAL_SECONDS,
                'conditional_recipes_by_selected_lr': {str(lr): followup_configs(cache, pin, lr) for lr in LEARNING_RATES}}
    (output/'protocol.json').write_bytes(json_bytes(protocol))
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    cases, records = [], []
    integrity_failed = False

    def execute(config):
        nonlocal integrity_failed
        name = config['name']
        config_path = output/'configs'/f'{name}.json'
        config_path.write_bytes(json_bytes(config))
        remaining = TOTAL_SECONDS - (time.monotonic()-started)
        if remaining <= 0 or integrity_failed:
            return {'name': name, 'status': 'not_run', 'reason': 'budget_or_source_integrity'}, None
        before = time.monotonic()
        try:
            if source_identity(root) != identity:
                integrity_failed = True
                raise ValueError('executed source identity changed')
            run_dir = output/'runs'/name
            with (output/'logs'/f'{name}.log').open('xb') as log:
                subprocess.run(child_command(root, config_path, run_dir, output/'protocol.json'),
                               cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               timeout=min(remaining, PER_CELL_SECONDS), check=True)
            if source_identity(root) != identity:
                integrity_failed = True
                raise ValueError('executed source identity changed')
            entries = json.loads((run_dir/'index.json').read_text())['runs']
            if len(entries) != 1 or entries[0]['status'] != 'ok':
                raise ValueError('child did not complete exactly one case')
            filename = entries[0]['file']
            if Path(filename).name != filename:
                raise ValueError('invalid child filename')
            path = run_dir/filename
            raw = path.read_bytes()
            record = json.loads(raw)
            meta = record['meta']
            if (meta['config'] != config or meta['provenance']['git_commit'] != identity['git_commit']
                    or meta['config_hash'] != config_hash(config)):
                raise ValueError('config/source identity mismatch')
            summary = summarize_record(record)
            return {'name': name, 'status': 'ok', 'elapsed_seconds': time.monotonic()-before,
                    'record_file': path.relative_to(output).as_posix(), 'record_sha256': hashlib.sha256(raw).hexdigest(),
                    'summary': summary}, record
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            # A dirty tree can raise inside source_identity before dictionary comparison.
            try:
                integrity_failed = integrity_failed or source_identity(root) != identity
            except (OSError, ValueError, subprocess.SubprocessError):
                integrity_failed = True
            return {'name': name, 'status': 'timeout' if isinstance(error, subprocess.TimeoutExpired) else 'failed',
                    'error': str(error), 'elapsed_seconds': time.monotonic()-before}, None

    def retain(config, *, skip=None):
        case, record = execute(config) if skip is None else ({'name': config['name'], 'status': 'not_run', 'reason': skip}, None)
        cases.append(case)
        _checkpoint(output/'index.json', {'cases': cases})
        print(json.dumps({'name': case['name'], 'status': case['status']}), flush=True)
        return record

    for config in initial_configs(cache, pin):
        record = retain(config)
        if record is not None:
            records.append(record)
    selected = select_learning_rate(records) if len(records) == 6 and not integrity_failed else None
    for config in followup_configs(cache, pin, selected or LEARNING_RATES[0]):
        retain(config, skip=None if selected is not None else 'initial_qualification_or_eligibility_failed')
    complete = all(c['status'] == 'ok' for c in cases)
    core = [c['summary']['eligible'] for c in cases if c['status'] == 'ok'
            and (c['name'].startswith('clients80_repeat_') or
                 (c['name'].startswith('clients80_lr_') and c['summary']['lr'] == selected))]
    result = {'pilot_version': protocol['pilot_version'], 'source_identity': identity,
              'protocol_document_sha256': protocol['protocol_document_sha256'], 'features_report_sha256': pin,
              'selected_lr': selected, 'complete': complete, 'source_integrity_failed': integrity_failed,
              'core_optimization_gate_passed': len(core) == 3 and all(core) and not integrity_failed,
              'elapsed_seconds': time.monotonic()-started, 'cases': cases, 'official_test_evaluated': False}
    (output/'summary.json').write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--expected-report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_pilot(args.cache, args.expected_report_sha256, args.output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'source_identity')}))
    if not result['complete']:
        parser.exit(2, 'pilot not complete; retain failed/unrun records\n')


if __name__ == '__main__':
    main()
