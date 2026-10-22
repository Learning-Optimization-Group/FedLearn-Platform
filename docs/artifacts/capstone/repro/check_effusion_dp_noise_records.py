"""Check every central-DP round of the Effusion run records: recorded noise std == z * replace-one sensitivity.

The sensitivity is 2C/n for the mean and 2C/(n - 2k) for a trimmed mean with k = floor(trim_ratio * n),
where n is the round's accepted cohort. Prints a JSON summary; exits 1 on any mismatch.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def expected_std(meta, line):
    server = meta['config']['server']
    n = int(line['participants']['accepted'])
    k = math.floor(float(server['rule_params']['trim_ratio']) * n) if server['rule'] == 'trimmed_mean' else 0
    return float(meta['noise_multiplier']) * 2.0 * float(server['clip_norm']) / (n - 2 * k)


def check(directories, tolerance=1e-9):
    out = {'rounds': 0, 'runs': 0, 'mismatches': [], 'worstRelativeError': 0.0, 'perStudy': {}}
    for directory in directories:
        for path in sorted(Path(directory).glob('*.json')):
            if path.name == 'summary.json':
                continue
            record = json.loads(path.read_text())
            meta = record['meta']
            if meta.get('privacy_mode') != 'central':
                continue
            out['runs'] += 1
            for line in record['rounds']:
                want, got = expected_std(meta, line), float(line['privacy']['noise_std'])
                rel = abs(got - want) / want
                out['worstRelativeError'] = max(out['worstRelativeError'], rel)
                out['rounds'] += 1
                out['perStudy'][Path(directory).name] = out['perStudy'].get(Path(directory).name, 0) + 1
                if rel > tolerance:
                    out['mismatches'].append({'file': path.name, 'round': line['round'], 'want': want, 'got': got})
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    result = check(parser.parse_args().directories)
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(1 if result['mismatches'] else 0)
