"""Build the Simulation Lab's Effusion R0 bundle from pinned inputs: train + selection only, never holdout or test.

Features are the historical R0 arrays standardised with the gate's train-only mean/std (verified against a
recomputation); labels are the protocol's exact-token Effusion rule. The framework writer refuses an existing
output, so a published bundle is never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_effusion_federated as federated  # noqa: E402  (also pins this worktree's fedlearn)

from fedlearn.simulation.effusion_bundle import write_effusion_bundle  # noqa: E402


def main(args):
    import effusion_labels
    import fit_effusion_gate
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise SystemExit('gate hash mismatch')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R0']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R0_mean']) and np.array_equal(std, heads['R0_std'])):
        raise SystemExit('standardiser mismatch')

    def split(name, x):
        rows = labels[name]['rows']
        return {'x': ((x - mean) / std).astype(np.float32), 'y': labels[name]['y'].astype(np.int64),
                'patient_ids': [r['patient_id'] for r in rows], 'image_ids': [r['image_index'] for r in rows]}

    sha = write_effusion_bundle(args.output, split('train', train_x), split('selection', selection_x), provenance={
        **provenance, 'gateSha256': args.gate_sha256, 'representation': 'R0 historical ImageNet ResNet-18 (512)',
        'standardiser': 'gate R0 train-only mean/std (float64), applied then cast to float32',
        'labelRule': 'positive iff exact token Effusion in pipe-split findings',
        'protocol': 'effusion-baseline-protocol-v1 (A6 R0 primary); holdout and official test excluded',
        'builderSha256': federated._sha256_file(HERE / 'build_effusion_bundle.py')})
    print(json.dumps({'status': 'built', 'reportSha256': sha}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256'):
        parser.add_argument(flag, required=True)
    main(parser.parse_args())
