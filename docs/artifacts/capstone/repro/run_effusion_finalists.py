"""Freeze or execute the finalist design after verified, completed diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_effusion_pilots import _load_stage
from finalist_design import cells
import run_effusion_pilots as pilot


def verify_parents(screen_dir, screen_sha256, bucket_dir, bucket_sha256):
    screen_dir, bucket_dir = Path(screen_dir), Path(bucket_dir)
    screen_reference = pilot.pinned_screen_reference(screen_dir/'summary.json', screen_sha256)
    screen = _load_stage(screen_dir, screen_sha256, 'screen')
    buckets = _load_stage(bucket_dir, bucket_sha256, 'buckets', screen_sha256)
    if any(r['status'] != 'completed' for stage in (screen, buckets) for r in stage['rows']):
        raise ValueError('finalist parents must be complete')
    bucket_summary = buckets['summary']
    if not bucket_summary.get('checkpoint_manifest'):
        raise ValueError('bucket parent lacks its committed manifest')
    pilot._checkpoint(buckets['plan'], bucket_summary['checkpoint_commit'],
                      pilot.ROOT/bucket_summary['checkpoint_manifest'])
    if bucket_summary['reference']['plan_sha256'] != screen['summary']['plan_sha256']:
        raise ValueError('bucket parent is bound to another screen')
    rows = screen['rows']
    def mean(tag, private):
        selected = [r['metrics']['auroc'] for r in rows if r['tag'] == tag and r['private'] == private and r['backend'] == 'byzfl']
        if len(selected) != 2:
            raise ValueError('candidate selection lacks two complete seeds')
        return float(np.mean(selected))
    width = max((64, 128, 256), key=lambda k: mean(f'mean_k{k}', True))
    eligible = [ratio for ratio in (.25, 1., 4.)
                if mean('mean', False)-mean(f'center_r{ratio:g}_clean', False) <= .01]
    ratio = max(eligible, key=lambda r: mean(f'center_r{r:g}_owned_pgd', False))
    if width != 128 or ratio != .25:
        raise ValueError('fixed finalists differ from the recorded diagnostic selection')
    return {'screen_summary_sha256': screen_sha256, 'bucket_summary_sha256': bucket_sha256,
            'screen_plan_sha256': screen['summary']['plan_sha256'],
            'bucket_plan_sha256': buckets['summary']['plan_sha256'],
            'screen_checkpoint_commit': screen_reference['checkpoint_commit'],
            'bucket_checkpoint_commit': bucket_summary['checkpoint_commit'],
            'phase3_verification_sha256': pilot.sha(pilot.PHASE3),
            'selected_width': width, 'selected_center_tau': .05*ratio,
            'selection_scope': 'exploratory development policy selection; fresh seeds do not create a new evaluation cohort'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prepare = sub.add_parser('preflight')
    for name in ('bundle', 'screen-dir', 'bucket-dir', 'output'):
        prepare.add_argument('--'+name, type=Path, required=True)
    prepare.add_argument('--screen-sha256', required=True)
    prepare.add_argument('--bucket-sha256', required=True)
    launch = sub.add_parser('execute')
    for name in ('plan-dir', 'output', 'checkpoint-manifest'):
        launch.add_argument('--'+name, type=Path, required=True)
    launch.add_argument('--plan-sha256', required=True)
    launch.add_argument('--checkpoint-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'preflight':
        reference = verify_parents(args.screen_dir, args.screen_sha256, args.bucket_dir, args.bucket_sha256)
        pilot.preflight(args.bundle, pilot.BUNDLE_PIN, args.output, cells=cells(), stage='finalists', reference=reference)
        print(json.dumps({'status': 'preflight_complete', 'cells': len(cells()), 'model_fits': 0,
                          'plan_sha256': pilot.sha(args.output/'plan.json'),
                          'public_manifest_sha256': pilot.sha(args.output/'public-manifest.json')}))
        return 0
    result = pilot.execute(args.plan_dir, args.plan_sha256, args.output,
                           checkpoint_commit=args.checkpoint_commit, checkpoint_manifest=args.checkpoint_manifest)
    print(json.dumps({'status': result['status'], 'cells': len(result['runs']),
                      'completed': sum(r['status'] == 'completed' for r in result['runs'])}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
