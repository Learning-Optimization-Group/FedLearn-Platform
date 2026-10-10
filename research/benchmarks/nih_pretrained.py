"""Frozen ImageNet ResNet18 NIH baseline on verified, patient-separated packages.

Uses the existing prepared 224-pixel grayscale images, repeated to RGB and
normalized per ImageNet channel. This is a research benchmark, not a mobile recipe.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nih_learning import bind_package, require_disjoint, digest, state_digest, train_control, evaluate

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def imagenet_input(x):
    if x.ndim != 4 or x.shape[1:] != (1, 224, 224) or not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
        raise ValueError('expected finite NIH grayscale tensors normalized by mean=.5/std=.5')
    unit = x * .5 + .5
    return (unit.repeat(1, 3, 1, 1) - x.new_tensor(MEAN)[None, :, None, None]) / x.new_tensor(STD)[None, :, None, None]


def extract_features(backbone, images, *, batch_size=16):
    backbone.eval()
    backbone.requires_grad_(False)
    features = []
    with torch.inference_mode():
        for (x,) in DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False):
            value = backbone(imagenet_input(x))
            if not torch.isfinite(value).all():
                raise ValueError('non-finite pretrained features')
            features.append(value)
    # Training a head requires ordinary tensors rather than inference-mode storage.
    return torch.cat(features).clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    torch.set_num_threads(4)
    import torchvision
    bound = [bind_package(args.packages / (n + '.zip'), split=n, purpose='train' if n != 'validation' else 'eval')
             for n in ('client-0', 'client-1', 'validation')]
    reference = bind_package(args.reference, split='validation', purpose='eval')
    for held in (bound[2], reference):
        groups = bound[:2] + [held]
        require_disjoint([b[2] for b in groups])
        require_disjoint([dict(patients=b[2]['patients'], images=b[2]['prepared_images']) for b in groups])
    if any(b[1]['source_metadata_hashes'] != bound[0][1]['source_metadata_hashes']
           or b[1]['preparation_sha256'] != bound[0][1]['preparation_sha256'] for b in bound + [reference]):
        raise ValueError('different source preparation')
    weight_sha = digest(args.weights.read_bytes())
    # Torchvision's official IMAGENET1K_V1 file is named by this hash prefix.
    if not weight_sha.startswith('f37072fd'):
        raise ValueError('weights are not the pinned official ResNet18 ImageNet V1 checkpoint')
    backbone = torchvision.models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(args.weights, weights_only=True, map_location='cpu'), strict=True)
    backbone.fc = torch.nn.Identity()
    original = state_digest(backbone)
    feature_sets = []
    for b in bound + [reference]:
        feature_sets.append(TensorDataset(extract_features(backbone, b[0].tensors[0]), b[0].tensors[1]))
        print(json.dumps(dict(extracted=len(b[0]))), flush=True)
    assert state_digest(backbone) == original
    training = TensorDataset(*(torch.cat([feature_sets[i].tensors[j] for i in (0, 1)]) for j in (0, 1)))
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        head = torch.nn.Linear(512, 2)
    initial = dict(training=evaluate(head, training, 16), validation=evaluate(head, feature_sets[2], 16),
                   proportional_reference=evaluate(head, feature_sets[3], 16), state_sha256=state_digest(head))
    result = train_control(head, training, feature_sets[2], epochs=10, learning_rate=.001, seed=args.seed, batch_size=16)
    args.out.mkdir(parents=True)
    path = args.out / 'head.npz'
    np.savez(path, **{k: v.detach().numpy() for k, v in head.state_dict().items()})
    reload = torch.nn.Linear(512, 2)
    with np.load(path, allow_pickle=False) as archive:
        reload.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
    measured = evaluate(reload, feature_sets[2], 16)
    assert measured == result['epochs'][-1]['validation'] and state_digest(reload) == result['final_state_sha256']
    report = dict(torch_version=torch.__version__, torchvision_version=torchvision.__version__, torch_num_threads=4,
                  architecture='ResNet18 frozen ImageNet V1 backbone + two-class linear head',
                  weights_sha256=weight_sha, backbone_state_sha256=original, backbone_unchanged=True,
                  transform=dict(size=[224,224], source='existing NIH grayscale tensors; undo mean=.5/std=.5 to unit range', channels='repeat to RGB', mean=MEAN, std=STD, additional_resize_or_crop=False),
                  packages=[dict(package_sha256=b[1]['package_sha256'], selected_rows_sha256=b[1]['selected_rows_sha256'], snapshot_id=b[3], records=len(b[0])) for b in bound + [reference]],
                  source_metadata_hashes=bound[0][1]['source_metadata_hashes'], preparation_sha256=bound[0][1]['preparation_sha256'],
                  official_test_used=False, train_heldout_overlap=0,
                  heldout_patient_overlap=len(bound[2][2]['patients'] & reference[2]['patients']),
                  heldout_image_overlap=len(bound[2][2]['images'] & reference[2]['images']),
                  seed=args.seed, selection='Fixed ten epochs and .001 rate before evaluation; no holdout selection',
                  initial=initial, control=result, checkpoint_sha256=digest(path.read_bytes()),
                  final_validation=measured, final_proportional_reference=evaluate(reload, feature_sets[3], 16), head_reload_exact=True)
    (args.out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
