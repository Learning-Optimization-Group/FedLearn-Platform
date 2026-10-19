"""Validation-only threshold diagnosis of a fixed NIH checkpoint; no further training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def _arrays(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('invalid binary labels or probability shape')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('invalid probabilities')
    return y, p


def validation_threshold(labels, probabilities, sensitivity):
    y, p = _arrays(labels, probabilities)
    positive = np.sort(p[y == 1])
    if not len(positive):
        raise ValueError('validation requires positive labels')
    if not np.isfinite(sensitivity) or not 0 < sensitivity <= 1:
        raise ValueError('invalid target sensitivity')
    required = int(np.ceil(sensitivity * len(positive)))
    # Classification uses p > threshold. Retain every example tied at the selected positive score.
    return float(np.nextafter(positive[-required], -np.inf))


def threshold_metrics(labels, probabilities, threshold):
    y, p = _arrays(labels, probabilities)
    if not np.isfinite(threshold):
        raise ValueError('invalid threshold')
    prediction = p > threshold
    tp = int(((y == 1) & prediction).sum()); fn = int(((y == 1) & ~prediction).sum())
    tn = int(((y == 0) & ~prediction).sum()); fp = int(((y == 0) & prediction).sum())
    return dict(threshold=threshold, count=len(y), true_positive=tp, false_negative=fn,
                true_negative=tn, false_positive=fp, accuracy=(tp+tn)/len(y),
                sensitivity=tp/(tp+fn) if tp+fn else None,
                specificity=tn/(tn+fp) if tn+fp else None,
                positive_predictive_value=tp/(tp+fp) if tp+fp else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sensitivity', type=float, default=.8)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    import torch
    import torchvision
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nih_learning import bind_package, require_disjoint, digest, state_digest, binary_metrics
    from nih_finetune import GrayscaleResNet
    torch.set_num_threads(4)
    bound = [bind_package(args.packages/(name+'.zip'), split=name,
             purpose='eval' if name=='validation' else 'train')
             for name in ('client-0', 'client-1', 'validation')]
    reference = bind_package(args.reference, split='validation', purpose='eval')
    for held in (bound[2], reference):
        require_disjoint([b[2] for b in bound[:2]+[held]])
        require_disjoint([dict(patients=b[2]['patients'], images=b[2]['prepared_images'])
                          for b in bound[:2]+[held]])
    if any(b[1]['source_metadata_hashes'] != bound[0][1]['source_metadata_hashes']
           or b[1]['preparation_sha256'] != bound[0][1]['preparation_sha256']
           for b in bound+[reference]):
        raise ValueError('NIH source preparations differ')
    backbone = torchvision.models.resnet18(weights=None); backbone.fc = torch.nn.Linear(512, 2)
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        backbone.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
    net = GrayscaleResNet(backbone); net.eval(); before = state_digest(net)
    def scores(dataset):
        from torch.utils.data import DataLoader
        probabilities = []
        with torch.no_grad():
            for x, _ in DataLoader(dataset, batch_size=16, shuffle=False):
                probabilities.append(net(x).softmax(1)[:,1].numpy())
        return np.concatenate(probabilities)
    validation_scores = scores(bound[2][0]); validation_labels = bound[2][0].tensors[1].numpy()
    threshold = validation_threshold(validation_labels, validation_scores, args.sensitivity)
    # The reference is evaluated only after the threshold is fixed from balanced validation.
    reference_scores = scores(reference[0]); reference_labels = reference[0].tensors[1].numpy()
    if state_digest(net) != before: raise ValueError('inference changed checkpoint state')
    repeated = scores(bound[2][0])
    if not np.array_equal(repeated, validation_scores): raise ValueError('repeated inference differs')
    report = dict(scope='Threshold diagnosis of an existing fixed checkpoint; no learning or deployment claim',
        checkpoint_sha256=digest(args.checkpoint.read_bytes()), state_sha256=before,
        driver_sha256=digest(Path(__file__).read_bytes()), official_test_used=False,
        target_validation_sensitivity=args.sensitivity, selected_threshold=threshold,
        selection='Largest score boundary retaining target positive validation fraction; reference not used',
        heldout_patient_overlap=len(bound[2][2]['patients'] & reference[2]['patients']),
        train_heldout_patient_and_prepared_image_overlap=0, inference_state_unchanged=True,
        repeated_validation_inference_exact=True,
        packages=[dict(package_sha256=b[1]['package_sha256'], snapshot_id=b[3], records=len(b[0])) for b in bound+[reference]],
        validation=dict(ranking=binary_metrics(validation_labels, validation_scores),
                        default=threshold_metrics(validation_labels, validation_scores, .5),
                        calibrated=threshold_metrics(validation_labels, validation_scores, threshold)),
        reference=dict(ranking=binary_metrics(reference_labels, reference_scores),
                       default=threshold_metrics(reference_labels, reference_scores, .5),
                       calibrated=threshold_metrics(reference_labels, reference_scores, threshold)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(threshold=threshold, validation=report['validation']['calibrated'],
                         reference=report['reference']['calibrated']), allow_nan=False))


if __name__ == '__main__':
    main()
