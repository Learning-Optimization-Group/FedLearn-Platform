"""Bounded NIH learning controls using packages verified at their source host.

Package receipts bind labels, patient identities and image hashes to the copied
bytes. They do not independently re-verify the full raw NIH inventory on the
training host. Official test data is never used by this benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'framework/src'), str(ROOT / 'fl-runtime')]

from contracted_dropout import SeededDropoutMasks
from fedlearn.communication.safetensors_codec import save_safetensors
from fedlearn.contract.image_package import read_package
from local_snapshot import canonical, digest, load_local_snapshot, SeededSnapshotLoader
from nih_runtime import binary_metrics as recipe_metrics


def binary_metrics(labels, probabilities):
    result = recipe_metrics(labels, probabilities)
    result['confusion'] = dict(tn=result['true_negative'], fp=result['false_positive'],
                               fn=result['false_negative'], tp=result['true_positive'])
    result['majority_baseline_accuracy'] = result['majority_accuracy']
    result['predicted_positive_fraction'] = float((np.asarray(probabilities) > .5).mean())
    result['positive_probability_quantiles'] = np.quantile(probabilities, [0, .25, .5, .75, 1]).tolist()
    return result


def require_disjoint(bindings):
    for i, first in enumerate(bindings):
        for second in bindings[i + 1:]:
            for kind in ('patients', 'images'):
                if first[kind] & second[kind]:
                    raise ValueError(f'cross-partition {kind} overlap')


def bind_package(path, *, split, purpose):
    import execution_plan
    from nih_dataset import CLASSES
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    rows, hashes = receipt['records'], receipt['image_sha256']
    if (receipt.get('dataset') != 'NIH_ChestXray14' or receipt.get('recipe') != 'NIH_PNEUMONIA_CNN'
            or receipt.get('classes') != CLASSES or receipt.get('split') != split
            or receipt.get('purpose') != purpose or purpose not in ('train', 'eval')
            or split == 'test' or digest(path.read_bytes()) != receipt.get('package_sha256')
            or digest(canonical(rows)) != receipt.get('selected_rows_sha256')
            or len(rows) != receipt.get('record_count') or len(hashes) != len(rows)):
        raise ValueError('NIH package differs from its source receipt or requested split')
    pixels, labels, classes = read_package(path)
    if classes != CLASSES or labels != [row['label'] for row in rows]:
        raise ValueError('NIH package labels differ from source receipt')
    patients = {row['patient_id'] for row in rows}
    if purpose == 'train' and (not receipt.get('patient_unique') or len(patients) != len(rows)):
        raise ValueError('training control requires unique patients')
    requirement = execution_plan.resolve_model_training(
        'NIH_PNEUMONIA_CNN', 'FedAvg', 'FULL', data_source='LOCAL_SNAPSHOT').data
    snapshot = load_local_snapshot(path, requirement, CLASSES)
    binding = dict(patients=patients, images=set(hashes),
                   prepared_images={digest(image.tobytes()) for image in pixels})
    return snapshot.dataset, receipt, binding, snapshot.snapshot_id


def state_digest(net):
    return digest(save_safetensors([(name, value.detach().cpu().numpy())
                                   for name, value in net.state_dict().items()]))


def evaluate(net, dataset, batch_size):
    net.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            scores.extend(net(x).softmax(dim=1)[:, 1].tolist())
            labels.extend(y.tolist())
    return binary_metrics(np.asarray(labels), np.asarray(scores))


def train_control(net, training, validation, *, epochs, learning_rate, seed,
                  batch_size, reset_optimizer_each_epoch=False):
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('invalid learning control settings')
    loader = SeededSnapshotLoader(training, batch_size=batch_size, seed=seed, drop_last=False)
    layers = [(name, float(layer.p)) for name, layer in net.named_modules()
              if isinstance(layer, torch.nn.Dropout)]
    report = dict(optimizer='Adam', learning_rate=learning_rate, seed=seed,
                  optimizer_lifetime='reset_each_epoch' if reset_optimizer_each_epoch else 'persistent',
                  batch_size=batch_size, initial_state_sha256=state_digest(net), epochs=[])
    optimizer = None
    for epoch in range(1, epochs + 1):
        if optimizer is None or reset_optimizer_each_epoch:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        loader.set_round(epoch)
        net.train()
        total_loss, examples = 0., 0
        with SeededDropoutMasks(net, layers, seed=seed, round_=epoch) as masks:
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(net(x), y)
                if not torch.isfinite(loss):
                    raise ValueError('non-finite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise ValueError('non-finite training gradients')
                optimizer.step()
                masks.advance()
                total_loss += float(loss.detach()) * len(y)
                examples += len(y)
        if any(not torch.isfinite(p).all() for p in net.parameters()):
            raise ValueError('non-finite trained state')
        record = dict(epoch=epoch, training_loss=total_loss / examples, examples=examples,
                      state_sha256=state_digest(net), training=evaluate(net, training, batch_size),
                      validation=evaluate(net, validation, batch_size))
        report['epochs'].append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    report['final_state_sha256'] = state_digest(net)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', required=True, type=Path)
    parser.add_argument('--initial', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    bound = [bind_package(args.packages / (name + '.zip'), split=name,
                          purpose='eval' if name == 'validation' else 'train')
             for name in ('client-0', 'client-1', 'validation')]
    require_disjoint([item[2] for item in bound])
    require_disjoint([dict(patients=b[2]['patients'], images=b[2]['prepared_images']) for b in bound])
    if any(item[1]['source_metadata_hashes'] != bound[0][1]['source_metadata_hashes']
           or item[1]['preparation_sha256'] != bound[0][1]['preparation_sha256'] for item in bound):
        raise ValueError('packages have different source preparation')
    training = TensorDataset(*(torch.cat([bound[i][0].tensors[j] for i in (0, 1)]) for j in (0, 1)))
    validation = bound[2][0]
    import recipes
    net = recipes.build_pneumonia_cnn()
    with np.load(args.initial, allow_pickle=False) as archive:
        initial = {name.replace('__DOT__', '.'): torch.from_numpy(archive[name].copy()) for name in archive.files}
    net.load_state_dict(initial, strict=True)
    output = dict(torch_version=torch.__version__, numpy_version=np.__version__,
                  initial_file_sha256=digest(args.initial.read_bytes()), training_count=len(training),
                  validation_count=len(validation), official_test_used=False,
                  source_receipt_verification='Full raw source verified when packages were prepared; copied package receipts checked here.',
                  packages=[dict(package_sha256=b[1]['package_sha256'], snapshot_id=b[3],
                                 selected_rows_sha256=b[1]['selected_rows_sha256']) for b in bound], controls=[])
    args.out.mkdir(parents=True)
    for rate, reset in ((.001, False), (.0001, False), (.001, True)):
        net.load_state_dict(initial, strict=True)
        result = train_control(net, training, validation, epochs=3, learning_rate=rate,
                               seed=args.seed, batch_size=16, reset_optimizer_each_epoch=reset)
        name = f'adam-{rate}-{result["optimizer_lifetime"]}'
        checkpoint = args.out / (name + '.npz')
        np.savez(checkpoint, **{k: v.detach().numpy() for k, v in net.state_dict().items()})
        result['checkpoint_file_sha256'] = digest(checkpoint.read_bytes())
        output['controls'].append(result)
        (args.out / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
