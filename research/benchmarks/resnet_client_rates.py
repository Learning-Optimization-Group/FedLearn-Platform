"""Matched two-round, two-partition CPU controls for adaptive ResNet client rates.

Uses production fitting and aggregation with isolated counterfactual contracts.
Both partitions run on Python CPU; this does not reproduce the native runtime.
"""
import contextlib, hashlib, json, sys, time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / 'fl-runtime'), str(root / 'framework/src')]
import client, fl_server
from fedlearn.contract import parse_contract_json
from google.protobuf.json_format import MessageToJson


def finite_cross_entropy(logits, targets):
    """Reject non-finite diagnostics rather than retaining an invalid receipt."""
    if not torch.isfinite(logits).all():
        raise ValueError("diagnostic logits are non-finite")
    loss = torch.nn.functional.cross_entropy(logits, targets)
    if not torch.isfinite(loss):
        raise ValueError("diagnostic loss is non-finite")
    return loss.item()


def require_validation_disjoint(training, validation):
    def identities(dataset):
        return {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x, _ in dataset}
    heldout = identities(validation)
    if any(identities(dataset) & heldout for dataset in training):
        raise ValueError('training/validation image overlap')


def validation_metrics(net, dataset):
    from torch.utils.data import DataLoader
    net.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=32, shuffle=False):
            logits.append(net(x)); labels.append(y)
    scores, targets = torch.cat(logits), torch.cat(labels)
    prediction = scores.argmax(1)
    confusion = torch.bincount(targets * 10 + prediction, minlength=100).reshape(10, 10)
    return dict(loss=finite_cross_entropy(scores, targets), correct=int((prediction == targets).sum()),
                count=len(targets), confusion=confusion.tolist(),
                recall=[float(confusion[i,i] / confusion[i].sum()) if confusion[i].sum() else None for i in range(10)])


def main():
    parser = ArgumentParser(description='Bounded offline ResNet client-rate controls; no official test selection')
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--phone-package', type=Path, required=True)
    parser.add_argument('--desktop-package', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--longer', action='store_true', help='Fixed six-round .001 FedAvg/FedProx/FedOpt profiles')
    parser.add_argument('--validation-package', type=Path)
    options = parser.parse_args()
    if options.longer != (options.validation_package is not None):
        parser.error('longer controls require an explicit disjoint validation package')
    out = options.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    client.USE_MLP = client.USE_LLM = client.USE_LLM_LORA = client.USE_PNEUMONIA = False
    client.USE_DERIVED = client.USE_WIRE_SUBSET = False
    client.MODEL_TYPE = 'CIFAR_RESNET18'
    client.TRAINING_ARM = 'FULL'
    client.DEVICE = torch.device('cpu')
    base = parse_contract_json(options.contract.read_text())
    from execution_plan import pb
    assert base.recipe == pb.RECIPE_CIFAR_RESNET18
    assert base.strategy == pb.STRATEGY_FEDPROX and base.model_training.fedprox_mu == 0.1
    assert base.num_rounds == 2 and base.model_training.local_training.batch_size == 32
    assert base.clients_per_round == 2, "the diagnostic requires exactly two clients"
    assert base.model_training.data.source == pb.DATA_SOURCE_LOCAL_SNAPSHOT
    with np.load(options.initial_state) as z:
        initial = OrderedDict(((k.replace('__DOT__', '.'), torch.from_numpy(z[k].copy())) for k in z.files if z[k].dtype == np.float32))
    layout = [p.name for p in base.model_training.trainable] + [b.module + '.running_' + kind for b in base.model_training.batch_norm for kind in ('mean', 'var')]
    assert set(initial) == set(layout)
    initial = OrderedDict(((name, initial[name]) for name in layout))
    from fedlearn.communication.safetensors_codec import save_safetensors
    assert hashlib.sha256(save_safetensors([(k, t.numpy()) for k, t in initial.items()])).hexdigest() == base.model_training.initial_state_sha256
    report = {'seed': str(base.seed), 'initial_sha256': hashlib.sha256(options.initial_state.read_bytes()).hexdigest(), 'scope': 'offline CPU counterfactuals; both partitions use Python, not a phone run; official test unused', 'contract_sha256': hashlib.sha256(options.contract.read_bytes()).hexdigest(), 'package_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('phone', options.phone_package), ('desktop', options.desktop_package)]}, 'torch': torch.__version__, 'torch_num_threads': torch.get_num_threads(), 'server_learning_rate': 0.01, 'proximal_mu': 0.1, 'rates': []}
    report['validation_package_sha256'] = hashlib.sha256(options.validation_package.read_bytes()).hexdigest() if options.longer else None
    report['selection'] = 'Fixed six-round profiles and final-epoch metrics; official test unused' if options.longer else 'fixed two-round rate controls'
    for strategy in (('FedAvg', 'FedProx', 'FedOpt') if options.longer else ('FedProx', 'FedOpt')):
        for rate in ((0.001,) if options.longer else (0.01, 0.001, 0.0001)):
            import strategy_client_settings
            # These counterfactual settings exist only in this diagnostic process.
            strategy_client_settings.RESNET_CLIENT_LEARNING_RATE = rate
            strategy_client_settings.FEDOPT_CLIENT_LEARNING_RATE = rate
            strategy_client_settings.FEDPROX_CLIENT_LEARNING_RATE = rate
            tag = f'{strategy}-{rate:g}'
            contract = parse_contract_json(options.contract.read_text())
            from execution_plan import pb
            contract.strategy = getattr(pb, {'FedAvg':'STRATEGY_FEDAVG','FedOpt':'STRATEGY_FEDOPT','FedProx':'STRATEGY_FEDPROX'}[strategy])
            if options.longer: contract.num_rounds = 6
            contract.model_training.local_training.adam.learning_rate = rate
            if strategy != 'FedProx':
                contract.model_training.ClearField('fedprox_mu')
            path = out / (tag + '-contract.json')
            path.write_text(MessageToJson(contract))
            log = (out / (tag + '.log')).open('w')
            entry = {'strategy': strategy, 'rate': rate, 'rounds': []}
            with contextlib.redirect_stdout(log):
                participants = []
                for part, name in enumerate(('phone', 'desktop')):
                    args = client.parse_args(['--project-id', contract.project_id, '--run-id', contract.run_id, '--server-address', 'localhost:50000', '--partition-id', str(part), '--model-type', 'CIFAR_RESNET18', '--strategy', strategy, '--execution-contract', str(path), '--dataset-path', str(options.phone_package if name == 'phone' else options.desktop_package)])
                    client.enforce_execution_contract(args, 'CIFAR_RESNET18', 'FULL')
                    participants.append(client.ZOSLClient(partition_id=part, dataset_name='cifar10'))
                    assert len(participants[-1].trainloader.dataset) == 64
                validation = None
                if options.longer:
                    from local_snapshot import load_local_snapshot
                    import recipes
                    validation = load_local_snapshot(options.validation_package, contract.model_training.data,
                        recipes.get_recipe('CIFAR_RESNET18').classes).dataset
                    require_validation_disjoint([p.trainloader.dataset for p in participants], validation)
                current = OrderedDict(((k, v.clone()) for k, v in initial.items()))
                aggregator = fl_server.select_strategy(Namespace(strategy=strategy, min_clients=2, clients_per_round=2, model_type='CIFAR_RESNET18', fedopt_server_learning_rate=0.01), current, None)
                for r in range(1, contract.num_rounds + 1):
                    updates = []
                    round_result = {'round': r, 'clients': []}
                    for part, p in enumerate(participants):
                        losses = []
                        handle = p.net.register_forward_hook(lambda m, inp, output: losses.append(output.detach().clone()))
                        p.trainloader.set_round(r)
                        labels = [y.clone() for x, y in p.trainloader]
                        start = time.monotonic()
                        actual, count = p.fit(current, {'server_round': r, 'learning_rate': str(rate), 'local_epochs': '1', 'proximal_mu': '.1' if strategy == 'FedProx' else '0'})
                        handle.remove()
                        batch_losses = [finite_cross_entropy(v, y) for v, y in zip(losses, labels)]
                        assert len(batch_losses) == 2 and count == 64
                        assert all((torch.isfinite(v).all() for v in actual.values()))
                        updates.append((str(part), OrderedDict(((k, v.detach().clone()) for k, v in actual.items())), count))
                        round_result['clients'].append({'partition': part, 'batch_losses': batch_losses, 'seconds': time.monotonic() - start})
                    current = aggregator.aggregate_fit(r, updates)
                    # Diagnostics reuse training examples; no held-out accuracy claim.
                    evals = []
                    with torch.no_grad():
                        for p in participants:
                            client.load_federated_state(p.net, current)
                            p.net.eval()
                            p.trainloader.set_round(r)
                            logits = []
                            ys = []
                            for x, y in p.trainloader:
                                logits.append(p.net(x))
                                ys.append(y)
                            logits = torch.cat(logits)
                            ys = torch.cat(ys)
                            evals.append({'loss': finite_cross_entropy(logits, ys), 'correct': int((logits.argmax(1) == ys).sum()), 'count': len(ys)})
                    assert all((torch.isfinite(v).all() for v in current.values()))
                    round_result['aggregate_training_diagnostic'] = evals
                    if validation is not None:
                        round_result['validation'] = validation_metrics(participants[0].net, validation)
                    entry['rounds'].append(round_result)
                if validation is not None:
                    checkpoint = out / (tag + '.npz')
                    np.savez(checkpoint, **{k:v.detach().numpy() for k,v in participants[0].net.state_dict().items()})
                    restored = client.ZOSLClient(partition_id=0, dataset_name='cifar10').net
                    with np.load(checkpoint, allow_pickle=False) as archive:
                        restored.load_state_dict({k:torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
                    if validation_metrics(restored, validation) != entry['rounds'][-1]['validation']:
                        raise ValueError('checkpoint validation reload differs')
                    entry.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), reload_exact=True)
            log.close()
            report['rates'].append(entry)
            (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(entry, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
