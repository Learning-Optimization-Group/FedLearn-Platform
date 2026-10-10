"""Fixed last-block ImageNet ResNet18 follow-up on patient-separated NIH packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nih_learning import bind_package, require_disjoint, digest, state_digest, evaluate, SeededSnapshotLoader
from nih_pretrained import imagenet_input, configure_fine_tuning, fine_tuning_mode


class GrayscaleResNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(imagenet_input(x))


def frozen_state(net):
    return {name: value.detach().clone() for name, value in net.state_dict().items()
            if not name.startswith(('layer4.', 'fc.')) or 'running_' in name or name.endswith('num_batches_tracked')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): parser.error('output already exists')
    torch.set_num_threads(4)
    import torchvision
    bound = [bind_package(args.packages/(n+'.zip'), split=n, purpose='eval' if n=='validation' else 'train')
             for n in ('client-0','client-1','validation')]
    reference = bind_package(args.reference, split='validation', purpose='eval')
    for held in (bound[2], reference):
        require_disjoint([b[2] for b in bound[:2]+[held]])
        require_disjoint([dict(patients=b[2]['patients'],images=b[2]['prepared_images']) for b in bound[:2]+[held]])
    if any(b[1]['source_metadata_hashes'] != bound[0][1]['source_metadata_hashes']
           or b[1]['preparation_sha256'] != bound[0][1]['preparation_sha256'] for b in bound+[reference]):
        raise ValueError('different NIH source preparation')
    weight_sha = digest(args.weights.read_bytes())
    if not weight_sha.startswith('f37072fd'): raise ValueError('requires pinned official ImageNet V1 ResNet18 weights')
    backbone = torchvision.models.resnet18(weights=None)
    backbone.load_state_dict(torch.load(args.weights, weights_only=True, map_location='cpu'), strict=True)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        backbone.fc = torch.nn.Linear(512, 2)
    configure_fine_tuning(backbone)
    frozen = frozen_state(backbone)
    net = GrayscaleResNet(backbone)
    training = TensorDataset(*(torch.cat([bound[i][0].tensors[j] for i in (0,1)]) for j in (0,1)))
    loader = SeededSnapshotLoader(training,batch_size=16,seed=42,drop_last=False)
    optimizer = torch.optim.Adam([dict(params=backbone.layer4.parameters(), lr=.00001),
                                  dict(params=backbone.fc.parameters(), lr=.001)])
    report = dict(torch_version=torch.__version__,torchvision_version=torchvision.__version__,threads=4,
                  architecture='ImageNet V1 ResNet18 last block and two-class head; all BatchNorm statistics frozen',
                  profile=dict(epochs=10,layer4_rate=.00001,head_rate=.001,batch_size=16,seed=42,optimizer='persistent Adam'),
                  selection='Fixed budget and rates before evaluation; final epoch reported; official test unused',
                  official_test_used=False,train_heldout_patient_and_image_overlap=0,
                  weights_sha256=weight_sha,driver_sha256=digest(Path(__file__).read_bytes()),
                  packages=[dict(package_sha256=b[1]['package_sha256'],snapshot_id=b[3],records=len(b[0])) for b in bound+[reference]],
                  heldout_patient_overlap=len(bound[2][2]['patients'] & reference[2]['patients']),
                  initial=dict(state_sha256=state_digest(net),validation=evaluate(net,bound[2][0],16)),epochs=[])
    args.out.mkdir(parents=True)
    for epoch in range(1,11):
        start = time.monotonic(); loader.set_round(epoch); fine_tuning_mode(backbone)
        total, count = 0., 0
        for x,y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(x),y)
            if not torch.isfinite(loss): raise ValueError('non-finite loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise ValueError('non-finite gradient')
            optimizer.step(); total += float(loss.detach())*len(y); count += len(y)
        if any(not torch.isfinite(v).all() for v in net.state_dict().values()): raise ValueError('non-finite state')
        for name,value in frozen.items():
            if not torch.equal(value,backbone.state_dict()[name]): raise ValueError('frozen backbone or BatchNorm state changed')
        record=dict(epoch=epoch,training_loss=total/count,examples=count,seconds=time.monotonic()-start,
                    validation=evaluate(net,bound[2][0],16),state_sha256=state_digest(net))
        report['epochs'].append(record)
        (args.out/'progress.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    checkpoint=args.out/'last-block.npz'
    np.savez(checkpoint,**{k:v.detach().numpy() for k,v in backbone.state_dict().items()})
    restored=torchvision.models.resnet18(weights=None);restored.fc=torch.nn.Linear(512,2)
    with np.load(checkpoint,allow_pickle=False) as archive:
        restored.load_state_dict({k:torch.from_numpy(archive[k].copy()) for k in archive.files},strict=True)
    restored=GrayscaleResNet(restored)
    final=evaluate(restored,bound[2][0],16)
    if final != report['epochs'][-1]['validation'] or state_digest(restored)!=state_digest(net):
        raise ValueError('checkpoint reload differs')
    report.update(checkpoint_sha256=digest(checkpoint.read_bytes()),reload_exact=True,frozen_state_unchanged=True,
                  final_validation=final,final_proportional_reference=evaluate(restored,reference[0],16))
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
