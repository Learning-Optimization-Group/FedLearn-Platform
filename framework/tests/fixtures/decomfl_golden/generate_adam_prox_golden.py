"""Freeze multi-step Adam endpoints with a fixed FedProx round anchor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generate_fedavg_golden import build_initial_net, load_committed_batch
from fedlearn.estimators.params import flat_params

HERE = Path(__file__).resolve().parent
STEPS, LR, MU = 12, .01, .1


def endpoint(control=None):
    net = build_initial_net()
    x, y = load_committed_batch()
    parameters = [p for p in net.parameters() if p.requires_grad]
    anchor = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=LR)
    for step in range(STEPS):
        start = 4 * (step % 2)
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(net(x[start:start + 4]), y[start:start + 4]).backward()
        if control != 'zero_mu' and control != 'after_adam':
            for p, initial in zip(parameters, anchor):
                p.grad.add_(p - initial, alpha=MU)
        optimizer.step()
        if control == 'after_adam':
            with torch.no_grad():
                for p, initial in zip(parameters, anchor):
                    p.add_(p - initial, alpha=-LR * MU)
        if control == 'moving_anchor':
            anchor = [p.detach().clone() for p in parameters]
    return flat_params(net).detach().numpy().astype('<f4')


def main():
    golden = endpoint()
    golden.tofile(HERE / 'adam_prox_local_final.f32')
    controls = {name: float(np.abs(endpoint(name) - golden).max())
                for name in ('zero_mu', 'moving_anchor', 'after_adam')}
    assert min(controls.values()) > 1e-4, controls
    report = dict(torch_version=torch.__version__, steps=STEPS, batch_size=4,
                  learning_rate=LR, proximal_mu=MU, anchor='fixed round-start trainable parameters',
                  sha256=hashlib.sha256(golden.tobytes()).hexdigest(), control_separation=controls,
                  endpoint_atol=1e-5)
    (HERE / 'adam_prox_local_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
