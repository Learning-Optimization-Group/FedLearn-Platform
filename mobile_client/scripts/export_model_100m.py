#!/usr/bin/env python3
"""Export Model100M (~100M params) as TorchScript for mobile deployment.

Usage:  python scripts/export_model_100m.py
Output: mobile_client/assets/model_100m.pt
"""
import sys, os

framework_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'framework'))
sys.path.insert(0, os.path.join(framework_root, 'examples', 'simple_federation'))

import torch
from model_100m import Model100M

model = Model100M()
model.eval()

example_input = torch.randn(1, 1, 28, 28)
scripted = torch.jit.script(model)

with torch.no_grad():
    out = scripted(example_input)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(output_dir, exist_ok=True)
path = os.path.join(output_dir, 'model_100m.pt')
scripted.save(path)
print(f"Exported to: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
