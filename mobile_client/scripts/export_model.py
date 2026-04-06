#!/usr/bin/env python3
"""
Export SimpleCNN from the framework as TorchScript for mobile deployment.

Usage:
    python scripts/export_model.py

Output:
    mobile_client/assets/simple_cnn.pt

This script imports the SimpleCNN model from the framework examples
and exports it as a TorchScript module that can be loaded by libtorch C++
via torch::jit::load().
"""
import sys
import os

# Add framework to path
framework_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'framework')
)
sys.path.insert(0, os.path.join(framework_root, 'src'))

import torch

# Try importing from framework examples
try:
    sys.path.insert(0, os.path.join(framework_root, 'examples', 'simple_federation'))
    from model import SimpleCNN
    print("Loaded SimpleCNN from framework/examples/simple_federation/model.py")
except ImportError:
    print("SimpleCNN not found in framework. Creating a default SimpleCNN.")

    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return x


# Create model and export
model = SimpleCNN()
model.eval()

# Create example input for tracing
example_input = torch.randn(1, 1, 28, 28)

# Use torch.jit.script for full TorchScript export
scripted_model = torch.jit.script(model)

# Verify the scripted model works
with torch.no_grad():
    output = scripted_model(example_input)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Save
output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'simple_cnn.pt')
scripted_model.save(output_path)
print(f"\nExported TorchScript model to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
