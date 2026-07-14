"""Throwaway tiny frozen-backbone + linear-head net for the DA-11 Phase-1 vertical slice."""
import torch
import torch.nn as nn


class TinyFrozenNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A trivial "backbone": 1x8x8 grayscale -> 4 features. Frozen after build.
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # -> [B, 2]
        )
        self.head = nn.Linear(2, 3)  # 3 classes; the only trainable part

    def forward(self, x):
        return self.head(self.backbone(x))


def build_tiny_frozen_net(seed: int = 0) -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = TinyFrozenNet()
    for p in net.backbone.parameters():
        p.requires_grad_(False)
    return net
