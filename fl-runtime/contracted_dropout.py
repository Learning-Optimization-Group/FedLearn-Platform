"""Seeded dropout masks for contracted training (Stage 4 S0).

When a run's execution contract states dropout layers (ModelTraining.dropout, DROPOUT_MASKS_SEEDED_V1), every
participant draws its masks from the contract's stream instead of torch's RNG, so an update is reproducible from the
contract and the data. SeededDropoutMasks swaps the model's nn.Dropout layers for seeded ones for one round; the
training loop calls advance() after each optimizer step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fedlearn.contract.dropout_masks import dropout_mask


class _SeededDropout(nn.Module):
    def __init__(self, owner: "SeededDropoutMasks", layer: int, rate: float) -> None:
        super().__init__()
        self._owner, self.layer, self.rate = owner, layer, rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mask = dropout_mask(x.numel(), self.rate, self._owner.seed, self._owner.round, self._owner.step, self.layer)
        return x * torch.from_numpy(mask.reshape(tuple(x.shape))).to(x.device)


class SeededDropoutMasks:
    """For one round: the model's dropout layers draw DROPOUT_MASKS_SEEDED_V1 masks for (seed, round, step, layer).

    ``layers`` is the contract's dropout list, (module, rate) in forward order; it must be exactly the model's dropout
    layers, or the model is not the one the contract describes.
    """

    def __init__(self, net: nn.Module, layers, seed: int, round_: int) -> None:
        actual = [(name, float(m.p)) for name, m in net.named_modules() if isinstance(m, nn.Dropout)]
        stated = [(name, float(rate)) for name, rate in layers]
        if actual != stated:
            raise ValueError(f"the model's dropout layers {actual} are not the contract's {stated}")
        self.net, self.seed, self.round, self.step = net, seed, round_, 0
        self._originals: list[tuple[str, nn.Module]] = []

    def _parent(self, name: str) -> tuple[nn.Module, str]:
        parent_name, _, child = name.rpartition(".")
        return (self.net.get_submodule(parent_name) if parent_name else self.net), child

    def __enter__(self) -> "SeededDropoutMasks":
        for layer, (name, module) in enumerate(
                [(n, m) for n, m in self.net.named_modules() if isinstance(m, nn.Dropout)]):
            parent, child = self._parent(name)
            self._originals.append((name, module))
            setattr(parent, child, _SeededDropout(self, layer, float(module.p)))
        return self

    def advance(self) -> None:
        """The next local SGD step of the round."""
        self.step += 1

    def __exit__(self, *exc) -> None:
        for name, module in self._originals:
            parent, child = self._parent(name)
            setattr(parent, child, module)
        self._originals.clear()
