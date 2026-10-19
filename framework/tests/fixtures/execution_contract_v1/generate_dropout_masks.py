"""Write dropout_masks_v1.golden: the DROPOUT_MASKS_SEEDED_V1 masks every runtime must reproduce.

Lines ``mask n rate seed round step layer : bits`` give which of n elements are kept (1) or dropped (0); lines
``scale rate : hex`` give the float32 bits of the kept elements' scale. Read by the Python and native tests.

    PYTHONPATH=framework/src .venv/bin/python framework/tests/fixtures/execution_contract_v1/generate_dropout_masks.py
"""
from __future__ import annotations

import os
import struct

from fedlearn.contract.dropout_masks import dropout_keep, dropout_scale

HERE = os.path.dirname(os.path.abspath(__file__))

MASKS = [
    (0, 0.3, 42, 1, 0, 0), (1, 0.3, 42, 1, 0, 0), (64, 0.0, 42, 1, 0, 0), (64, 0.3, 42, 1, 0, 0),
    (64, 0.3, 42, 1, 0, 1), (64, 0.3, 42, 1, 1, 0), (64, 0.3, 42, 2, 0, 0), (64, 0.3, 43, 1, 0, 0),
    (64, 0.5, 0, 1, 0, 0), (64, 0.9, (1 << 64) - 1, 1, 0, 0), (257, 0.3, 42, (1 << 32) - 1, 1000, 7),
]
SCALES = [0.0, 0.1, 0.3, 0.5, 0.9]


def main() -> None:
    lines = ["# mask n rate seed round step layer : kept bits | scale rate : float32 hex "
             "(DROPOUT_MASKS_SEEDED_V1; generate_dropout_masks.py)"]
    for n, rate, seed, round_, step, layer in MASKS:
        bits = "".join("1" if k else "0" for k in dropout_keep(n, rate, seed, round_, step, layer))
        lines.append(f"mask {n} {rate!r} {seed} {round_} {step} {layer} : {bits}")
    for rate in SCALES:
        lines.append(f"scale {rate!r} : {struct.pack('<f', dropout_scale(rate)).hex()}")
    with open(os.path.join(HERE, "dropout_masks_v1.golden"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
