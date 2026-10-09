#!/usr/bin/env python3
"""Prepare NIH pneumonia-label manifests without copying images or changing labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fl-runtime"))
from nih_dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clients", default=2, type=int)
    parser.add_argument("--validation-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    try:
        report = prepare_dataset(args.source, args.out, num_clients=args.clients,
                                 validation_fraction=args.validation_fraction, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"NIH preparation refused: {exc}\n")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
