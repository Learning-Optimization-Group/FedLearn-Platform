"""Own-data CSV loading with a fixed ordered feature schema."""
import csv
import hashlib
from pathlib import Path

import numpy as np

from .model import FEATURES


def load_csv(path):
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError("local tree CSV must be a nonempty file of at most 256 MiB")
    raw = path.read_bytes()
    reader = csv.reader(raw.decode("utf-8").splitlines(), strict=True)
    if next(reader, None) != [*FEATURES, "label"]:
        raise ValueError("CSV header must be f0,f1,f2,f3,label in that order")
    rows, labels = [], []
    for row in reader:
        if not row:
            continue
        if len(row) != 5 or row[-1] not in ("0", "1") or len(rows) >= 1_000_000:
            raise ValueError("invalid CSV width, label or row count")
        values = [float(value) for value in row[:4]]
        if not all(np.isfinite(value) and abs(value) <= float(np.finfo(np.float32).max) for value in values):
            raise ValueError("features must be finite float32 values")
        rows.append(values)
        labels.append(int(row[-1]))
    if not rows:
        raise ValueError("local tree CSV contains no examples")
    x = np.asarray(rows, dtype="float32")
    y = np.asarray(labels, dtype="float32")
    return x, y, hashlib.sha256(x.astype("<f4").tobytes() + y.astype("<f4").tobytes()).hexdigest()
