"""Standalone CPU XGBoost execution and model-IO probe; no federation."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import xgboost as xgb


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    feature_names: tuple[str, ...]
    seed: int


def make_fixture(seed: int = 42) -> dict[str, Dataset]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer between 0 and 2**31 - 1")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(448, 4)).astype(np.float32)
    labels = (features @ np.array([1.0, 0.6, -0.3, 0.2], dtype=np.float32) > 0).astype(np.int64)
    order = rng.permutation(448)
    names = ("signal_0", "signal_1", "signal_2", "signal_3")
    return {
        name: Dataset(features[ids], labels[ids], ids, names, seed)
        for name, ids in (("train", order[:256]), ("validation", order[256:320]), ("test", order[320:]))
    }


def _matrix(data: Dataset) -> xgb.DMatrix:
    x, y, names = data.features, data.labels, data.feature_names
    if x.ndim != 2 or len(x) == 0 or x.shape[1] != len(names):
        raise ValueError("features must be nonempty rows matching the named feature schema")
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("feature names must be unique nonempty strings")
    if x.dtype.kind not in "fi" or np.isinf(x).any():
        raise ValueError("features must be numeric, with NaN as the only missing-value marker")
    if y.ndim != 1 or len(y) != len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a binary vector matching the feature rows")
    return xgb.DMatrix(x, label=y, feature_names=list(names), missing=np.nan, nthread=1)


def _params(seed: int) -> dict:
    return {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu", "nthread": 1,
            "seed": seed, "eta": 0.2, "max_depth": 3, "base_score": 0.5,
            "subsample": 1.0, "colsample_bytree": 1.0, "eval_metric": "logloss"}


def train_probe(data: Dataset, rounds: int, parent: xgb.Booster | None = None) -> xgb.Booster:
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if parent is not None and parent.feature_names != list(data.feature_names):
        raise ValueError("parent model feature order differs from the training data")
    return xgb.train(_params(data.seed), _matrix(data), num_boost_round=rounds, xgb_model=parent)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def save_checkpoint(model: xgb.Booster, path: Path) -> dict:
    path = Path(path)
    if path.suffix not in (".json", ".ubj"):
        raise ValueError("checkpoint format must be JSON or UBJSON")
    manifest_path = _manifest_path(path)
    if path.exists() or manifest_path.exists():
        raise FileExistsError(f"checkpoint already exists: {path.name}")
    model.save_model(path)
    payload = path.read_bytes()
    metadata = {"schema_version": 1, "format": path.suffix[1:], "xgboost_version": xgb.__version__,
                "feature_names": model.feature_names, "rounds": model.num_boosted_rounds(),
                "bytes": len(payload), "sha256": _digest(payload)}
    manifest_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def load_checkpoint(path: Path, feature_names: tuple[str, ...]) -> xgb.Booster:
    path = Path(path)
    metadata = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("format") not in ("json", "ubj"):
        raise ValueError("unsupported checkpoint metadata")
    if metadata["format"] != path.suffix[1:] or metadata.get("xgboost_version") != xgb.__version__:
        raise ValueError("checkpoint format or runtime version differs from this probe")
    if metadata.get("feature_names") != list(feature_names):
        raise ValueError("checkpoint feature order differs from the requested schema")
    payload = path.read_bytes()
    if len(payload) != metadata.get("bytes") or _digest(payload) != metadata.get("sha256"):
        raise ValueError("checkpoint size or checksum mismatch")
    model = xgb.Booster(params={"nthread": 1})
    model.load_model(path)
    if model.feature_names != list(feature_names) or model.num_boosted_rounds() != metadata.get("rounds"):
        raise ValueError("native model differs from its checkpoint metadata")
    return model


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    tp = int(np.sum(predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    positive, negative = probabilities[labels == 1], probabilities[labels == 0]
    p = np.clip(probabilities.astype(np.float64), 1e-15, 1 - 1e-15)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2 if len(positive) and len(negative) else None,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "roc_auc": float(np.mean((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)))
        if len(positive) and len(negative) else None,
        "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def run_probe(output_dir: Path, seed: int = 42) -> dict:
    fixture = make_fixture(seed)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    train = fixture["train"]
    initial = train_probe(train, 8)
    matrices = {name: _matrix(data) for name, data in fixture.items()}
    initial_predictions = {name: initial.predict(matrix) for name, matrix in matrices.items()}
    uninterrupted = train_probe(train, 12)
    reference_predictions = {name: uninterrupted.predict(matrix) for name, matrix in matrices.items()}
    reload_error = resume_error = 0.0
    artifacts = {}
    for format in ("json", "ubj"):
        path = directory / f"initial.{format}"
        artifacts[path.name] = save_checkpoint(initial, path)
        loaded = load_checkpoint(path, train.feature_names)
        continued = train_probe(train, 4, parent=loaded)
        for name, matrix in matrices.items():
            reload_error = max(reload_error, float(np.max(np.abs(loaded.predict(matrix) - initial_predictions[name]))))
            resume_error = max(resume_error, float(np.max(np.abs(continued.predict(matrix) - reference_predictions[name]))))
    final_path = directory / "continued.ubj"
    artifacts[final_path.name] = save_checkpoint(continued, final_path)
    final = load_checkpoint(final_path, train.feature_names)
    final_predictions = {name: final.predict(matrix) for name, matrix in matrices.items()}
    for name in matrices:
        reload_error = max(reload_error, float(np.max(np.abs(final_predictions[name] - continued.predict(matrices[name])))))
    if reload_error != 0.0 or resume_error != 0.0:
        raise RuntimeError("checkpoint reload or continuation disagrees with the pinned CPU reference")
    report = {
        "scope": "single_process_cpu_probe", "federation_implemented": False,
        "dataset": "synthetic_binary_v1", "seed": seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "xgboost": xgb.__version__},
        "parameters": _params(seed), "feature_names": list(train.feature_names),
        "rounds": {"initial": initial.num_boosted_rounds(), "continued": final.num_boosted_rounds()},
        "splits": {name: {"rows": len(data.labels), "class_counts": np.bincount(data.labels, minlength=2).tolist(),
                          "row_ids_sha256": _digest(data.indices.astype("<i8").tobytes()),
                          "features_sha256": _digest(data.features.astype("<f4").tobytes()),
                          "labels_sha256": _digest(data.labels.astype("<i8").tobytes())}
                   for name, data in fixture.items()},
        "metrics": {stage: {name: _metrics(data.labels, predictions[name]) for name, data in fixture.items()}
                    for stage, predictions in (("initial", initial_predictions), ("continued", final_predictions))},
        "checks": {"reload_max_abs_difference": reload_error, "resume_max_abs_difference": resume_error},
        "artifacts": artifacts, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_probe(args.output_dir, args.seed)
    print(json.dumps({"scope": report["scope"], "checks": report["checks"],
                      "test_metrics": report["metrics"]["continued"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
