"""
benchmarks.py — the metric-computation core of the FedLearn benchmarking suite.

Single source of truth for *how every benchmark metric is computed*. The FL
server (fl_server.py) imports this to enrich each round's server-side evaluation,
and the standalone `run_benchmark.py` CLI imports it to score a saved model on a
held-out test set. Keeping the math in one place means the online (per-round) and
offline (deliberate) benchmarks report identical, directly-comparable numbers.

Design constraints:
  * ARM64 / Jetson safe — only numpy + scikit-learn (both already in the scripts
    env; client.py pre-imports sklearn to dodge the ARM64 static-TLS libgomp
    issue). Every sklearn use is guarded so a missing install degrades to a
    pure-numpy fallback rather than crashing a real FL run.
  * No new pip dependency is introduced.
  * Pure functions over (y_true, y_pred[, y_score]) so they are unit-testable
    with no torch / no GPU.

Metric taxonomy (macro = run/round-level aggregate, micro = per-class/per-client):
  Classification (macro): accuracy, balanced_accuracy, precision/recall/f1 in
    macro|micro|weighted averaging, matthews_corrcoef (MCC), cohen_kappa,
    log_loss & roc_auc (need probabilities).
  Classification (micro): per-class precision/recall/f1/support, confusion matrix.
  Generative / causal-LM: eval_loss, perplexity = exp(loss), token_accuracy.
  Distribution / FL-heterogeneity: per-class label distribution + normalized
    Shannon entropy (a cheap, label-only non-IID proxy).

All public builders return JSON-ready dicts whose keys are **camelCase** so the
payload maps 1:1 onto the Java BenchmarkRoundDto with no renaming.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

import numpy as np

# scikit-learn is present in the scripts env (client.py pre-imports it for the
# ARM64 TLS fix). Guard anyway: a missing install must not break a live run —
# we fall back to numpy implementations of the core metrics.
try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    _SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn is absent
    _SKLEARN = False


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _as_int_array(x: Sequence) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).ravel()


def _round(x: Optional[float], ndigits: int = 6) -> Optional[float]:
    """Round, mapping NaN/inf to None so JSON stays valid and the UI can blank it."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, ndigits)


def _prf_numpy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray):
    """Pure-numpy per-class precision/recall/f1/support (sklearn fallback)."""
    precision, recall, f1, support = [], [], [], []
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(int(np.sum(y_true == c)))
    return (np.array(precision), np.array(recall), np.array(f1), np.array(support))


def _confusion_numpy(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    idx = {int(c): i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if int(t) in idx and int(p) in idx:
            m[idx[int(t)], idx[int(p)]] += 1
    return m


def _expected_calibration_error(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """ECE: confidence-binned |accuracy - mean confidence|, size-weighted.

    Calibration is a distinct quality axis from accuracy and is specifically
    fragile under FedAvg (averaging well-calibrated client models can yield a
    miscalibrated global model), so it is tracked first-class.
    """
    conf = score.max(axis=1)
    pred = score.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    n = y_true.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = int(mask.sum())
        if m:
            ece += (m / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return ece


def _brier_score(y_true: np.ndarray, score: np.ndarray, num_classes: int) -> float:
    """Multiclass Brier score: mean squared error between one-hot truth and probs.

    Sum-over-classes convention (Brier 1950); range [0, 2], lower is better.
    """
    onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((score - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_score: Optional[Sequence] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Compute the full classification benchmark for one evaluation pass.

    Args:
        y_true: ground-truth integer labels, shape (N,).
        y_pred: predicted integer labels, shape (N,).
        y_score: optional class probabilities/logits, shape (N, C). Enables
            log_loss and roc_auc (ovr/macro for multiclass, score-of-positive
            for binary). Pass softmax probabilities for a correct log_loss.
        num_classes: total class count C. Inferred from labels if omitted.
        class_names: human labels for the C classes (for per-class + confusion).

    Returns:
        camelCase dict with macro scalars, the perClass list, confusionMatrix,
        classLabels, and samplesEvaluated. Values that cannot be computed are
        None (never NaN), so the payload is always valid JSON.
    """
    yt = _as_int_array(y_true)
    yp = _as_int_array(y_pred)
    n = int(yt.shape[0])
    if n == 0:
        return {"samplesEvaluated": 0}

    if num_classes is None:
        num_classes = int(max(yt.max(), yp.max())) + 1
    labels = np.arange(num_classes)
    if class_names is not None and len(class_names) == num_classes:
        names = [str(c) for c in class_names]
    else:
        names = [str(int(c)) for c in labels]

    out: dict[str, Any] = {"samplesEvaluated": n, "numClasses": int(num_classes)}

    if _SKLEARN:
        out["accuracy"] = _round(accuracy_score(yt, yp))
        try:
            out["balancedAccuracy"] = _round(balanced_accuracy_score(yt, yp))
        except Exception:
            out["balancedAccuracy"] = None
        for avg in ("macro", "micro", "weighted"):
            p, r, f, _ = precision_recall_fscore_support(
                yt, yp, labels=labels, average=avg, zero_division=0
            )
            cap = avg.capitalize()
            out[f"precision{cap}"] = _round(p)
            out[f"recall{cap}"] = _round(r)
            out[f"f1{cap}"] = _round(f)
        try:
            out["mcc"] = _round(matthews_corrcoef(yt, yp))
        except Exception:
            out["mcc"] = None
        try:
            out["cohenKappa"] = _round(cohen_kappa_score(yt, yp))
        except Exception:
            out["cohenKappa"] = None
        pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(
            yt, yp, labels=labels, average=None, zero_division=0
        )
        cm = confusion_matrix(yt, yp, labels=labels)
    else:  # numpy fallback
        out["accuracy"] = _round(float(np.mean(yt == yp)))
        pc_p, pc_r, pc_f, pc_s = _prf_numpy(yt, yp, labels)
        # macro = unweighted mean; micro = global; weighted = support-weighted
        out["precisionMacro"] = _round(float(np.mean(pc_p)))
        out["recallMacro"] = _round(float(np.mean(pc_r)))
        out["f1Macro"] = _round(float(np.mean(pc_f)))
        out["precisionMicro"] = out["recallMicro"] = out["f1Micro"] = out["accuracy"]
        w = pc_s / max(pc_s.sum(), 1)
        out["precisionWeighted"] = _round(float(np.sum(pc_p * w)))
        out["recallWeighted"] = _round(float(np.sum(pc_r * w)))
        out["f1Weighted"] = _round(float(np.sum(pc_f * w)))
        recalls = [pc_r[i] for i in range(len(labels)) if pc_s[i] > 0]
        out["balancedAccuracy"] = _round(float(np.mean(recalls))) if recalls else None
        out["mcc"] = None
        out["cohenKappa"] = None
        cm = _confusion_numpy(yt, yp, labels)

    # Probability-based metrics — need stored softmax probabilities, not just
    # argmax. ECE/Brier are pure-numpy (no sklearn); roc_auc/log_loss use sklearn.
    out["rocAuc"] = None
    out["logLoss"] = None
    out["ece"] = None
    out["brier"] = None
    if y_score is not None:
        score = np.asarray(y_score, dtype=np.float64)
        if score.ndim == 2 and score.shape[1] == num_classes and score.shape[0] == n:
            out["ece"] = _round(_expected_calibration_error(yt, score))
            out["brier"] = _round(_brier_score(yt, score, int(num_classes)))
            if _SKLEARN:
                try:
                    out["logLoss"] = _round(log_loss(yt, score, labels=labels))
                except Exception:
                    pass
                try:
                    if num_classes == 2:
                        out["rocAuc"] = _round(roc_auc_score(yt, score[:, 1]))
                    else:
                        out["rocAuc"] = _round(
                            roc_auc_score(yt, score, multi_class="ovr", average="macro", labels=labels)
                        )
                except Exception:
                    pass

    out["perClass"] = [
        {
            "label": names[i],
            "precision": _round(float(pc_p[i])),
            "recall": _round(float(pc_r[i])),
            "f1": _round(float(pc_f[i])),
            "support": int(pc_s[i]),
        }
        for i in range(len(labels))
    ]
    out["confusionMatrix"] = cm.astype(int).tolist()
    out["classLabels"] = names
    return out


# ---------------------------------------------------------------------------
# Generative / causal-LM metrics
# ---------------------------------------------------------------------------
def perplexity_from_loss(avg_loss: float) -> float:
    """exp(mean cross-entropy). Clamped so a diverged run reports inf, not overflow."""
    return math.exp(avg_loss) if avg_loss is not None and avg_loss < 30 else float("inf")


def generative_metrics(
    avg_loss: float,
    correct_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Causal-LM benchmark: eval loss, perplexity, and (optional) token accuracy.

    token_accuracy = correct next-token predictions / non-padding target tokens.
    """
    out: dict[str, Any] = {
        "loss": _round(avg_loss),
        "perplexity": _round(perplexity_from_loss(avg_loss)),
    }
    if total_tokens:
        out["tokenAccuracy"] = _round(correct_tokens / total_tokens)
        out["samplesEvaluated"] = int(total_tokens)
    return out


# ---------------------------------------------------------------------------
# Distribution / FL-heterogeneity helpers
# ---------------------------------------------------------------------------
def label_distribution(labels: Sequence, num_classes: Optional[int] = None) -> dict[str, Any]:
    """Per-class counts + normalized Shannon entropy (label-only non-IID proxy).

    normalizedEntropy in [0,1]: 1.0 = perfectly uniform label mix (IID-like),
    near 0 = a single class dominates (severe quantity/label skew). This is the
    cheapest heterogeneity signal computable from labels alone, with no access to
    other clients' data.
    """
    yt = _as_int_array(labels)
    if num_classes is None:
        num_classes = int(yt.max()) + 1 if yt.size else 0
    counts = np.bincount(yt, minlength=num_classes).astype(int)
    total = int(counts.sum())
    if total == 0 or num_classes <= 1:
        return {"counts": counts.tolist(), "total": total, "normalizedEntropy": None}
    p = counts / total
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "counts": counts.tolist(),
        "total": total,
        "normalizedEntropy": _round(entropy / math.log(num_classes)),
    }


def model_size(params) -> dict[str, Any]:
    """Parameter count + float32 size (MB) from a state-dict-like mapping of tensors/arrays."""
    total = 0
    for v in params.values():
        try:
            total += int(v.numel())  # torch tensor
        except AttributeError:
            total += int(np.asarray(v).size)  # numpy array
    return {"paramCount": total, "modelSizeMb": _round(total * 4 / (1024 * 1024))}


# ---------------------------------------------------------------------------
# Round-record assembly (the wire contract with the backend)
# ---------------------------------------------------------------------------
def build_round_record(
    server_round: int,
    *,
    model_type: Optional[str] = None,
    task_type: Optional[str] = None,
    quality: Optional[dict] = None,
    loss: Optional[float] = None,
    round_duration_ms: Optional[int] = None,
    eval_duration_ms: Optional[int] = None,
    param_count: Optional[int] = None,
    model_size_mb: Optional[float] = None,
    client_count: Optional[int] = None,
    target_accuracy: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """Assemble one per-round benchmark record (camelCase) for POST to the backend.

    `quality` is the dict returned by classification_metrics / generative_metrics;
    its keys are merged at the top level. perClass / confusionMatrix / classLabels
    and the open-ended `extra` bag are JSON-serialized by the caller as needed.
    """
    rec: dict[str, Any] = {"serverRound": int(server_round)}
    if model_type is not None:
        rec["modelType"] = model_type
    if task_type is not None:
        rec["taskType"] = task_type
    if quality:
        rec.update(quality)
    if loss is not None and "loss" not in rec:
        rec["loss"] = _round(loss)
    if round_duration_ms is not None:
        rec["roundDurationMs"] = int(round_duration_ms)
    if eval_duration_ms is not None:
        rec["evalDurationMs"] = int(eval_duration_ms)
    if param_count is not None:
        rec["paramCount"] = int(param_count)
    if model_size_mb is not None:
        rec["modelSizeMb"] = _round(model_size_mb, 4)
    if client_count is not None:
        rec["clientCount"] = int(client_count)
    if target_accuracy is not None:
        rec["targetAccuracy"] = _round(target_accuracy)
    if extra:
        rec["extraMetrics"] = extra
    return rec


# ---------------------------------------------------------------------------
# Standalone CLI: score a predictions file and print a benchmark report.
# ---------------------------------------------------------------------------
def _cli(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute a classification/generative benchmark from a predictions file."
    )
    ap.add_argument("--predictions", required=True,
                    help="JSON file with {yTrue:[...], yPred:[...], yScore?:[[...]], "
                         "classNames?:[...], taskType?:'CLASSIFICATION'|'CAUSAL_LM', avgLoss?:float}")
    ap.add_argument("--out", help="Write the metric report JSON here (else stdout).")
    args = ap.parse_args(argv)

    with open(args.predictions) as f:
        payload = json.load(f)

    task = str(payload.get("taskType", "CLASSIFICATION")).upper()
    if task == "CAUSAL_LM":
        report = generative_metrics(
            payload["avgLoss"],
            payload.get("correctTokens"),
            payload.get("totalTokens"),
        )
    else:
        report = classification_metrics(
            payload["yTrue"],
            payload["yPred"],
            y_score=payload.get("yScore"),
            num_classes=payload.get("numClasses"),
            class_names=payload.get("classNames"),
        )

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"benchmark report written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
