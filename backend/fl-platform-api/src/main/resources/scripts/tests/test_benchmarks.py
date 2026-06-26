import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import benchmarks  # noqa: E402


def test_perfect_classifier():
    yt = [0, 1, 2, 0, 1, 2]
    m = benchmarks.classification_metrics(yt, yt, num_classes=3, class_names=["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["f1Macro"] == 1.0
    assert m["f1Micro"] == 1.0
    assert m["balancedAccuracy"] == 1.0
    assert m["samplesEvaluated"] == 6
    assert m["confusionMatrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert [c["label"] for c in m["perClass"]] == ["a", "b", "c"]
    assert all(c["precision"] == 1.0 and c["recall"] == 1.0 for c in m["perClass"])


def test_known_binary_metrics_match_hand_calc():
    # 4 samples: TP=1 (idx0), FN=1 (idx1), FP=1 (idx2), TN=1 (idx3) for the positive class (1)
    yt = [1, 1, 0, 0]
    yp = [1, 0, 1, 0]
    m = benchmarks.classification_metrics(yt, yp, num_classes=2, class_names=["neg", "pos"])
    assert m["accuracy"] == 0.5  # 2/4 correct
    pos = next(c for c in m["perClass"] if c["label"] == "pos")
    assert pos["precision"] == 0.5  # TP/(TP+FP) = 1/2
    assert pos["recall"] == 0.5     # TP/(TP+FN) = 1/2
    assert pos["f1"] == 0.5
    # micro precision == micro recall == accuracy for single-label multiclass
    assert m["precisionMicro"] == m["recallMicro"] == m["accuracy"]
    assert m["confusionMatrix"] == [[1, 1], [1, 1]]


def test_roc_auc_and_logloss_with_scores():
    yt = [0, 0, 1, 1]
    # well-separated probabilities -> perfect ranking -> AUC 1.0
    score = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    yp = [0, 0, 1, 1]
    m = benchmarks.classification_metrics(yt, yp, y_score=score, num_classes=2)
    assert m["rocAuc"] == 1.0
    assert m["logLoss"] is not None and m["logLoss"] > 0


def test_generative_metrics():
    g = benchmarks.generative_metrics(0.0)
    assert math.isclose(g["perplexity"], 1.0, rel_tol=1e-6)
    g2 = benchmarks.generative_metrics(1.0, correct_tokens=80, total_tokens=100)
    assert math.isclose(g2["perplexity"], math.e, rel_tol=1e-6)
    assert g2["tokenAccuracy"] == 0.8
    assert benchmarks.generative_metrics(9999.0)["perplexity"] is None  # inf -> None


def test_calibration_metrics():
    yt = [0, 0, 1, 1]
    # Perfectly confident AND correct -> ECE 0, Brier 0.
    perfect = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    m = benchmarks.classification_metrics(yt, [0, 0, 1, 1], y_score=perfect, num_classes=2)
    assert m["ece"] == 0.0
    assert m["brier"] == 0.0
    # Confidently WRONG -> ECE saturates near 1, Brier near 2 (worst case).
    bad = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    m2 = benchmarks.classification_metrics(yt, [1, 1, 0, 0], y_score=bad, num_classes=2)
    assert m2["ece"] is not None and m2["ece"] > 0.5
    assert m2["brier"] is not None and m2["brier"] > 1.0


def test_label_distribution_entropy():
    uniform = benchmarks.label_distribution([0, 1, 2, 3], num_classes=4)
    assert uniform["normalizedEntropy"] == 1.0  # perfectly uniform
    skewed = benchmarks.label_distribution([0, 0, 0, 0, 1], num_classes=4)
    assert skewed["normalizedEntropy"] < 0.5    # one class dominates
    assert skewed["counts"] == [4, 1, 0, 0]


def test_model_size():
    params = {"w": np.zeros((10, 10), dtype=np.float32), "b": np.zeros(10, dtype=np.float32)}
    s = benchmarks.model_size(params)
    assert s["paramCount"] == 110
    assert s["modelSizeMb"] == round(110 * 4 / (1024 * 1024), 6)


def test_build_round_record_merges_quality():
    q = benchmarks.classification_metrics([0, 1], [0, 1], num_classes=2)
    rec = benchmarks.build_round_record(
        3, model_type="CNN", quality=q, round_duration_ms=1200, client_count=2, param_count=110
    )
    assert rec["serverRound"] == 3
    assert rec["modelType"] == "CNN"
    assert rec["accuracy"] == 1.0
    assert rec["roundDurationMs"] == 1200
    assert rec["clientCount"] == 2


def test_empty_input_is_safe():
    assert benchmarks.classification_metrics([], [])["samplesEvaluated"] == 0
