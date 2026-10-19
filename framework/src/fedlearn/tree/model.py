"""Pinned XGBoost model IO and bounded, numeric-only tree concatenation."""
from __future__ import annotations

import copy
import json
import math

import numpy as np

from .schema import FEATURES, require_schema
VERSION = "3.4.1"
MAX_MODEL_BYTES = 16 * 1024 * 1024
MAX_DELTA_BYTES = 64 * 1024
MAX_TREES = 10_000


def _xgb():
    import xgboost
    if xgboost.__version__ != VERSION:
        raise ValueError(f"XGBoost {VERSION} is required")
    return xgboost


def canonical(model):
    return json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("unsupported model fields")


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid tree integer")
    return value


def _finite(value, minimum=-float(np.finfo(np.float32).max), maximum=float(np.finfo(np.float32).max)):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("invalid tree numeric value")


def _validate_tree(tree, index, width):
    numeric = ("base_weights", "loss_changes", "split_conditions", "sum_hessian")
    integer = ("default_left", "left_children", "right_children", "parents", "split_indices", "split_type")
    categories = ("categories", "categories_nodes", "categories_segments", "categories_sizes")
    _keys(tree, (*numeric, *integer, *categories, "id", "tree_param"))
    if type(tree["id"]) is not int or tree["id"] != index:
        raise ValueError("tree ID is not canonical")
    param = tree["tree_param"]
    _keys(param, ("num_deleted", "num_feature", "num_nodes", "size_leaf_vector"))
    if param["num_deleted"] != "0" or param["num_feature"] != str(width) or param["size_leaf_vector"] != "1":
        raise ValueError("unsupported tree parameters")
    nodes = param["num_nodes"]
    if type(nodes) is not str or nodes not in {str(n) for n in range(1, 16)}:
        raise ValueError("tree node limit exceeded")
    count = int(nodes)
    if any(tree[key] != [] for key in categories):
        raise ValueError("categorical splits are unsupported")
    for key in (*numeric, *integer):
        if not isinstance(tree[key], list) or len(tree[key]) != count:
            raise ValueError("inconsistent tree arrays")
        for value in tree[key]:
            if key in numeric:
                _finite(value, 0 if key == "sum_hessian" else -float(np.finfo(np.float32).max))
            elif key in ("default_left", "split_type"):
                _integer(value, 0, 1 if key == "default_left" else 0)
            elif key == "split_indices":
                _integer(value, 0, width - 1)
            elif key == "parents":
                _integer(value, 0, 2147483647)
            else:
                _integer(value, -1, count - 1)
    if tree["parents"][0] != 2147483647:
        raise ValueError("invalid root parent")
    visited = set()

    def visit(node, depth):
        if node in visited or depth > 3:
            raise ValueError("cycle, duplicate child or excessive tree depth")
        visited.add(node)
        left, right = tree["left_children"][node], tree["right_children"][node]
        if (left == -1) != (right == -1):
            raise ValueError("incomplete split")
        if left != -1:
            for child in (left, right):
                if tree["parents"][child] != node:
                    raise ValueError("inconsistent parent")
                visit(child, depth + 1)
    visit(0, 0)
    if len(visited) != count:
        raise ValueError("unreachable tree node")


def validate_model(blob: bytes, expected_trees=None, *, feature_names=None):
    limit = MAX_DELTA_BYTES if expected_trees == 1 else MAX_MODEL_BYTES
    if not isinstance(blob, bytes) or not 0 < len(blob) <= limit:
        raise ValueError("model byte limit exceeded")
    try:
        model = json.loads(blob, object_pairs_hook=_unique)
        _keys(model, ("learner", "version"))
        if model["version"] != [3, 4, 1] or any(type(value) is not int for value in model["version"]):
            raise ValueError("wrong XGBoost model version")
        learner = model["learner"]
        _keys(learner, ("attributes", "feature_names", "feature_types", "gradient_booster", "learner_model_param", "objective"))
        names = require_schema(learner["feature_names"])
        if feature_names is not None and names != require_schema(feature_names):
            raise ValueError("model differs from the expected feature schema")
        width = len(names)
        if learner["attributes"] != {} or learner["feature_types"] != ["float"] * width:
            raise ValueError("incompatible feature schema")
        if learner["objective"] != {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}:
            raise ValueError("incompatible objective")
        if learner["learner_model_param"] != {"base_score": "[5E-1]", "boost_from_average": "0", "num_class": "0", "num_feature": str(width), "num_target": "1"}:
            raise ValueError("incompatible base prediction or learner shape")
        booster = learner["gradient_booster"]
        _keys(booster, ("name", "model"))
        if booster["name"] != "gbtree":
            raise ValueError("unsupported booster")
        ensemble = booster["model"]
        _keys(ensemble, ("cats", "gbtree_model_param", "iteration_indptr", "tree_info", "trees"))
        if ensemble["cats"] != {"enc": [], "feature_segments": [], "sorted_idx": []}:
            raise ValueError("categorical encodings are unsupported")
        trees = ensemble["trees"]
        if not isinstance(trees, list) or len(trees) > MAX_TREES or (expected_trees is not None and len(trees) != expected_trees):
            raise ValueError("wrong number of contributed trees")
        count = len(trees)
        if ensemble["gbtree_model_param"] != {"num_parallel_tree": "1", "num_trees": str(count)}:
            raise ValueError("inconsistent tree count")
        if ensemble["tree_info"] != [0] * count or ensemble["iteration_indptr"] != list(range(count + 1)):
            raise ValueError("inconsistent tree groups")
        if any(type(value) is not int for value in ensemble["tree_info"] + ensemble["iteration_indptr"]):
            raise ValueError("tree groups require integers")
        for index, tree in enumerate(trees):
            _validate_tree(tree, index, width)
        return model
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("malformed XGBoost model") from exc


def load_model(blob):
    validate_model(blob)
    booster = _xgb().Booster(params={"device": "cpu", "nthread": 1})
    booster.load_model(bytearray(blob))
    return booster


def initial_model(*, feature_names=FEATURES):
    names = require_schema(feature_names)
    xgb = _xgb()
    # Zero rounds establishes the schema and explicit intercept without fitting data.
    matrix = xgb.DMatrix(np.zeros((1, len(names)), dtype="float32"), label=[0],
                         feature_names=list(names), feature_types=["float"] * len(names))
    booster = xgb.train({"objective": "binary:logistic", "base_score": .5,
                         "device": "cpu", "nthread": 1}, matrix, num_boost_round=0)
    raw = bytes(booster.save_raw(raw_format="json"))
    return canonical(validate_model(raw, expected_trees=0))


def train_delta(parent, features, labels, *, cohort_size, seed):
    names = tuple(validate_model(parent)["learner"]["feature_names"])
    x, y = np.asarray(features, dtype="float32"), np.asarray(labels, dtype="float32")
    if x.ndim != 2 or x.shape[1] != len(names) or not 1 <= len(x) <= 1_000_000 or y.shape != (len(x),):
        raise ValueError("expected contracted features and one binary label per row")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("features must be finite and labels binary")
    _integer(cohort_size, 1, 100)
    _integer(seed, 0, 2**64 - 1)
    xgb = _xgb()
    prior = load_model(parent)
    matrix = xgb.DMatrix(x, label=y, feature_names=list(names), feature_types=["float"] * len(names))
    trained = xgb.train({"objective": "binary:logistic", "base_score": .5, "max_depth": 3,
                         "eta": .3 / cohort_size, "tree_method": "hist", "device": "cpu",
                         "nthread": 1, "seed": seed % (2**31), "subsample": 1,
                         "colsample_bytree": 1}, matrix, num_boost_round=1, xgb_model=prior)
    rounds = trained.num_boosted_rounds()
    delta = bytes(trained[rounds - 1:rounds].save_raw(raw_format="json"))
    return canonical(validate_model(delta, expected_trees=1, feature_names=names))


def merge_trees(parent, contributions):
    model = copy.deepcopy(validate_model(parent))
    ids = [client_id for client_id, _ in contributions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate client contributions")
    ensemble = model["learner"]["gradient_booster"]["model"]
    for client_id, blob in sorted(contributions):
        delta = validate_model(blob, expected_trees=1, feature_names=model["learner"]["feature_names"])
        tree = copy.deepcopy(delta["learner"]["gradient_booster"]["model"]["trees"][0])
        tree["id"] = len(ensemble["trees"])
        ensemble["trees"].append(tree)
        ensemble["tree_info"].append(0)
        ensemble["iteration_indptr"].append(len(ensemble["trees"]))
    ensemble["gbtree_model_param"]["num_trees"] = str(len(ensemble["trees"]))
    blob = canonical(model)
    validate_model(blob)
    return blob
