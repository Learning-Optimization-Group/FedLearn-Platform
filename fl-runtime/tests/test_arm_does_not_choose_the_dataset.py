"""The training arm must not decide which DATASET the client loads.

Found by a live federation. The client built the correct model for its recipe, then crashed::

    RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
                  but got input of size: [32, 256]

A CNN receiving a batch of 256-dim vectors: it had been handed FROZEN_DEMO's synthetic *vector*
dataset while holding a CIFAR-10 convolutional model.

The cause is mine, from P1-1b. `USE_DERIVED` used to mean "this is the FROZEN_DEMO recipe", so
keying dataset selection on it was correct-by-accident. P1-1b redefined it as "this arm federates a
trainable subset" — true for FROZEN_HEAD on *any* recipe. The model-building site was generalised
to build the selected recipe; `load_data` was not, and kept loading FROZEN_DEMO's data for every
frozen run.

The invariant, stated plainly: **the dataset is a property of the RECIPE; the arm only decides which
parameters are trainable and federated.** Freezing a backbone does not change what you train on.
"""

import ast
import os

HERE = os.path.dirname(__file__)
CLIENT = os.path.join(HERE, "..", "client.py")


def _load_data_fn():
    tree = ast.parse(open(CLIENT).read())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_data"]
    assert fns, "client.load_data() not found"
    return fns[0]


def _guards_mentioning(fn, name):
    """Line numbers of `if` guards inside fn whose test references `name`."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Name) and sub.id == name:
                    out.append(node.lineno)
    return out


def _fn(name):
    tree = ast.parse(open(CLIENT).read())
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert fns, f"client.{name}() not found"
    return fns[0]


def test_batch_handling_does_not_branch_on_the_arm():
    """The same defect, one layer deeper: train() chose the batch SHAPE from the arm.

    Fixing load_data alone got a CIFAR batch to a frozen CNN and then died differently --
    `'str' object has no attribute 'to'` -- because train()'s USE_DERIVED branch unpacks
    `features, labels = batch`, and a CIFAR batch is a dict, so that unpacked its KEYS.
    Batch shape is a property of the dataset, which follows the recipe.
    """
    offenders = _guards_mentioning(_fn("train"), "USE_DERIVED")
    assert not offenders, (
        f"client.train() branches on USE_DERIVED at line(s) {offenders} to decide how to unpack a "
        f"batch. That is a dataset property, not an arm property.")


def test_dataset_selection_does_not_branch_on_the_arm():
    """THE regression. USE_DERIVED describes the ARM; it must not select the dataset."""
    offenders = _guards_mentioning(_load_data_fn(), "USE_DERIVED")
    assert not offenders, (
        f"client.load_data() branches on USE_DERIVED at line(s) {offenders}. USE_DERIVED means "
        f"'this arm federates a subset', which is true for FROZEN_HEAD on ANY recipe — so this "
        f"hands FROZEN_DEMO's synthetic vector data to whatever model the recipe built. The "
        f"dataset must follow the recipe (MODEL_TYPE), not the arm.")


def test_the_frozen_demo_dataset_is_reached_by_recipe_not_by_arm():
    """FROZEN_DEMO still needs its synthetic shard — reached because it IS that recipe."""
    src = ast.unparse(_load_data_fn())
    assert "FROZEN_DEMO" in src, "the FROZEN_DEMO shard is no longer reachable at all"
    assert "MODEL_TYPE" in src, \
        "load_data must select on the recipe (MODEL_TYPE), which is what makes the shard reachable"


def test_the_model_build_site_still_honours_the_arm():
    """The arm SHOULD drive model construction — that half of P1-1b was right, and must stay."""
    tree = ast.parse(open(CLIENT).read())
    cls_fns = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "_build_model")]
    src = "\n".join(ast.unparse(f) for f in cls_fns)
    assert "apply_arm" in src, "the client no longer applies the arm when building its model"
