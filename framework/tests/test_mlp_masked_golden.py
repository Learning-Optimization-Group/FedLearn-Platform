"""Stage 4 S3b: the MLP masked-dropout golden the native trainer replays is torch's, and far from every wrong mask."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN_DIR = os.path.join(REPO, "framework", "tests", "fixtures", "mlp_golden")
sys.path.insert(0, GOLDEN_DIR)
sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))
sys.path.insert(0, os.path.join(REPO, "fl-runtime"))


def test_the_masked_mlp_golden_is_reproduced_by_torch_and_far_from_every_control():
    import generate_mlp_golden as g
    from fedlearn.contract.dropout_masks import dropout_mask
    with open(os.path.join(GOLDEN_DIR, "mlp_manifest.json")) as fh:
        manifest = json.load(fh)
    got = g.train(lambda step, layer, rows, width: dropout_mask(rows * width, 0.3, g.SEED, g.ROUND, step, layer))
    golden = np.fromfile(os.path.join(GOLDEN_DIR, manifest["final_flat_file"]), dtype="<f4")
    assert np.abs(got - golden).max() < 1e-6
    assert set(manifest["control_separation"]) == {"dropout_off", "step_counter_reset_each_epoch", "layers_swapped",
                                                   "wrong_seed"}
    assert min(manifest["control_separation"].values()) > 10 * manifest["endpoint_atol"]


def test_the_mlp_recipe_states_its_two_dropout_layers():
    import pte_export
    import generate_mlp_golden as g
    assert pte_export.dropout_layers(g.build_net()) == [("dropout1", 0.3), ("dropout2", 0.3)]
