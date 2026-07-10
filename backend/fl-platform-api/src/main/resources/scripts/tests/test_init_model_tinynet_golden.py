import os, sys, subprocess
import numpy as np

HERE = os.path.dirname(__file__)
SCRIPTS = os.path.join(HERE, "..")


def test_init_model_tinynet_golden_saves_trainable_only(tmp_path):
    """DeComFL federates the trainable params ONLY; the frozen fc2 is deterministically rebuilt on
    every peer (server + phone + desktop clients). So init_model must persist the requires_grad-
    filtered trainable layout (25 scalars = fc1) — NOT the full 43-param state_dict — otherwise the
    server's DeComFL model_dim becomes 43 and rejects every 25-dim client
    (see framework decomfl_strategy.py:281-288 + estimators.params.trainable_state).
    """
    out = str(tmp_path / "tinynet.npz")
    subprocess.run(
        [sys.executable, "init_model.py", "--model-type", "TINYNET_GOLDEN",
         "--model-name", "tinynet_golden", "--optimizer", "SGD", "--out", out],
        cwd=SCRIPTS, check=True)
    npz = np.load(out)
    dec = {k.replace("__DOT__", ".") for k in npz.keys()}
    # trainable-only: exactly fc1 (weight + bias); the frozen fc2 must NOT be saved.
    assert dec == {"fc1.weight", "fc1.bias"}, f"expected fc1-only, got {sorted(dec)}"
    total = int(sum(npz[k].size for k in npz.keys()))
    assert total == 25, f"expected 25 trainable scalars (fc1: 5x4 + 5), got {total}"
