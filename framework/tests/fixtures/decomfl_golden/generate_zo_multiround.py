"""Freeze a MULTI-ROUND DeComFL trajectory golden (Python <-> C++ endpoint parity).

Extends the single-step ``zo_*`` golden. Runs the tiny frozen net through N DeComFL
rounds (single client, K local steps x P perturbations, ONE-SIDED forward-difference
g-scalar, float32 arithmetic) and freezes the final flat params + the per-round
g-scalars + the per-round seed matrices, so the C++ mobile core can replay the SAME
trajectory and assert a tolerance-bounded endpoint match.

Why this exists: the single-step goldens prove every DeComFL *kernel* agrees across
Python and C++ (perturbation, g-scalar, forward, safetensors). This freezes their
*composition over rounds* — the one conformance claim the harness did not yet make.
See research/notes/2026-07-15-cross-language-conformance-contract.md.

Contract mirrored from the running code (all citations verified 2026-07-15):
  * update (single client, N=1 so eta/(N*P) = eta/P):
        x_r = x_{r-1} - (eta/P) * sum_{p=0..P-1} g_{r,p} * z_{r,p}
    with K local steps compounding within a round (no 1/K averaging).
    Python: decomfl_client.py:266-290 / decomfl_strategy.py:233-252 (server replay).
    C++:    DeComFLClient.cpp:15-27, rebuildModel :34-45; decomfl_equivalence_test.cpp:39-49.
  * g (one-sided forward diff): g = (L(x + mu*z) - L(x)) / mu, at the round/step-start x.
    Python zeroth_order.py:102-115 ; C++ EtZeroOrder.h:32-39.
  * z = canonical_perturbation(seed, d) (CPU torch.randn, float32). == C++ flat_randn.
  * SEEDS are frozen here (numpy PCG64 default_rng(SEED_BASE), k-outer/p-inner, one
    advancing stream, K*P draws per round) because the C++ side cannot reproduce the
    numpy stream — the gtest READS zo_multiround_seeds.i64.

Run ONLY on an intentional torch bump (torch pinned 2.12.0, matching zo_manifest.json):
    cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate_zo_multiround.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform

import numpy as np
import torch

from fedlearn.estimators.perturbation import canonical_perturbation
from fedlearn.estimators.zeroth_order import ZerothOrderEstimator

from generate_zo import TinyNet  # same seed-0 net the single-step golden freezes

HERE = os.path.dirname(os.path.abspath(__file__))

# --- trajectory config (single client). Kept SMALL: cross-language endpoint drift
#     (forward backend + float32-vs-double g division + z arch-ULP) compounds over
#     N*K*P evals, so few rounds/perturbations keep the tolerance meaningful. ---
N_ROUNDS = 3
K = 1          # local steps / round (framework default)
P = 4          # perturbations / step (small; single-step golden uses 4 seeds)
ETA = 0.02     # learning rate (meaningful movement without amplifying drift)
MU = 0.001     # smoothing param (matches zo_manifest.json)
SEED_BASE = 42  # DeComFL.__init__ default (decomfl_strategy.py:54)


def _load_committed_batch():
    """The SAME batch the single-step golden + the C++ tests read (zo_inputs/zo_targets)."""
    inputs = torch.from_numpy(np.fromfile(os.path.join(HERE, "zo_inputs.f32"), dtype="<f4").reshape(8, 4).copy())
    targets = torch.from_numpy(np.fromfile(os.path.join(HERE, "zo_targets.i64"), dtype="<i8").reshape(8).copy())
    return inputs, targets


def compute_multiround_trajectory(net, zo, x0, inputs, targets, *, n_rounds=N_ROUNDS,
                                  k_steps=K, p_perturb=P, eta=ETA, seed_base=SEED_BASE):
    """Pure trajectory (no disk). Returns {final_flat<f4[d], per_round_g[N][K][P], seeds[N][K][P]}.

    float32 discipline mirrors the C++ core: g cast to float32 before multiply,
    step = float32(eta/P), delta accumulated in float32, x advanced in float32.
    """
    d = int(x0.shape[0])
    x = x0.astype(np.float32).copy()
    rng = np.random.default_rng(seed_base)
    step_coeff = np.float32(eta / p_perturb)
    all_seeds, all_g = [], []
    for _r in range(n_rounds):
        seeds_r, g_r = [], []
        for _k in range(k_steps):
            # all P perturbations are evaluated at the SAME pre-step x, then x advances once
            flat_t = torch.from_numpy(x.copy())
            seeds_kp = [int(rng.integers(0, 2 ** 31 - 1)) for _ in range(p_perturb)]
            delta = np.zeros(d, dtype=np.float32)
            g_kp = []
            for p in range(p_perturb):
                z_t = canonical_perturbation(seeds_kp[p], d)  # float32 CPU tensor
                g = float(zo.compute_gradient_scalar(net, flat_t, z_t, inputs, targets))
                g_kp.append(g)
                delta += np.float32(g) * z_t.numpy().astype(np.float32)
            x = (x - step_coeff * delta).astype(np.float32)
            seeds_r.append(seeds_kp)
            g_r.append(g_kp)
        all_seeds.append(seeds_r)
        all_g.append(g_r)
    return {"final_flat": x.astype("<f4"), "per_round_g": all_g, "seeds": all_seeds}


def main() -> None:
    from fedlearn.communication.safetensors_codec import save_safetensors

    torch.manual_seed(0)
    net = TinyNet().eval()  # identical construction/seed to generate_zo.py -> same x0
    zo = ZerothOrderEstimator(smoothing_param=MU, device="cpu")

    # start from the committed initial flat (== zo._get_flat_params(net)); load the file so
    # the C++ test (readF32("zo_flat.f32")) and Python start from byte-identical params.
    x0 = np.fromfile(os.path.join(HERE, "zo_flat.f32"), dtype="<f4")
    inputs, targets = _load_committed_batch()

    traj = compute_multiround_trajectory(net, zo, x0, inputs, targets)
    final_flat = traj["final_flat"]  # <f4 [d]
    d = int(final_flat.shape[0])

    # raw f32 endpoint (C++ EXPECT_NEAR loop reads this via fedtest::readF32)
    final_flat.tofile(os.path.join(HERE, "zo_multiround_final.f32"))
    final_sha = hashlib.sha256(final_flat.tobytes()).hexdigest()

    # per-round seeds, flattened row-major [r][k][p] -> i64 (C++ readI64 + reshape by N,K,P)
    seeds_flat = np.array(
        [s for rd in traj["seeds"] for st in rd for s in st], dtype="<i8"
    )
    seeds_flat.tofile(os.path.join(HERE, "zo_multiround_seeds.i64"))

    # per-round g-scalars, flattened [r][k][p] -> f64 (C++ reads for a per-round diagnostic
    # assertion; the C++ recomputes its own g and checks it against these within g_atol).
    g_flat = np.array(
        [g for rd in traj["per_round_g"] for st in rd for g in st], dtype="<f8"
    )
    g_flat.tofile(os.path.join(HERE, "zo_multiround_g.f64"))

    # safetensors state-dict of the final flat (byte-exact codec contract, same layout as zo_state)
    named_tensors = []
    off = 0
    for _name, _p in net.named_parameters():
        if not _p.requires_grad:
            continue
        _k = int(_p.numel())
        named_tensors.append((_name, final_flat[off:off + _k].reshape(list(_p.shape))))
        off += _k
    state_blob = save_safetensors(named_tensors, {"num_examples": "8", "num_rounds": str(N_ROUNDS)})
    with open(os.path.join(HERE, "zo_multiround_state.safetensors"), "wb") as fh:
        fh.write(state_blob)
    state_sha = hashlib.sha256(state_blob).hexdigest()

    # final unperturbed loss at the endpoint (diagnostic)
    with torch.no_grad():
        # set net params to final flat via the estimator's own setter, then forward
        zo._set_flat_params(net, torch.from_numpy(final_flat.astype(np.float32)))
        final_loss = float(torch.nn.functional.cross_entropy(net(inputs), targets))

    manifest = {
        "description": "Multi-round DeComFL trajectory golden (Python<->C++ endpoint parity). "
                       "N rounds of x -= (eta/P)*sum_p g_p*z_p, one-sided forward-diff g, float32.",
        "torch_version": torch.__version__.split("+")[0],
        # freeze platform (this trajectory is tolerance-based, never asserted bit-exact —
        # forward-backend + float32-vs-double g + z arch-ULP drift are absorbed by *_atol below).
        "platform_machine": platform.machine(),
        "num_rounds": N_ROUNDS,
        "K": K,
        "P": P,
        "eta": ETA,
        "mu": MU,
        "seed_base": SEED_BASE,
        "flat_dim": d,
        "initial_flat_file": "zo_flat.f32",
        "inputs_file": "zo_inputs.f32",
        "targets_file": "zo_targets.i64",
        "seeds_file": "zo_multiround_seeds.i64",
        "g_file": "zo_multiround_g.f64",
        "seeds": traj["seeds"],           # [N][K][P] ints (also frozen as .i64 for C++)
        "per_round_g": traj["per_round_g"],  # [N][K][P] doubles
        "final_flat_file": "zo_multiround_final.f32",
        "final_flat_sha256": final_sha,
        "state_file": "zo_multiround_state.safetensors",
        "state_sha256": state_sha,
        "final_loss": final_loss,
        # provisional: sized from the single-eval forward tolerance (1e-4) propagated over
        # N*K*P evals; NOT yet validated against a real C++/ExecuTorch run (CI is the first).
        "endpoint_atol": 2e-3,
        "g_atol": 2e-3,
    }
    with open(os.path.join(HERE, "zo_multiround_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"N={N_ROUNDS} K={K} P={P} eta={ETA} mu={MU} d={d}")
    print("final_flat[:5] =", final_flat[:5].tolist())
    print("final_loss =", final_loss)
    print("final_flat_sha256 =", final_sha[:12], "| state_sha256 =", state_sha[:12])
    print("per_round_g =", traj["per_round_g"])


if __name__ == "__main__":
    main()
