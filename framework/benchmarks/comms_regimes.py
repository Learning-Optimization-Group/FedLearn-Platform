"""DA-14 / TE-15 unified — three-regime per-round communication-cost comparison.

TE-15 measured full-model FedAvg vs DeComFL zeroth-order; DA-14 measured full-model vs head-only
(frozen-backbone) FedAvg. Those are two disjoint two-way contrasts. This module unifies them: it puts
ALL THREE per-round wire regimes side by side in ONE comparative table across a backbone-dominant
model-size sweep, so the paper can cite the communication axis once, coherently.

Every byte is a REAL measurement from the SAME production codecs the socket uses
(``benchmarks.wire_bytes``) — nothing here is an analytic estimate:

  (a) full-model FedAvg      — per round a client uploads the entire model state_dict, serialized by
                               the deterministic safetensors codec (``first_order_model_bytes`` over
                               the full state).
  (b) head-only FedAvg       — frozen backbone (the DA-11 subset-federation contract): only the small
                               trainable head rides the wire (``first_order_model_bytes`` over
                               ``estimators.params.trainable_state``). The win grows with backbone size.
  (c) DeComFL (zeroth-order) — per round a client uploads only K*P float64 gradient scalars + K*P
                               int64 seeds, measured on the real ``SubmitGradientScalarsRequest``
                               protobuf (``decomfl_upload_bytes(K, P)``). This payload is O(K*P) and
                               **INDEPENDENT of the model size d** — it does not grow with the backbone.

Honest framing (inherited verbatim from ``benchmarks.wire_bytes``): these are protobuf payload bytes,
before HTTP/2 framing/headers (which add ~1% identically across all three regimes). DeComFL also pays
a ONE-SHOT O(d) initial model download (the full model, materialized on device once at join) — it is
reported SEPARATELY here, never folded into the per-round column, exactly as the DeComFL paper accounts
for it. Over R rounds that one-shot cost amortizes toward zero per round while the first-order regimes
keep paying their full/head payload every single round.

Scope: this module measures ONLY the communication axis. The utility/accuracy trade-off of the
frozen-backbone regime lives in ``benchmarks.frozen_backbone_fl``; the DeComFL model itself is
architecture-agnostic here — the sweep models are backbone-dominant Linear ``_Derived`` shapes chosen
so the three regimes are directly comparable on the wire. Fully deterministic (seeded model init;
byte counts are exact, not sampled).

Reproduce:
    cd framework && PYTHONPATH=src python benchmarks/comms_regimes.py
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from fedlearn.estimators.params import trainable_state
from benchmarks.wire_bytes import decomfl_upload_bytes, decomfl_download_config_bytes, first_order_model_bytes

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Representative DeComFL round shape: K local zeroth-order steps, each probing P perturbation
# directions -> K*P uploaded scalars + K*P seeds. These, NOT the model size, set the DeComFL payload.
DEFAULT_LOCAL_STEPS = 10
DEFAULT_PERTURBATIONS = 10

# A backbone-dominant sweep: d_hidden grows monotonically (so the trainable head grows too) while the
# frozen backbone d_in*d_hidden grows faster (so the head-vs-full win widens). Chosen so all three
# comparison ratios are strictly increasing across the sweep.
DEFAULT_SIZES = [(256, 64, 4), (1024, 256, 4), (4096, 512, 8), (8192, 1024, 10)]

# The honest per-table footnote (kept in sync with benchmarks.wire_bytes's framing).
FOOTNOTE = (
    "Protobuf payload bytes, before HTTP/2 framing/headers (~1% identical across regimes). "
    "DeComFL's per-round upload is O(K*P) scalars+seeds, INDEPENDENT of model size d; its one-shot "
    "O(d) initial model download (full model, once at join) is reported separately and amortizes "
    "toward zero per round over R rounds."
)


class _Derived(nn.Module):
    """A frozen Linear backbone + a trainable Linear head (the DA-14 derived-model shape)."""

    def __init__(self, d_in, d_hidden, n_classes):
        super().__init__()
        self.backbone = nn.Linear(d_in, d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.head(torch.relu(self.backbone(x)))


def _build(d_in, d_hidden, n_classes, seed=0):
    """Deterministic model construction (seeded, isolated RNG) — byte counts depend only on shapes."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return _Derived(d_in, d_hidden, n_classes)


def regime_bytes(sizes, num_local_steps=DEFAULT_LOCAL_STEPS, num_perturbations=DEFAULT_PERTURBATIONS):
    """The three-regime per-round upload wire cost for each ``(d_in, d_hidden, n_classes)``.

    All measured with the production ``benchmarks.wire_bytes`` codecs (real serialized payload bytes):

      * ``full_bytes``    — (a) full-model FedAvg: the entire model state_dict on the wire, per round.
      * ``head_bytes``    — (b) head-only FedAvg: only the trainable head (frozen backbone), per round.
      * ``decomfl_bytes`` — (c) DeComFL: K*P gradient scalars + K*P seeds, per round; O(K*P), NOT O(d).

    Plus the three cross-regime ratios and DeComFL's separately-accounted one-shot O(d) download.

    Returns ``[{size, full_bytes, head_bytes, decomfl_bytes, ratio_full_head, ratio_full_decomfl,
    ratio_head_decomfl, decomfl_oneshot_download_bytes}]``.
    """
    # DeComFL's per-round upload depends ONLY on (K, P) — compute it once; it is the same for every
    # model size in the sweep (the whole point of dimension-free zeroth-order communication).
    decomfl = decomfl_upload_bytes(num_local_steps, num_perturbations)
    out = []
    for (d_in, d_hidden, n_classes) in sizes:
        m = _build(d_in, d_hidden, n_classes)
        full = first_order_model_bytes(OrderedDict(m.state_dict()))
        head = first_order_model_bytes(trainable_state(m))
        out.append({
            "size": [d_in, d_hidden, n_classes],
            "full_bytes": full,
            "head_bytes": head,
            "decomfl_bytes": decomfl,
            "ratio_full_head": round(full / head, 2),
            "ratio_full_decomfl": round(full / decomfl, 2),
            "ratio_head_decomfl": round(head / decomfl, 2),
            # DeComFL still materializes the full model ONCE (O(d)); reported apart from the per-round
            # column so the amortized-to-zero-per-round nature is explicit, not hidden.
            "decomfl_oneshot_download_bytes": full,
        })
    return out


def _write_markdown(path, rows, num_local_steps, num_perturbations):
    with open(path, "w") as f:
        f.write("# Three-regime per-round communication cost (real safetensors + protobuf wire bytes)\n\n")
        f.write(f"DeComFL round shape: K={num_local_steps} local steps x P={num_perturbations} "
                f"perturbations => {num_local_steps * num_perturbations} scalars + seeds uploaded.\n\n")
        f.write("| backbone d_in->d_hidden->classes | (a) full FedAvg | (b) head-only FedAvg | "
                "(c) DeComFL upload | full/head | full/decomfl | head/decomfl |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            s = r["size"]
            f.write(f"| {s[0]}->{s[1]}->{s[2]} | {r['full_bytes']:,} B | {r['head_bytes']:,} B | "
                    f"{r['decomfl_bytes']:,} B | {r['ratio_full_head']}x | "
                    f"{r['ratio_full_decomfl']}x | {r['ratio_head_decomfl']}x |\n")
        f.write("\n## DeComFL one-shot O(d) initial model download (accounted separately)\n\n")
        f.write("| backbone d_in->d_hidden->classes | one-shot model download (once at join) |\n|---|---|\n")
        for r in rows:
            s = r["size"]
            f.write(f"| {s[0]}->{s[1]}->{s[2]} | {r['decomfl_oneshot_download_bytes']:,} B |\n")
        f.write(f"\n> {FOOTNOTE}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-steps", type=int, default=DEFAULT_LOCAL_STEPS,
                    help="K: DeComFL local zeroth-order steps per round")
    ap.add_argument("--perturbations", type=int, default=DEFAULT_PERTURBATIONS,
                    help="P: DeComFL perturbation directions per local step")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "comms_regimes"))
    args = ap.parse_args()

    rows = regime_bytes(DEFAULT_SIZES, args.local_steps, args.perturbations)
    payload = {
        "decomfl_round": {"local_steps": args.local_steps, "perturbations": args.perturbations,
                          "scalars_uploaded": args.local_steps * args.perturbations},
        "per_round_upload": rows,
        "decomfl_per_round_download_config_bytes": decomfl_download_config_bytes(
            args.local_steps, args.perturbations),
        "footnote": FOOTNOTE,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(payload, f, indent=2)
    _write_markdown(args.out + ".md", rows, args.local_steps, args.perturbations)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
