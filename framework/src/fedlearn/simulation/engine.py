"""Config-driven simulation engine: one JSON config in, one complete record out.

Wires a workload, the production ``ComposedStrategy`` and ``FLCoordinator`` (via
``SimulatedFederation``), bad-client behaviours, local or central DP, and cost and metric
accounting. Measured wall-clock (``time.wall_s``) and the simulated deployment clock
(``time.simulated_round_s``) are separate fields and must not be compared with each other.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from fedlearn.estimators.params import trainable_state
from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon, required_noise_multiplier
from fedlearn.privacy.local_dp import privatize
from fedlearn.server.composed_strategy import ComposedStrategy
from fedlearn.server.subset_federation import apply_trainable_subset
from fedlearn.simulation import costs
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import ConfigError, config_hash, resolve, tolerance_flags
from fedlearn.simulation.federation import SimulatedFederation
from fedlearn.simulation.metrics import (
    client_fairness, deviation_ratio, exclusion_rates, flat, mean_pairwise_cosine, state_digest,
)
from fedlearn.simulation.workloads import FeatureHeadWorkload

UTILITY_KEYS = ("accuracy", "auc", "loss", "sensitivity", "specificity", "balanced_accuracy", "ece",
                "backdoor_success")


class HeadClient:
    """Trains the head on one client's (possibly poisoned) data; full batch unless batch_size is set.

    Training is the same computation every round, so its FLOPs are measured once per client with
    ``FlopCounterMode`` and the analytic count is stored beside it as the cross-check.
    """

    def __init__(self, workload, cid, plan, client_cfg, data_seed, flops, clock):
        self.workload, self.cid, self.plan, self.cfg = workload, cid, plan, client_cfg
        self.data_seed, self.flops, self.clock = data_seed, flops, clock

    def _train(self, net, x, y, global_params):
        opt = torch.optim.SGD(net.parameters(), lr=float(self.cfg["lr"]))
        mu = float(self.cfg["proximal_mu"]) if self.cfg["algorithm"] == "fedprox" else 0.0
        anchors = {k: global_params[k].detach().float() for k in trainable_state(net)} if mu > 0 else None
        bs = self.cfg["batch_size"]
        net.train()
        for _ in range(int(self.cfg["local_epochs"])):
            batches = [slice(None)] if bs is None else list(torch.randperm(len(y)).split(int(bs)))
            for sel in batches:
                opt.zero_grad()
                loss = F.cross_entropy(net(x[sel]), y[sel])
                if mu > 0:
                    loss = loss + 0.5 * mu * sum(((p - anchors[name]) ** 2).sum()
                                                 for name, p in net.named_parameters() if name in anchors)
                loss.backward()
                opt.step()

    def fit(self, global_params, config):
        net = self.workload.build_model()
        apply_trainable_subset(net, global_params)
        x, y, idx = self.workload.client_data(self.cid)
        x, y, train = self.plan.training_data(self.cid, x, y, self.workload.trigger_rows(idx), self.data_seed)
        if not train:  # free-rider: no local computation at all
            self.flops["measured"].setdefault(self.cid, 0)
            self.flops["analytic"].setdefault(self.cid, 0)
            return trainable_state(net), int(len(y))
        with self.clock.stage("train"):
            if self.cid in self.flops["measured"]:
                self._train(net, x, y, global_params)
            else:
                with FlopCounterMode(display=False) as fc:
                    self._train(net, x, y, global_params)
                self.flops["measured"][self.cid] = int(fc.get_total_flops())
                self.flops["analytic"][self.cid] = costs.analytic_train_flops(
                    self.workload.layer_dims(), int(len(y)), int(self.cfg["local_epochs"]))
        return trainable_state(net), int(len(y))


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=__file__.rsplit("/src/", 1)[0], timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _start_checks(resolved: Dict, workload) -> None:
    """Fail at start, not mid-run, on what only the features file can tell: the dataset it holds,
    its class count (a backdoor target must be a class), and whether it has triggered features
    whose trigger agrees with the config's."""
    have_dataset = workload.info.get("dataset")
    if have_dataset is not None and have_dataset != resolved["workload"]["dataset"]:
        raise ConfigError("workload.dataset",
                          f"config says {resolved['workload']['dataset']!r} but the features file holds {have_dataset!r}")
    for i, b in enumerate(resolved["behaviors"]):
        if b["type"] != "backdoor":
            continue
        if not 0 <= int(b["target_label"]) < workload.n_classes:
            raise ConfigError(f"behaviors[{i}].target_label",
                              f"must be a class index in [0, {workload.n_classes})")
        if not workload.has_trigger:
            raise ValueError("backdoor needs triggered features: prepare the features file with --trigger-patch")
        if b["trigger"] is not None:
            have = workload.info.get("trigger") or {}
            wrong = {k: v for k, v in b["trigger"].items() if have.get(k) != v}
            if wrong:
                raise ValueError(f"backdoor trigger {wrong} does not match the features file's trigger {have}")


def run(cfg: Dict, on_round: Optional[Callable[[Dict], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None) -> Dict:
    r = resolve(cfg)
    rss_before = costs.peak_rss_bytes()
    seeds, fed, sv, pv = r["seeds"], r["federation"], r["server"], r["privacy"]
    net_cfg = fed["network"]
    wl = FeatureHeadWorkload(r)
    plan = BehaviorPlan(r["behaviors"], int(fed["num_clients"]), int(seeds["attack"]), wl.n_classes)
    _start_checks(r, wl)
    rounds = int(fed["rounds"])
    z = (float(required_noise_multiplier(float(pv["epsilon"]), 1.0, rounds, float(pv["delta"])))
         if pv["mode"] != "none" else 0.0)
    rule_params = dict(sv["rule_params"])
    if rule_params["byzantine_fraction"] is None:
        rule_params["byzantine_fraction"] = plan.malicious_fraction
    init_params = trainable_state(wl.build_model(int(seeds["init"])))
    d = sum(v.numel() for v in init_params.values())
    # The server bounds every upload at the norm an honest upload can have: C, or under local DP the
    # privatized norm C + ||N(0, (2zC)^2 I_d)||, bounded by 2zC(sqrt(d) + 4) except w.p. <= e^-8.
    # Re-clipping privatized uploads at C would shrink honest steps by ~1/(2z sqrt(d)).
    server_clip = sv["clip_norm"]
    if pv["mode"] == "local":
        c = float(sv["clip_norm"])
        server_clip = c + 2.0 * z * c * (math.sqrt(d) + 4.0)
    clock = costs.StageClock()
    evaluate = wl.evaluate_fn(plan.backdoor_target())
    state = {"round0_digest": None, "params": None, "refusals": [],
             "client_bytes": {c: 0 for c in range(int(fed["num_clients"]))}}

    def timed_eval(server_round, params):
        state["params"] = params
        with clock.stage("eval"):
            return evaluate(server_round, params)

    strategy = ComposedStrategy(
        init_params, rule=sv["rule"], clip_norm=server_clip, rule_params=rule_params,
        central_noise_multiplier=z if pv["mode"] == "central" else 0.0, noise_seed=int(seeds["noise"]),
        optimizer=sv["optimizer"]["type"], optimizer_lr=float(sv["optimizer"]["lr"]),
        evaluate_fn=timed_eval, clients_per_round=int(fed["clients_per_round"]))
    flops: Dict[str, Dict[int, int]] = {"measured": {}, "analytic": {}}
    local_gen = torch.Generator().manual_seed(int(seeds["noise"]) + 1)
    per_transfer = costs.transfer_bytes(init_params)
    attackers = plan.malicious_ids()

    def client_factory(cid, client_rng):
        return HeadClient(wl, cid, plan, r["client"], client_rng.torch_seed(), flops, clock)

    def preview(subs):
        return strategy.preview([(str(c), p, n) for c, p, n in subs]) if subs else None

    def round_hook(round_num, collected, global_params, selected, dropped):
        strategy.last_round_stats = {}
        state["params"] = None
        keys = list(global_params.keys())
        records = {cid: {"client_state": p, "n_examples": n,
                         "honest_delta": OrderedDict((k, p[k].float() - global_params[k].float()) for k in keys)}
                   for cid, p, n in collected}
        uploads = {cid: rec["client_state"] for cid, rec in records.items()}
        with clock.stage("local_dp"):
            if pv["mode"] == "local":
                for cid in sorted(records):
                    if cid not in attackers:
                        delta = privatize(records[cid]["honest_delta"], float(sv["clip_norm"]), z, local_gen)
                        uploads[cid] = OrderedDict((k, global_params[k].float() + delta[k]) for k in keys)
        with clock.stage("attack"):
            uploads.update(plan.upload_overrides(records, global_params, round_num))
        if state["round0_digest"] is None:
            state["round0_digest"] = state_digest(uploads[c] for c in sorted(uploads))
        for c in selected:
            state["client_bytes"][c] += per_transfer
        for c in records:
            state["client_bytes"][c] += per_transfer
        finish = {cid: costs.finish_seconds(flops["measured"].get(cid, 0), plan.slowdown(cid), per_transfer,
                                            per_transfer, net_cfg) for cid in records}
        deadline = net_cfg["round_deadline_s"]
        late = {cid for cid, t in finish.items() if deadline is not None and t > float(deadline)}
        submissions = [(cid, uploads[cid], records[cid]["n_examples"]) for cid in sorted(uploads) if cid not in late]
        agg_all = preview(submissions)
        agg_honest = preview([s for s in submissions if s[0] not in attackers])
        dev = (deviation_ratio(agg_all, agg_honest, global_params)
               if agg_all is not None and agg_honest is not None else None)
        up_norms = {cid: float(flat(OrderedDict((k, uploads[cid][k].float() - global_params[k].float())
                                                for k in keys), keys).norm()) for cid in uploads}
        honest_norms = [v for c, v in up_norms.items() if c not in attackers]
        attack_norms = [v for c, v in up_norms.items() if c in attackers]
        extra = {
            "participants": {"sampled": len(selected), "dropped": len(dropped), "straggled": len(late),
                             "aggregated": len(submissions)},
            "comm": {"down_bytes": per_transfer * len(selected), "up_bytes": per_transfer * len(records)},
            # A server with a deadline cannot know a client is late or gone until the deadline passes.
            "simulated_round_s": (float(deadline) if deadline is not None and (late or dropped)
                                  else max([finish[c] for c, _, _ in submissions], default=0.0)),
            "attack_note": plan.last_attack_note,
            "client_flops": sum(flops["measured"].get(c, 0) for c in records),
            "client_flops_analytic": sum(flops["analytic"].get(c, 0) for c in records),
            "robustness": {"deviation_ratio": dev},
            "updates": {"honest_norm_mean": float(np.mean(honest_norms)) if honest_norms else None,
                        "attacker_norm_mean": float(np.mean(attack_norms)) if attack_norms else None,
                        "honest_cosine": mean_pairwise_cosine(
                            [flat(records[c]["honest_delta"], keys) for c in sorted(records) if c not in attackers])},
        }
        return submissions, extra

    lines: List[Dict] = []

    def to_line(rec) -> None:
        st = strategy.last_round_stats
        m = rec.metrics or {}
        extra = rec.extra or {}
        every = int(r["eval"]["per_client_every"])
        fairness = None
        if every > 0 and rec.round % every == 0 and state["params"] is not None:
            with clock.stage("eval"):
                fairness = client_fairness(wl.per_client_accuracy(state["params"]))
        t = clock.take()
        if st.get("failed"):
            refusal = st.get("message")
        elif not st:  # the strategy never ran: the coordinator resolved the round without aggregating
            refusal = sim.coordinator.last_round_message or "round not aggregated"
        else:
            refusal = None
        if refusal is not None:
            state["refusals"].append({"round": rec.round, "reason": refusal})
        releases = len(lines) + 1
        eps_spent = (float(get_epsilon(compute_rdp(1.0, z, releases), float(pv["delta"]))[0])
                     if pv["mode"] != "none" else None)
        noise_std = st.get("noise_std") or 0.0
        agg_norm = st.get("aggregate_delta_norm")
        line = {
            "round": rec.round, "forced": rec.forced,
            # A refused round has no model of its own; the coordinator's last metrics belong to an
            # earlier round and must not be reported as this one's.
            **{k: (None if refusal is not None else (rec.loss if k == "loss" else m.get(k))) for k in UTILITY_KEYS},
            "attack_note": extra.get("attack_note"),
            "fairness": fairness,
            "participants": extra.get("participants"), "comm": extra.get("comm"),
            "time": {"wall_s": {**t["wall_s"], **st.get("timings", {})}, "cpu_s": t["cpu_s"],
                     "simulated_round_s": extra.get("simulated_round_s")},
            "compute": {"client_flops": extra.get("client_flops"),
                        "client_flops_analytic": extra.get("client_flops_analytic"),
                        "server_flops_est": costs.server_flops_estimate(
                            sv["rule"], int(st.get("aggregated", 0)), d, int(rule_params["iterations"])),
                        "cpu_s": sum(t["cpu_s"].values())},
            "privacy": {"epsilon_spent": eps_spent, "noise_std": noise_std, "sensitivity": st.get("sensitivity"),
                        "clip_rate": st.get("clip_rate"), "mean_shrink": st.get("mean_shrink"),
                        "noise_to_signal": (noise_std * math.sqrt(d) / agg_norm) if (noise_std and agg_norm) else None},
            "robustness": {**extra.get("robustness", {}),
                           "exclusion": exclusion_rates(st.get("kept_share"), {str(c) for c in attackers})},
            "updates": extra.get("updates"),
            "refusal": refusal,
        }
        lines.append(line)
        if on_round is not None:
            on_round(line)

    sim = SimulatedFederation(strategy, client_factory, int(fed["num_clients"]), int(fed["clients_per_round"]),
                              seed=int(seeds["client"]), initial_parameters=init_params,
                              dropout_rate=plan.dropout_fraction, round_hook=round_hook)
    result = sim.run(rounds, on_round=to_line, should_stop=should_stop)
    final = result.final_params
    # ru_maxrss is a process-lifetime peak: it is this run's own peak only if the run raised it.
    rss_after = costs.peak_rss_bytes()
    totals = {
        "down_bytes": sum((l["comm"] or {}).get("down_bytes", 0) for l in lines),
        "up_bytes": sum((l["comm"] or {}).get("up_bytes", 0) for l in lines),
        "per_client_bytes": {str(c): b for c, b in state["client_bytes"].items()},
        "simulated_seconds": sum(l["time"]["simulated_round_s"] or 0.0 for l in lines),
        "client_flops": sum(l["compute"]["client_flops"] or 0 for l in lines),
        "client_flops_analytic": sum(l["compute"]["client_flops_analytic"] or 0 for l in lines),
        "cpu_seconds": sum(l["compute"]["cpu_s"] for l in lines),
    }
    meta = {
        "config": r, "config_hash": config_hash(r), "seeds": seeds, "flags": tolerance_flags(r),
        "workload": {**wl.info, "feat_dim": wl.feat_dim, "n_classes": wl.n_classes,
                     "n_train": int(len(wl.train_y)), "n_test": int(len(wl.test_y)), "parameters": d},
        "noise_multiplier": z, "adjacency": pv["adjacency"], "privacy_mode": pv["mode"],
        "server_clip_norm": server_clip,
        "accounted_epsilon": (float(get_epsilon(compute_rdp(1.0, z, rounds), float(pv["delta"]))[0])
                              if pv["mode"] != "none" else None),
        "behavior_assignment": plan.assignment(), "totals": totals, "refusals": state["refusals"],
        "round0_upload_digest": state["round0_digest"],
        "final_digest": state_digest([final]) if final else None,
        "stopped_early": result.stopped_early, "stop_reason": result.stop_reason,
        "peak_rss_bytes": rss_after, "peak_rss_scope": "run" if rss_after > rss_before else "process",
        "wall_seconds": result.meta.get("wall_seconds"),
        "provenance": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                       "platform": platform.platform(), "host": platform.node(), "git_commit": _git_commit()},
        "clocks": "time.wall_s is measured on this machine; time.simulated_round_s is the deployment model",
    }
    return {"meta": meta, "rounds": lines}


def _file_name(name: str, resolved: Dict) -> str:
    """A safe, unique file name for a matrix run. Labels built from axis values can hold path
    separators or grow past filesystem limits; those get a slug plus the run's config hash, and the
    full label stays in the record and the index."""
    import re

    slug = re.sub(r"[^A-Za-z0-9._=+-]", "-", name)
    if slug == name and len(name) <= 150:
        return f"{name}.json"
    return f"{slug[:150]}__{config_hash(resolved)[:8]}.json"


def run_matrix(cfg: Dict, out_dir: str, stream: Optional[Callable[[str, Dict], None]] = None) -> List[Dict]:
    """Run every matrix cell; write one JSON per run plus ``index.json``. A failing run is recorded
    with its traceback and the remaining runs continue (failures are results)."""
    import os
    import traceback
    from fedlearn.simulation.config import expand_matrix

    os.makedirs(out_dir, exist_ok=True)
    index = []
    for name, resolved in expand_matrix(cfg):
        file = _file_name(name, resolved)
        try:
            rec = run(resolved, on_round=(lambda line, n=name: stream(n, line)) if stream else None)
            status = {"name": name, "file": file, "status": "ok",
                      "final_accuracy": rec["rounds"][-1]["accuracy"] if rec["rounds"] else None}
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            rec = {"meta": {"config": resolved}, "error": repr(exc), "traceback": traceback.format_exc()}
            status = {"name": name, "file": file, "status": "failed", "error": repr(exc)}
        try:
            with open(os.path.join(out_dir, file), "w") as fh:
                json.dump(rec, fh, indent=2)
        except OSError as exc:  # one unwritable record must not lose the rest of the matrix
            status = {**status, "status": "write_failed", "error": repr(exc)}
        index.append(status)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump({"runs": index}, fh, indent=2)
    return index


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Run a FedLearn simulation from a JSON config.")
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", action="store_true", help="print one JSON line per round to stdout")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    stream = (lambda name, line: (print(json.dumps({"run": name, **line}, default=float)), sys.stdout.flush())) \
        if a.stream else None
    run_matrix(cfg, a.out, stream=stream)


if __name__ == "__main__":
    main()
