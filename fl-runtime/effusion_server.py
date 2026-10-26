"""Private selection evaluator around the canonical full-cohort gRPC server."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import ipaddress
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from effusion_runtime import _freeze, _integer, _settings, _sha256, _unique_object, _verified_bytes
from fedlearn.server.composed_strategy import ComposedStrategy

BUNDLE_PIN = "dec54dfb580232ed6bf8da16ea9466248db58de4925285c8fb0d07a381562ffa"
_PLAN_KEYS = {
    "schema",
    "synthetic",
    "run_id",
    "method",
    "width",
    "num_clients",
    "seed",
    "rounds",
    "settings",
    "client_ids",
    "client_hosts",
    "bind_address",
    "selection",
}
_SELECTION_KEYS = {
    "x_path",
    "y_path",
    "x_sha256",
    "y_sha256",
    "split",
    "source_bundle_report_sha256",
}


@dataclass(frozen=True)
class ServerPlan:
    """Verified private settings and Mac-only selection tensors."""

    manifest: Mapping
    selection_x: torch.Tensor
    selection_y: torch.Tensor


def _bind_address(value):
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("bind_address must name a loopback host and port")
    host, port = value.rsplit(":", 1)
    host = host.removeprefix("[").removesuffix("]")
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("bind_address must name a loopback host and valid port")


def load_server_plan(path: Path, expected_sha256: str) -> ServerPlan:
    """Verify one pinned plan and its two derived selection arrays without opening a bundle."""
    path = Path(path)
    try:
        manifest = json.loads(
            _verified_bytes(path, expected_sha256), object_pairs_hook=_unique_object
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid server plan JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _PLAN_KEYS:
        raise ValueError("server plan differs from the frozen schema")
    if (
        manifest["schema"] != "fedlearn.effusion_server.v1"
        or type(manifest["synthetic"]) is not bool
    ):
        raise ValueError("invalid server plan schema or synthetic classification")
    run_id = manifest["run_id"]
    if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 200:
        raise ValueError("run_id must be a nonempty string of at most 200 characters")
    width = _integer(manifest["width"], "width", 1)
    if (
        not isinstance(manifest["method"], str)
        or {"mean512": 512, "projected128": 128}.get(manifest["method"]) != width
    ):
        raise ValueError("method and feature width disagree")
    n = _integer(manifest["num_clients"], "num_clients", 1)
    _integer(manifest["seed"], "seed")
    rounds = _integer(manifest["rounds"], "rounds", 1)
    settings = _settings(manifest["settings"])
    if rounds != settings["rounds"]:
        raise ValueError("round count differs from the frozen client settings")
    ids = manifest["client_ids"]
    if not isinstance(ids, list) or ids != [f"{run_id}:client-{cid:05d}" for cid in range(n)]:
        raise ValueError("client_ids must be the entire ordered logical cohort")
    hosts = manifest["client_hosts"]
    if (
        not isinstance(hosts, dict)
        or set(hosts) != set(ids)
        or any(host not in ("mac", "g14", "orin") for host in hosts.values())
    ):
        raise ValueError("every logical identity requires one known assigned host")
    _bind_address(manifest["bind_address"])
    selection = manifest["selection"]
    if (
        not isinstance(selection, dict)
        or set(selection) != _SELECTION_KEYS
        or selection["split"] != "selection"
    ):
        raise ValueError("only the exact selection input schema is permitted")
    source_pin = _sha256(selection["source_bundle_report_sha256"])
    if not manifest["synthetic"] and source_pin != BUNDLE_PIN:
        raise ValueError("medical selection source is not the frozen canonical bundle")
    if manifest["synthetic"] and source_pin == BUNDLE_PIN:
        raise ValueError("canonical medical inputs cannot be classified as synthetic")
    arrays = {}
    for key in ("x", "y"):
        name = selection[key + "_path"]
        if not isinstance(name, str) or not name:
            raise ValueError("selection input paths must be nonempty strings")
        payload = Path(name)
        if not payload.is_absolute():
            payload = path.parent / payload
        if any(
            part.lower() in ("train", "training", "test", "holdout") for part in payload.parts
        ) or payload.name.lower().startswith(
            ("train_", "train-", "test_", "test-", "holdout_", "holdout-")
        ):
            raise ValueError("train/holdout/test paths are forbidden to the selection server")
        raw = BytesIO(_verified_bytes(payload, selection[key + "_sha256"]))
        array = np.load(raw, allow_pickle=False)
        if not isinstance(array, np.ndarray) or raw.tell() != len(raw.getbuffer()):
            raise ValueError("selection payload must contain exactly one NumPy array")
        arrays[key] = array
    x, y = arrays["x"], arrays["y"]
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError("selection features must be finite float32 of the declared width")
    if (
        y.dtype != np.int64
        or y.ndim != 1
        or len(y) != len(x)
        or not len(y)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError("selection labels must be matching nonempty binary int64")
    if not x.flags.c_contiguous or not y.flags.c_contiguous:
        raise ValueError("selection arrays must be C contiguous")
    return ServerPlan(_freeze(manifest), torch.from_numpy(x), torch.from_numpy(y))


def rank_auc(labels, scores):
    """Mann–Whitney AUROC with average tied ranks; a single class has no defined AUROC."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or not len(labels)
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("AUROC requires matching finite scores and binary labels")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    ranks = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
    positive_rank_sum = ranks[labels[order] == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def head_sha256(parameters):
    """Hash the exact float32 layout and values, preserving canonical head-key order."""
    digest = hashlib.sha256()
    for key in ("head.weight", "head.bias"):
        digest.update(key.encode())
        array = parameters[key].detach().cpu().contiguous().numpy()
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_head(parameters, width):
    if not isinstance(parameters, Mapping) or set(parameters) != {"head.weight", "head.bias"}:
        raise ValueError("incompatible binary head keys")
    for key, shape in (("head.weight", (2, width)), ("head.bias", (2,))):
        value = parameters[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("incompatible binary head layout, dtype, device or values")


class EffusionStrategy(ComposedStrategy):
    """The existing uniform clipped-mean pipeline with a fixed complete logical cohort."""

    def __init__(self, plan: ServerPlan):
        self.plan = plan
        self.started = time.perf_counter()
        manifest = plan.manifest
        self.allowed_client_ids = tuple(manifest["client_ids"])
        self.receipts, self.evaluations = [], []
        initial = OrderedDict(
            [
                ("head.weight", torch.zeros((2, manifest["width"]), dtype=torch.float32)),
                ("head.bias", torch.zeros(2, dtype=torch.float32)),
            ]
        )
        super().__init__(
            initial,
            rule="mean",
            clip_norm=0.05,
            min_fit_clients=manifest["num_clients"],
            clients_per_round=manifest["num_clients"],
        )

    def get_client_config(self):
        p, settings = self.plan.manifest, self.plan.manifest["settings"]
        return dict(
            learning_rate=str(settings["lr"]),
            local_epochs=str(settings["local_epochs"]),
            batch_size=str(settings["batch_size"]),
            optimizer="sgd",
            run_id=p["run_id"],
            num_clients=str(p["num_clients"]),
            seed=str(p["seed"]),
            width=str(p["width"]),
            rounds=str(p["rounds"]),
        )

    def aggregate_fit(self, server_round, results):
        _integer(server_round, "server_round", 1)
        if server_round != len(self.receipts) + 1 or server_round > self.plan.manifest["rounds"]:
            raise ValueError("aggregation round is stale, ahead or beyond the declared budget")
        if not isinstance(results, (list, tuple)) or len(results) != len(self.allowed_client_ids):
            raise ValueError("aggregation requires the complete cohort")
        updates = {}
        for result in results:
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                raise ValueError("aggregation requires identified three-field client updates")
            cid, parameters, examples = result
            if not isinstance(cid, str) or cid not in self.allowed_client_ids or cid in updates:
                raise ValueError("unknown or duplicate logical client update")
            _integer(examples, "num_examples", 1)
            _validate_head(parameters, self.plan.manifest["width"])
            updates[cid] = (cid, parameters, examples)
        ordered = [updates[cid] for cid in self.allowed_client_ids]
        started = time.perf_counter()
        output = super().aggregate_fit(server_round, ordered)
        if (
            output is None
            or self.last_round_stats.get("aggregated") != len(ordered)
            or self.last_round_stats.get("failed")
        ):
            raise RuntimeError("canonical aggregation did not complete the fixed cohort")
        _validate_head(output, self.plan.manifest["width"])
        assigned_hosts = sorted(
            {self.plan.manifest["client_hosts"][cid] for cid in self.allowed_client_ids}
        )
        self.receipts.append(
            dict(
                round=server_round,
                client_ids=list(self.allowed_client_ids),
                client_count=len(ordered),
                num_examples=[u[2] for u in ordered],
                assigned_hosts=assigned_hosts,
                assigned_host_count=len(assigned_hosts),
                head_sha256=head_sha256(output),
                aggregation_elapsed_s=time.perf_counter() - started,
                stats=json.loads(json.dumps(self.last_round_stats, allow_nan=False)),
            )
        )
        return output

    def evaluate(self, server_round, parameters):
        if server_round != len(self.evaluations) + 1 or len(self.receipts) != server_round:
            raise ValueError("selection evaluation must follow exactly one completed aggregation")
        _validate_head(parameters, self.plan.manifest["width"])
        if head_sha256(parameters) != self.receipts[-1]["head_sha256"]:
            raise ValueError("selection head differs from the aggregated head")
        started = time.perf_counter()
        with torch.no_grad():
            logits = F.linear(
                self.plan.selection_x, parameters["head.weight"], parameters["head.bias"]
            )
            loss = float(F.cross_entropy(logits, self.plan.selection_y))
            scores = (logits[:, 1].double() - logits[:, 0].double()).numpy()
        if not np.isfinite(loss):
            raise ValueError("selection loss is nonfinite")
        auc = rank_auc(self.plan.selection_y.numpy(), scores)
        self.evaluations.append(
            dict(
                round=server_round,
                loss=loss,
                auc=auc,
                head_sha256=head_sha256(parameters),
                evaluation_elapsed_s=time.perf_counter() - started,
                round_completed_elapsed_s=time.perf_counter() - self.started,
            )
        )
        return loss, {"auc": auc}


@contextmanager
def _server_environment(n):
    updates = {"MAX_CLIENTS": str(n), "FEDLEARN_ROUND_TIMEOUT_S": "600"}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        platform=platform.platform(),
        device="cpu",
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        peak_rss_bytes=int(peak) if sys.platform == "darwin" else int(peak) * 1024,
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execute(plan_path: Path, expected_sha256: str, output: Path):
    """Run the canonical server; publish a head only after full receipt/history reconciliation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, strategy = time.perf_counter(), None
    summary = dict(schema="fedlearn.effusion_server_result.v1", status="failed", completed_rounds=0)
    try:
        plan = load_server_plan(plan_path, expected_sha256)
        p = plan.manifest
        summary.update(
            run_id=p["run_id"],
            synthetic=p["synthetic"],
            method=p["method"],
            num_clients=p["num_clients"],
            seed=p["seed"],
            declared_rounds=p["rounds"],
            server_plan_sha256=expected_sha256,
        )
        strategy = EffusionStrategy(plan)
        from fedlearn.server.server import ServerConfig, start_server

        with _server_environment(p["num_clients"]):
            history, final = start_server(
                p["bind_address"], ServerConfig(num_rounds=p["rounds"]), strategy
            )
        expected_rounds = list(range(1, p["rounds"] + 1))
        if (
            [r["round"] for r in strategy.receipts] != expected_rounds
            or [r["round"] for r in strategy.evaluations] != expected_rounds
            or [r[0] for r in history] != expected_rounds
        ):
            raise RuntimeError(
                "server returned without every declared cohort, evaluation and history round"
            )
        _validate_head(final, p["width"])
        if head_sha256(final) != strategy.receipts[-1]["head_sha256"]:
            raise RuntimeError("returned head differs from the last completed aggregate")
        for (_, metrics), evaluation in zip(history, strategy.evaluations):
            if metrics.get("loss") != evaluation["loss"] or metrics.get("auc") != evaluation["auc"]:
                raise RuntimeError(
                    "server history differs from independent selection-stage receipts"
                )
        np.savez(output / "head.npz", **{k: v.detach().cpu().numpy() for k, v in final.items()})
        _write_json(output / "receipts.json", strategy.receipts)
        _write_json(output / "rounds.json", strategy.evaluations)
        summary.update(status="complete", head_sha256=head_sha256(final))
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        for name in ("head.npz", "receipts.json", "rounds.json"):
            (output / name).unlink(missing_ok=True)
        if strategy is not None:
            _write_json(output / "failed-receipts.json", strategy.receipts)
            _write_json(output / "failed-rounds.json", strategy.evaluations)
        raise RuntimeError("Effusion server failed; see private summary.json") from exc
    finally:
        summary["completed_rounds"] = len(strategy.evaluations) if strategy is not None else 0
        summary["aggregated_rounds"] = len(strategy.receipts) if strategy is not None else 0
        summary["elapsed_s"] = time.perf_counter() - started
        summary["runtime"] = _runtime()
        _write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-plan", required=True, type=Path)
    parser.add_argument("--server-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        execute(args.server_plan, args.server_plan_sha256, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
