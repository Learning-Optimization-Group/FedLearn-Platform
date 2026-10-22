"""Process-owned, bounded exploratory validation runs for Simulation Lab."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import tempfile
import threading
import time
import traceback
import uuid


MESSAGES = {
    "invalid_recipe": "The simulation recipe is unsupported or inconsistent.",
    "invalid_cache": "The configured workload is unavailable or unqualified.",
    "qualification_failed": "The run did not pass complete execution checks.",
    "execution_failed": "Execution failed; diagnostic evidence was retained.",
    "memory_limit": "Observed engine-completion memory exceeded the qualification limit.",
    "artifact_failed": "The run evidence could not be saved completely.",
    "stop_requested": "The run was stopped by request.",
    "parent_disconnected": "The owning service disconnected.",
    "deadline_exceeded": "The worker reached its execution deadline.",
}


def _atomic_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _read_json(path):
    def nonfinite(_):
        raise ValueError("nonfinite JSON value")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("invalid input file")
    return json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=nonfinite)


def _checked_config(cfg, report_sha256):
    from fedlearn.simulation.effusion_bundle import is_effusion_bundle
    from fedlearn.simulation.lab_contract import RECIPES, resolve_choices
    try:
        wl, fed, sv, pv = (cfg[key] for key in ("workload", "federation", "server", "privacy"))
        head = wl["head"]
        selected_head = "linear" if head["type"] == "linear" else f'mlp{head["hidden"]}'
        behaviors = cfg["behaviors"]
        behavior = ("clean" if not behaviors else
                    "sign_flip" if behaviors[0]["type"] == "sign_flip_scale" else "label_flip")
        choices = dict(head=selected_head, numClients=fed["num_clients"], rounds=fed["rounds"],
                       localEpochs=cfg["client"]["local_epochs"], aggregation=sv["rule"],
                       clipNorm=sv["clip_norm"], privacy=pv["mode"], epsilon=pv["epsilon"],
                       behavior=behavior, seed=cfg["seed"])
        if is_effusion_bundle(wl["features_path"]):
            # The recipe is identified by its (LR, clip) pair; any other pair re-resolves differently and is refused.
            recipe = next((name for name, pair in RECIPES.items()
                           if pair == (cfg["client"]["lr"], sv["clip_norm"])), "unrecognised")
            choices.update(workloadId="nih-effusion-r0", recipe=recipe)
            choices.pop("clipNorm")
        resolved = resolve_choices(choices, cache_path=Path(wl["features_path"]),
                                   report_sha256=report_sha256)
        if json.dumps(cfg, sort_keys=True, allow_nan=False) != json.dumps(resolved["config"], sort_keys=True):
            raise ValueError("resolved recipe differs")
        return resolved
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise ValueError("invalid recipe") from error


def run_worker(config_path: Path, output: Path, *, report_sha256: str,
               run_id: str, parent_stream, deadline_seconds: float = 180) -> int:
    """Run in a dedicated process; parent EOF and deadline do not depend on core callbacks.

    Run this lightweight entrypoint as ``python -m fedlearn.lab_worker`` so the
    monitors precede heavy simulation imports. This dedicated process retains its
    terminal signal handlers until shutdown; do not call it inside an API process.
    The parent must keep its input pipe open. SIGTERM requests a round-boundary stop;
    an unresponsive stopped core exits after two seconds. The independent deadline
    exits immediately, retaining the last atomic completed-round snapshot.
    """
    started = time.monotonic()
    if (type(run_id) is not str or str(uuid.UUID(run_id)) != run_id
            or type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
            or not 0 < deadline_seconds <= 180 or parent_stream is None):
        raise ValueError("invalid worker ownership or deadline")
    config_path, output = Path(config_path), Path(output)
    parent_fd = parent_stream.fileno()
    parent = output.parent.resolve(strict=True)
    if (output.name != "execution" or output.parent.is_symlink() or parent.name != run_id
            or config_path.parent.resolve(strict=True) != parent or config_path.is_symlink()):
        raise ValueError("worker paths must belong to the declared job")
    output.mkdir(mode=0o700)
    lock = threading.RLock()
    finished = threading.Event()
    control = {"reason": None, "requested_at": None, "sealed": False}
    progress = dict(version=1, run_id=run_id, config_hash=None, state="starting",
                    completed_rounds=0, requested_rounds=0, rounds=[], record_encoding=None,
                    record_file=None, error_code=None, error_message=None)
    execution = dict(version=1, run_id=run_id, config_hash=None, state="starting",
                     worker_pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=0.0, deadline_seconds=deadline_seconds, cpu_threads=4,
                     exit_code=None)

    def persist():
        _atomic_json(output / "progress.json", progress)
        execution.update(state=progress["state"], config_hash=progress["config_hash"],
                         elapsed_seconds=time.monotonic() - started)
        _atomic_json(output / "execution.json", execution)

    def request(reason):
        with lock:
            if not finished.is_set() and not control["sealed"] and control["reason"] is None:
                control.update(reason=reason, requested_at=time.monotonic())

    def apply_stop():
        reason = control["reason"]
        progress.update(state={"stop_requested": "stopped", "parent_disconnected": "interrupted",
                               "deadline_exceeded": "timed_out"}[reason],
                        error_code=reason, error_message=MESSAGES[reason])
        return 124 if reason == "deadline_exceeded" else 3

    def hard_exit(reason):
        if not lock.acquire(timeout=.1):
            os._exit(124 if reason == "deadline_exceeded" else 3)
        try:
            if finished.is_set():return
            if control["reason"] is None:
                control.update(reason=reason, requested_at=time.monotonic())
            code = apply_stop()
            execution["exit_code"] = code
            persist()
            os._exit(code)
        finally:
            lock.release()

    def watch():
        while not finished.wait(.02):
            now = time.monotonic()
            if now - started >= deadline_seconds:
                hard_exit("deadline_exceeded")
            with lock:
                requested_at = control["requested_at"]
            if requested_at is not None and now - requested_at >= 2:
                hard_exit(control["reason"])

    def watch_parent():
        try:
            while not finished.is_set():
                readable, _, _ = select.select([parent_fd], [], [], .05)
                if readable and not os.read(parent_fd, 1024):
                    request("parent_disconnected")
                    return
        except (OSError, ValueError):
            request("parent_disconnected")

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: request("stop_requested"))
    persist()
    deadline_thread = threading.Thread(target=watch, daemon=True, name="simulation-deadline")
    parent_thread = threading.Thread(target=watch_parent, daemon=True, name="simulation-owner")
    deadline_thread.start()
    parent_thread.start()
    record, code, error_code = None, 2, "invalid_recipe"
    try:
        os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "4"
        import torch
        from fedlearn.simulation import core_qualification as core
        from fedlearn.simulation.lab_contract import _tag_nonfinite, write_research_record
        from fedlearn.simulation.effusion_bundle import is_effusion_bundle, load_effusion_bundle
        from fedlearn.simulation.nih_cache import load_nih_cache
        torch.set_num_threads(4)
        resolved = _checked_config(_read_json(config_path), report_sha256)
        cfg = resolved["config"]
        with lock:
            progress.update(config_hash=resolved["config_hash"], requested_rounds=cfg["federation"]["rounds"])
            persist()
        error_code = "invalid_cache"
        features_path = Path(cfg["workload"]["features_path"])
        if is_effusion_bundle(features_path):
            bundle = load_effusion_bundle(features_path, expected_report_sha256=report_sha256)
            if bundle["report"]["pretrained"] is not True:
                raise ValueError("non-pretrained bundle")
            del bundle
        else:
            cache = load_nih_cache(features_path, expected_report_sha256=report_sha256)
            if cache["provenance"]["pretrained"] is not True:
                raise ValueError("non-pretrained cache")
            del cache

        def on_round(line):
            with lock:
                if line["round"] != len(progress["rounds"]) + 1:
                    raise ValueError("nonsequential core callback")
                progress["rounds"].append(_tag_nonfinite(line))
                progress.update(completed_rounds=len(progress["rounds"]), state="running")
                persist()

        error_code = "execution_failed"
        if control["reason"] is None:
            with lock:
                progress["state"] = "running"
                persist()
            try:
                record = core.run_core(cfg, feature_report_sha256=report_sha256,
                                       clients=cfg["federation"]["num_clients"], evaluation_split="validation",
                                       on_round=on_round, should_stop=lambda: control["reason"] is not None)
            except core.CoreQualificationError as error:
                record = error.record
                error_code = "qualification_failed"
                raise
            rss = record["meta"]["peak_rss_bytes"]
            error_code = "memory_limit"
            if type(rss) is not int or not 0 < rss <= 2**31:
                raise ValueError("observed RSS qualification failed")
            code, error_code = 0, None
    except Exception:
        try:
            (output / "diagnostics.txt").write_text(traceback.format_exc()[-16384:])
        except OSError:
            pass
    finally:
        partial_path = output / ".partial-record.json"
        qualified_path = output / ".qualified-record.json"
        encodings = {}
        # Serialization and fsync must not hold the watchdog's state lock. Stage an
        # unqualified record first so cancellation can never publish a success marker.
        if record is not None:
            try:
                partial = {**record, "meta": {key: value for key, value in record["meta"].items()
                                              if key != "core_qualification"}}
                encodings[partial_path] = write_research_record(partial_path, partial)
                if code == 0 and control["reason"] is None:
                    encodings[qualified_path] = write_research_record(qualified_path, record)
            except Exception:
                code, error_code = 2, "artifact_failed"
        with lock:
            # Successful execution and staged evidence are ready. Cancellation is
            # linearized here; a later request observes this terminal decision.
            control["sealed"] = True
            if control["reason"] is not None:
                code = apply_stop()
            else:
                progress.update(state="completed" if code == 0 else "failed", error_code=error_code,
                                error_message=MESSAGES.get(error_code))
            candidate = qualified_path if code == 0 else partial_path
            if candidate in encodings:
                try:
                    os.link(candidate, output / "record.json")
                    progress.update(record_encoding=encodings[candidate], record_file="record.json")
                except Exception:
                    code = 2
                    progress.update(state="failed", error_code="artifact_failed",
                                    error_message=MESSAGES["artifact_failed"])
            execution["exit_code"] = code
            persist()
            finished.set()
        for candidate in encodings:
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        deadline_thread.join(timeout=.25)
        parent_thread.join(timeout=.25)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--choices", type=Path, required=True)
    resolve.add_argument("--cache", type=Path, required=True)
    resolve.add_argument("--report-sha256", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report-sha256", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--deadline-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        if args.mode == "resolve":
            from fedlearn.simulation.lab_contract import resolve_choices
            print(json.dumps(resolve_choices(_read_json(args.choices), cache_path=args.cache,
                                             report_sha256=args.report_sha256), allow_nan=False))
            return 0
        return run_worker(args.config, args.output, report_sha256=args.report_sha256,
                          run_id=args.run_id, parent_stream=sys.stdin.buffer,
                          deadline_seconds=args.deadline_seconds)
    except (ValueError, OSError):
        print(json.dumps({"error_code": "invalid_input", "error_message": MESSAGES["invalid_recipe"]}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
