"""Prospective private preparation for the fixed multi-host benchmark.

Preparation loads only the qualified training/selection bundle, projects on the Mac,
and emits training-only host packs. It performs no model fitting or network actions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile

import numpy as np
import torch

from capstone_pilot_mechanisms import public_projection
from fedlearn.simulation.effusion_bundle import load_effusion_bundle
from fedlearn.simulation.partition import patient_iid_partition

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
REPRO = Path("docs/artifacts/capstone/repro")
BUNDLE_PIN = "dec54dfb580232ed6bf8da16ea9466248db58de4925285c8fb0d07a381562ffa"
PROJECTION_SEED = 20261025
HOSTS = ("mac", "g14", "orin")
BUNDLE_FILES = {f"{split}-{kind}.npy" for split in ("train", "selection") for kind in ("x", "y")} | {
    "train.csv", "selection.csv", "report.json"
}
SETTINGS = dict(lr=0.03, local_epochs=7, batch_size=256, clip_norm=0.05, optimizer="sgd", device="cpu")
_PLAN_KEYS = {"schema", "synthetic", "bundle_path", "bundle_report_sha256", "port", "sources",
              "public_manifest_sha256", "cells"}


def cells():
    """The approved 48 systems cells and four historical-seed replay anchors."""
    result = []
    for stage, counts, seeds, rounds in (("systems", (3, 6, 24, 80), (14001, 14002, 14003), 5),
                                         ("anchor", (80,), (13001,), 20)):
        for n in counts:
            for method, width in (("mean512", 512), ("projected128", 128)):
                for seed in seeds:
                    for placement in ("mac_only", "three_hosts"):
                        result.append(dict(name=f"{stage}_n{n:03d}_{method}_s{seed}_{placement}",
                                           stage=stage, num_clients=n, method=method, width=width,
                                           seed=seed, rounds=rounds, placement=placement))
    return result


def assignments(num_clients, placement):
    if type(num_clients) is not int or num_clients < 1 or placement not in ("mac_only", "three_hosts"):
        raise ValueError("assignment requires a positive client count and known placement")
    result = {host: [] for host in HOSTS}
    for cid in range(num_clients):
        result["mac" if placement == "mac_only" else HOSTS[cid % 3]].append(cid)
    return result


def _digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path.name}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return _digest(path)


def runtime():
    """Public local preparation runtime facts; no hostname or network inventory."""
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                system=platform.system(), machine=platform.machine(), cpu_count=os.cpu_count(),
                device="cpu", torch_num_threads=torch.get_num_threads(),
                torch_num_interop_threads=torch.get_num_interop_threads())


def source_inventory():
    """Pin the live runtime, protocol definitions and preparation/runner sources."""
    paths = set((ROOT / "framework/src").rglob("*.py")) | set((ROOT / "fl-runtime").glob("*.py"))
    paths |= set((ROOT / "proto").rglob("*.proto"))
    paths |= {ROOT / REPRO / name for name in (
        "multihost_manifest.py", "test_multihost_manifest.py", "capstone_pilot_mechanisms.py")}
    for relative in (REPRO / 'run_multihost_benchmark.py', REPRO / 'test_run_multihost_benchmark.py',
                     REPRO / 'run_effusion_byzfl_qualification.py',
                     Path('docs/artifacts/capstone/multihost-protocol-2026-10-26.md'),
                     Path('docs/superpowers/specs/2026-10-25-capstone-multihost-training-design.md')):
        if (ROOT / relative).exists():
            paths.add(ROOT / relative)
    for relative in ('fl-runtime/tests/test_effusion_runtime.py','fl-runtime/tests/test_effusion_host.py',
                     'fl-runtime/tests/test_effusion_server.py','fl-runtime/tests/test_effusion_transport.py',
                     'framework/tests/test_multihost_cohort_contract.py'):
        if (ROOT / relative).exists():
            paths.add(ROOT / relative)
    return {path.relative_to(ROOT).as_posix(): _digest(path) for path in sorted(paths)}


def _bundle_report(bundle, pin, synthetic):
    if type(synthetic) is not bool or not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise ValueError("synthetic classification and bundle report SHA-256 are required")
    if not synthetic and pin != BUNDLE_PIN:
        raise ValueError("medical preparation requires the canonical bundle report pin")
    if synthetic and pin == BUNDLE_PIN:
        raise ValueError("the canonical medical bundle cannot be classified as synthetic")
    if bundle.is_symlink() or not bundle.is_dir() or {path.name for path in bundle.iterdir()} != BUNDLE_FILES:
        raise ValueError("bundle must contain only the exact training/selection files")
    if _digest(bundle / "report.json") != pin:
        raise ValueError("bundle report hash mismatch")
    report = json.loads((bundle / "report.json").read_text())
    if report.get("holdoutIncluded") is not False or report.get("testIncluded") is not False:
        raise ValueError("bundle must exclude holdout and test data")
    hashes = report.get("payload_hashes")
    if not isinstance(hashes, dict) or set(hashes) != BUNDLE_FILES - {"report.json"}:
        raise ValueError("bundle payload names differ from the training/selection schema")
    for name, expected in hashes.items():
        if _digest(bundle / name) != expected:
            raise ValueError(f"bundle payload hash mismatch: {name}")
    return report


def _private_path(output, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("prepared artifact path must stay inside its output directory")
    path = output / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != output.parent):
        raise ValueError("prepared artifact paths cannot contain symlinks")
    return path


def _pack(path, cell, host, ids, parts, x, y, projection_sha256):
    path.mkdir(parents=True)
    rows = np.concatenate([parts[cid] for cid in ids])
    np.save(path / "x.npy", np.ascontiguousarray(x[rows]))
    np.save(path / "y.npy", np.ascontiguousarray(y[rows]))
    members, cursor = [], 0
    for cid in ids:
        stop = cursor + len(parts[cid])
        members.append(dict(client_id=cid, start=cursor, stop=stop,
                            identity=f"{cell['name']}:client-{cid:05d}"))
        cursor = stop
    manifest = dict(schema="fedlearn.effusion_host_data.v1", dataset="NIH_ChestXray14", target="Effusion",
                    training_only=True, selection_included=False, holdout_included=False, test_included=False,
                    feat_dim=cell["width"], original_feat_dim=512, projection_seed=PROJECTION_SEED,
                    projection_sha256=projection_sha256, width=cell["width"], run_id=cell["name"],
                    host_label=host, num_clients=cell["num_clients"], seed=cell["seed"], clients=members,
                    payload_sha256={name: _digest(path / name) for name in ("x.npy", "y.npy")},
                    settings={**SETTINGS, "rounds": cell["rounds"]})
    return _json(path / "host.json", manifest)


def prepare(bundle_path, bundle_sha256, output, *, port=54051, synthetic=False, launch_runtimes=None):
    """Write a fresh source-bound 52-cell plan and assigned training packs; never fit."""
    output, bundle = Path(output).absolute(), Path(bundle_path).absolute()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer from 1 through 65535")
    _bundle_report(bundle, bundle_sha256, synthetic)
    if not synthetic and (not isinstance(launch_runtimes,dict) or set(launch_runtimes)!={'mac','g14','orin'}):
        raise ValueError('medical preparation requires the three actual selected interpreter runtimes')
    data = load_effusion_bundle(bundle, expected_report_sha256=bundle_sha256)
    train, selection = data["train"], data["evaluation"]
    if len(set(train["patient_ids"])) < 80:
        raise ValueError("the frozen 80-client cohort requires at least 80 training patients")
    if not len(selection["y"]):
        raise ValueError("selection must contain at least one row")
    sources = source_inventory()
    matrices = {width: public_projection(512, width, seed=PROJECTION_SEED) for width in (512, 128)}
    projections = {str(width): hashlib.sha256(matrix.numpy().tobytes()).hexdigest()
                   for width, matrix in matrices.items()}
    train_x = {512: train["x"], 128: (torch.from_numpy(train["x"]).double() @ matrices[128]).float().numpy()}
    selection_x = {512: selection["x"],
                   128: (torch.from_numpy(selection["x"]).double() @ matrices[128]).float().numpy()}
    for array in (*train_x.values(), *selection_x.values()):
        array.flags.writeable = False
    public = dict(schema="fedlearn.multihost_public.v1", synthetic=synthetic,
                  scope=dict(training="fixed whole-patient train cohort; selection-only evaluation",
                             systems="five-round execution diagnostics; three fresh paired training seeds",
                             anchors="historical-seed replay; not fresh confirmation",
                             logical_clients="multiplexed on three physical failure domains"),
                  bundle_report_sha256=bundle_sha256, source_sha256=sources, cells=cells(),
                  execution_order=[c['name'] for c in sorted(cells(),key=lambda c:
                      hashlib.sha256(('20261026:'+c['name']).encode()).hexdigest())],
                  settings_templates={stage: {**SETTINGS, "rounds": rounds}
                                      for stage, rounds in (("systems", 5), ("anchor", 20))},
                  projection_seed=PROJECTION_SEED, projection_sha256=projections,
                  fit_concurrency_per_host=2, round_timeout_s=600, runtime=runtime(),launch_runtimes=launch_runtimes)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".multihost-prepare-", dir=output.parent))
    try:
        for relative, expected in sources.items():
            destination = staging / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            if _digest(destination) != expected:
                raise ValueError("source changed while preparing its snapshot")
        for width in (512, 128):
            directory = staging / "selection" / str(width)
            directory.mkdir(parents=True)
            np.save(directory / "selection_x.npy", selection_x[width])
            np.save(directory / "selection_y.npy", selection["y"])
        private_cells = []
        for cell in cells():
            parts = patient_iid_partition(train["patient_ids"], cell["num_clients"], cell["seed"])
            mapping = assignments(cell["num_clients"], cell["placement"])
            hosts = {}
            for host, ids in mapping.items():
                if not ids:
                    continue
                relative = Path("cells") / cell["name"] / "data" / host
                manifest_pin = _pack(staging / relative, cell, host, ids, parts, train_x[cell["width"]],
                                     train["y"], projections[str(cell["width"])])
                hosts[host] = dict(manifest_path=(relative / "host.json").as_posix(), manifest_sha256=manifest_pin)
            ids = [f"{cell['name']}:client-{cid:05d}" for cid in range(cell["num_clients"])]
            relative_selection = Path("selection") / str(cell["width"])
            selection_plan = dict(split="selection", source_bundle_report_sha256=bundle_sha256)
            for name in ("x", "y"):
                relative = relative_selection / f"selection_{name}.npy"
                selection_plan[name + "_path"] = str(output / relative)
                selection_plan[name + "_sha256"] = _digest(staging / relative)
            server = dict(schema="fedlearn.effusion_server.v1", synthetic=synthetic, run_id=cell["name"],
                          method=cell["method"], width=cell["width"], num_clients=cell["num_clients"],
                          seed=cell["seed"], rounds=cell["rounds"], settings={**SETTINGS, "rounds": cell["rounds"]},
                          client_ids=ids, client_hosts={f"{cell['name']}:client-{cid:05d}": host
                                                     for host, members in mapping.items() for cid in members},
                          bind_address=f"127.0.0.1:{port}", selection=selection_plan)
            relative_server = Path("cells") / cell["name"] / "server.json"
            server_pin = _json(staging / relative_server, server)
            private_cells.append({**cell, "server_plan_path": relative_server.as_posix(),
                                  "server_plan_sha256": server_pin, "hosts": hosts})
        public['input_commitments'] = {cell['name']: dict(server_plan_sha256=cell['server_plan_sha256'],
            hosts={host: ref['manifest_sha256'] for host,ref in cell['hosts'].items()}) for cell in private_cells}
        public_pin = _json(staging / "public-manifest.json", public)
        plan = dict(schema="fedlearn.multihost_private.v1", synthetic=synthetic, bundle_path=str(bundle),
                    bundle_report_sha256=bundle_sha256, port=port, sources=sources,
                    public_manifest_sha256=public_pin, cells=private_cells)
        _json(staging / "plan.json", plan)
        if source_inventory() != sources or runtime() != public["runtime"]:
            raise ValueError("source or runtime changed during preparation")
        staging.rename(output)
        return plan
    except BaseException:
        shutil.rmtree(staging)
        raise


def verify_prepared(output):
    """Recheck sources, runtime and all bound input payloads before any execution."""
    output = Path(output).absolute()
    plan = json.loads((output / "plan.json").read_text())
    if not isinstance(plan, dict) or set(plan) != _PLAN_KEYS or plan["schema"] != "fedlearn.multihost_private.v1":
        raise ValueError("private plan schema differs from preparation")
    if _digest(output / "public-manifest.json") != plan["public_manifest_sha256"]:
        raise ValueError("public manifest hash mismatch")
    public = json.loads((output / "public-manifest.json").read_text())
    if public["source_sha256"] != plan["sources"] or source_inventory() != plan["sources"]:
        raise ValueError("live source inventory changed after preparation")
    if public["runtime"] != runtime():
        raise ValueError("preparation runtime changed")
    for relative, expected in plan["sources"].items():
        if _digest(_private_path(output, "sources/" + relative)) != expected:
            raise ValueError("source snapshot hash mismatch")
    _bundle_report(Path(plan["bundle_path"]), plan["bundle_report_sha256"], plan["synthetic"])
    if len(plan["cells"]) != 52 or public["cells"] != cells():
        raise ValueError("prepared cell budget changed")
    for cell, expected_cell in zip(plan["cells"], cells()):
        if {key: cell[key] for key in expected_cell} != expected_cell:
            raise ValueError("prepared cell definition changed")
        commitment = dict(server_plan_sha256=cell['server_plan_sha256'],
                          hosts={host: ref['manifest_sha256'] for host,ref in cell['hosts'].items()})
        if public['input_commitments'].get(cell['name']) != commitment:
            raise ValueError('private input commitment differs from the public seal')
        server_path = _private_path(output, cell["server_plan_path"])
        if _digest(server_path) != cell["server_plan_sha256"]:
            raise ValueError("server plan hash mismatch")
        server = json.loads(server_path.read_text())
        for kind in ("x", "y"):
            relative = f"selection/{cell['width']}/selection_{kind}.npy"
            if (server["selection"][kind + "_path"] != str(output / relative)
                    or _digest(_private_path(output, relative)) != server["selection"][kind + "_sha256"]):
                raise ValueError("selection input hash or path changed")
        mapping = assignments(cell["num_clients"], cell["placement"])
        if set(cell["hosts"]) != {host for host, ids in mapping.items() if ids}:
            raise ValueError("prepared host ownership changed")
        for host, reference in cell["hosts"].items():
            path = _private_path(output, reference["manifest_path"])
            if _digest(path) != reference["manifest_sha256"]:
                raise ValueError("host manifest hash mismatch")
            manifest = json.loads(path.read_text())
            if [entry["client_id"] for entry in manifest["clients"]] != mapping[host]:
                raise ValueError("logical client ownership changed")
            if set(entry.name for entry in path.parent.iterdir()) != {"host.json", "x.npy", "y.npy"}:
                raise ValueError("unexpected file in a training-only host pack")
            for name in ("x.npy", "y.npy"):
                if _digest(path.parent / name) != manifest["payload_sha256"][name]:
                    raise ValueError("host training payload hash mismatch")
    return plan
