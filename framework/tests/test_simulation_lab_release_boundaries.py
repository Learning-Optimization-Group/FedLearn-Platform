"""Actual process exit and startup must agree with the worker's lifecycle evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import pytest

from fedlearn.simulation.lab_contract import resolve_choices
from tests.test_simulation_lab_worker import lab_cache


ENTRYPOINT = "fedlearn.lab_worker"
SLOW_IMPORT = '''
import importlib.abc,os,sys,time
from pathlib import Path
class SlowTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch':
            Path(os.environ['IMPORT_SENTINEL']).write_text('before torch initialization')
            time.sleep(5)
        return None
sys.meta_path.insert(0,SlowTorch())
'''


def job(tmp_path, lab_cache):
    cache, pin = lab_cache
    run_id = str(uuid.uuid4())
    folder = tmp_path / run_id
    folder.mkdir()
    cfg = resolve_choices({"rounds": 2}, cache_path=cache, report_sha256=pin)["config"]
    config = folder / "config.json"
    config.write_text(json.dumps(cfg))
    return folder, config, pin


def source_environment(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(source)]),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}


@pytest.mark.parametrize("owner_lost", [False, True])
def test_actual_entrypoint_bounds_slow_import_and_parent_loss(tmp_path, lab_cache, owner_lost):
    folder, config, pin = job(tmp_path, lab_cache)
    (tmp_path / "sitecustomize.py").write_text(SLOW_IMPORT)
    env = {**source_environment(tmp_path), "IMPORT_SENTINEL": str(folder / "import-started")}
    log = (folder / "process.log").open("w")
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, "-m", ENTRYPOINT, "run", "--config", str(config),
                              "--output", str(folder / "execution"), "--report-sha256", pin,
                              "--run-id", folder.name, "--deadline-seconds", "15" if owner_lost else ".05"],
                             cwd=tmp_path, env=env, stdin=subprocess.PIPE, stdout=log, stderr=log)
    try:
        if owner_lost:child.stdin.close()
        code = child.wait(timeout=9)
        elapsed = time.monotonic() - started
        assert (folder / "import-started").exists()
        assert code == (3 if owner_lost else 124)
        assert elapsed < (3.5 if owner_lost else .7)
        output = folder / "execution"
        progress = json.loads((output / "progress.json").read_text())
        execution = json.loads((output / "execution.json").read_text())
        assert progress["state"] == ("interrupted" if owner_lost else "timed_out")
        assert progress["completed_rounds"] == 0 and not (output / "record.json").exists()
        assert abs(elapsed - execution["elapsed_seconds"]) < .4
    finally:
        if child.poll() is None:child.kill();child.wait()
        child.stdin.close()
        log.close()


def test_late_sigterm_after_worker_return_keeps_completed_exit_and_evidence(tmp_path, lab_cache):
    folder, config, pin = job(tmp_path, lab_cache)
    bootstrap = '''
import importlib,sys,time
from pathlib import Path
module,config,folder,pin=sys.argv[1:]
folder=Path(folder)
worker=importlib.import_module(module)
code=worker.run_worker(Path(config),folder/'execution',report_sha256=pin,run_id=folder.name,
                       parent_stream=sys.stdin.buffer,deadline_seconds=15)
(folder/'returned').write_text(str(code))
time.sleep(.5)
raise SystemExit(code)
'''
    with (folder / "process.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-c", bootstrap, ENTRYPOINT, str(config), str(folder), pin],
                                 cwd=tmp_path, env=source_environment(tmp_path), stdin=subprocess.PIPE,
                                 stdout=log, stderr=log)
        try:
            until = time.monotonic() + 20
            while child.poll() is None and time.monotonic() < until:
                if (folder / "returned").exists():break
                time.sleep(.01)
            assert (folder / "returned").read_text() == "0"
            before = (folder / "execution/progress.json").read_bytes()
            time.sleep(.02)
            assert child.poll() is None
            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=5) == 0
            assert (folder / "execution/progress.json").read_bytes() == before
            execution = json.loads((folder / "execution/execution.json").read_text())
            record = json.loads((folder / "execution/record.json").read_text())
            assert execution["state"] == "completed" and execution["exit_code"] == 0
            assert record["meta"]["core_qualification"]["scope"] == "simulation_execution_only"
        finally:
            if child.poll() is None:child.kill();child.wait()
            child.stdin.close()
