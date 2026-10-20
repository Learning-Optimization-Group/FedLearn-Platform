"""Real worker processes over synthetic features; no NIH learning-quality claim."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

from fedlearn.simulation.lab_contract import resolve_choices
from fedlearn.simulation.nih_cache import CSV_FIELDS, TRANSFORM, json_bytes, load_nih_cache


@pytest.fixture
def lab_cache(tmp_path):
    """Construct a schema-valid synthetic cache, including synthetic pretrained metadata."""
    cache = tmp_path / "synthetic-cache"
    cache.mkdir()
    counts, hashes = {}, {}
    rng = np.random.default_rng(16)
    for split, start, patients in [("train", 1, 20), ("validation", 101, 4), ("test", 201, 4)]:
        rows = []
        for patient in range(start, start + patients):
            for frame in range(2):
                name = f"{patient:08d}_{frame:03d}.png"
                rows.append(dict(image_index=name, patient_id=str(patient),
                                 image_path="images_001/images/" + name, label=frame,
                                 findings="Mass" if frame else "No Finding"))
        with (cache / f"{split}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        y = np.array([row["label"] for row in rows], dtype=np.int64)
        x = rng.normal(size=(len(rows), 512)).astype(np.float32)
        np.save(cache / f"{split}-x.npy", x, allow_pickle=False)
        np.save(cache / f"{split}-y.npy", y, allow_pickle=False)
        counts[split] = dict(images=len(rows), patients=patients,
                             positive_images=int(y.sum()), negative_images=int(len(y) - y.sum()))
        for name in [f"{split}.csv", f"{split}-x.npy", f"{split}-y.npy"]:
            hashes[name] = hashlib.sha256((cache / name).read_bytes()).hexdigest()
    shared = dict(dataset="synthetic-schema-control", classes=["NO_FINDING", "FINDING"],
                  label_definition="synthetic control, not patient observations",
                  source_hashes={name: "a" * 64 for name in
                                 ["Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"]},
                  original_preparation_sha256="b" * 64, image_audit_sha256="c" * 64)
    cohort = dict(**shared, schema_version=1, cohort_version="capstone-nih-curated-cohort-v1",
                  policy_version="nih-development-quarantine-v1", development_content_qualified=True,
                  official_test_unchanged=True, independent_split_rgb_overlap_groups=0,
                  selected_splits=counts, payload_hashes={k: v for k, v in hashes.items() if k.endswith(".csv")})
    (cache / "cohort-report.json").write_bytes(json_bytes(cohort))
    cohort_sha = hashlib.sha256((cache / "cohort-report.json").read_bytes()).hexdigest()
    hashes["cohort-report.json"] = cohort_sha
    report = dict(**shared, schema_version=1, cache_version="capstone-nih-frozen-features-v1",
                  backbone="resnet18", feat_dim=512, n_classes=2, feature_dtype="float32",
                  label_dtype="int64", pretrained=True, backbone_seed=16, model_state_sha256="d" * 64,
                  weight_identity="ResNet18_Weights.IMAGENET1K_V1", transform=TRANSFORM,
                  device="cpu", batch_size=4, audit_workers=1, image_workers=1, torch_num_threads=4,
                  curated_cohort_sha256=cohort_sha, payload_hashes=hashes,
                  selected_manifest_hashes={k: v for k, v in hashes.items() if k.endswith(".csv")},
                  splits=counts)
    (cache / "report.json").write_bytes(json_bytes(report))
    pin = hashlib.sha256((cache / "report.json").read_bytes()).hexdigest()
    load_nih_cache(cache, expected_report_sha256=pin)
    return cache, pin


BOOTSTRAP = r'''
import json,sys,time
from pathlib import Path
from fedlearn.simulation import core_qualification as core
from fedlearn.simulation import nih_cache
from fedlearn.simulation import lab_contract
from fedlearn.lab_worker import run_worker
config,output,pin,run_id,scenario,deadline=sys.argv[1:]
output=Path(output)
original=core.run_core
def controlled(cfg,**kwargs):
    (output/'engine-entered').write_text('actual core boundary')
    if scenario=='before_first':raise RuntimeError('/private/sensitive/before-first')
    if scenario=='hung':time.sleep(30)
    callback=kwargs['on_round']
    def observe(line):
        if scenario=='callback_nonfinite':line['loss']=float('inf')
        callback(line)
        if scenario in ('hold_first','hold_last','hung_after_first') and line['round']==(2 if scenario=='hold_last' else 1):
            (output/'boundary').write_text(str(line['round']))
            if scenario=='hung_after_first':time.sleep(30)
            while not kwargs['should_stop']():time.sleep(.01)
    kwargs['on_round']=observe
    record=original(cfg,**kwargs)
    if scenario=='after_last':raise core.CoreQualificationError('after final callback',record=record)
    if scenario=='nonfinite':
        record['rounds'][-1]['loss']=float('inf')
        raise core.CoreQualificationError('nonfinite postflight control',record=record)
    if scenario=='rss':record['meta']['peak_rss_bytes']=2**31+1
    return record
core.run_core=controlled
if scenario=='cache_hung':nih_cache.load_nih_cache=lambda *a,**k:time.sleep(30)
if scenario=='record_hung':
    def blocked_record(*a,**k):
        (output/'boundary').write_text('record publication')
        time.sleep(30)
    lab_contract.write_research_record=blocked_record
if scenario=='hold_partial':
    writer=lab_contract.write_research_record
    def hold_partial(path,record):
        encoding=writer(path,record)
        if path.name=='.partial-record.json':
            (output/'boundary').write_text('partial record ready')
            while not (output/'release').exists():time.sleep(.01)
        return encoding
    lab_contract.write_research_record=hold_partial
raise SystemExit(run_worker(Path(config),output,report_sha256=pin,run_id=run_id,
                           parent_stream=sys.stdin.buffer,deadline_seconds=float(deadline)))
'''


def launch(tmp_path, lab_cache, *, scenario="normal", deadline=15, mutation=None):
    cache, pin = lab_cache
    run_id = str(uuid.uuid4())
    job = tmp_path / run_id
    job.mkdir()
    cfg = resolve_choices({"rounds": 2}, cache_path=cache, report_sha256=pin)["config"]
    if mutation:
        mutation(cfg)
    config = job / "config.json"
    config.write_text(json.dumps(cfg))
    output = job / "execution"
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    log = (job / "process.log").open("w")
    child = subprocess.Popen([sys.executable, "-c", BOOTSTRAP, str(config), str(output), pin,
                              run_id, scenario, str(deadline)], stdin=subprocess.PIPE,
                             stdout=log, stderr=subprocess.STDOUT, env=env)
    return child, output, log


def finish(child, output, log, timeout=25):
    try:
        code = child.wait(timeout=timeout)
        log.flush()
        progress = json.loads((output / "progress.json").read_text())
        execution = json.loads((output / "execution.json").read_text())
        return code, progress, execution
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdin.close()
        log.close()


def boundary(child, output, timeout=20):
    until = time.monotonic() + timeout
    while child.poll() is None and time.monotonic() < until:
        if (output / "boundary").exists():return
        time.sleep(.02)
    raise AssertionError((output.parent / "process.log").read_text())


def cli(tmp_path, *arguments):
    source = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run([sys.executable, "-m", "fedlearn.lab_worker", *arguments],
                          cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(source)},
                          capture_output=True, text=True, timeout=15)


def test_resolve_cli_uses_installed_library_only_and_returns_one_json_object(tmp_path):
    choices = tmp_path / "choices.json"
    choices.write_text('{"head":"mlp8","rounds":1}')
    result = cli(tmp_path, "resolve", "--choices", str(choices), "--cache", str(tmp_path / "cache"),
                 "--report-sha256", "a" * 64)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["display_recipe"]["parameters"] == 4122
    assert value["config"]["federation"]["rounds"] == 1


@pytest.mark.parametrize("text", ['{"seed":1,"seed":2}', '{"epsilon":NaN}', '[]'])
def test_resolve_cli_rejects_bad_json_without_returning_a_recipe(tmp_path, text):
    choices = tmp_path / "choices.json"
    choices.write_text(text)
    result = cli(tmp_path, "resolve", "--choices", str(choices), "--cache", str(tmp_path / "cache"),
                 "--report-sha256", "a" * 64)
    assert result.returncode == 2 and result.stdout == ""
    assert str(tmp_path) not in result.stderr


@pytest.mark.parametrize("kind", ["existing", "symlink", "foreign_parent", "noncanonical_id"])
def test_owned_output_refusal_preserves_existing_files_before_any_worker_thread(tmp_path, kind):
    from fedlearn.lab_worker import run_worker
    run_id = str(uuid.uuid4())
    job = tmp_path / run_id
    job.mkdir()
    config = job / "config.json"
    config.write_text("{}")
    output = job / "execution"
    protected = tmp_path / "protected"
    protected.mkdir()
    sentinel = protected / "sentinel"
    sentinel.write_text("preserved")
    if kind == "existing":output.mkdir()
    elif kind == "symlink":output.symlink_to(protected, target_is_directory=True)
    elif kind == "foreign_parent":config = protected / "config.json"
    else:run_id = run_id.upper()
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb") as owner:
            with pytest.raises((ValueError, FileExistsError)):
                run_worker(config, output, report_sha256="a" * 64, run_id=run_id, parent_stream=owner)
    finally:
        os.close(write_fd)
    assert sentinel.read_text() == "preserved"
    assert not (output / "progress.json").exists()


@pytest.mark.parametrize("deadline", [True, 0, -1, 181, float("nan"), float("inf")])
def test_worker_rejects_an_unbounded_or_nonfinite_deadline_before_output(tmp_path, deadline):
    from fedlearn.lab_worker import run_worker
    job = tmp_path / str(uuid.uuid4())
    job.mkdir()
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "rb") as owner:
            with pytest.raises(ValueError):
                run_worker(job / "config.json", job / "execution", report_sha256="a" * 64,
                           run_id=job.name, parent_stream=owner, deadline_seconds=deadline)
    finally:
        os.close(write_fd)
    assert not (job / "execution").exists()


def test_actual_two_round_worker_qualifies_and_persists_exact_completed_events(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache)
    code, progress, execution = finish(child, output, log)
    record = json.loads((output / "record.json").read_text())
    assert code == 0 and progress["state"] == execution["state"] == "completed"
    assert progress["version"] == 1 and progress["run_id"] == output.parent.name
    assert progress["config_hash"] == record["meta"]["config_hash"] == execution["config_hash"]
    assert progress["completed_rounds"] == progress["requested_rounds"] == 2
    assert progress["rounds"] == record["rounds"] and [r["round"] for r in progress["rounds"]] == [1, 2]
    assert record["meta"]["core_qualification"]["evaluation_split"] == "validation"
    assert record["meta"]["workload"]["torch_num_threads"] == 4
    assert progress["record_encoding"] == "json" and progress["record_file"] == "record.json"
    assert progress["error_code"] is progress["error_message"] is None
    assert execution["elapsed_seconds"] > 0


@pytest.mark.parametrize("mutation", [
    lambda c:c["workload"].update(evaluation_split="test"),
    lambda c:c["workload"].update(features_report_sha256="0" * 64),
    lambda c:c["client"].update(lr=.5),
    lambda c:c["seeds"].update(noise=17),
])
def test_recipe_tampering_refuses_before_actual_core_entry(tmp_path, lab_cache, mutation):
    child, output, log = launch(tmp_path, lab_cache, mutation=mutation)
    code, progress, _ = finish(child, output, log)
    assert code == 2 and progress["state"] == "failed"
    assert progress["error_code"] == "invalid_recipe" and progress["completed_rounds"] == 0
    assert not (output / "engine-entered").exists() and not (output / "record.json").exists()


@pytest.mark.parametrize("corruption", ["payload", "random"])
def test_unqualified_cache_refuses_before_engine(tmp_path, lab_cache, corruption):
    cache, pin = lab_cache
    if corruption == "payload":(cache / "train-x.npy").write_bytes(b"corrupt")
    else:
        report = json.loads((cache / "report.json").read_text())
        report.update(pretrained=False, weight_identity="random-seed:16")
        (cache / "report.json").write_bytes(json_bytes(report))
        lab_cache = cache, hashlib.sha256((cache / "report.json").read_bytes()).hexdigest()
    child, output, log = launch(tmp_path, lab_cache)
    code, progress, _ = finish(child, output, log)
    assert code == 2 and progress["error_code"] == "invalid_cache"
    assert not (output / "engine-entered").exists()


@pytest.mark.parametrize("scenario,completed", [("hold_first", 1), ("hold_last", 2)])
def test_sigterm_after_real_callback_retains_work_without_success_marker(tmp_path, lab_cache, scenario, completed):
    child, output, log = launch(tmp_path, lab_cache, scenario=scenario)
    try:
        boundary(child, output)
        child.send_signal(signal.SIGTERM)
    finally:
        code, progress, _ = finish(child, output, log)
    record = json.loads((output / "record.json").read_text())
    assert code == 3 and progress["state"] == "stopped" and progress["error_code"] == "stop_requested"
    assert progress["completed_rounds"] == completed == len(record["rounds"])
    assert "core_qualification" not in record["meta"]


def test_parent_pipe_loss_stops_after_real_round_and_cannot_report_completion(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="hold_first")
    try:
        boundary(child, output)
        child.stdin.close()
    finally:
        code, progress, _ = finish(child, output, log)
    assert code == 3 and progress["state"] == "interrupted"
    assert progress["error_code"] == "parent_disconnected" and progress["completed_rounds"] == 1
    assert "core_qualification" not in json.loads((output / "record.json").read_text())["meta"]


@pytest.mark.parametrize("scenario", ["hung", "cache_hung"])
def test_deadline_is_independent_of_round_and_cache_callbacks(tmp_path, lab_cache, scenario):
    child, output, log = launch(tmp_path, lab_cache, scenario=scenario, deadline=.5)
    code, progress, execution = finish(child, output, log, timeout=8)
    assert code == 124 and progress["state"] == execution["state"] == "timed_out"
    assert progress["error_code"] == "deadline_exceeded" and progress["completed_rounds"] == 0
    assert execution["elapsed_seconds"] < 4
    assert not (output / "record.json").exists()


def test_parent_loss_also_terminates_a_core_that_never_checks_stop(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="hung_after_first")
    try:
        boundary(child, output)
        child.stdin.close()
    finally:
        code, progress, _ = finish(child, output, log, timeout=8)
    assert code == 3 and progress["state"] == "interrupted" and progress["completed_rounds"] == 1


def test_deadline_also_bounds_final_record_storage(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="record_hung", deadline=1.5)
    try:
        boundary(child, output)
    finally:
        code, progress, _ = finish(child, output, log, timeout=6)
    assert code == 124 and progress["state"] == "timed_out"
    assert progress["completed_rounds"] == 2 and not (output / "record.json").exists()


def test_stop_during_record_staging_keeps_the_full_unqualified_record(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="hold_partial")
    try:
        boundary(child, output)
        child.send_signal(signal.SIGTERM)
        (output / "release").touch()
    finally:
        code, progress, _ = finish(child, output, log)
    record = json.loads((output / "record.json").read_text())
    assert code == 3 and progress["state"] == "stopped" and len(record["rounds"]) == 2
    assert "core_qualification" not in record["meta"]


def test_nonfinite_completed_callback_and_failed_record_are_retained_as_tags(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="callback_nonfinite")
    code, progress, _ = finish(child, output, log)
    raw = json.loads((output / "record.json").read_text())
    assert code == 2 and progress["error_code"] == "qualification_failed"
    assert progress["completed_rounds"] == 2 and progress["record_encoding"] == "nonfinite_tagged_v1"
    assert all(line["loss"] == {"__nonfinite_float__": "inf"} for line in progress["rounds"])
    assert raw["encoding"] == "nonfinite_tagged_v1" and "core_qualification" not in raw["record"]["meta"]


@pytest.mark.parametrize("scenario,completed", [("before_first", 0), ("after_last", 2), ("nonfinite", 2), ("rss", 2)])
def test_failure_never_promotes_callbacks_into_success_and_retains_available_record(tmp_path, lab_cache, scenario, completed):
    child, output, log = launch(tmp_path, lab_cache, scenario=scenario)
    code, progress, _ = finish(child, output, log)
    assert code == 2 and progress["state"] == "failed" and progress["completed_rounds"] == completed
    assert "/private" not in json.dumps(progress)
    if scenario == "before_first":assert not (output / "record.json").exists()
    else:
        raw = json.loads((output / "record.json").read_text())
        record = raw["record"] if scenario == "nonfinite" else raw
        assert "core_qualification" not in record["meta"]
        if scenario == "nonfinite":
            assert progress["record_encoding"] == "nonfinite_tagged_v1"
            assert record["rounds"][-1]["loss"] == {"__nonfinite_float__": "inf"}


def test_actual_process_kill_retains_atomic_last_completed_snapshot(tmp_path, lab_cache):
    child, output, log = launch(tmp_path, lab_cache, scenario="hold_first")
    try:
        boundary(child, output)
        child.kill()
    finally:
        code, progress, execution = finish(child, output, log)
    assert code == -signal.SIGKILL and progress["state"] == "running"
    assert progress["completed_rounds"] == 1 and [r["round"] for r in progress["rounds"]] == [1]
    assert execution["state"] in ("starting", "running")
    assert not (output / "record.json").exists()
