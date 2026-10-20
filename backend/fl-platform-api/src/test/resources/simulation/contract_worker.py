"""Synthetic process-contract controls, not the numerical simulation engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("mode")
for key in ("choices", "cache", "config", "output", "report-sha256", "run-id", "deadline-seconds"):
    p.add_argument("--" + key)
a = p.parse_args()
if a.mode == "resolve":
    choices = json.loads(Path(a.choices).read_text())
    if choices.get("seed") == 50:
        print("/private/control-only-diagnostic", file=sys.stderr)
    allowed = {"workloadId", "head", "numClients", "rounds", "localEpochs", "aggregation", "clipNorm", "privacy", "epsilon", "behavior", "seed"}
    if set(choices) - allowed:
        raise ValueError("/private/unsupported-choice")
    if len(str(choices.get("epsilon", 4))) > 100:
        raise OverflowError("/private/resolver-diagnostic")
    recipe = {"workloadId": "nih-qualified", "head": "linear", "numClients": 10, "rounds": 2,
              "localEpochs": 1, "aggregation": "mean", "clipNorm": .05, "privacy": "none",
              "epsilon": 4, "behavior": "clean", "seed": 42, **choices,
              "featureReportSha256": a.report_sha256, "evaluationSplit": "validation"}
    config = {"federation": {"rounds": recipe["rounds"], "num_clients": recipe["numClients"],
                             "network": {"downlink_mbps": 50.0, "uplink_mbps": 10.0, "latency_ms": 50.0}}, "seed": recipe["seed"],
              "seeds": {"init": recipe["seed"], "partition": recipe["seed"], "client": recipe["seed"],
                        "attack": recipe["seed"] + 1000, "noise": recipe["seed"] + 2000},
              "client": {"local_epochs": recipe["localEpochs"], "lr": .003, "batch_size": 256},
              "workload": {"features_path": a.cache, "features_report_sha256": a.report_sha256,
                           "evaluation_split": "validation"}, "recipe": recipe}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"config": config, "config_hash": digest, "display_recipe": recipe}))
    sys.exit(0)

config = json.loads(Path(a.config).read_text())
digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
output = Path(a.output)
output.mkdir()
seed = config["seed"]
stop = False


def stopped(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, stopped)


def write(name, value):
    path = output / name
    temporary = output / ("." + name)
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


progress = dict(version=1, run_id=a.run_id, config_hash=digest, state="running",
                completed_rounds=0, requested_rounds=config["federation"]["rounds"], rounds=[],
                record_encoding=None, record_file=None, error_code=None, error_message=None)
write("progress.json", progress)
if seed == 46:
    time.sleep(30)
if seed == 47:
    sys.exit(2)
for index in range(1, progress["requested_rounds"] + 1):
    if stop:
        break
    time.sleep(.15)
    progress["rounds"].append(dict(round=index, loss=.6, auc=.7, accuracy=.6,
                                  participants={"sampled": 10, "submitted": 10, "admitted": 10,
                                                "accepted": 10, "aggregated": 10, "dropped": 0, "straggled": 0}))
    progress["completed_rounds"] = index
    if seed == 45:
        progress["config_hash"] = "c" * 64
    write("progress.json", progress)
    if seed == 43:
        while not stop:
            time.sleep(.01)

code = 3 if stop else 0
state = "stopped" if stop else "completed"
meta = dict(config=config, config_hash=digest, seeds=config["seeds"], peak_rss_bytes=100000000,
            peak_rss_scope="run", wall_seconds=.5, stopped_early=stop, stop_reason="external_stop" if stop else None,
            participation_schema_version=2,
            totals={"up_bytes": 85000, "down_bytes": 85000, "cpu_seconds": .1, "simulated_seconds": .2,
                    "per_client_bytes": {"private-client-identity": 8500}},
            workload={"dataset": "nih", "source_dataset": "synthetic-contract-control", "features_report_sha256": a.report_sha256,
                      "curated_cohort_sha256": "c" * 64, "model_state_sha256": "d" * 64,
                      "evaluation_split": "validation", "pretrained": True, "feat_dim": 512, "parameters": 1026},
            provenance={"host": "private-contract-host", "python": "3.12.7", "torch": "2.12.0", "numpy": "2.1.2", "git_commit": None})
if not stop and seed != 44:
    meta["core_qualification"] = dict(version=1, scope="simulation_execution_only", evaluation_split="validation",
                                      accepted_clients=config["recipe"]["numClients"], privacy_mode=config["recipe"]["privacy"],
                                      research_record_dp_safe=False)
write("record.json", dict(meta=meta, rounds=progress["rounds"]))
progress.update(state=state, record_encoding="json", record_file="record.json")
write("progress.json", progress)
write("execution.json", dict(version=1, run_id=a.run_id, config_hash=digest, state=state, exit_code=code,
                             worker_pid=os.getpid(), cpu_threads=4))
sys.exit(code)
