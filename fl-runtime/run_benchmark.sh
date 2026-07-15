#!/bin/bash
set -e # Exit immediately if a command fails.

# Wrapper for benchmarks.py — the standalone benchmark scorer. Mirrors
# run_recipes.sh / run_infer.sh so the same cross-platform invocation works.
#
# Scores a predictions file and prints the full metric report (the same metrics
# the online per-round suite computes):
#
#   bash run_benchmark.sh --predictions preds.json [--out report.json]
#
# preds.json (classification):
#   {"taskType":"CLASSIFICATION","yTrue":[...],"yPred":[...],
#    "yScore":[[...]],"classNames":["..."]}
# preds.json (generative):
#   {"taskType":"CAUSAL_LM","avgLoss":1.23,"correctTokens":80,"totalTokens":100}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"
PYTHON="${FEDLEARN_PYTHON:-python3}"
"$PYTHON" benchmarks.py "$@"

EXIT_CODE=$?
exit $EXIT_CODE
