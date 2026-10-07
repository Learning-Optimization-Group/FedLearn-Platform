# XGBoost CPU execution probe

This standalone experiment checks the XGBoost runtime and model-serialization
foundations for future platform integration. It does not implement federation,
tree aggregation, Android training, or a production recipe.

## Run

From the repository root, using Python 3.12:

```bash
python -m venv .temp/xgboost-venv
.temp/xgboost-venv/bin/python -m pip install -r framework/benchmarks/xgboost_cpu/requirements.txt
.temp/xgboost-venv/bin/python -m pytest -c framework/benchmarks/xgboost_cpu/pytest.ini framework/benchmarks/xgboost_cpu/tests
.temp/xgboost-venv/bin/python framework/benchmarks/xgboost_cpu/benchmark.py \
  --output-dir .temp/xgboost-probe --seed 42
```

Use a new output directory for every run. An existing directory is refused to
preserve previous experiments. The separate requirements leave the platform's
production dependency set unchanged.

## What is checked

- A deterministic synthetic binary-classification fixture: 256 training rows,
  64 validation rows and 128 test rows, with four ordered numeric features.
  Splits use disjoint source-row identities. No real user data is downloaded.
- Eight initial CPU histogram boosting rounds, with a fixed base prediction,
  learning rate, seed and one thread. Validation/test data do not train the model.
- XGBoost-produced JSON and UBJSON checkpoints with manifests recording runtime
  version, ordered feature names, round count, payload size and SHA-256.
- Exact prediction agreement after saving and loading both formats.
- Four additional rounds after loading each checkpoint, compared with twelve
  uninterrupted rounds under the same pinned configuration.
- Feature-order, checksum and runtime/format checks before loading a checkpoint.

Checkpoint validation is for locally produced experiment files. Checksums do not
authenticate a model, and this is not an ingestion boundary for untrusted models.
Federated model admission and security require the later execution-contract design.

## Outputs and interpretation

`report.json` records versions, parameters, split hashes/class counts, checkpoint
hashes/sizes, initial/continued metrics, reload/continuation differences and total
probe wall time. The directory also contains `initial.json`, `initial.ubj`,
`continued.ubj` and their manifests. Metrics include accuracy, balanced accuracy,
MCC, ROC-AUC, log loss and the confusion matrix.

The probe raises an error if checkpoint reload or continuation differs from the
same-runtime CPU reference. Exact agreement here is not a cross-device tolerance
or a claim that federated XGBoost will equal centralized training. Synthetic
accuracy is execution evidence, not a result on ECG or another research dataset.

Next integration work: select the federation method and target data schema, then
design typed contributions, a compatible execution contract, coordination,
artifact registration and inference through the platform.
