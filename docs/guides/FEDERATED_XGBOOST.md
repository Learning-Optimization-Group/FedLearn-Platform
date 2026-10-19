# Federated XGBoost

The initial tree workload trains a binary classifier through the platform with
Python CPU clients. Each client keeps its own rows and trains one new tree per
round against the current ensemble. The server appends the trees in client-ID
order. This is horizontal tree bagging, following the
[Flower parallel-tree approach](https://flower.ai/docs/examples/xgboost-comprehensive.html).
It is not histogram-sharing distributed XGBoost or neural FedAvg.

## Supported configuration

- Recipe `XGBOOST`, model `xgboost_binary`, optimizer `TreeBoosting`, arm `FULL`.
- Strategy `XgbBagging`, zero server pretraining, `LOCAL_SNAPSHOT` data source.
- XGBoost 3.4.1; CPU `hist`, binary logistic loss, base prediction 0.5, depth 3.
- Four ordered float32 features `f0,f1,f2,f3`, binary labels `0,1`, no missing
  values, categorical features or preprocessing fitted on server data.
- One new tree per client per round; local learning rate is 0.3 divided by the
  fixed cohort size. All cohort members must contribute before the deadline.
- Execution contract version 2 and tree RPC protocol 3. Neural clients and the
  Android app refuse this workload. Differential privacy, secure aggregation,
  frozen arms and model derivation are unavailable.

Install the optional workload in the runtime environment:

```sh
python -m pip install -e 'framework[xgboost]'
```

Linux installs `xgboost-cpu` to avoid GPU/NCCL dependencies. Other systems install
the matching `xgboost` package. Tree entry points select their workload before
importing PyTorch: the Mac OpenMP runtimes cannot safely coexist in these builds.
Keep tree tests in their separate process:

```sh
PYTHONPATH=framework/src python -m pytest -q framework/tree_tests
```

## Client data and execution

Create and start the project in the dashboard or through the normal project API.
Enroll each client as a separate user; one enrollment cannot impersonate several
cohort members. Save that run's published `executionContract` as ProtoJSON and
use the enrolled partition, run ID, endpoint and connection token.

Each client owns a CSV with this exact header:

```csv
f0,f1,f2,f3,label
0.2,1.1,-0.4,2.0,0
1.3,0.1,0.6,-0.2,1
```

```sh
python fl-runtime/client.py --model-type XGBOOST --model-name xgboost_binary \
  --project-id "$PROJECT_ID" --run-id "$RUN_ID" \
  --server-address "$SERVER_ADDRESS" --partition-id "$PARTITION_ID" \
  --execution-contract contract.json --dataset-path own-data.csv \
  --strategy XgbBagging --device cpu
```

Provide `FEDLEARN_CONNECTION_TOKEN` through the process environment when the run
requires authentication. Follow the existing enrollment certificate flow and
`FEDLEARN_GRPC_*` settings for TLS. Never put credentials in commits or logs.
The client checks the contract before opening data or connecting; it does not
partition, normalize or hold out rows from the supplied CSV.

## Acceptance and artifacts

Contributions carry the run, contract, round, parent digest, delta digest and
example count. Exact retries are idempotent; conflicting duplicates, stale
parents, wrong schemas and malformed tree graphs are rejected. JSON fields,
array lengths, node counts, depth, feature indices and numeric values are bounded
and checked before contributed trees enter native model loading. Trees can still
reveal training information; this path makes no differential-privacy claim and
cannot prove that a client used the prescribed training procedure.

The final ensemble is a UBJ `FULL_CHECKPOINT` in the content-addressed registry.
Continued runs and the normal vector inference API reload that registry head.
The evaluation card records tree counts and accepted-update digests. The server
has no client rows and does not fabricate loss or accuracy; evaluate a separately
held-out dataset and record its split provenance.

NIH ChestX-ray14 PNGs require a separate image adapter. A tree experiment would
first need a fixed feature extractor and a common embedding schema. The current
four-feature binary recipe cannot directly train on those images.


## Real-data qualification — 9 October 2026

A mixed Mac/g14 run trained collaboratively on binary Iris versicolor/virginica
rows: 30/31 training rows in separate clients, 20 validation and 19 test rows.
Duplicate measurement groups were separated before the deterministic split.
Four rounds produced eight trees through the normal enrollment, contribution,
aggregation, artifact and inference paths. Both original CPU runtimes replayed
their contributions exactly; all aggregate parents matched and changed-label
controls differed.

External test evaluation classified 18 of 19 rows correctly, with AUROC
0.99444. Two fixed centralized eight-tree controls also classified 18 of 19
correctly. Sequential pooled boosting and parallel federated tree growth are
different algorithms; equal tree counts do not make this a controlled efficacy
comparison. The small test set does not establish broad model quality.

The [benchmark instructions](../../framework/benchmarks/xgboost_iris/README.md)
explain feature mapping, provenance and reproduction. The
[qualification receipt](../../research/results/xgboost/iris-mixed-cpu-2026-10-09.json)
records metrics, hashes, exact replays and platform inference. Android tree
training, multiclass tasks, clinical features and arbitrary schemas remain
unsupported by this qualification.
