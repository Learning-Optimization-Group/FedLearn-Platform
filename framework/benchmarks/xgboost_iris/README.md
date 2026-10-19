# Real-data XGBoost qualification

This benchmark prepares binary Iris data for the platform's current four-feature
XGBoost contract. It keeps versicolor and virginica, maps their labels to `0` and
`1`, and excludes setosa. Features `f0` through `f3` mean sepal length, sepal
width, petal length and petal width, respectively, in centimetres.

```sh
PYTHONPATH=framework/src python framework/benchmarks/xgboost_iris/benchmark.py --out output/iris-binary
cd framework/benchmarks/xgboost_iris
python -m pytest
```

Preparation uses the locally installed scikit-learn Iris copy and records its
version, source-data digest, original sample identities, CSV hashes and feature
mapping. Identical measurement rows form groups before splitting. Five
stratified group folds reserve fold zero for testing and fold one for validation;
the remaining groups are split into two client partitions. This is an explicit
benchmark split; Iris has no official train/test partition.

For seed 42, the partitions contain 30 and 31 training rows, 20 validation rows and
19 test rows. Neither held-out partition trains any model. The test set is small,
so percentages from it are coarse and do not establish general model quality.

`centralized_comparison(data_directory, output, seed=42)` verifies each CSV
against its receipt and the bundled source, pools only the two client CSVs, and
trains two fixed eight-tree CPU controls. Both use depth 3 and histogram training.
Learning rate 0.15 matches each federated client's tree shrinkage; 0.3 is the
unscaled control. Centralized trees train sequentially on pooled rows, whereas
two federated clients grow trees in parallel from the same parent each round.
Equal tree counts therefore do not make the algorithms equivalent.

The actual platform qualification uses the normal project creation, enrollment,
client training, tree aggregation, artifact registration and inference paths.
Its retained report is
`research/results/xgboost/iris-mixed-cpu-2026-10-09.json`. Both original CPU
runtimes reproduced their four contributed trees exactly, all aggregate parents
matched, and changed-label controls differed. The final model has eight trees.
Held-out test accuracy was 18/19 and AUROC 0.99444. Both centralized controls also
classified 18/19 correctly. These runs do not qualify Android tree training,
multiclass models, clinical data, arbitrary schemas or packaged desktop clients.

Dataset reference:
[scikit-learn Iris documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
