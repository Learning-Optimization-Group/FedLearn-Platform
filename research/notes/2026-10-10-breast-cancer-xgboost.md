# Thirty-feature collaborative XGBoost — 10 October 2026

The platform completed ten authenticated rounds between the native client
embedded in the fresh Mac desktop app and a Linux CPU client. Each participant
trained 182 local rows and contributed one depth-three tree per round. The
registered model contains twenty trees. The Mac participant used the normal packaged-app login, native CSV picker,
project selection and Start controls; it exited successfully and returned to idle.

The explicit `xgboost_breast_cancer` preset binds an ordered thirty-feature schema
through CSV loading, execution contract, parent and delta validation, aggregation
and inference. The original four-feature `xgboost_binary` preset remains supported.
Arbitrary or reordered schemas remain refused. The engine is pinned to XGBoost
3.4.1, CPU histogram training, binary logistic prediction, intercept0.5 and
client-tree shrinkage0.15 for this two-client cohort. Trees grown in parallel
from a shared parent are appended in canonical client-ID order.

[Preparation](../benchmarks/breast_cancer.py) uses the locally bundled
[scikit-learn diagnostic dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
It maps labels to 0=benign and 1=malignant, casts raw measurements to float32 and
separates identical feature groups before stratified splitting. The fixed split
contains 182/182 client rows, 91 validation rows and 114 test rows. Source CSV
digest, original indices, feature mapping and every CSV hash are retained. No
patient IDs are provided, so patient separation is not claimed. Clients send
trees and metrics through the training protocol; their CSV rows stay local.

Each original CPU runtime reproduced all ten contributed trees exactly; changed
labels produced different trees. All twenty accepted tree digests and every
aggregate parent reconstruct the registered model exactly. Independent artifact
reload preserves predictions, normal API inference matches, and a four-value
request to the thirty-feature model is refused.

Final test accuracy is 106/114 (92.98%), with AUROC0.977513. Six of 42 malignant
samples are missed and two of 72 benign samples are false positives at0.5.
Two fixed pooled controls train only the 364 training rows: twenty sequential
trees at rate0.15 score105/114, and ten sequential trees at rate0.3 score106/114.
Parallel federated trees and sequential pooled trees differ algorithmically;
these small results do not establish superiority or clinical efficacy. Budgets
were fixed before evaluation, with no test-driven tuning.

The [qualification receipt](../results/xgboost/breast-cancer-mixed-cpu-2026-10-10.json)
contains provenance, all contribution receipts, exact replay checks, complete
heldout metrics, fixed controls and executable identity. A prior waiting attempt
failed before collaborative training; it is retained without an unverified cause
claim. Transport was authenticated development plaintext. Android XGBoost,
multiclass training, arbitrary schemas and deployed TLS remain separate work.
