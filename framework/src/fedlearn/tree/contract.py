"""Strict acceptance of the first CPU tree execution contract."""
import hashlib
import re
import uuid

from google.protobuf.json_format import Parse

from fedlearn.communication.generated import execution_contract_pb2 as p
from .model import FEATURES, VERSION


def validate_contract(contract, *, run_id=None, project_id=None):
    def require(condition, detail):
        if not condition:
            raise ValueError("unsupported tree contract: " + detail)
    require(contract.contract_version == 2 and contract.min_client_protocol_version == 3, "version")
    require(contract.WhichOneof("workload") == "tree_training", "workload")
    for name, expected in (("run_id", run_id), ("project_id", project_id)):
        value = getattr(contract, name)
        try:
            require(str(uuid.UUID(value)) == value and (expected is None or value == expected), name)
        except (ValueError, AttributeError) as exc:
            raise ValueError("invalid tree contract identity") from exc
    require(contract.recipe == p.RECIPE_XGBOOST and contract.strategy == p.STRATEGY_XGB_BAGGING, "recipe/strategy")
    require(1 <= contract.num_rounds <= 100 and 1 <= contract.clients_per_round <= 100, "round sizes")
    require(contract.partitioning == p.PARTITIONING_LOCAL and contract.HasField("seed"), "partition/seed")
    policy = contract.round
    require(1 <= policy.timeout_ms <= 86_400_000 and policy.one_accepted_update_per_round, "round policy")
    require(policy.HasField("max_transient_retries") and policy.max_transient_retries <= 3
            and 1 <= policy.retry_backoff_ms <= 60_000, "retry policy")
    security = contract.security
    require(security.transport in (p.TRANSPORT_TLS_REQUIRED, p.TRANSPORT_PLAINTEXT_DEV), "transport")
    require(security.client_auth in (p.CLIENT_AUTH_CONNECTION_TOKEN, p.CLIENT_AUTH_DISABLED_DEV), "client auth")
    require(security.secure_aggregation == p.SECAGG_NONE and not security.HasField("secure_agg_threshold")
            and not security.HasField("central_dp"), "secure aggregation/DP")
    tree = contract.tree_training
    require(tree.xgboost_version == VERSION and tree.objective == "binary:logistic"
            and tree.booster == "gbtree" and tree.tree_method == "hist", "tree engine")
    require(list(tree.feature_names) == list(FEATURES) and list(tree.classes) == ["0", "1"], "features/classes")
    require(tree.base_score == .5 and tree.learning_rate == .3 and tree.max_depth == 3
            and tree.local_boost_rounds == 1, "training parameters")
    require(tree.input_dtype == "float32" and tree.missing_policy == "reject"
            and tree.contribution_codec == "xgboost-json-v1"
            and tree.data_source == p.DATA_SOURCE_LOCAL_SNAPSHOT, "data/update protocol")
    require(bool(re.fullmatch("[a-f0-9]{64}", tree.initial_model_sha256)), "initial digest")
    return contract


def parse_contract(text, **expected):
    if len(text) > 64 * 1024:
        raise ValueError("tree contract is oversized")
    value = Parse(text, p.ExecutionContract())
    return validate_contract(value, **expected)


def contract_id(contract):
    return hashlib.sha256(contract.SerializeToString(deterministic=True)).hexdigest()
