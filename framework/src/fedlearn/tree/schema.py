"""Named, bounded ordered schemas shared by tree data, models and contracts."""

FEATURES = ("f0", "f1", "f2", "f3")
BREAST_CANCER_FEATURES = tuple(
    prefix + "_" + measure
    for prefix in ("mean", "error", "worst")
    for measure in ("radius", "texture", "perimeter", "area", "smoothness",
                    "compactness", "concavity", "concave_points", "symmetry",
                    "fractal_dimension")
)
MODEL_SCHEMAS = {"xgboost_binary": FEATURES,
                 "xgboost_breast_cancer": BREAST_CANCER_FEATURES}


def require_schema(feature_names):
    names = tuple(feature_names)
    if names not in MODEL_SCHEMAS.values():
        raise ValueError("unsupported ordered tree feature schema")
    return names


def model_schema(model_name):
    try:
        return MODEL_SCHEMAS[model_name]
    except KeyError as exc:
        raise ValueError("unsupported tree model preset") from exc
