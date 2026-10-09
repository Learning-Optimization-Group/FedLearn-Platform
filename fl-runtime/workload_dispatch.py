"""Select the non-neural workload before initializing native tensor libraries."""
import argparse


def dispatch_tree_cli(role):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-type")
    args, _ = parser.parse_known_args()
    if args.model_type == "XGBOOST":
        import tree_runtime
        raise SystemExit(getattr(tree_runtime, role + "_main")())
