# src/fedlearn/server/__init__.py

from .server import start_server, ServerConfig

# DeComFL components
from .decomfl_strategy import DeComFL

# Strategy classes
from .strategy import FedAvg, FedLoRA, FedProx, FedOpt

# Robust aggregation (FR-12): coordinate-wise median / trimmed-mean + norm clipping
from .robust_aggregation import RobustAggregator

# Factory construction imports the composed privacy pipeline. Load its public
# exports on demand so importing a privacy helper does not recurse into itself.
__all__ = ["start_server", "ServerConfig", "DeComFL", "FedAvg", "FedLoRA", "FedProx", "FedOpt",
           "RobustAggregator", "create_strategy", "STRATEGY_REGISTRY"]


def __getattr__(name):
    if name not in ("create_strategy", "STRATEGY_REGISTRY"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(".strategy_factory", __name__), name)
    globals()[name] = value
    return value
