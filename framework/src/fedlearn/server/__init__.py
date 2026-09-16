# src/fedlearn/server/__init__.py

from .server import start_server, ServerConfig

# DeComFL components
from .decomfl_strategy import DeComFL

# Strategy classes
from .strategy import FedAvg, FedLoRA, FedProx, FedOpt

# Robust aggregation (FR-12): coordinate-wise median / trimmed-mean + norm clipping
from .robust_aggregation import RobustAggregator

# Strategy factory (name -> Strategy)
from .strategy_factory import create_strategy, STRATEGY_REGISTRY
