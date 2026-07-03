# src/fedlearn/server/strategy_factory.py
"""Strategy factory: name -> Strategy instance.

Single dispatch point the server entrypoint / coordinator wiring selects from, so adding a
strategy is a one-line registry entry rather than another if/elif branch scattered across the
launch code. Lives in its own module to avoid a circular import: ``strategy.py`` must not import
``decomfl_strategy`` (which imports ``strategy``), so the factory — which needs both — sits
above them.

Usage:
    from fedlearn.server.strategy_factory import create_strategy
    strategy = create_strategy("fedprox", initial_parameters=params, proximal_mu=0.1)

Names are matched case-insensitively; hyphens/underscores are ignored ("fed_avg" == "FedAvg").
"""

from typing import Callable, Dict

from .strategy import Strategy, FedAvg, FedLoRA, FedProx, FedOpt
from .decomfl_strategy import DeComFL
from .robust_aggregation import RobustAggregator

# Registry of canonical strategy name -> constructor. Extend here to register a new strategy.
STRATEGY_REGISTRY: Dict[str, Callable[..., Strategy]] = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedopt": FedOpt,
    "fedlora": FedLoRA,
    "decomfl": DeComFL,
    "robust": RobustAggregator,
}


def _normalize(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


def create_strategy(name: str, **kwargs) -> Strategy:
    """Instantiate the strategy registered under ``name`` with ``kwargs``.

    Raises:
        ValueError: if ``name`` is not registered.
    """
    key = _normalize(name)
    ctor = STRATEGY_REGISTRY.get(key)
    if ctor is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown strategy {name!r}; available: {available}")
    return ctor(**kwargs)
