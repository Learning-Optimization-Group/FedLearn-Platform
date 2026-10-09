"""Public neural entry points, loaded only when requested.

Tree workloads must not initialize PyTorch or its OpenMP runtime.
"""
from importlib import import_module

_EXPORTS = {
    "server": (".server", None), "client": (".client", None),
    "Client": (".client.client", "Client"),
    "Strategy": (".server.strategy", "Strategy"),
    "FedAvg": (".server.strategy", "FedAvg"),
    "FedProx": (".server.strategy", "FedProx"),
    "FedOpt": (".server.strategy", "FedOpt"),
    "DeComFLClient": (".client.decomfl_client", "DeComFLClient"),
    "DeComFL": (".server.decomfl_strategy", "DeComFL"),
    "LocalTrainer": (".client.local_trainer", "LocalTrainer"),
    "create_strategy": (".server.strategy_factory", "create_strategy"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value
    return value
