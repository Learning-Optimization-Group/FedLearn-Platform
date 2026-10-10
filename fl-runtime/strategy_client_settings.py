"""Client-side training a first-order strategy's FL server asks for each round.

One source for both sides of the execution contract: fl_server.py hands these to FedOpt, which sends them to
every client each round, and execution_plan.py publishes them as the contract's local training. The rate a
contract states is therefore the rate the server sends. Torch-free so the resolver can import it cheaply.
"""
# FedOpt adapts on the server (FedAdam); its clients use their recipe optimizer at this rate for this many epochs.
FEDOPT_CLIENT_LEARNING_RATE = 0.01
FEDOPT_CLIENT_LOCAL_EPOCHS = 1

# FedProx aggregates like FedAvg; its clients use their recipe optimizer at this rate for this many epochs and add the proximal
# gradient mu * (w - w_global) before each optimizer step.
FEDPROX_CLIENT_PROXIMAL_MU = 0.1
FEDPROX_CLIENT_LEARNING_RATE = 0.01
FEDPROX_CLIENT_LOCAL_EPOCHS = 1


# Full ResNet uses the same Adam rate as its base recipe. The legacy .01 override
# destabilized its adaptive runs; this does not change other recipes' settings.
RESNET_CLIENT_LEARNING_RATE = 0.001


def client_learning_rate(strategy: str, recipe: str | None = None) -> float:
    """Resolve a new run's rate identically for its plan and server round config."""
    defaults = {
        "fedopt": FEDOPT_CLIENT_LEARNING_RATE,
        "fedprox": FEDPROX_CLIENT_LEARNING_RATE,
    }
    key = strategy.lower()
    if key not in defaults:
        raise ValueError(f"No adaptive client rate for strategy {strategy!r}")
    if (recipe or "").upper() == "CIFAR_RESNET18":
        return RESNET_CLIENT_LEARNING_RATE
    return defaults[key]
