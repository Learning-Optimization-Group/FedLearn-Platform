# src/fedlearn/client/local_trainer.py
"""First-order local trainer for the FedAvg / FedProx / FedOpt family.

Runs ordinary minibatch SGD on the client's local data and returns the updated model
parameters (the params-based submit path, unlike DeComFL's gradient-scalar path). It is the
client counterpart to :class:`fedlearn.server.strategy.FedProx` / ``FedOpt``:

  * FedProx — when ``config["proximal_mu"] > 0`` the trainer adds the proximal penalty
    ``(mu/2) * || w - w_global ||^2`` to the local objective. Its exact gradient contribution,
    ``mu * (w - w_global)``, is added to each parameter's ``.grad`` before the optimiser step,
    where ``w_global`` is the round's starting global model (the ``parameters`` handed to
    ``fit``). ``mu = 0`` skips the term entirely, so the trainer reduces to plain local SGD
    (i.e. the FedAvg client).
  * FedOpt — clients train plainly (``mu = 0``); the adaptive step happens server-side.

Hyperparameters arrive through ``config`` (``learning_rate``, ``proximal_mu``,
``local_epochs``), mirroring how ``DeComFLClient.fit`` reads ``config["learning_rate"]``.
Values may be str or float (they flow through a protobuf ``map<string,string>``); the trainer
coerces them.
"""

import logging
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .client import Client

log = logging.getLogger(__name__)


class LocalTrainer(Client):
    """SGD trainer with an optional FedProx proximal term (see module docstring)."""

    def __init__(
            self,
            model: nn.Module,
            train_loader,
            device: str = "cpu",
            criterion: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: PyTorch model to train.
            train_loader: Finite iterable of (inputs, targets) batches; must expose
                ``.dataset`` for the num-examples count (matches the FedAvg client contract).
            device: Compute device.
            criterion: Loss function; defaults to CrossEntropyLoss (classification).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.grpc_client = None

    def set_grpc_client(self, grpc_client):
        """Set gRPC client for heartbeat updates (parity with DeComFLClient)."""
        self.grpc_client = grpc_client

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        return self.model.state_dict()

    def fit(
            self,
            parameters: Optional[OrderedDict[str, torch.Tensor]],
            config: dict,
    ) -> Tuple[OrderedDict[str, torch.Tensor], int]:
        """Train locally and return (updated state_dict, num_examples)."""
        if parameters is not None:
            self.model.load_state_dict(parameters)

        lr = float(config.get("learning_rate", 0.01))
        mu = float(config.get("proximal_mu", 0.0))
        local_epochs = int(config.get("local_epochs", 1))

        # FedProx anchor: snapshot the round's starting global model w_global. Only needed when
        # the proximal term is active; skipped for plain SGD so mu=0 is exactly the FedAvg client.
        global_anchor = (
            [p.detach().clone() for p in self.model.parameters()] if mu > 0.0 else None
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.model.train()

        total_steps = max(1, local_epochs)
        step = 0
        # FR-10: server-driven stop — polled between minibatch steps so a stop request arriving
        # on the heartbeat thread aborts within ~one heartbeat interval + one optimiser step.
        # On abort we return the current (partially trained) state; the caller must check
        # should_stop_training() and skip the submit — the server has stopped the run.
        stopped = False
        for _ in range(local_epochs):
            if stopped:
                break
            for batch in self.train_loader:
                if self.grpc_client is not None and self.grpc_client.should_stop_training():
                    log.info("Server requested stop; aborting local training at epoch %d/%d",
                             step, total_steps)
                    stopped = True
                    break
                inputs, targets = batch
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                loss = self.criterion(self.model(inputs), targets)
                loss.backward()

                # FedProx: add the proximal gradient mu*(w - w_global) in-place before stepping.
                if mu > 0.0:
                    for p, w0 in zip(self.model.parameters(), global_anchor):
                        if p.grad is not None:
                            p.grad.add_(p.detach() - w0, alpha=mu)

                optimizer.step()

            step += 1
            if self.grpc_client:
                self.grpc_client.update_status("training", step, total_steps)

        num_examples = len(self.train_loader.dataset)
        new_params = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items()
        )
        log.debug("Local training complete: mu=%g lr=%g epochs=%d n=%d", mu, lr, local_epochs, num_examples)
        return new_params, num_examples
