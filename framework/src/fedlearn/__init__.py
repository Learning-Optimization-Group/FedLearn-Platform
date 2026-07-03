# src/fedlearn/__init__.py

# Expose the main entry points for server and client
from . import server
from . import client

# Expose the base Client class for users to inherit from
from .client.client import Client

# Expose the base Strategy class and the default FedAvg strategy
from .server.strategy import Strategy, FedAvg, FedProx, FedOpt

# Expose the DeComFL components
from .client.decomfl_client import DeComFLClient
from .server.decomfl_strategy import DeComFL

# Expose the first-order local trainer (FedAvg / FedProx / FedOpt client) + strategy factory
from .client.local_trainer import LocalTrainer
from .server.strategy_factory import create_strategy