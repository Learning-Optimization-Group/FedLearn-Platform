

# Expose the main entry points for server and client
from . import server
from . import client

# Expose the base Client class for users to inherit from
from .client.client import Client

# Expose the base Strategy class and the default FedAvg strategy
from .server.strategy import Strategy, FedAvg