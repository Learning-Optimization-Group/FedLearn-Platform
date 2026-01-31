# src/fedlearn/client/__init__.py
"""
Client-side components for federated learning
"""

from .client import Client, start_client
from .grpc_client import GrpcClient
from .decomfl_client import DeComFLClient
from .decomfl_start import start_decomfl_client

__all__ = [
    'Client',
    'start_client',
    'GrpcClient',
    'DeComFLClient',
    'start_decomfl_client',
]