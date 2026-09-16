"""Single-node federated-learning simulation (P0).

This package makes it possible to run a federation of hundreds or thousands of clients
in one process, with no gRPC, no port pool, and no OS processes — while reusing the
*production* :class:`fedlearn.server.coordinator.FLCoordinator` and the production
strategies, so a simulated result exercises the same aggregation code a deployed run does.

Why it exists: the deployed FL path reserves a real TCP port per server from the range
``50000-50010`` (``application.properties``), which caps concurrent federations at 11 and
makes a 1000-client experiment impossible to express. Every published FL result is quoted
at client counts far above that, so without this package the platform cannot produce a
comparable experiments table.

Two design commitments, both load-bearing:

* **Determinism is a property, not an accident.** Each client draws from its own
  :class:`~fedlearn.simulation.rng.ClientRng` stream, derived from a single run seed, so a
  run is bitwise reproducible from its ``meta`` block and adding/removing a client does not
  perturb the others' randomness.
* **The wire stays in the loop.** :class:`~fedlearn.simulation.federation.SimulatedFederation`
  can route any fraction of clients through the real safetensors encode/decode path, so
  simulation never silently stops testing serialization.
"""

from .partition import (
    dirichlet_partition,
    iid_partition,
    partition_report,
    pathological_partition,
    shard_partition,
)
from .rng import ClientRng, RunRng

__all__ = [
    "dirichlet_partition",
    "iid_partition",
    "pathological_partition",
    "shard_partition",
    "partition_report",
    "ClientRng",
    "RunRng",
]
