"""Isolated per-client randomness for federated simulation (P0-1b).

A federated simulation that draws every client's randomness from one global stream is not
reproducible in the way experiments need. The trajectory of client 7 ends up depending on how
many clients ran before it, so ``clients_per_round=10`` and ``clients_per_round=100`` are not
comparable, and a dropped client silently perturbs every remaining one. The bug is invisible
in a passing test suite and fatal in an experiments table.

The fix is to make a client's stream a **pure function of (run_seed, client_id)** — and, where
a round-local draw is needed, of ``(run_seed, client_id, round)``. Nothing else. In particular
it must not depend on cohort size or construction order, which rules out the otherwise natural
``numpy.random.SeedSequence.spawn(n)`` design: spawning ``n`` children folds ``n`` into each
child's entropy, so the same client draws differently in a bigger cohort.

Here the identity is folded into ``SeedSequence``'s entropy *tuple* instead, which is
order-free and count-free by construction::

    SeedSequence(entropy=(run_seed, client_id))
    SeedSequence(entropy=(run_seed, client_id, round))

This mirrors the determinism discipline BlazeFL (arXiv:2604.03606) established for single-node
FL simulation — isolated per-client generators yielding bitwise-identical reruns — and matches
this repo's existing commitment to a byte-deterministic wire.

Torch needs separate handling: PyTorch's global RNG drives dropout, weight init and DataLoader
shuffling, and seeding it inside one client would leak into the next. :func:`torch_rng_scope`
therefore saves and restores global torch state around any block that seeds it.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import Iterator, Tuple

import numpy as np
import torch

__all__ = ["ClientRng", "RunRng", "torch_rng_scope"]

# Domain separator, so a torch seed and a numpy stream derived from the same identity are not
# the same number reused in two places.
_TORCH_DOMAIN = 0x7F0C_1D2E

# The server's own stream identity. numpy's SeedSequence rejects negative entropy, so the
# obvious "-1 means the server" sentinel is not available; this is the top of the uint32 range
# instead, which no realistic client_id reaches.
SERVER_STREAM_ID = 0xFFFF_FFFF


def _seed_sequence(entropy: Tuple[int, ...]) -> np.random.SeedSequence:
    """Build a SeedSequence from an identity tuple.

    ``SeedSequence`` accepts only non-negative integers, and fails deep inside Cython with a
    bare ``ValueError: expected non-negative integer`` that names neither the offending value
    nor the caller. Validate here so a negative seed or client id is reported where it can be
    acted on.
    """
    for i, value in enumerate(entropy):
        if value < 0:
            raise ValueError(
                f"RNG identity components must be non-negative; entropy[{i}] = {value}. "
                f"(Seeds and client ids are identities, not offsets.)"
            )
    return np.random.SeedSequence(entropy=list(entropy))


class ClientRng:
    """The randomness a single simulated client is allowed to use.

    Args:
        run_seed: the run-level seed, recorded in the result's ``meta`` block.
        client_id: the client's stable integer identity within the federation.

    The ``numpy`` generator is created lazily and memoised, so repeated access continues one
    stream rather than restarting it — ``rng.numpy.random(2)`` twice yields four distinct
    values, as a caller would expect.
    """

    __slots__ = ("run_seed", "client_id", "_np")

    def __init__(self, run_seed: int, client_id: int):
        self.run_seed = int(run_seed)
        self.client_id = int(client_id)
        self._np: np.random.Generator | None = None

    # -- streams ------------------------------------------------------------------------

    @property
    def numpy(self) -> np.random.Generator:
        """This client's persistent numpy stream."""
        if self._np is None:
            self._np = np.random.default_rng(
                _seed_sequence((self.run_seed, self.client_id))
            )
        return self._np

    def for_round(self, round_num: int) -> "ClientRng":
        """A fresh round-scoped view whose stream depends on ``(run_seed, client_id, round)``.

        Round-scoped rather than sequential so that round 5 is reproducible *without* replaying
        rounds 1-4 — which is what makes a single anomalous round re-examinable, and a crashed
        run resumable, without re-running everything before it.
        """
        view = ClientRng.__new__(ClientRng)
        view.run_seed = self.run_seed
        view.client_id = self.client_id
        view._np = np.random.default_rng(
            _seed_sequence((self.run_seed, self.client_id, int(round_num)))
        )
        return view

    # -- torch --------------------------------------------------------------------------

    def torch_seed(self, round_num: int | None = None) -> int:
        """A stable 63-bit torch seed for this identity (optionally round-scoped)."""
        entropy: Tuple[int, ...] = (self.run_seed, self.client_id, _TORCH_DOMAIN)
        if round_num is not None:
            entropy = entropy + (int(round_num),)
        # generate_state is the documented way to get well-mixed bits out of a SeedSequence.
        words = _seed_sequence(entropy).generate_state(2, dtype=np.uint32)
        value = (int(words[0]) << 32) | int(words[1])
        return value & ((1 << 63) - 1)

    def torch_generator(self, round_num: int | None = None) -> torch.Generator:
        """A private ``torch.Generator`` — e.g. for a DataLoader's shuffle order.

        Preferred over seeding torch globally: a generator passed explicitly cannot leak into
        another client's draws.
        """
        g = torch.Generator()
        g.manual_seed(self.torch_seed(round_num))
        return g

    # -- provenance ---------------------------------------------------------------------

    def provenance(self) -> dict:
        """JSON-serializable identity for a result's ``meta`` block.

        ``stream_digest`` lets a note assert "this run used the same client streams" without
        dumping raw draws, and makes an accidental reseed visible in a diff.
        """
        state = _seed_sequence((self.run_seed, self.client_id)).generate_state(
            4, dtype=np.uint32
        )
        digest = hashlib.sha256(state.tobytes()).hexdigest()[:16]
        return {
            "run_seed": self.run_seed,
            "client_id": self.client_id,
            "stream_digest": digest,
        }


class RunRng:
    """Factory for the per-client streams of one run.

    Holds no cross-client state by design: :meth:`client` is a pure function of its argument,
    so constructing clients in any order — or constructing only some of them — yields the same
    streams. Any caching here would be an optimisation, never a semantic.
    """

    __slots__ = ("seed",)

    def __init__(self, seed: int):
        self.seed = int(seed)

    def client(self, client_id: int) -> ClientRng:
        return ClientRng(run_seed=self.seed, client_id=client_id)

    def server_rng(self, round_num: int | None = None) -> np.random.Generator:
        """The server's own stream — client sampling, dropout selection, strategy noise.

        Keyed on :data:`SERVER_STREAM_ID` so it can never collide with a real client's stream.
        Pass ``round_num`` for a round-scoped stream, so round 5's cohort is reproducible
        without replaying rounds 1-4.
        """
        entropy: Tuple[int, ...] = (self.seed, SERVER_STREAM_ID)
        if round_num is not None:
            entropy = entropy + (int(round_num),)
        return np.random.default_rng(_seed_sequence(entropy))

    def server_torch_seed(self) -> int:
        """Torch seed for server-side user code — chiefly the strategy's ``evaluate_fn``.

        Server-side callbacks are ordinary user code and routinely touch torch's global RNG
        (constructing a model to load parameters into is enough). Seeding the whole run from
        here, under a restoring scope, keeps those draws derived from the run seed rather than
        from whatever global state the caller left behind.
        """
        return ClientRng(run_seed=self.seed, client_id=SERVER_STREAM_ID).torch_seed()

    def provenance(self) -> dict:
        return {"run_seed": self.seed}


@contextlib.contextmanager
def torch_rng_scope(seed: int) -> Iterator[None]:
    """Seed torch's global RNG for a block, then restore the previous state exactly.

    Local training touches torch's global RNG through dropout, weight init and any DataLoader
    that was not handed an explicit generator. Seeding it without restoring re-couples clients
    — the precise failure this module exists to prevent — so every seeded block is scoped.

    CUDA state is restored too when available, since a device-side generator is just as global.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else None
    )
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
