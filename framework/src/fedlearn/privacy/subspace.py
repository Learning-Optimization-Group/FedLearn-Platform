"""Intrinsic Subspace Projection for High-Dimensional Differential Privacy.

Mitigates the Curse of Dimensionality in federated differential privacy (SNR proportional to 1 / sqrt(d))
by projecting client update deltas into a shared, seed-synchronized pseudo-random orthonormal subspace:
    z_i = P_t^T * Delta_i in R^k  (k << d)
where P_t in R^{d x k} is generated deterministically from the round seed.

DP clipping and calibrated Gaussian noise are injected strictly in R^k, reducing total injected noise
energy by a factor of d / k (e.g. 204x less noise variance for d=26,112, k=128), boosting SNR by
sqrt(d / k) ~= 14.3x. Reconstructing Delta_global = P_t * z_global in R^d is a linear post-processing
step that incurs ZERO additional privacy loss by the Post-Processing Theorem of Differential Privacy.

References:
    - Yu, D., et al. (2021). Large Scale Private Learning via Low-rank Reparametrization. ICML 2021.
    - Aghajanyan, A., et al. (2020). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. ACL 2021.
    - Li, C., et al. (2018). Measuring the Intrinsic Dimension of Objective Landscapes. ICLR 2018.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def build_orthonormal_basis(d: int, k: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Construct an orthonormal basis P in R^{d x k} (P^T P = I_k) from a deterministic seed.

    Uses reduced QR decomposition of a Gaussian random matrix G ~ N(0, 1)^{d x k}.
    Enforces canonical positive diagonal on R to eliminate cross-platform Householder
    reflection sign ambiguities (Gram-Schmidt uniqueness theorem).
    """
    if d < 1:
        raise ValueError(f"ambient dimension d must be >= 1, got {d}")
    if k < 1:
        raise ValueError(f"subspace dimension k must be >= 1, got {k}")
    if k > d:
        raise ValueError(f"subspace dimension k ({k}) cannot exceed ambient dimension d ({d})")

    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    # Generate Gaussian matrix on CPU for deterministic cross-device numerical agreement
    g_mat = torch.randn(d, k, generator=gen, dtype=torch.float32, device="cpu")
    q_mat, r_mat = torch.linalg.qr(g_mat, mode="reduced")
    # Resolve QR column sign ambiguity by forcing positive diagonal entries on R
    d_signs = torch.sign(torch.diag(r_mat))
    d_signs = torch.where(d_signs == 0.0, torch.tensor(1.0, dtype=q_mat.dtype), d_signs)
    q_mat = q_mat * d_signs.unsqueeze(0)
    if device != "cpu":
        q_mat = q_mat.to(device)
    return q_mat


class SubspaceProjector:
    """Manages seed-synchronized orthonormal subspace projection and reconstruction across rounds.

    Args:
        in_dim: Ambient parameter dimension d.
        subspace_dim: Subspace dimension k <= d.
        base_seed: Base PRG seed.
        rotating: If True, round t uses seed = (base_seed + t * 10007), exploring different subspaces
            across rounds while preserving per-round DP noise compression. If False, uses a fixed seed.
    """

    def __init__(self, in_dim: int, subspace_dim: int, base_seed: int = 42, rotating: bool = True):
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if subspace_dim < 1:
            raise ValueError(f"subspace_dim must be >= 1, got {subspace_dim}")
        if subspace_dim > in_dim:
            raise ValueError(f"subspace_dim ({subspace_dim}) cannot exceed in_dim ({in_dim})")

        self.in_dim = int(in_dim)
        self.subspace_dim = int(subspace_dim)
        self.base_seed = int(base_seed)
        self.rotating = bool(rotating)

        self._cached_round: Optional[int] = None
        self._cached_basis: Optional[torch.Tensor] = None

    @property
    def noise_norm_reduction_factor(self) -> float:
        """Ratio of expected subspace noise norm to ambient noise norm: sqrt(k / d)."""
        return math.sqrt(float(self.subspace_dim) / float(self.in_dim))

    @property
    def snr_amplification(self) -> float:
        """Theoretical upper-bound SNR amplification factor sqrt(d / k) for signals fully in-subspace."""
        return math.sqrt(float(self.in_dim) / float(self.subspace_dim))

    @property
    def noise_reduction_factor(self) -> float:
        """Ratio of subspace noise energy to full parameter noise energy: k / d."""
        return float(self.subspace_dim) / float(self.in_dim)

    def get_basis(self, server_round: int = 0) -> torch.Tensor:
        """Get or compute the orthonormal basis P_t in R^{d x k} for the given round."""
        round_key = server_round if self.rotating else 0
        if self._cached_round == round_key and self._cached_basis is not None:
            return self._cached_basis

        seed = (self.base_seed + round_key * 10007) & 0xFFFFFFFF
        basis = build_orthonormal_basis(self.in_dim, self.subspace_dim, seed=seed)
        self._cached_round = round_key
        self._cached_basis = basis
        return basis

    def project_vector(self, flat_vector: torch.Tensor, server_round: int = 0) -> torch.Tensor:
        """Project flat update delta in R^d to coordinate vector z in R^k: z = P_t^T * x.

        Uses double-precision dot products to avoid intermediate float32 overflow on large inputs.
        """
        basis = self.get_basis(server_round)
        v = flat_vector.double().reshape(-1)
        if v.numel() != self.in_dim:
            raise ValueError(f"Expected vector of length {self.in_dim}, got {v.numel()}")
        z = basis.double().T @ v
        return z.float()

    def reconstruct_vector(self, z: torch.Tensor, server_round: int = 0) -> torch.Tensor:
        """Reconstruct update delta in R^d from subspace coordinate vector z in R^k: x_recon = P_t * z."""
        basis = self.get_basis(server_round)
        coord = z.double().reshape(-1)
        if coord.numel() != self.subspace_dim:
            raise ValueError(f"Expected coordinate vector of length {self.subspace_dim}, got {coord.numel()}")
        recon = basis.double() @ coord
        return recon.float()

    @staticmethod
    def flatten_dict(
        params: "OrderedDict[str, torch.Tensor]", keys: Sequence[str]
    ) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size, int]], Dict[str, torch.dtype]]:
        """Flatten tensors for ``keys`` into a single 1D tensor in canonical sorted key order."""
        canonical_keys = sorted(keys)
        tensors = []
        layout = []
        dtypes = {}
        for k in canonical_keys:
            t = params[k]
            tensors.append(t.detach().float().cpu().reshape(-1))
            layout.append((k, t.shape, t.numel()))
            dtypes[k] = t.dtype
        flat = torch.cat(tensors, dim=0) if tensors else torch.empty(0, dtype=torch.float32)
        return flat, layout, dtypes

    @staticmethod
    def unflatten_dict(
        flat: torch.Tensor,
        layout: List[Tuple[str, torch.Size, int]],
        dtypes: Optional[Dict[str, torch.dtype]] = None,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Unflatten a 1D tensor back into an OrderedDict matching ``layout``."""
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        offset = 0
        for name, shape, numel in layout:
            slice_t = flat[offset : offset + numel].reshape(shape).clone()
            if dtypes is not None and name in dtypes:
                slice_t = slice_t.to(dtypes[name])
            out[name] = slice_t
            offset += numel
        return out

    def project_dict(
        self,
        update: "OrderedDict[str, torch.Tensor]",
        keys: Sequence[str],
        server_round: int = 0,
    ) -> torch.Tensor:
        """Flatten ``keys`` from ``update`` in canonical sorted order and project into R^k."""
        flat, _, _ = self.flatten_dict(update, keys)
        return self.project_vector(flat, server_round)

    def reconstruct_dict(
        self,
        z: torch.Tensor,
        template: "OrderedDict[str, torch.Tensor]",
        keys: Sequence[str],
        server_round: int = 0,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Reconstruct model tensors in R^d matching ``template`` in canonical sorted order from z."""
        recon_flat = self.reconstruct_vector(z, server_round)
        canonical_keys = sorted(keys)
        layout = [(k, template[k].shape, template[k].numel()) for k in canonical_keys]
        dtypes = {k: template[k].dtype for k in canonical_keys}
        return self.unflatten_dict(recon_flat, layout, dtypes)
