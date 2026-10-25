"""Prospective pilot projection and promoted owned-data attack API."""
import torch

from fedlearn.simulation.owned_data_attack import ascent_upload


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def public_projection(input_dim, output_dim, *, seed):
    """Public orthonormal columns; reduced widths share one seeded basis.

    Full width is the exact identity control. No private fitting, centering,
    variance correction or coordinate rescaling is performed here.
    """
    _integer(input_dim, "input_dim")
    _integer(output_dim, "output_dim")
    _integer(seed, "seed", 0)
    if output_dim > input_dim:
        raise ValueError("output_dim exceeds input_dim")
    if output_dim == input_dim:
        return torch.eye(input_dim, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(input_dim, input_dim, dtype=torch.float64, generator=generator)
    basis, triangular = torch.linalg.qr(matrix)
    signs = torch.where(triangular.diag() < 0, -1., 1.)
    return (basis * signs)[:, :output_dim].contiguous()
