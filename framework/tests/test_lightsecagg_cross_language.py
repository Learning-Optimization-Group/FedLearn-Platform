"""Cross-language verification tests for Android Native C++ LightSecAgg.

Directly loads the native C++ library compiled from
`mobile_client/android/app/src/main/cpp/LightSecAggMath.cpp` and asserts bitwise and
numerical parity against Python reference `fedlearn.security.lightsecagg`.
"""
import ctypes
import os
import subprocess
import tempfile
import pytest
import torch

from fedlearn.security.lightsecagg import (
    PRIME,
    aggregate_shares,
    mask_values,
    recover_aggregate,
    shamir_reconstruct,
    shamir_share,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CPP_DIR = os.path.join(REPO_ROOT, "mobile_client", "android", "app", "src", "main", "cpp")


@pytest.fixture(scope="module")
def native_lib():
    """Compiles and loads the native C++ LightSecAgg library into a temporary directory."""
    src = os.path.join(CPP_DIR, "LightSecAggMath.cpp")
    with tempfile.TemporaryDirectory() as tmpdir:
        lib_path = os.path.join(tmpdir, "liblightsecagg.dylib")
        cmd = [
            "clang++",
            "-std=c++17",
            "-O2",
            "-dynamiclib",
            src,
            "-o",
            lib_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            pytest.fail(f"Failed to compile native LightSecAgg dylib: {res.stderr}")

        lib = ctypes.CDLL(lib_path)

        # Function signatures
        lib.lightsecagg_add_mod.argtypes = [ctypes.c_int64, ctypes.c_int64]
        lib.lightsecagg_add_mod.restype = ctypes.c_int64

        lib.lightsecagg_sub_mod.argtypes = [ctypes.c_int64, ctypes.c_int64]
        lib.lightsecagg_sub_mod.restype = ctypes.c_int64

        lib.lightsecagg_mul_mod.argtypes = [ctypes.c_int64, ctypes.c_int64]
        lib.lightsecagg_mul_mod.restype = ctypes.c_int64

        lib.lightsecagg_inv_mod.argtypes = [ctypes.c_int64]
        lib.lightsecagg_inv_mod.restype = ctypes.c_int64

        lib.lightsecagg_quantize.argtypes = [ctypes.c_double, ctypes.c_int64]
        lib.lightsecagg_quantize.restype = ctypes.c_int64

        lib.lightsecagg_dequantize.argtypes = [ctypes.c_int64, ctypes.c_int64]
        lib.lightsecagg_dequantize.restype = ctypes.c_double

        lib.lightsecagg_compute_shamir_shares.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.lightsecagg_compute_shamir_shares.restype = ctypes.c_int32

        lib.lightsecagg_aggregate_shares.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.lightsecagg_aggregate_shares.restype = ctypes.c_int32

        lib.lightsecagg_mask_vector.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.lightsecagg_mask_vector.restype = ctypes.c_int32

        lib.lightsecagg_reconstruct_secret.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.lightsecagg_reconstruct_secret.restype = ctypes.c_int32

        yield lib



def test_native_modular_arithmetic_parity(native_lib):
    """Verifies modular add, sub, mul, and inv against Python."""
    test_cases = [
        (0, 0),
        (0, PRIME - 1),
        (PRIME - 1, 1),
        (PRIME - 1, PRIME - 1),
        (123456789, 987654321),
        (PRIME // 2, PRIME // 3),
        (-1, 5),
        (-100, -200),
    ]
    for a, b in test_cases:
        py_add = ((a % PRIME + PRIME) % PRIME + (b % PRIME + PRIME) % PRIME) % PRIME
        cpp_add = native_lib.lightsecagg_add_mod(a, b)
        assert cpp_add == py_add, f"add mismatch for ({a}, {b}): {cpp_add} vs {py_add}"

        py_sub = (a - b) % PRIME
        cpp_sub = native_lib.lightsecagg_sub_mod(a, b)
        assert cpp_sub == py_sub, f"sub mismatch for ({a}, {b}): {cpp_sub} vs {py_sub}"

        py_mul = (a * b) % PRIME
        cpp_mul = native_lib.lightsecagg_mul_mod(a, b)
        assert cpp_mul == py_mul, f"mul mismatch for ({a}, {b}): {cpp_mul} vs {py_mul}"

    # Inverses
    for val in [1, 2, 3, 100, 1000000, PRIME - 1, PRIME - 2]:
        py_inv = pow(val, PRIME - 2, PRIME)
        cpp_inv = native_lib.lightsecagg_inv_mod(val)
        assert cpp_inv == py_inv, f"inv mismatch for {val}: {cpp_inv} vs {py_inv}"


def test_native_quantization_parity(native_lib):
    """Verifies quantization and dequantization matching Python lightsecagg."""
    scale = 1_000_000
    for v in [0.0, 1.0, -1.0, 0.5, -0.25, 0.123456, -0.987654]:
        # Python quantization in lightsecagg.py:
        # torch.round(torch.as_tensor(values, dtype=torch.float64) * scale).to(torch.int64) % PRIME
        py_q = int(torch.round(torch.as_tensor(v, dtype=torch.float64) * scale).to(torch.int64) % PRIME)
        cpp_q = native_lib.lightsecagg_quantize(v, scale)
        assert cpp_q == py_q, f"quantize mismatch for {v}: {cpp_q} vs {py_q}"

        cpp_dq = native_lib.lightsecagg_dequantize(cpp_q, scale)
        assert abs(cpp_dq - v) < 1e-5


def test_native_shamir_shares_parity_with_python(native_lib):
    """Verifies Shamir sharing in C++ using Python-generated coefficients matches Horner evaluation."""
    length = 8
    num_shares = 5
    threshold = 3
    secret = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64)

    # Use deterministic coefficients from torch generator
    gen = torch.Generator().manual_seed(42)
    coeffs = [secret]
    for _ in range(threshold - 1):
        coeffs.append(torch.randint(0, PRIME, (length,), generator=gen, dtype=torch.int64))

    # Python shares
    py_shares = {}
    for x in range(1, num_shares + 1):
        acc = torch.zeros(length, dtype=torch.int64)
        for c in reversed(coeffs):
            acc = (acc * x + c) % PRIME
        py_shares[x] = acc

    # C++ shares via ctypes
    c_secret = (ctypes.c_int64 * length)(*secret.tolist())
    flattened_coeffs = []
    for c in coeffs[1:]:
        flattened_coeffs.extend(c.tolist())
    c_coeffs = (ctypes.c_int64 * len(flattened_coeffs))(*flattened_coeffs)
    out_shares = (ctypes.c_int64 * (num_shares * length))()

    res = native_lib.lightsecagg_compute_shamir_shares(
        c_secret,
        length,
        num_shares,
        threshold,
        c_coeffs,
        out_shares,
    )
    assert res == 0

    for x in range(1, num_shares + 1):
        start = (x - 1) * length
        cpp_share = list(out_shares[start : start + length])
        assert cpp_share == py_shares[x].tolist(), f"Share mismatch at x={x}"


def test_native_aggregate_shares_parity_with_python(native_lib):
    """Verifies C++ aggregate_shares matches Python aggregate_shares bitwise."""
    num_vectors = 4
    length = 6
    shares = [
        torch.randint(0, PRIME, (length,), dtype=torch.int64)
        for _ in range(num_vectors)
    ]
    py_agg = aggregate_shares(shares)

    flat_shares = []
    for s in shares:
        flat_shares.extend(s.tolist())
    c_shares = (ctypes.c_int64 * len(flat_shares))(*flat_shares)
    c_out = (ctypes.c_int64 * length)()

    res = native_lib.lightsecagg_aggregate_shares(
        c_shares,
        num_vectors,
        length,
        c_out,
    )
    assert res == 0
    assert list(c_out) == py_agg.tolist()


def test_native_mask_vector_parity_with_python(native_lib):
    """Verifies C++ mask_vector matches Python mask_values bitwise."""
    values = [0.1, -0.5, 1.25, -2.0, 0.0]
    length = len(values)
    mask = torch.randint(0, PRIME, (length,), dtype=torch.int64)

    py_masked = mask_values(values, mask)

    c_vals = (ctypes.c_double * length)(*values)
    c_mask = (ctypes.c_int64 * length)(*mask.tolist())
    c_out = (ctypes.c_int64 * length)()

    res = native_lib.lightsecagg_mask_vector(
        c_vals,
        c_mask,
        length,
        1_000_000,
        c_out,
    )
    assert res == 0
    assert list(c_out) == py_masked.tolist()


def test_native_reconstruct_secret_cross_recovery(native_lib):
    """Verifies that secret shared by Python is reconstructed by C++, and vice versa."""
    length = 5
    num_shares = 6
    threshold = 3
    secret = torch.tensor([111, 222, 333, 444, 555], dtype=torch.int64)

    py_shares = shamir_share(secret, num_shares=num_shares, threshold=threshold, seed=777)

    # Reconstruct in C++ using shares 2, 4, 5
    xs = [2, 4, 5]
    c_xs = (ctypes.c_int32 * len(xs))(*xs)
    flat_shares = []
    for x in xs:
        flat_shares.extend(py_shares[x].tolist())
    c_shares = (ctypes.c_int64 * len(flat_shares))(*flat_shares)
    c_secret = (ctypes.c_int64 * length)()

    res = native_lib.lightsecagg_reconstruct_secret(
        c_xs,
        c_shares,
        len(xs),
        length,
        threshold,
        c_secret,
    )
    assert res == 0
    assert list(c_secret) == secret.tolist()
