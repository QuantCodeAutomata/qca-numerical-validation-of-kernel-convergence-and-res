"""Tests for FFT vs direct benchmark."""
import numpy as np
import pytest

from src.fft_benchmark import compute_phi_n, simulate_direct, simulate_fft_accelerated

# ---------------------------------------------------------------------------
# Shared parameters for benchmark tests
# ---------------------------------------------------------------------------
ALPHA = 0.62
N_SCALE = 50     # time-scaling parameter n used for benchmark comparisons
N_STEPS = 100    # number of simulation time steps


# ---------------------------------------------------------------------------
# Kernel weights
# ---------------------------------------------------------------------------

def test_phi_n_positivity():
    """All scaled kernel weights phi_n are non-negative."""
    phi_n = compute_phi_n(ALPHA, N_SCALE, L=1000)
    assert np.all(phi_n >= 0), "Found negative phi_n weight(s)"


def test_phi_n_sum_less_than_one():
    """Sum of phi_n weights is strictly less than 1 (a_n < 1 by construction)."""
    phi_n = compute_phi_n(ALPHA, N_SCALE, L=1000)
    total = np.sum(phi_n)
    assert total < 1.0, f"sum(phi_n)={total:.6f}, expected < 1"


# ---------------------------------------------------------------------------
# Simulation equivalence
# ---------------------------------------------------------------------------

def test_direct_fft_numerical_equivalence():
    """Direct and FFT-accelerated simulations agree to within 1e-8 max abs error.

    For n=50 and a shared random seed, both methods must produce the same
    log-volatility path up to floating-point rounding differences.
    """
    seed = 0
    Lambda_direct = simulate_direct(ALPHA, N_SCALE, N_STEPS, seed=seed)
    Lambda_fft = simulate_fft_accelerated(ALPHA, N_SCALE, N_STEPS, seed=seed)
    max_diff = np.max(np.abs(Lambda_direct - Lambda_fft))
    assert max_diff < 1e-8, (
        f"max |direct - fft| = {max_diff:.2e}, expected < 1e-8"
    )


def test_direct_deterministic():
    """simulate_direct produces identical output for identical inputs (seed fixed)."""
    seed = 42
    Lambda1 = simulate_direct(ALPHA, N_SCALE, N_STEPS, seed=seed)
    Lambda2 = simulate_direct(ALPHA, N_SCALE, N_STEPS, seed=seed)
    assert np.all(Lambda1 == Lambda2), (
        "simulate_direct is not deterministic: outputs differ for same seed"
    )


def test_simulation_length():
    """simulate_direct returns an array of length N + 1 (including t=0)."""
    Lambda = simulate_direct(ALPHA, N_SCALE, N_STEPS, seed=0)
    assert len(Lambda) == N_STEPS + 1, (
        f"len(Lambda)={len(Lambda)}, expected {N_STEPS + 1}"
    )
