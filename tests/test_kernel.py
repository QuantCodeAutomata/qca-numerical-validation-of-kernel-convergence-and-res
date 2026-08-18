"""Tests for kernel computation and Mittag-Leffler utilities."""
import numpy as np
import pytest
from scipy.special import gamma

from src.kernel import (
    compute_phi,
    compute_phi_sequence,
    compute_a_n,
    compute_scaled_kernel,
    mittag_leffler,
    mittag_leffler_kernel,
    compute_psi_direct,
    compute_psi_fft,
    compute_rho_n,
    compute_kernel_errors,
)


def test_phi_sum_approaches_one():
    """Sum of phi_1 … phi_L converges to 1 as L grows for any alpha in (0.5, 1)."""
    L = 10_000
    for alpha in [0.501, 0.62, 0.999]:
        phi = compute_phi_sequence(alpha, L)
        total = np.sum(phi)
        assert abs(total - 1.0) < 0.01, (
            f"alpha={alpha}: sum(phi)={total:.6f}, expected within 0.01 of 1.0"
        )


def test_phi_positivity():
    """All kernel weights phi_i are non-negative for i = 1 … 1000."""
    alpha = 0.62
    phi = compute_phi_sequence(alpha, 1000)
    assert np.all(phi >= 0), "Found negative phi weight(s)"


def test_phi_monotone_decreasing():
    """phi_i is monotonically non-increasing for i >= 2 (index 1 onward, 0-indexed)."""
    alpha = 0.62
    phi = compute_phi_sequence(alpha, 1000)
    # phi[0] = phi_1 may be special; strict decrease starts from phi_2 (index 1)
    tail = phi[1:]
    assert np.all(np.diff(tail) <= 0), (
        "phi sequence is not monotonically non-increasing from i=2 onward"
    )


def test_phi_1_formula():
    """phi_1 equals 1 - 1 / (Gamma(1 - alpha) * 2^alpha) for alpha = 0.62."""
    alpha = 0.62
    expected = 1.0 - 1.0 / (gamma(1.0 - alpha) * 2.0**alpha)
    phi_1 = compute_phi(alpha, 1)
    assert abs(phi_1 - expected) < 1e-10, (
        f"phi_1={phi_1:.12f}, expected={expected:.12f}"
    )


def test_a_n_in_unit_interval():
    """a_n lies strictly in (0, 1) for all combinations of alpha and n tested."""
    for alpha in [0.501, 0.62, 0.999]:
        for n in [100, 500, 5000]:
            a = compute_a_n(alpha, n)
            assert 0.0 < a < 1.0, (
                f"alpha={alpha}, n={n}: a_n={a} is outside (0, 1)"
            )


def test_scaled_kernel_sum_less_than_one():
    """Sum of phi_i^(n) = a_n * phi_i is strictly less than 1 (since a_n < 1)."""
    alpha = 0.62
    n = 100
    L = 1000
    phi_scaled = compute_scaled_kernel(alpha, n, L)
    total = np.sum(phi_scaled)
    assert total < 1.0, f"sum(phi^(n))={total:.6f} should be < 1"


def test_mittag_leffler_at_zero():
    """E_{alpha, alpha}(0) = 1 / Gamma(alpha).

    The two-parameter Mittag-Leffler function evaluated at z=0 reduces to
    the reciprocal of the Gamma function: E_{a,b}(0) = 1/Gamma(b).
    """
    alpha = 0.62
    result = mittag_leffler(0.0, alpha, alpha)
    expected = 1.0 / gamma(alpha)
    assert abs(result - expected) < 1e-6, (
        f"E_{{alpha,alpha}}(0)={result:.8f}, expected 1/Gamma(alpha)={expected:.8f}"
    )


def test_mittag_leffler_kernel_positive():
    """The Mittag-Leffler kernel f^{alpha, kappa}(x) is strictly positive for x in (0, 5]."""
    x_vals = np.linspace(0.01, 5.0, 50)
    for alpha in [0.501, 0.62, 0.999]:
        for kappa in [0.1, 1.0]:
            for x in x_vals:
                val = mittag_leffler_kernel(x, alpha, kappa)
                assert val > 0.0, (
                    f"alpha={alpha}, kappa={kappa}, x={x:.3f}: "
                    f"kernel value={val} is not positive"
                )


def test_psi_direct_vs_fft_equivalence():
    """Direct and FFT-based psi computations agree to within 1e-6 max absolute error."""
    alpha = 0.62
    n = 100
    kappa = 0.1
    L = 100
    psi_direct = compute_psi_direct(alpha, n, kappa, L)
    psi_fft = compute_psi_fft(alpha, n, kappa, L)
    max_diff = np.max(np.abs(psi_direct - psi_fft))
    assert max_diff < 1e-6, (
        f"max |psi_direct - psi_fft| = {max_diff:.2e}, expected < 1e-6"
    )


def test_psi_total_mass():
    """Total mass of the renewal measure satisfies sum(psi) * (1 - a_n) ≈ 1.

    When psi is the renewal measure U = delta + phi^(n) + phi^(n)*phi^(n) + …,
    its total mass is 1 / (1 - a_n), so sum(psi) * (1 - a_n) → 1 as L → ∞.
    """
    alpha = 0.62
    n = 100
    kappa = 0.1
    L = 2000
    psi = compute_psi_direct(alpha, n, kappa, L)
    a_n = compute_a_n(alpha, n)
    total = np.sum(psi) * (1.0 - a_n)
    assert abs(total - 1.0) < 0.05, (
        f"sum(psi)*(1-a_n)={total:.5f}, expected within 0.05 of 1.0"
    )


def test_rho_n_shape():
    """rho_n array has the same shape as the input x_grid."""
    x_grid = np.linspace(0.01, 5.0, 100)
    rho = compute_rho_n(x_grid, 0.62, 100, 0.1)
    assert rho.shape == x_grid.shape, (
        f"rho_n shape {rho.shape} does not match x_grid shape {x_grid.shape}"
    )


def test_kernel_errors_non_negative():
    """All error metrics returned by compute_kernel_errors are non-negative."""
    errors = compute_kernel_errors(0.62, 100, 0.1, 1000)
    for key, val in errors.items():
        assert val >= 0.0, f"Error metric '{key}' = {val} is negative"


def test_edge_case_phi_large_i():
    """phi_i for i = 1_000_000 is negligibly small (< 1e-10) for alpha = 0.62.

    The kernel weights decay as i^{-(1+alpha)} so large indices contribute
    essentially nothing to the sum.
    """
    alpha = 0.62
    phi_large = compute_phi(alpha, 1_000_000)
    assert phi_large < 1e-10, (
        f"phi(1_000_000) = {phi_large:.2e}, expected < 1e-10"
    )
