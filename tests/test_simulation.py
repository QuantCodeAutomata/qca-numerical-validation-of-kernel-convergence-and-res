"""Tests for path simulation."""
import numpy as np
import pytest

from src.simulation import (
    decompose_mu_zeta,
    compute_omega_n,
    compute_omega_t_array,
    simulate_lambda_path,
    simulate_path,
    simulate_log_price_path,
)

# ---------------------------------------------------------------------------
# Shared calibration constants used across tests
# ---------------------------------------------------------------------------
ALPHA = 0.62
N_SCALE = 100          # time-scaling parameter n
N_STEPS = 500          # number of discrete time steps
KAPPA = 1.0
THETA = np.log(np.sqrt(0.3156))   # long-run log-vol mean; exp(THETA) ≈ 0.5618
MU = 0.4
RHO = -0.681
LAMBDA0 = np.log(np.sqrt(0.0392))  # initial log-vol; exp(LAMBDA0) ≈ 0.1980


# ---------------------------------------------------------------------------
# Parameter decomposition
# ---------------------------------------------------------------------------

def test_decompose_mu_zeta_constraint():
    """sqrt(rho^2 + zeta^2) * mu must equal 0.331 after decomposition.

    The function returns a 2-tuple (rho_component, zeta) where zeta is the
    orthogonal vol-of-vol component satisfying the identification constraint.
    """
    _, zeta = decompose_mu_zeta(MU, RHO)
    lhs = np.sqrt(RHO**2 + zeta**2) * MU
    assert abs(lhs - 0.331) < 1e-6, (
        f"sqrt(rho^2+zeta^2)*mu = {lhs:.8f}, expected 0.331"
    )


def test_decompose_mu_zeta_invalid_mu():
    """mu=0.7 exceeds the feasibility bound 0.331/|rho| ≈ 0.486 → ValueError."""
    with pytest.raises(ValueError):
        decompose_mu_zeta(0.7, RHO)


# ---------------------------------------------------------------------------
# Omega (intercept) utilities
# ---------------------------------------------------------------------------

def test_omega_n_positive():
    """omega_n is strictly positive for the calibrated parameter set."""
    omega = compute_omega_n(THETA, MU, N_SCALE)
    assert omega > 0.0, f"omega_n = {omega}, expected > 0"


def test_omega_t_array_shape():
    """compute_omega_t_array returns an array of shape (N,)."""
    N = 100
    omega_t = compute_omega_t_array(THETA, MU, N_SCALE, N)
    assert omega_t.shape == (N,), (
        f"omega_t shape {omega_t.shape}, expected ({N},)"
    )


# ---------------------------------------------------------------------------
# Lambda path simulation
# ---------------------------------------------------------------------------

def test_initial_volatility():
    """exp(Lambda[0]) equals the initial volatility sqrt(0.0392) ≈ 0.1980.

    The first element of the simulated Lambda path must match Lambda0, so
    exp(Lambda[0]) = exp(Lambda0) = sqrt(0.0392).
    """
    Lambda = simulate_lambda_path(
        alpha=ALPHA, n=N_SCALE, N=N_STEPS,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, seed=0,
    )
    assert abs(np.exp(Lambda[0]) - np.sqrt(0.0392)) < 1e-3, (
        f"exp(Lambda[0]) = {np.exp(Lambda[0]):.6f}, expected {np.sqrt(0.0392):.6f}"
    )


def test_lambda_path_finite():
    """All elements of the simulated Lambda path are finite (no NaN or Inf)."""
    Lambda = simulate_lambda_path(
        alpha=ALPHA, n=N_SCALE, N=N_STEPS,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, seed=1,
    )
    assert np.all(np.isfinite(Lambda)), "Lambda path contains NaN or Inf"


def test_sigma_positive():
    """All instantaneous volatilities sigma_t = exp(Lambda_t) are positive."""
    Lambda = simulate_lambda_path(
        alpha=ALPHA, n=N_SCALE, N=N_STEPS,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, seed=2,
    )
    sigma = np.exp(Lambda)
    assert np.all(sigma > 0.0), "Found non-positive sigma values"


# ---------------------------------------------------------------------------
# Log-price path simulation
# ---------------------------------------------------------------------------

def test_log_price_path_starts_at_X0():
    """The log-price path begins exactly at X0."""
    X0 = 0.0
    X = simulate_log_price_path(
        alpha=ALPHA, n=N_SCALE, N=N_STEPS,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, X0=X0, seed=3,
    )
    assert X[0] == X0, f"X[0] = {X[0]}, expected {X0}"


def test_log_price_path_length():
    """simulate_log_price_path returns an array of length N + 1."""
    N = 200
    X = simulate_log_price_path(
        alpha=ALPHA, n=N_SCALE, N=N,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, X0=0.0, seed=4,
    )
    assert len(X) == N + 1, f"len(X) = {len(X)}, expected {N + 1}"


def test_seed_reproducibility():
    """Two calls with the same seed produce byte-identical log-price paths."""
    kwargs = dict(
        alpha=ALPHA, n=N_SCALE, N=N_STEPS,
        kappa=KAPPA, theta=THETA, mu=MU,
        Lambda0=LAMBDA0, X0=0.0, seed=42,
    )
    X1 = simulate_log_price_path(**kwargs)
    X2 = simulate_log_price_path(**kwargs)
    assert np.all(X1 == X2), "Paths differ despite identical seeds"


# ---------------------------------------------------------------------------
# Roughness ordering
# ---------------------------------------------------------------------------

def test_roughness_ordering():
    """Lower alpha yields higher quadratic variation (rougher paths).

    For a fixed seed and the same simulation parameters, the discrete quadratic
    variation of the log-volatility increments must satisfy:
        QV(alpha=0.501) > QV(alpha=0.62) > QV(alpha=0.999)
    confirming that smaller alpha → smaller Hurst index → rougher paths.
    """
    seed = 0
    N = 1000
    common = dict(n=N_SCALE, N=N, kappa=KAPPA, theta=THETA, mu=MU,
                  Lambda0=LAMBDA0, seed=seed)

    def quadratic_variation(alpha):
        Lambda = simulate_lambda_path(alpha=alpha, **common)
        return float(np.sum(np.diff(Lambda) ** 2))

    qv_rough = quadratic_variation(0.501)
    qv_mid = quadratic_variation(0.62)
    qv_smooth = quadratic_variation(0.999)

    assert qv_rough > qv_mid, (
        f"QV(0.501)={qv_rough:.4f} should exceed QV(0.62)={qv_mid:.4f}"
    )
    assert qv_mid > qv_smooth, (
        f"QV(0.62)={qv_mid:.4f} should exceed QV(0.999)={qv_smooth:.4f}"
    )
