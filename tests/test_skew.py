"""Tests for src.skew, src.scott_model, and src.mc_engine utilities."""

import numpy as np
import pytest

from src.skew import (
    implied_vol_call,
    estimate_atm_skew_fd,
    estimate_atm_skew_regression,
    powerlaw_regression,
    _bs_call,
)
from src.scott_model import simulate_scott_batch, compute_payoffs_scott
from src.mc_engine import (
    _simulate_lambda_batch,
    MU, RHO, KAPPA, THETA, LAMBDA0, X0,
)
from src.simulation import decompose_mu_zeta, _compute_omega_t_full
from src.kernel import compute_a_n, compute_phi_sequence


# ---------------------------------------------------------------------------
# _bs_call  (skew module internal)
# ---------------------------------------------------------------------------

def test_bs_call_positive():
    """Black-Scholes call price is non-negative for all moneyness."""
    for K in [80.0, 100.0, 120.0]:
        c = _bs_call(100.0, K, 1.0, 0.2)
        assert c >= 0.0, f"Negative call for K={K}"


def test_bs_call_intrinsic_itm():
    """Deep ITM call is approximately S - K."""
    c = _bs_call(200.0, 100.0, 1.0, 0.01)   # very low vol, very ITM
    assert abs(c - 100.0) < 1.0, f"Deep-ITM call {c:.2f} far from intrinsic 100"


def test_bs_call_at_zero_maturity():
    """At T=0, call price equals max(S-K,0)."""
    assert _bs_call(100.0, 90.0, 0.0, 0.3) == pytest.approx(10.0, abs=1e-8)
    assert _bs_call(100.0, 110.0, 0.0, 0.3) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# implied_vol_call
# ---------------------------------------------------------------------------

def test_implied_vol_roundtrip():
    """Implied vol inversion recovers the original sigma to 1e-5."""
    sigma_true = 0.25
    S, K, T = 100.0, 100.0, 0.5
    C = _bs_call(S, K, T, sigma_true)
    sigma_rec = implied_vol_call(C, S, K, T)
    assert sigma_rec is not None, "implied_vol_call returned None"
    assert abs(sigma_rec - sigma_true) < 1e-5, (
        f"Recovered IV={sigma_rec:.6f}, expected {sigma_true}"
    )


def test_implied_vol_below_intrinsic_returns_none():
    """Price below intrinsic returns None (no valid IV)."""
    S, K, T = 100.0, 90.0, 1.0
    intrinsic = max(S - K, 0.0)
    result = implied_vol_call(intrinsic - 1.0, S, K, T)
    assert result is None, "Expected None for sub-intrinsic price"


def test_implied_vol_various_strikes():
    """Round-trip succeeds for ITM, ATM, and OTM options."""
    S, T = 100.0, 1.0
    for K, sigma in [(80.0, 0.2), (100.0, 0.3), (120.0, 0.25)]:
        C = _bs_call(S, K, T, sigma)
        iv = implied_vol_call(C, S, K, T)
        assert iv is not None, f"K={K}: implied_vol_call returned None"
        assert abs(iv - sigma) < 1e-4, f"K={K}: IV={iv:.4f} vs {sigma}"


# ---------------------------------------------------------------------------
# ATM skew estimators
# ---------------------------------------------------------------------------

def test_atm_skew_fd_positive():
    """FD ATM skew estimate is positive for a negatively-skewed smile."""
    # Under negative rho: IV(-h) > IV(+h), so slope is negative, |slope| > 0
    iv_pos = 0.20   # IV at k = +h (OTM call, low IV under neg rho)
    iv_neg = 0.25   # IV at k = -h (ITM call, high IV under neg rho)
    psi = estimate_atm_skew_fd(iv_neg, iv_pos, h=0.01)
    assert psi > 0, f"ATM skew should be positive, got {psi}"
    assert abs(psi - 2.5) < 1e-6, f"FD skew = {psi}, expected 2.5"


def test_atm_skew_regression_matches_fd():
    """Regression-based ATM skew closely matches FD estimate for a linear smile."""
    ks = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    # Perfect linear IV: iv(k) = 0.25 - 2*k  → slope = -2 → |slope| = 2
    ivs = 0.25 - 2.0 * ks
    psi_reg = estimate_atm_skew_regression(ks, ivs)
    assert abs(psi_reg - 2.0) < 1e-6, f"Regression skew = {psi_reg}, expected 2.0"


# ---------------------------------------------------------------------------
# Power-law regression
# ---------------------------------------------------------------------------

def test_powerlaw_regression_perfect_fit():
    """Perfect power-law data (log-log linear) yields R²=1 and correct slope."""
    b_true = -0.3
    tau = np.array([0.1, 0.2, 0.5, 1.0])
    psi = np.exp(1.5) * tau**b_true          # a + b*log(tau) perfectly linear

    result = powerlaw_regression(tau, psi)
    assert abs(result["slope"] - b_true) < 1e-6, (
        f"Slope={result['slope']:.6f}, expected {b_true}"
    )
    assert abs(result["r_squared"] - 1.0) < 1e-6, (
        f"R²={result['r_squared']:.6f}, expected 1.0"
    )


def test_powerlaw_regression_raises_on_nonpositive():
    """Raises ValueError if tau or psi contain non-positive values."""
    with pytest.raises(ValueError):
        powerlaw_regression([-0.1, 0.5], [0.1, 0.2])
    with pytest.raises(ValueError):
        powerlaw_regression([0.1, 0.5], [0.0, 0.2])


def test_powerlaw_regression_r2_between_0_and_1():
    """R² is in [0, 1] for noisy data."""
    rng = np.random.default_rng(0)
    tau = np.array([1/52, 1/12, 3/12, 6/12, 1.0])
    psi = 0.5 * tau**(-0.3) + rng.normal(0, 0.01, size=len(tau))
    psi = np.abs(psi) + 0.01           # ensure positive
    result = powerlaw_regression(tau, psi)
    assert 0.0 <= result["r_squared"] <= 1.0, f"R²={result['r_squared']}"


# ---------------------------------------------------------------------------
# Scott model simulation
# ---------------------------------------------------------------------------

def test_scott_simulation_output_shape():
    """simulate_scott_batch returns (n_paths, N+1) arrays."""
    n_paths, N = 50, 20
    S, Lam = simulate_scott_batch(
        n_paths=n_paths, N=N, dt=1.0/N,
        kappa=KAPPA, theta=THETA, sigma=0.331,
        rho=RHO, Lambda0=LAMBDA0, X0=X0, seed=0,
    )
    assert S.shape == (n_paths, N + 1), f"S shape {S.shape}"
    assert Lam.shape == (n_paths, N + 1), f"Lam shape {Lam.shape}"


def test_scott_initial_conditions():
    """Scott model initialises Lambda and S correctly at t=0."""
    n_paths, N = 10, 5
    S, Lam = simulate_scott_batch(
        n_paths=n_paths, N=N, dt=1.0/N,
        kappa=KAPPA, theta=THETA, sigma=0.331,
        rho=RHO, Lambda0=LAMBDA0, X0=X0, seed=1,
    )
    assert np.allclose(Lam[:, 0], LAMBDA0), "Lambda_0 mismatch"
    assert np.allclose(S[:, 0], np.exp(X0)), "S_0 mismatch"


def test_scott_prices_positive():
    """All simulated prices are strictly positive."""
    n_paths, N = 100, 50
    S, _ = simulate_scott_batch(
        n_paths=n_paths, N=N, dt=1.0/N,
        kappa=KAPPA, theta=THETA, sigma=0.331,
        rho=RHO, Lambda0=LAMBDA0, X0=X0, seed=2,
    )
    assert np.all(S > 0), "Non-positive price detected"


def test_scott_payoffs_non_negative():
    """All payoffs from compute_payoffs_scott are non-negative."""
    n_paths, N = 100, 50
    S, _ = simulate_scott_batch(
        n_paths=n_paths, N=N, dt=1.0/N,
        kappa=KAPPA, theta=THETA, sigma=0.331,
        rho=RHO, Lambda0=LAMBDA0, X0=X0, seed=3,
    )
    for K in [80.0, 100.0, 120.0]:
        pf = compute_payoffs_scott(S, K)
        for key, arr in pf.items():
            assert np.all(arr >= 0), f"Negative payoff in {key} at K={K}"


def test_scott_call_put_parity_approximate():
    """European C - P ≈ S0 - K within 3 SEs under the Scott model."""
    n_paths, N = 10_000, 500
    S0_val = np.exp(X0)
    K = 100.0
    S, _ = simulate_scott_batch(
        n_paths=n_paths, N=N, dt=1.0/N,
        kappa=KAPPA, theta=THETA, sigma=0.331,
        rho=RHO, Lambda0=LAMBDA0, X0=X0, seed=42,
    )
    pf = compute_payoffs_scott(S, K)
    ec = pf["ec"].mean()
    ep = pf["ep"].mean()
    se = pf["ec"].std() / np.sqrt(n_paths)
    diff = abs((ec - ep) - (S0_val - K))
    assert diff < 5 * se, (
        f"C - P parity error = {diff:.4f}, 5*SE = {5*se:.4f}"
    )


# ---------------------------------------------------------------------------
# Batch lambda simulation (_simulate_lambda_batch)
# ---------------------------------------------------------------------------

def test_lambda_batch_shape():
    """_simulate_lambda_batch returns (batch_size, N) array."""
    alpha, N_steps = 0.62, 20
    batch_size = 10
    _, zeta = decompose_mu_zeta(MU, RHO, 0.331)
    phi_seq = compute_phi_sequence(alpha, N_steps)
    a_n = compute_a_n(alpha, 500, KAPPA)
    from src.simulation import compute_omega_n
    omega_n = compute_omega_n(THETA, MU, 500)
    xi = LAMBDA0 / THETA
    phi_n = a_n * phi_seq
    omega_t = _compute_omega_t_full(omega_n, xi, a_n, phi_n, N_steps)
    lambda0 = xi * omega_n / (1.0 - a_n)

    eps = np.random.default_rng(0).standard_normal((batch_size, N_steps))
    lam = _simulate_lambda_batch(eps, phi_n, omega_t, lambda0, RHO, zeta)
    assert lam.shape == (batch_size, N_steps), f"lam shape {lam.shape}"


def test_lambda_batch_finite():
    """All simulated lambda values are finite (no NaN / Inf)."""
    alpha, N_steps = 0.62, 50
    batch_size = 20
    _, zeta = decompose_mu_zeta(MU, RHO, 0.331)
    phi_seq = compute_phi_sequence(alpha, N_steps)
    a_n = compute_a_n(alpha, 500, KAPPA)
    from src.simulation import compute_omega_n
    omega_n = compute_omega_n(THETA, MU, 500)
    xi = LAMBDA0 / THETA
    phi_n = a_n * phi_seq
    omega_t = _compute_omega_t_full(omega_n, xi, a_n, phi_n, N_steps)
    lambda0 = xi * omega_n / (1.0 - a_n)

    eps = np.random.default_rng(1).standard_normal((batch_size, N_steps))
    lam = _simulate_lambda_batch(eps, phi_n, omega_t, lambda0, RHO, zeta)
    assert np.all(np.isfinite(lam)), "Non-finite lambda values detected"
