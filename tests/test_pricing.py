"""Tests for option pricing and Black-Scholes utilities."""
import numpy as np
import pytest

from src.bs_utils import bs_call_price, bs_implied_vol, bs_put_price
from src.pricing import compute_phi_kernel, price_options_batch


# ---------------------------------------------------------------------------
# Black-Scholes utilities
# ---------------------------------------------------------------------------

def test_bs_call_put_parity():
    """Put-call parity: C - P = S - K * exp(-rT). For S=K=100, r=0, parity = 0."""
    S, K, T, sigma, r = 100.0, 100.0, 1.0, 0.2, 0.0
    C = bs_call_price(S, K, T, sigma, r=r)
    P = bs_put_price(S, K, T, sigma, r=r)
    # S - K*exp(-rT) = 100 - 100 = 0
    assert abs(C - P) < 1e-8, f"C - P = {C - P:.2e}, expected 0 (put-call parity)"


def test_bs_call_lower_bound():
    """European call price satisfies C >= max(S - K * exp(-rT), 0)."""
    S, K, T, sigma, r = 100.0, 90.0, 1.0, 0.2, 0.0
    C = bs_call_price(S, K, T, sigma, r=r)
    lower = max(S - K * np.exp(-r * T), 0.0)
    assert C >= lower - 1e-12, f"C={C:.6f} violates lower bound {lower:.6f}"


def test_bs_call_at_expiry():
    """At expiry (T=0), the call price equals the intrinsic value max(S-K, 0)."""
    # In-the-money: intrinsic = 10
    price = bs_call_price(100.0, 90.0, 0.0, 0.2)
    assert abs(price - 10.0) < 1e-10, f"Call at expiry = {price}, expected 10.0"


def test_bs_implied_vol_round_trip():
    """Inverting bs_call_price recovers the original sigma to within 1e-6."""
    sigma_true = 0.25
    S, K, T, r = 100.0, 100.0, 1.0, 0.0
    C = bs_call_price(S, K, T, sigma_true, r=r)
    iv = bs_implied_vol(C, S, K, T, r=r)
    assert abs(iv - sigma_true) < 1e-6, (
        f"Recovered IV={iv:.8f}, expected {sigma_true}"
    )


def test_bs_implied_vol_otm():
    """Implied vol round-trip works for an OTM call (K=120 > S=100)."""
    sigma_true = 0.30
    S, K, T, r = 100.0, 120.0, 1.0, 0.0
    C = bs_call_price(S, K, T, sigma_true, r=r)
    iv = bs_implied_vol(C, S, K, T, r=r)
    assert abs(iv - sigma_true) < 1e-6, (
        f"OTM recovered IV={iv:.8f}, expected {sigma_true}"
    )


# ---------------------------------------------------------------------------
# Pricing kernel
# ---------------------------------------------------------------------------

def test_phi_kernel_sum():
    """For alpha=0.62 and L=10000 the kernel weights sum to approximately 1."""
    alpha = 0.62
    L = 10_000
    phi = compute_phi_kernel(alpha, L)
    total = np.sum(phi)
    assert abs(total - 1.0) < 0.01, (
        f"sum(phi_kernel)={total:.5f}, expected within 0.01 of 1.0"
    )


# ---------------------------------------------------------------------------
# Monte Carlo option pricing via price_options_batch
# ---------------------------------------------------------------------------

def test_price_options_european_call_payoff():
    """European call price equals mean(S_T - K) when every path has S_T > K.

    We construct synthetic paths with a constant terminal value of 150 so the
    payoff is deterministic: max(150 - 100, 0) = 50.
    """
    n_paths = 1000
    N = 10
    S0 = 100.0
    K = 100.0
    S_T_const = 150.0
    # paths: shape (n_paths, N+1); all terminal values fixed at S_T_const
    paths = np.full((n_paths, N + 1), S_T_const)
    paths[:, 0] = S0

    price = price_options_batch(paths, K=K, T=1.0, r=0.0, option_type="european_call")
    expected = S_T_const - K  # = 50
    assert abs(price - expected) / expected < 0.01, (
        f"European call price={price:.4f}, expected≈{expected:.4f}"
    )


def test_price_options_put_call_parity():
    """C - P ≈ S0 - K (within 3 standard errors) for European options with r=0."""
    rng = np.random.default_rng(0)
    n_paths = 5000
    N = 1
    S0 = 100.0
    K = 100.0
    sigma = 0.2
    T = 1.0

    # Risk-neutral log-normal terminal prices (r=0 → drift = -0.5*sigma^2*T)
    log_returns = rng.normal(-0.5 * sigma**2 * T, sigma * np.sqrt(T), n_paths)
    S_T = S0 * np.exp(log_returns)
    paths = np.column_stack([np.full(n_paths, S0), S_T])

    call = price_options_batch(paths, K=K, T=T, r=0.0, option_type="european_call")
    put = price_options_batch(paths, K=K, T=T, r=0.0, option_type="european_put")

    # C - P = mean(S_T) - K; SE = std(S_T) / sqrt(n_paths)
    se = np.std(S_T) / np.sqrt(n_paths)
    parity_value = np.mean(S_T) - K
    assert abs((call - put) - parity_value) < 3 * se, (
        f"Put-call parity deviation = {abs((call - put) - parity_value):.4f}, "
        f"3*SE = {3 * se:.4f}"
    )


def test_european_call_moneyness():
    """ITM call (K=80) > ATM call (K=100) > OTM call (K=120) for S0=100.

    Paths are drawn from a log-normal distribution via numpy, satisfying the
    'numpy not full MC' requirement (no rough-volatility path simulation).
    """
    rng = np.random.default_rng(0)
    n_paths = 10_000
    S0 = 100.0
    sigma = 0.2
    T = 1.0

    S_T = S0 * np.exp(rng.normal(-0.5 * sigma**2 * T, sigma * np.sqrt(T), n_paths))
    paths = np.column_stack([np.full(n_paths, S0), S_T])

    call_itm = price_options_batch(paths, K=80.0, T=T, r=0.0, option_type="european_call")
    call_atm = price_options_batch(paths, K=100.0, T=T, r=0.0, option_type="european_call")
    call_otm = price_options_batch(paths, K=120.0, T=T, r=0.0, option_type="european_call")

    assert call_itm > call_atm, (
        f"ITM call {call_itm:.4f} should exceed ATM call {call_atm:.4f}"
    )
    assert call_atm > call_otm, (
        f"ATM call {call_atm:.4f} should exceed OTM call {call_otm:.4f}"
    )


def test_barrier_down_out_put_edge():
    """Down-and-out put has near-zero value when every path crosses the barrier.

    For K=80 and barrier B_d=90, a down-and-out put is knocked out whenever
    the price falls below 90. If all simulated paths drop below B_d immediately
    after inception and stay there, every option is knocked out and the price ≈ 0.
    """
    n_paths = 100
    N = 10
    S0 = 100.0
    K = 80.0
    B_d = 90.0  # down-and-out barrier

    # All paths drop below B_d immediately; every option is knocked out.
    paths = np.full((n_paths, N + 1), 85.0)  # all below barrier throughout
    paths[:, 0] = S0                          # start above barrier

    price = price_options_batch(
        paths, K=K, T=1.0, r=0.0,
        option_type="barrier_down_out_put", barrier=B_d,
    )
    assert price < 0.01, (
        f"Down-out put = {price:.6f}, expected ≈ 0 (all paths crossed barrier)"
    )


def test_asian_call_cheaper_than_european():
    """Asian (average-price) call is at most as expensive as a European call.

    By Jensen's inequality applied to the concavity of the averaging operator,
    E[max(avg(S) − K, 0)] ≤ E[max(S_T − K, 0)] for the same underlying paths.
    """
    rng = np.random.default_rng(42)
    n_paths = 2000
    N = 50
    S0 = 100.0
    K = 100.0
    T = 1.0
    sigma = 0.2

    # Simulate log-normal paths (simple, not rough MC)
    dt = T / N
    increments = rng.normal((- 0.5 * sigma**2) * dt, sigma * np.sqrt(dt),
                             (n_paths, N))
    log_paths = np.hstack([
        np.zeros((n_paths, 1)),
        np.cumsum(increments, axis=1),
    ])
    paths = S0 * np.exp(log_paths)

    asian_price = price_options_batch(paths, K=K, T=T, r=0.0, option_type="asian_call")
    european_price = price_options_batch(paths, K=K, T=T, r=0.0, option_type="european_call")

    assert asian_price <= european_price + 1e-10, (
        f"Asian call {asian_price:.4f} exceeds European call {european_price:.4f}"
    )
