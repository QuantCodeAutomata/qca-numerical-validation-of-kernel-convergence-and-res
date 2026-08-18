"""Scott lognormal stochastic-volatility model (alpha = 1 Markov benchmark).

Continuous-time model (Euler–Maruyama discretisation):

    dX_t = -(1/2) exp(2*Lambda_t) dt + exp(Lambda_t) dB_t
    dLambda_t = kappa*(theta - Lambda_t) dt + sigma * dW_t

with Cov(dB_t, dW_t) = rho * dt.

Parameters are exactly those reported in the paper:
    kappa = 0.1,  theta = log(sqrt(0.3156)),  sigma = 0.331,
    rho = -0.681, Lambda_0 = log(sqrt(0.0392)),  S_0 = 100.

The effective vol-of-vol sigma ≡ sqrt(rho^2 + zeta^2) * mu = 0.331 is
directly identified from the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _correlated_normals(
    rng: np.random.Generator,
    rho: float,
    n_paths: int,
    N: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two correlated standard-normal arrays.

    Z_B = rho * Z_W + sqrt(1 - rho^2) * Z_perp

    Parameters
    ----------
    rng:
        NumPy random Generator.
    rho:
        Target correlation.
    n_paths:
        Number of Monte Carlo paths.
    N:
        Number of time steps.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Z_W, Z_B), each of shape (n_paths, N).
    """
    Z_W = rng.standard_normal((n_paths, N))
    Z_perp = rng.standard_normal((n_paths, N))
    Z_B = rho * Z_W + np.sqrt(max(1.0 - rho**2, 0.0)) * Z_perp
    return Z_W, Z_B


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------


def simulate_scott_batch(
    n_paths: int,
    N: int,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    Lambda0: float,
    X0: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the Scott model for a batch of paths via Euler–Maruyama.

    The discretisation uses left-point evaluation of Lambda within each Euler
    step and is documented as the primary scheme.  For a single-step update:

        Lambda_{t+1} = Lambda_t + kappa*(theta - Lambda_t)*dt
                       + sigma*sqrt(dt)*Z_t^W
        X_{t+1}      = X_t - (1/2)*exp(2*Lambda_t)*dt
                       + exp(Lambda_t)*sqrt(dt)*Z_t^B

    Parameters
    ----------
    n_paths:
        Number of Monte Carlo paths.
    N:
        Number of time steps (total steps = N, so final time = N*dt).
    dt:
        Time-step size (e.g. 1/500 for daily with n=500 steps/year).
    kappa:
        Mean-reversion speed.
    theta:
        Long-run log-volatility level (log scale).
    sigma:
        Vol-of-vol: sqrt(rho^2 + zeta^2) * mu = 0.331.
    rho:
        Leverage correlation.
    Lambda0:
        Initial log-volatility.
    X0:
        Initial log-price.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (S, Lambda) where:
        - S has shape (n_paths, N+1), S[:, 0] = exp(X0).
        - Lambda has shape (n_paths, N+1), Lambda[:, 0] = Lambda0.
    """
    rng = np.random.default_rng(seed)
    Z_W, Z_B = _correlated_normals(rng, rho, n_paths, N)

    sqrt_dt = np.sqrt(dt)
    # Allocate arrays including t=0
    Lam = np.empty((n_paths, N + 1), dtype=np.float64)
    X = np.empty((n_paths, N + 1), dtype=np.float64)
    Lam[:, 0] = Lambda0
    X[:, 0] = X0

    for t in range(N):
        l_t = Lam[:, t]
        sig_t = np.exp(l_t)
        Lam[:, t + 1] = (
            l_t + kappa * (theta - l_t) * dt + sigma * sqrt_dt * Z_W[:, t]
        )
        X[:, t + 1] = (
            X[:, t] - 0.5 * sig_t**2 * dt + sig_t * sqrt_dt * Z_B[:, t]
        )

    return np.exp(X), Lam


# ---------------------------------------------------------------------------
# Payoff computation (reused by exp4)
# ---------------------------------------------------------------------------


def compute_payoffs_scott(
    S: np.ndarray,
    K: float,
    barrier_up: float = 110.0,
    barrier_down: float = 90.0,
) -> dict:
    """Compute all required option payoffs from a price-path matrix.

    Monitoring is discrete over all time points t = 0, ..., N.

    Parameters
    ----------
    S:
        Price paths of shape (n_paths, N+1).  S[:, 0] = S_0 = 100.
    K:
        Strike.
    barrier_up:
        Up-In call barrier.
    barrier_down:
        Down-Out put barrier.

    Returns
    -------
    dict
        Keys: 'ec', 'ep', 'ac', 'ap', 'lc', 'lp', 'uic', 'dop'.
        Values: np.ndarray of shape (n_paths,) of undiscounted payoffs.
    """
    S_T = S[:, -1]
    # Arithmetic average over t = 1 ... N (exclude t=0)
    A = S[:, 1:].mean(axis=1)
    S_max = S.max(axis=1)
    S_min = S.min(axis=1)

    ec = np.maximum(S_T - K, 0.0)
    ep = np.maximum(K - S_T, 0.0)
    ac = np.maximum(A - K, 0.0)
    ap = np.maximum(K - A, 0.0)
    lc = np.maximum(S_max - K, 0.0)
    lp = np.maximum(K - S_min, 0.0)

    # Up-In call: pays if max S >= B_u
    touched_up = S_max >= barrier_up
    uic = np.where(touched_up, ec, 0.0)

    # Down-Out put: knocked out when min S <= B_d
    survived_down = S_min > barrier_down
    dop = np.where(survived_down, ep, 0.0)

    return {"ec": ec, "ep": ep, "ac": ac, "ap": ap,
            "lc": lc, "lp": lp, "uic": uic, "dop": dop}
