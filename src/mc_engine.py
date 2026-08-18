"""Batched Monte Carlo engine for the discrete rough-volatility approximation.

Core routine: :func:`simulate_paths_batch` runs the quasi-score AR(∞)
recursion for a mini-batch of paths and returns undiscounted payoffs for
all option types and strikes in one pass.

Design decisions (documented per methodology)
---------------------------------------------
* ``mu = 0.4``:  chosen so that zeta² = (0.331/0.4)² − 0.681² > 0.
  Verified by :func:`src.simulation.decompose_mu_zeta`.
* ``lambda0^{(n)} = xi * omega_n / (1 − a_n)``: consistency initialisation.
* Discrete arithmetic average over t = 1, …, N (exclude t = 0).
* Lookback extrema include t = 0.
* Up-In call triggered when max_t S_t >= B_u.
* Down-Out put knocked when min_t S_t <= B_d.
* r = 0 throughout (no discounting).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.kernel import compute_a_n, compute_phi_sequence
from src.simulation import (
    decompose_mu_zeta,
    compute_omega_n,
    _compute_omega_t_full,
)


# ---------------------------------------------------------------------------
# Shared parameter set
# ---------------------------------------------------------------------------

MU: float = 0.4          # see docstring for feasibility proof
PRODUCT: float = 0.331
RHO: float = -0.681
KAPPA: float = 0.1
THETA: float = float(np.log(np.sqrt(0.3156)))
LAMBDA0: float = float(np.log(np.sqrt(0.0392)))
X0: float = float(np.log(100.0))
S0: float = 100.0
BARRIER_UP: float = 110.0
BARRIER_DOWN: float = 90.0
SQRT2_PI: float = float(np.sqrt(2.0 / np.pi))


def get_zeta(mu: float = MU) -> float:
    """Return zeta from the (mu, zeta) decomposition at the default rho."""
    _, zeta = decompose_mu_zeta(mu, RHO, PRODUCT)
    return zeta


# ---------------------------------------------------------------------------
# Batch simulation of the AR(∞) quasi-score recursion (vectorised over paths)
# ---------------------------------------------------------------------------


def _simulate_lambda_batch(
    batch_eps: np.ndarray,
    phi_n: np.ndarray,
    omega_t: np.ndarray,
    lambda0: float,
    rho: float,
    zeta: float,
) -> np.ndarray:
    """Simulate lambda for a batch of paths (vectorised across paths).

    Parameters
    ----------
    batch_eps:
        iid N(0,1) innovations, shape (batch_size, N).
    phi_n:
        Scaled kernel weights of length N.
    omega_t:
        Time-varying intercept array of length N.
    lambda0:
        Initial latent state lambda_0^{(n)}.
    rho, zeta:
        Leverage and asymmetry parameters.

    Returns
    -------
    np.ndarray
        Shape (batch_size, N): lambda_1, …, lambda_N.
    """
    batch_size, N = batch_eps.shape
    n_phi = len(phi_n)

    lam = np.empty((batch_size, N + 1), dtype=np.float64)
    eps_ext = np.zeros((batch_size, N + 1), dtype=np.float64)
    eps_ext[:, 1:] = batch_eps
    lam[:, 0] = lambda0

    # h[i, t] = lam[i, t] + rho*eps_ext[i, t] + zeta*(|eps_ext[i,t]| - sqrt(2/pi))
    h_ext = np.empty_like(lam)
    h_ext[:, 0] = lambda0  # eps_ext[:, 0] == 0

    for t in range(1, N + 1):
        n_terms = min(t, n_phi)
        # h_ext[:, t-n_terms:t] reversed along axis-1 gives h_{t-1},...,h_{t-n_terms}
        h_win = h_ext[:, t - n_terms: t][:, ::-1]  # (batch_size, n_terms)
        w = phi_n[:n_terms]                          # (n_terms,)
        lam[:, t] = omega_t[t - 1] + (h_win * w).sum(axis=1)
        e = eps_ext[:, t]
        h_ext[:, t] = lam[:, t] + rho * e + zeta * (np.abs(e) - SQRT2_PI)

    return lam[:, 1:]   # (batch_size, N)


# ---------------------------------------------------------------------------
# Full pricing engine
# ---------------------------------------------------------------------------


def run_mc_pricing(
    alpha: float,
    n: int,
    N: int,
    M: int,
    strikes: list,
    barrier_up: float = BARRIER_UP,
    barrier_down: float = BARRIER_DOWN,
    batch_size: int = 5_000,
    mu: float = MU,
    seed: Optional[int] = 42,
) -> dict:
    """Price all option types for a grid of strikes using batch Monte Carlo.

    Parameters
    ----------
    alpha:
        Roughness parameter (e.g. 0.999, 0.62, 0.501).
    n:
        Discretisation steps per year.
    N:
        Total simulation steps (= n * T for T years).
    M:
        Total number of Monte Carlo paths.
    strikes:
        List of strike prices to price.
    barrier_up:
        Up-In call barrier level.
    barrier_down:
        Down-Out put barrier level.
    batch_size:
        Number of paths per mini-batch (memory vs. throughput trade-off).
    mu:
        Vol-of-vol decomposition parameter; must satisfy mu < 0.331/|rho|.
    seed:
        Global RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys are strike values.  Each value is a sub-dict with keys
        'ec','ep','ac','ap','lc','lp','uic','dop' mapping to
        (price, se) tuples.
    """
    _, zeta = decompose_mu_zeta(mu, RHO, PRODUCT)
    phi_seq = compute_phi_sequence(alpha, N)
    a_n = compute_a_n(alpha, n, KAPPA)
    omega_n = compute_omega_n(THETA, mu, n)
    xi = LAMBDA0 / THETA
    one_minus_a = 1.0 - a_n
    lambda0 = xi * omega_n / one_minus_a

    phi_n = a_n * phi_seq                          # scaled kernel
    scale = one_minus_a * THETA / omega_n          # rescaling factor
    omega_t = _compute_omega_t_full(omega_n, xi, a_n, phi_n, N)

    inv_2n = 1.0 / (2.0 * n)
    inv_sqrt_n = 1.0 / np.sqrt(n)

    # Accumulate payoffs per strike per option type
    option_keys = ["ec", "ep", "ac", "ap", "lc", "lp", "uic", "dop"]
    accum = {K: {k: [] for k in option_keys} for K in strikes}

    rng = np.random.default_rng(seed)
    n_batches = (M + batch_size - 1) // batch_size

    for b in range(n_batches):
        actual = min(batch_size, M - b * batch_size)
        eps = rng.standard_normal((actual, N))          # (actual, N)

        lam = _simulate_lambda_batch(
            eps, phi_n, omega_t, lambda0, RHO, zeta
        )  # (actual, N)

        Lambda = scale * lam                            # log-vol path

        # Vectorised log-price path:  X_t = X0 + sum_{k<t} dX_k
        sig_all = np.exp(Lambda)                        # (actual, N)
        dX = -inv_2n * sig_all**2 + inv_sqrt_n * sig_all * eps   # (actual, N)
        cum = np.cumsum(dX, axis=1)                     # (actual, N)
        X = np.empty((actual, N + 1), dtype=np.float64)
        X[:, 0] = X0
        X[:, 1:] = X0 + cum

        S = np.exp(X)                                   # (actual, N+1)

        S_T = S[:, -1]
        A = S[:, 1:].mean(axis=1)                      # arithmetic avg t=1..N
        S_max = S.max(axis=1)
        S_min = S.min(axis=1)
        touched_up = S_max >= barrier_up
        survived_down = S_min > barrier_down

        for K in strikes:
            ec = np.maximum(S_T - K, 0.0)
            ep = np.maximum(K - S_T, 0.0)
            ac = np.maximum(A - K, 0.0)
            ap = np.maximum(K - A, 0.0)
            lc = np.maximum(S_max - K, 0.0)
            lp = np.maximum(K - S_min, 0.0)
            uic = np.where(touched_up, ec, 0.0)
            dop = np.where(survived_down, ep, 0.0)

            for key, arr in zip(option_keys, [ec, ep, ac, ap, lc, lp, uic, dop]):
                accum[K][key].append(arr)

    # Aggregate across batches
    results = {}
    for K in strikes:
        results[K] = {}
        for key in option_keys:
            all_payoffs = np.concatenate(accum[K][key])
            price = float(all_payoffs.mean())
            se = float(all_payoffs.std() / np.sqrt(len(all_payoffs)))
            results[K][key] = (price, se)

    return results
