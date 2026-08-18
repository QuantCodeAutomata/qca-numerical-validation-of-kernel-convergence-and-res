"""Monte Carlo option pricing utilities for the rough-volatility model."""

import numpy as np
from scipy.special import gamma as sc_gamma
from typing import Optional

from src.kernel import compute_phi_sequence


def compute_phi_kernel(alpha: float, L: int) -> np.ndarray:
    """Return the unscaled lag-weight sequence phi_1, …, phi_L with tail correction.

    The weights are defined by Algorithm 1 of the paper::

        phi_1 = 1 - 1 / (Γ(1-alpha) · 2^alpha)
        phi_i = (1 / Γ(1-alpha)) · (i^{-alpha} - (i+1)^{-alpha})   for i ≥ 2

    The exact truncated tail ``sum_{i=L+1}^∞ phi_i = (L+1)^{-alpha} / Γ(1-alpha)``
    is absorbed into the last element so that ``sum(phi_kernel) == 1.0`` for
    any finite L.

    Parameters
    ----------
    alpha:
        Self-similarity exponent in (0.5, 1).
    L:
        Number of weights to compute.

    Returns
    -------
    np.ndarray
        Non-negative array of length L whose sum equals 1.0.
    """
    phi = compute_phi_sequence(alpha, L)
    # Exact tail: sum_{i=L+1}^inf phi_i = (L+1)^{-alpha} / Gamma(1-alpha)
    tail = (L + 1) ** (-alpha) / sc_gamma(1.0 - alpha)
    phi = phi.copy()
    phi[-1] += tail
    return phi


def price_options_batch(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = "european_call",
    barrier: Optional[float] = None,
) -> float:
    """Price a European or exotic option via Monte Carlo averaging.

    Parameters
    ----------
    paths:
        Simulated price paths of shape ``(n_paths, N+1)``, where ``paths[:, 0]``
        is the initial price and ``paths[:, -1]`` is the terminal price.
    K:
        Strike price.
    T:
        Time to expiry (years). Used solely for discounting.
    r:
        Continuously-compounded risk-free rate.
    option_type:
        One of:

        * ``"european_call"`` – max(S_T − K, 0)
        * ``"european_put"`` – max(K − S_T, 0)
        * ``"barrier_down_out_put"`` – max(K − S_T, 0) if S_t > barrier for
          all t, else 0 (down-and-out put). Requires ``barrier`` argument.
        * ``"asian_call"`` – max(mean(S_t, t=0…T) − K, 0)

    barrier:
        Knock-out barrier level. Required when ``option_type`` is
        ``"barrier_down_out_put"``.

    Returns
    -------
    float
        Discounted Monte Carlo price.

    Raises
    ------
    ValueError
        For unknown option types or a missing barrier.
    """
    paths = np.asarray(paths, dtype=float)
    S_T = paths[:, -1]
    discount = np.exp(-r * T)

    if option_type == "european_call":
        payoffs = np.maximum(S_T - K, 0.0)

    elif option_type == "european_put":
        payoffs = np.maximum(K - S_T, 0.0)

    elif option_type == "barrier_down_out_put":
        if barrier is None:
            raise ValueError(
                "barrier must be specified for barrier_down_out_put"
            )
        # Knock out if any path value (excluding t=0) falls at or below the barrier
        not_knocked = np.all(paths[:, 1:] > barrier, axis=1)
        payoffs = np.where(not_knocked, np.maximum(K - S_T, 0.0), 0.0)

    elif option_type == "asian_call":
        # Arithmetic average over all time steps including t=0
        avg_S = np.mean(paths, axis=1)
        payoffs = np.maximum(avg_S - K, 0.0)

    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")

    return float(discount * np.mean(payoffs))
