"""Black-Scholes pricing utilities."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """European call price under Black-Scholes.

    Parameters
    ----------
    S:
        Current spot price.
    K:
        Strike price.
    T:
        Time to expiry (years). When T == 0 returns the intrinsic value.
    sigma:
        Annualised volatility.
    r:
        Continuously-compounded risk-free rate.

    Returns
    -------
    float
        Call price C = S·N(d1) − K·e^{−rT}·N(d2).
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """European put price under Black-Scholes via put-call parity.

    Parameters
    ----------
    S:
        Current spot price.
    K:
        Strike price.
    T:
        Time to expiry (years). When T == 0 returns the intrinsic value.
    sigma:
        Annualised volatility.
    r:
        Continuously-compounded risk-free rate.

    Returns
    -------
    float
        Put price P = K·e^{−rT}·N(−d2) − S·N(−d1).
    """
    if T <= 0.0:
        return max(K - S, 0.0)
    call = bs_call_price(S, K, T, sigma, r=r)
    # put-call parity: P = C - S + K * exp(-rT)
    return call - S + K * np.exp(-r * T)


def bs_implied_vol(
    C: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Recover Black-Scholes implied volatility from a call price.

    Uses Brent's method on the interval [1e-6, 10].

    Parameters
    ----------
    C:
        Observed call price.
    S:
        Current spot price.
    K:
        Strike price.
    T:
        Time to expiry (years).
    r:
        Continuously-compounded risk-free rate.
    tol:
        Absolute tolerance on the recovered volatility.
    max_iter:
        Maximum number of function evaluations.

    Returns
    -------
    float
        Implied volatility sigma such that bs_call_price(S, K, T, sigma, r) == C.

    Raises
    ------
    ValueError
        If the root cannot be bracketed (price out of no-arbitrage bounds).
    """
    if T <= 0.0:
        raise ValueError("T must be positive for implied vol computation.")
    intrinsic = max(S * np.exp(-0.0) - K * np.exp(-r * T), 0.0)
    if C < intrinsic - tol:
        raise ValueError("Call price is below intrinsic value.")

    def objective(sigma: float) -> float:
        return bs_call_price(S, K, T, sigma, r=r) - C

    lo, hi = 1e-6, 10.0
    try:
        return brentq(objective, lo, hi, xtol=tol, maxiter=max_iter)
    except ValueError as exc:
        raise ValueError(
            f"Could not bracket implied vol for C={C:.6f}, S={S}, K={K}, T={T}"
        ) from exc
