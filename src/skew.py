"""ATM implied-volatility skew estimation and power-law regression.

Used by exp_5 to study whether the discrete rough-volatility approximation
generates an ATM skew with approximate power-law decay in maturity.

Public API
----------
estimate_atm_skew(smile_k, smile_iv, h=0.01)   -> float
powerlaw_regression(tau, psi)                   -> dict
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import linregress
from scipy.stats import norm as sp_norm
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Black-Scholes helpers (self-contained, no circular import with bs_utils)
# ---------------------------------------------------------------------------


def _bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes call price (r=0 convention matches exp_5)."""
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return float(S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2))


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    lo: float = 1e-6,
    hi: float = 10.0,
    tol: float = 1e-10,
) -> Optional[float]:
    """Invert a call price to Black-Scholes implied volatility.

    Returns None if the price lies outside the no-arbitrage band or if
    the root finder fails (instead of raising, to allow graceful skips).

    Parameters
    ----------
    price:
        Observed (or simulated) call price.
    S, K, T, r:
        Spot, strike, maturity, rate.
    lo, hi:
        Initial bracket for Brent's method.
    tol:
        Tolerance on the recovered sigma.

    Returns
    -------
    float or None
        Implied volatility, or None on failure.
    """
    intrinsic = max(S * np.exp(0.0) - K * np.exp(-r * T), 0.0)
    upper = S  # call price <= S always
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return None

    def obj(sigma: float) -> float:
        return _bs_call(S, K, T, sigma, r) - price

    try:
        return float(brentq(obj, lo, hi, xtol=tol, maxiter=500))
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# ATM skew estimators
# ---------------------------------------------------------------------------


def estimate_atm_skew_fd(
    iv_minus_h: float,
    iv_plus_h: float,
    h: float,
) -> float:
    """Central finite-difference estimate of |d sigma_IV / dk| at k = 0.

    psi ≈ |sigma_IV(h) - sigma_IV(-h)| / (2h)

    Here k = log(K/S) so positive k is OTM call, negative k is ITM call.
    Under negative rho (leverage effect) sigma_IV should increase as k
    decreases, so the skew is measured as the absolute value of the slope.

    Parameters
    ----------
    iv_minus_h:
        Implied vol at log-moneyness k = -h.
    iv_plus_h:
        Implied vol at log-moneyness k = +h.
    h:
        Finite-difference step in log-moneyness.

    Returns
    -------
    float
        ATM skew estimate psi.
    """
    return abs(iv_plus_h - iv_minus_h) / (2.0 * h)


def estimate_atm_skew_regression(
    ks: np.ndarray,
    ivs: np.ndarray,
) -> float:
    """Local linear regression slope at k = 0 as ATM skew estimate.

    Fits sigma_IV(k) ≈ a + b*k over the supplied (k, iv) pairs and
    returns |b| as the unsigned ATM slope.

    Parameters
    ----------
    ks:
        Log-moneyness values (1-D array).
    ivs:
        Corresponding implied volatilities (1-D array, same length).

    Returns
    -------
    float
        |slope| of the fitted line.
    """
    result = linregress(ks, ivs)
    return abs(float(result.slope))


# ---------------------------------------------------------------------------
# Log-log regression for power-law analysis
# ---------------------------------------------------------------------------


def powerlaw_regression(
    tau: Sequence[float],
    psi: Sequence[float],
) -> dict:
    """Fit log(psi) = a + b * log(tau) by OLS.

    Parameters
    ----------
    tau:
        Maturity values (positive).
    psi:
        Corresponding ATM skew estimates (positive).

    Returns
    -------
    dict with keys:
        slope (b), intercept (a), r_squared, p_value, stderr_slope,
        log_tau, log_psi (arrays used in the fit).

    Raises
    ------
    ValueError
        If tau or psi contain non-positive values.
    """
    tau_arr = np.asarray(tau, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)

    if np.any(tau_arr <= 0) or np.any(psi_arr <= 0):
        raise ValueError("All tau and psi values must be strictly positive.")

    log_tau = np.log(tau_arr)
    log_psi = np.log(psi_arr)

    result = linregress(log_tau, log_psi)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue**2),
        "p_value": float(result.pvalue),
        "stderr_slope": float(result.stderr),
        "log_tau": log_tau,
        "log_psi": log_psi,
    }
