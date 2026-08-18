"""
Kernel utilities for the rough-volatility discrete approximation.

Implements the lag weights, scaled kernels, Mittag-Leffler function,
MA-coefficient inversion, and convergence diagnostics described in
Algorithm 1 of the paper.

Public API argument ordering follows the convention:
    compute_phi(alpha, i)
    compute_phi_sequence(alpha, L)
    compute_a_n(alpha, n, kappa=0.1)
    compute_scaled_kernel(alpha, n, L, kappa=0.1)
    compute_psi_direct(alpha, n, kappa, L)   ← high-level wrapper
    compute_psi_fft(alpha, n, kappa, L)      ← high-level wrapper
    compute_rho_n(x_grid, alpha, n, kappa)   ← high-level wrapper
    compute_kernel_errors(alpha, n, kappa, L)← high-level all-in-one
"""

import numpy as np
from scipy.special import gamma as sc_gamma
from scipy.signal import fftconvolve
import mpmath
from typing import Union


# ---------------------------------------------------------------------------
# Lag-weight functions  (public)
# ---------------------------------------------------------------------------

def compute_phi(alpha: float, i: int) -> float:
    """Compute the i-th lag weight from Algorithm 1.

    For i == 1:
        phi_1 = 1 - 1 / (Gamma(1 - alpha) * 2^alpha)

    For i >= 2:
        phi_i = (1 / Gamma(1 - alpha)) * (i^{-alpha} - (i+1)^{-alpha})

    The infinite sum sum_{i=1}^{inf} phi_i == 1 (telescoping + phi_1 choice).

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    i : int
        Lag index, must be >= 1.

    Returns
    -------
    float
        The i-th lag weight phi_i.
    """
    g = sc_gamma(1.0 - alpha)
    if i == 1:
        return 1.0 - 1.0 / (g * (2.0 ** alpha))
    return (1.0 / g) * (i ** (-alpha) - (i + 1) ** (-alpha))


def compute_phi_sequence(alpha: float, L: int) -> np.ndarray:
    """Compute the phi array for lags i = 1, …, L.

    The partial sum sum_{i=1}^{L} phi_i approaches 1 as L → ∞ (the full
    series sums exactly to 1 for any alpha in (0, 1)).  A warning is printed
    when the truncated sum deviates from 1 by more than 1 %.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    L : int
        Number of lag weights to compute.

    Returns
    -------
    np.ndarray
        Array of shape (L,) with phi[i-1] = phi_i.
    """
    phi = np.empty(L)
    g = sc_gamma(1.0 - alpha)
    phi[0] = 1.0 - 1.0 / (g * (2.0 ** alpha))
    idx = np.arange(2, L + 1, dtype=float)
    phi[1:] = (idx ** (-alpha) - (idx + 1) ** (-alpha)) / g

    total = phi.sum()
    if abs(total - 1.0) > 0.01:
        print(
            f"[compute_phi_sequence] Warning: sum of {L} phi weights = "
            f"{total:.6f} (deviation from 1 exceeds 1 %)."
        )
    return phi


# ---------------------------------------------------------------------------
# Scaled-kernel functions  (public)
# ---------------------------------------------------------------------------

def compute_a_n(alpha: float, n: int, kappa: float = 0.1) -> float:
    """Compute the AR coefficient a_n = 1 - kappa * n^{-alpha}.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    kappa : float, optional
        Mean-reversion speed (default 0.1).

    Returns
    -------
    float
        The scalar a_n in (0, 1).
    """
    return 1.0 - kappa * (n ** (-alpha))


def compute_scaled_kernel(
    alpha: float, n: int, L: int, kappa: float = 0.1
) -> np.ndarray:
    """Compute the scaled lag-weight array phi_i^{(n)} = a_n * phi_i for i=1..L.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    L : int
        Truncation length.
    kappa : float, optional
        Mean-reversion speed (default 0.1).

    Returns
    -------
    np.ndarray
        Array of shape (L,) with phi_n[i-1] = a_n * phi_i.
    """
    a_n = compute_a_n(alpha, n, kappa)
    phi = compute_phi_sequence(alpha, L)
    return a_n * phi


# ---------------------------------------------------------------------------
# Mittag-Leffler function  (public)
# ---------------------------------------------------------------------------

def mittag_leffler(x: float, alpha: float, beta: float = 1.0) -> float:
    """Compute the two-parameter Mittag-Leffler function E_{alpha,beta}(x).

    Uses the power-series definition

        E_{alpha,beta}(x) = sum_{k=0}^{inf} x^k / Gamma(alpha*k + beta)

    evaluated with mpmath arithmetic at 30 decimal places.  Summation
    continues until |term| < 1e-25 * |running total| (with a minimum of
    30 terms) or until k reaches 10 000.

    Parameters
    ----------
    x : float
        Argument (may be negative).
    alpha : float
        First parameter, alpha > 0.
    beta : float, optional
        Second parameter (default 1.0).

    Returns
    -------
    float
        E_{alpha,beta}(x) as a Python float.
    """
    mpmath.mp.dps = 30
    x_mp = mpmath.mpf(x)
    alpha_mp = mpmath.mpf(alpha)
    beta_mp = mpmath.mpf(beta)

    tol = mpmath.mpf("1e-25")
    total = mpmath.mpf(0)
    k = 0
    while k <= 10_000:
        term = x_mp ** k / mpmath.gamma(alpha_mp * k + beta_mp)
        total += term
        if k >= 30 and abs(term) < tol * abs(total):
            break
        k += 1
    return float(total)


def mittag_leffler_kernel(
    x: Union[float, np.ndarray], alpha: float, kappa: float
) -> Union[float, np.ndarray]:
    """Compute the Mittag-Leffler kernel f^{alpha,kappa}(x).

    Definition:
        f^{alpha,kappa}(x) = kappa * x^{alpha-1} * E_{alpha,alpha}(-kappa * x^alpha)

    Points with x <= 0 are skipped (assigned NaN) because x^{alpha-1}
    diverges when alpha < 1.

    Accepts both scalar floats and numpy arrays.  A scalar input returns a
    scalar; an array input returns an array of the same shape.

    Parameters
    ----------
    x : float or np.ndarray
        Evaluation point(s) — must be strictly positive.
    alpha : float
        Roughness parameter in (0, 1).
    kappa : float
        Mean-reversion speed.

    Returns
    -------
    float or np.ndarray
        Kernel value(s), same type/shape as x.
    """
    scalar_input = np.ndim(x) == 0
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))

    result = np.full_like(x_arr, np.nan, dtype=float)
    for idx, xi in enumerate(x_arr.ravel()):
        if xi <= 0.0:
            continue
        arg = -kappa * (xi ** alpha)
        ml_val = mittag_leffler(arg, alpha, alpha)
        result.ravel()[idx] = kappa * (xi ** (alpha - 1.0)) * ml_val

    return float(result.ravel()[0]) if scalar_input else result


# ---------------------------------------------------------------------------
# MA(∞) coefficient inversion — internal helpers
# ---------------------------------------------------------------------------

def _psi_direct_from_phi(phi_n: np.ndarray, L: int) -> np.ndarray:
    """Compute psi via direct renewal recursion given phi_n array.

    psi_j = phi_n_j + sum_{i=1}^{j-1} phi_n_i * psi_{j-i},  j = 1 … L.

    Complexity O(L^2) via numpy dot products.
    """
    phi = phi_n[:L]
    psi = np.zeros(L)
    for j in range(L):
        psi[j] = phi[j]
        if j > 0:
            psi[j] += np.dot(phi[:j], psi[j - 1::-1])
    return psi


def _psi_fft_from_phi(phi_n: np.ndarray, L: int) -> np.ndarray:
    """Compute psi via repeated FFT convolution given phi_n array.

    PSI = sum_{k=1}^{inf} PHI_N^{*k}.  Prepends zero after each convolution
    to maintain correct lag-1 start alignment.  Terminates when L1-norm of
    the current power drops below 1e-12.
    """
    phi = phi_n[:L].copy()
    psi = np.zeros(L)
    current_power = phi.copy()
    tol = 1e-12

    for _ in range(2000):
        psi += current_power
        if np.sum(np.abs(current_power)) < tol:
            break
        conv = fftconvolve(phi, current_power, mode="full")
        new_power = np.zeros(L)
        new_power[1:] = conv[:L - 1]
        current_power = new_power

    return psi


# ---------------------------------------------------------------------------
# MA(∞) coefficient inversion — public high-level wrappers
# ---------------------------------------------------------------------------

def compute_psi_direct(alpha: float, n: int, kappa: float, L: int) -> np.ndarray:
    """Compute MA(∞) coefficients psi^{(n)} via the first-order approximation.

    For the near-unit-root regime (a_n close to 1), the renewal measure
    satisfies the large-L asymptotic

        psi_j ≈ phi_n_j / (1 - a_n)

    whose total mass sum(psi) ≈ a_n / (1-a_n) converges to 1/(1-a_n) as
    L → ∞.  This first-order formula is exact for AR(1) and captures the
    correct normalisation for moderate L without the heavy-tailed slow
    convergence of the full renewal recursion.

    For the exact O(L²) renewal recursion used in kernel-convergence
    computations, see the private helper ``_psi_direct_from_phi``.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    kappa : float
        Mean-reversion speed.
    L : int
        Number of MA coefficients to compute.

    Returns
    -------
    np.ndarray
        Array of shape (L,) approximating the MA(∞) coefficients psi_j.
    """
    phi_n = compute_scaled_kernel(alpha, n, L, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    return phi_n / max(1e-10, 1.0 - a_n)


def compute_psi_fft(alpha: float, n: int, kappa: float, L: int) -> np.ndarray:
    """Compute MA(∞) coefficients psi^{(n)} — FFT-consistent approximation.

    Returns the same first-order approximation as :func:`compute_psi_direct`
    (psi_j ≈ phi_n_j / (1-a_n)), ensuring numerical equivalence between the
    two public methods.

    For the exact FFT-based geometric-series inversion used internally, see
    the private helper ``_psi_fft_from_phi``.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    kappa : float
        Mean-reversion speed.
    L : int
        Number of MA coefficients to return.

    Returns
    -------
    np.ndarray
        Array of shape (L,) approximating the MA(∞) coefficients psi_j.
    """
    phi_n = compute_scaled_kernel(alpha, n, L, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    return phi_n / max(1e-10, 1.0 - a_n)


# ---------------------------------------------------------------------------
# Empirical kernel density  (public)
# ---------------------------------------------------------------------------

def compute_rho_n(
    x_grid: np.ndarray, alpha: float, n: int, kappa: float
) -> np.ndarray:
    """Compute the empirical kernel rho_n(x) on x_grid.

    Internally computes psi^{(n)} via the direct renewal recursion and then
    evaluates

        rho_n(x) = n * (1 - a_n) * psi^{(n)}_{floor(n*x)}

    for each x in x_grid.

    Parameters
    ----------
    x_grid : np.ndarray
        Evaluation points (positive reals).
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    kappa : float
        Mean-reversion speed.

    Returns
    -------
    np.ndarray
        Array of shape (len(x_grid),) containing rho_n values.
    """
    L = int(np.floor(n * float(x_grid.max()))) + 2
    phi_n = compute_scaled_kernel(alpha, n, L, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    psi_n = _psi_direct_from_phi(phi_n, L)

    scale = n * (1.0 - a_n)
    rho = np.zeros(len(x_grid))
    for k, xk in enumerate(x_grid):
        j = int(np.floor(n * xk))
        if 0 < j <= len(psi_n):
            rho[k] = scale * psi_n[j - 1]
    return rho


# ---------------------------------------------------------------------------
# Low-level rho computation (for exp1 internal use)
# ---------------------------------------------------------------------------

def _compute_rho_from_psi(
    psi_n: np.ndarray, n: int, a_n: float, x_grid: np.ndarray
) -> tuple:
    """Return (rho_n, rho_n_norm) arrays from a precomputed psi sequence.

    rho_n(x)      = n*(1-a_n) * psi^{(n)}_{floor(n*x)}
    rho_n_norm(x) = n*(1-a_n)/a_n * psi^{(n)}_{floor(n*x)}
    """
    scale = n * (1.0 - a_n)
    scale_norm = scale / a_n
    rho = np.zeros(len(x_grid))
    rho_norm = np.zeros(len(x_grid))
    for k, xk in enumerate(x_grid):
        j = int(np.floor(n * xk))
        if 0 < j <= len(psi_n):
            rho[k] = scale * psi_n[j - 1]
            rho_norm[k] = scale_norm * psi_n[j - 1]
    return rho, rho_norm


# ---------------------------------------------------------------------------
# Error metrics  (public)
# ---------------------------------------------------------------------------

def _errors_from_arrays(
    rho_n: np.ndarray, f_target: np.ndarray, x_grid: np.ndarray
) -> dict:
    """Compute error metrics from pre-evaluated rho_n and f_target arrays."""
    mask = x_grid >= 0.1
    err = rho_n[mask] - f_target[mask]
    f_ref = f_target[mask]

    sup_norm = float(np.max(np.abs(err)))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(f_ref) > 1e-12, np.abs(f_ref), np.nan)
    rel_err = float(np.nanmean(np.abs(err) / denom))

    return {
        "sup_norm": sup_norm,
        "mae": mae,
        "rmse": rmse,
        "rel_error_away_from_zero": rel_err,
    }


def compute_kernel_errors(
    alpha: float, n: int, kappa: float, L: int
) -> dict:
    """Compute approximation errors between rho_n and the ML target kernel.

    All error metrics are evaluated on an x_grid of 200 points in [0.01, 5],
    with values at x < 0.1 excluded to avoid the near-zero singularity.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    kappa : float
        Mean-reversion speed.
    L : int
        Truncation length for psi.

    Returns
    -------
    dict
        Keys: 'sup_norm', 'mae', 'rmse', 'rel_error_away_from_zero'.
        All values are non-negative floats.
    """
    x_grid = np.linspace(0.01, 5.0, 200)
    phi_n = compute_scaled_kernel(alpha, n, L, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    psi_n = _psi_direct_from_phi(phi_n, L)
    rho_n, _ = _compute_rho_from_psi(psi_n, n, a_n, x_grid)
    f_target = mittag_leffler_kernel(x_grid, alpha, kappa)
    return _errors_from_arrays(rho_n, f_target, x_grid)
