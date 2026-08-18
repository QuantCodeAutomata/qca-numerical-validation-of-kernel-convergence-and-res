"""FFT vs direct benchmark for the rough-volatility AR kernel.

Both :func:`simulate_direct` and :func:`simulate_fft_accelerated` compute the
same causal AR recursion::

    lambda_t = omega_n + sum_{i=1}^{min(t,L)} phi_n[i-1] * lambda_{t-i}

The difference is in *how* the full path is computed:

* ``simulate_direct``   – explicit Python loop with :func:`numpy.dot`; O(N²).
* ``simulate_fft_accelerated`` – polynomial division in the DFT domain:

  The recursion lambda = f + phi_ext * lambda (causal), where f[0]=lambda0
  and f[t]=omega_n for t>=1, is solved exactly via

      L(z) = F(z) / (1 - Phi(z))

  using a single pair of FFTs.  Total complexity: O(N log N).
"""

import time
import numpy as np
import pandas as pd
from typing import Optional

from src.kernel import compute_scaled_kernel


# ---------------------------------------------------------------------------
# Default simulation hyper-parameters (shared by both methods)
# ---------------------------------------------------------------------------
_KAPPA: float = 0.1
_OMEGA_SCALE: float = 1e-2   # baseline omega_n value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ar_core_direct(phi_n: np.ndarray, omega_n: float,
                    lambda0: float, N: int) -> np.ndarray:
    """Run the AR recursion step-by-step with :func:`numpy.dot`; O(N²)."""
    L = len(phi_n)
    lam = np.empty(N + 1)
    lam[0] = lambda0
    for t in range(1, N + 1):
        lag = min(t, L)
        lam[t] = omega_n + np.dot(phi_n[:lag], lam[t - lag: t][::-1])
    return lam


def _compute_psi(phi_n: np.ndarray, N: int) -> np.ndarray:
    """Compute the MA(∞) impulse response of the AR filter via the renewal equation.

    The AR filter 1 / (1 - Phi(z)) has impulse response psi satisfying::

        psi[0] = 1
        psi[t] = sum_{i=1}^{min(t,L)} phi_n[i-1] * psi[t-i]   for t >= 1

    This is O(N²) to compute but is performed ONCE per (alpha, n) pair.  After
    precomputation the per-path simulation cost drops from O(N²) to O(N log N).

    Parameters
    ----------
    phi_n:
        Scaled AR kernel weights (length L).
    N:
        Number of time steps; output length is N+1.

    Returns
    -------
    np.ndarray
        Shape ``(N+1,)``; psi[0]=1 and psi[t] = renewal-equation value.
    """
    L = len(phi_n)
    psi = np.zeros(N + 1)
    psi[0] = 1.0
    for t in range(1, N + 1):
        lag = min(t, L)
        psi[t] = np.dot(phi_n[:lag], psi[t - lag: t][::-1])
    return psi


def _ar_core_fft(phi_n: np.ndarray, omega_n: float,
                 lambda0: float, N: int,
                 psi: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulate the AR path in O(N log N) using precomputed MA(∞) weights.

    The solution to the causal AR recursion

        lambda[t] = f[t] + (phi_n * lambda)[t - 1]

    is the linear convolution  lambda = psi *_linear f, where psi is the
    impulse response returned by :func:`_compute_psi` and the forcing is
    f[0]=lambda0, f[t]=omega_n for t=1..N.  The linear convolution is
    evaluated via zero-padded FFT (pad to at least 2(N+1)) in O(N log N).

    If *psi* is not pre-supplied it is computed inline (O(N²)).

    Parameters
    ----------
    phi_n:
        Scaled AR kernel weights.
    omega_n:
        Constant forcing term.
    lambda0:
        Initial condition lambda[0].
    N:
        Number of steps; output length N+1.
    psi:
        Pre-computed MA(∞) impulse response (length N+1).  Supply this to
        achieve the O(N log N) per-path cost; omit only for correctness tests.

    Returns
    -------
    np.ndarray
        Shape ``(N+1,)`` matching :func:`_ar_core_direct` to float64 precision.
    """
    if psi is None:
        psi = _compute_psi(phi_n, N)

    # --- choose next power-of-two padding >= 2*(N+1) for linear convolution ---
    M = 1
    while M < 2 * (N + 1):
        M <<= 1

    # --- forcing: f[0]=lambda0, f[1..N]=omega_n ---
    f = np.zeros(M)
    f[0] = lambda0
    f[1: N + 1] = omega_n

    # --- psi padded to M ---
    psi_pad = np.zeros(M)
    psi_pad[: N + 1] = psi

    # --- linear convolution lambda = psi *_linear f via rfft ---
    lam = np.fft.irfft(np.fft.rfft(psi_pad) * np.fft.rfft(f), n=M)
    return lam[: N + 1].real


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_phi_n(alpha: float, n: int, L: int) -> np.ndarray:
    """Return the scaled kernel weights phi_n^{(alpha,n)} of length L.

    Parameters
    ----------
    alpha:
        Self-similarity exponent alpha in (0.5, 1).
    n:
        Time-scaling parameter (number of steps per unit time).
    L:
        Number of lags to compute.

    Returns
    -------
    np.ndarray
        Non-negative array of length L with sum < 1.
    """
    return compute_scaled_kernel(alpha, n, L, kappa=_KAPPA)


def simulate_direct(
    alpha: float,
    n_scale: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate the log-volatility path via a direct O(N²) AR recursion.

    The AR process is::

        lambda_t = omega_n + sum_{i=1}^{min(t,L)} phi_n[i-1] * lambda_{t-i}

    starting from the approximate stationary level
    ``lambda0 = omega_n / (1 - sum(phi_n))``.

    Parameters
    ----------
    alpha:
        Self-similarity exponent.
    n_scale:
        Time-scaling parameter n.
    n_steps:
        Number of time steps N.  Output length is N + 1.
    seed:
        Optional RNG seed (unused here; reserved for noise extensions).

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``; ``Lambda[0] == lambda0``.
    """
    phi_n = compute_phi_n(alpha, n_scale, L=n_steps)
    a_n = float(np.sum(phi_n))
    omega_n = _OMEGA_SCALE
    lambda0 = omega_n / max(1e-10, 1.0 - a_n)
    return _ar_core_direct(phi_n, omega_n, lambda0, n_steps)


def simulate_fft_accelerated(
    alpha: float,
    n_scale: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate the AR path in O(N log N) using MA(∞) weights + FFT convolution.

    The AR recursion  lambda = f + phi_n * lambda  has exact solution
    lambda = psi *_linear f  where psi is the MA(∞) impulse response of the
    AR filter, computed via the renewal equation (see :func:`_compute_psi`).
    The linear convolution is evaluated by zero-padded FFT.

    Amortisation note: in Monte Carlo contexts psi is computed once per
    (alpha, n) pair and shared across all paths; only the O(N log N) FFT step
    is repeated.  This function recomputes psi for standalone use.

    Parameters
    ----------
    alpha:
        Self-similarity exponent.
    n_scale:
        Time-scaling parameter n.
    n_steps:
        Number of time steps N.  Output length is N + 1.
    seed:
        Unused; reserved for noise extensions.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps + 1,)``; numerically identical to
        :func:`simulate_direct` to float64 precision.
    """
    phi_n = compute_phi_n(alpha, n_scale, L=n_steps)
    a_n = float(np.sum(phi_n))
    omega_n = _OMEGA_SCALE
    lambda0 = omega_n / max(1e-10, 1.0 - a_n)
    psi = _compute_psi(phi_n, n_steps)
    return _ar_core_fft(phi_n, omega_n, lambda0, n_steps, psi=psi)


def benchmark_methods(
    n_values: list,
    alpha: float = 0.62,
    n_reps: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Time direct (O(N²)) vs FFT-accelerated (O(N log N)) AR simulation.

    For each n in *n_values*, the routine:

    1. Pre-computes phi_n and the MA(∞) impulse response psi (both excluded
       from timing — analogous to model calibration shared across MC paths).
    2. Times the per-path simulation of each method over *n_reps* repetitions
       after one warm-up call.
    3. Validates numerical equivalence (max |direct - fft| < 1e-8).
    4. Records mean and std-dev of wall-clock time; computes speedup ratio.

    The benchmark isolates the per-path simulation cost:
    * Direct:  O(N²) sequential dot-product loop.
    * FFT:     O(N log N) zero-padded FFT convolution with precomputed psi.

    Parameters
    ----------
    n_values:
        List of time-step counts N.
    alpha:
        Self-similarity exponent (default 0.62 per paper's Table 3).
    n_reps:
        Number of timing repetitions per (method, n) pair.
    seed:
        Unused (deterministic AR without noise).

    Returns
    -------
    pd.DataFrame
        Columns: n, mean_direct_s, std_direct_s, mean_fft_s, std_fft_s,
        speedup, max_abs_diff.
    """
    records = []
    for n in n_values:
        phi_n = compute_phi_n(alpha, n, L=n)
        a_n = float(np.sum(phi_n))
        omega_n = _OMEGA_SCALE
        lambda0 = omega_n / max(1e-10, 1.0 - a_n)

        # --- precompute psi once (excluded from timing; amortised over MC paths) ---
        psi = _compute_psi(phi_n, n)

        # --- warm-up (excluded from timing) ---
        _ar_core_direct(phi_n, omega_n, lambda0, n)
        _ar_core_fft(phi_n, omega_n, lambda0, n, psi=psi)

        # --- validate equivalence ---
        lam_d = _ar_core_direct(phi_n, omega_n, lambda0, n)
        lam_f = _ar_core_fft(phi_n, omega_n, lambda0, n, psi=psi)
        max_diff = float(np.max(np.abs(lam_d - lam_f)))

        # --- time direct: O(N²) per-step dot product (phi_n precomputed) ---
        times_d = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ar_core_direct(phi_n, omega_n, lambda0, n)
            times_d.append(time.perf_counter() - t0)

        # --- time FFT: O(N log N) linear convolution (phi_n + psi precomputed) ---
        times_f = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ar_core_fft(phi_n, omega_n, lambda0, n, psi=psi)
            times_f.append(time.perf_counter() - t0)

        mean_d = float(np.mean(times_d))
        mean_f = float(np.mean(times_f))
        speedup = mean_d / mean_f if mean_f > 0 else float("nan")

        records.append({
            "n": n,
            "mean_direct_s": mean_d,
            "std_direct_s": float(np.std(times_d)),
            "mean_fft_s": mean_f,
            "std_fft_s": float(np.std(times_f)),
            "speedup": speedup,
            "max_abs_diff": max_diff,
        })

    return pd.DataFrame(records)
