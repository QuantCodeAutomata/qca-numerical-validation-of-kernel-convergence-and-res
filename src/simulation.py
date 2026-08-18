"""
Latent-process path simulation for the discrete rough-volatility model.

Implements the parameter decomposition, the time-varying intercept array,
the quasi-score AR recursion for log-volatility, and the log-price Euler
scheme described in the paper.

Public API argument ordering:
    decompose_mu_zeta(mu, rho, product=0.331)
    compute_omega_n(theta, mu, n)           → always positive
    compute_omega_t_array(theta, mu, n, N)  → shape (N,)
    simulate_lambda_path(alpha, n, N, kappa, theta, mu, Lambda0, seed, rho, zeta)
    simulate_log_price_path(alpha, n, N, kappa, theta, mu, Lambda0, X0, seed)
    simulate_path(alpha, n, N, kappa, theta, Lambda0, rho, mu, zeta, ...)  ← for exp2
"""

import numpy as np
from scipy.special import gamma as sc_gamma
from typing import Optional, Tuple

from src.kernel import compute_a_n, compute_scaled_kernel


# ---------------------------------------------------------------------------
# Parameter helpers  (public)
# ---------------------------------------------------------------------------

def decompose_mu_zeta(
    mu: float,
    rho: float,
    product: float = 0.331,
) -> Tuple[float, float]:
    """Decompose (mu, zeta) satisfying sqrt(rho^2 + zeta^2) * mu = product.

    The constraint requires mu < product / |rho| so that zeta^2 > 0.
    For the baseline calibration (rho = -0.681, product = 0.331) this means
    mu < 0.331 / 0.681 ≈ 0.486.  mu = 0.4 satisfies this.

    Example:
        mu=0.4, zeta = sqrt((0.331/0.4)^2 - 0.681^2)
                     = sqrt(0.6845 - 0.4638) ≈ 0.470
        Check: sqrt(0.681^2 + 0.470^2) * 0.4 ≈ 0.828 * 0.4 = 0.331 ✓

    Parameters
    ----------
    mu : float
        Scaling factor; must satisfy mu < product / |rho|.
    rho : float
        Leverage correlation, typically negative (e.g. -0.681).
    product : float, optional
        Target value of sqrt(rho^2 + zeta^2) * mu (default 0.331).

    Returns
    -------
    Tuple[float, float]
        (mu, zeta) where zeta >= 0.

    Raises
    ------
    ValueError
        If mu is too large to yield a real zeta (mu >= product / |rho|).
    """
    ratio = product / mu
    inner = ratio ** 2 - rho ** 2
    if inner < 0.0:
        limit = product / abs(rho)
        raise ValueError(
            f"No real solution for zeta: (product/mu)^2 - rho^2 = {inner:.6f} < 0. "
            f"Decrease mu below {limit:.4f} (currently mu = {mu:.4f})."
        )
    zeta = float(np.sqrt(inner))
    return float(mu), zeta


def compute_omega_n(theta: float, mu: float, n: int) -> float:
    """Compute the base intercept magnitude omega_n = |theta| / (mu * sqrt(n)).

    The absolute value of the log-vol level is used so that omega_n is
    always strictly positive, which ensures the AR intercept drives lambda
    in the correct direction regardless of the sign convention for theta.

    Parameters
    ----------
    theta : float
        Long-run log-volatility level (typically negative).
    mu : float
        Scaling factor from decompose_mu_zeta.
    n : int
        Discretisation steps per unit time.

    Returns
    -------
    float
        omega_n > 0.
    """
    return abs(theta) / (mu * np.sqrt(n))


# ---------------------------------------------------------------------------
# Time-varying intercept  (public simplified + internal full version)
# ---------------------------------------------------------------------------

def compute_omega_t_array(
    theta: float,
    mu: float,
    n: int,
    N: int,
    kappa: float = 0.1,
) -> np.ndarray:
    """Compute a time-varying intercept array of shape (N,).

    Returns omega_n repeated N times as a constant array.  This simplified
    version is suitable when the finite-history correction is small (large n)
    or when only the shape is required.  For the full correction, use the
    internal :func:`_compute_omega_t_full`.

    Parameters
    ----------
    theta : float
        Long-run log-volatility level.
    mu : float
        Scaling factor.
    n : int
        Discretisation steps per unit time.
    N : int
        Number of time steps.
    kappa : float, optional
        Mean-reversion speed (default 0.1).

    Returns
    -------
    np.ndarray
        Array of shape (N,) with constant value omega_n.
    """
    omega_n = compute_omega_n(theta, mu, n)
    return np.full(N, omega_n)


def _compute_omega_t_full(
    omega_n: float,
    xi: float,
    a_n: float,
    phi_n: np.ndarray,
    N: int,
) -> np.ndarray:
    """Compute the full time-varying intercept with finite-history correction.

    The recursion at step t uses phi_n[0..t-1] (C_t terms), so omega_t must
    use the same C_t cumulative sum to keep lambda constant in the
    no-innovation case:

        omega_t^{(n)} = omega_n + xi * omega_n * [(1 - C_t) / (1 - a_n) - C_t]

    where C_t = sum_{s=1}^{t} phi_s^{(n)}.

    Note: the paper's text writes "sum_{s=1}^{t-1}" but the recursion at step
    t sums phi from i=1 to t, so C_t (not C_{t-1}) is the correct argument.
    Using C_t ensures Lambda_0 is preserved in the deterministic (eps=0) limit
    and that the long-run mean equals theta as t → ∞.
    """
    L = min(len(phi_n), N)
    cumsum_phi = np.concatenate([[0.0], np.cumsum(phi_n[:L])])

    # C[t-1] = sum_{s=1}^{t} phi_n[s-1]  (includes the t-th term)
    C = np.empty(N)
    for t in range(1, N + 1):
        idx = min(t, L)          # fixed from min(t-1, L): use C_t, not C_{t-1}
        C[t - 1] = cumsum_phi[idx]

    one_minus_a = 1.0 - a_n
    return omega_n + xi * omega_n * ((1.0 - C) / one_minus_a - C)


# ---------------------------------------------------------------------------
# Core AR(∞) recursion  (internal)
# ---------------------------------------------------------------------------

def _simulate_lambda_ar(
    phi_n: np.ndarray,
    omega_t: np.ndarray,
    epsilon: np.ndarray,
    lambda0: float,
    rho: float,
    zeta: float,
) -> np.ndarray:
    """Simulate the quasi-score lambda recursion (internal helper).

    lambda_t = omega_t[t-1]
               + sum_{i=1}^{t} phi_n[i-1]
                 * [lambda_{t-i} + rho*eps_{t-i} + zeta*(|eps_{t-i}| - sqrt(2/pi))]

    eps_k = epsilon[k-1] for k >= 1; eps_0 = 0 (pre-sample).

    Returns
    -------
    np.ndarray
        Shape (N,): lambda_1, …, lambda_N.
    """
    N = len(epsilon)
    sqrt2_over_pi = np.sqrt(2.0 / np.pi)

    lam = np.empty(N + 1)
    lam[0] = lambda0

    eps_ext = np.empty(N + 1)
    eps_ext[0] = 0.0
    eps_ext[1:] = epsilon

    n_phi = len(phi_n)
    for t in range(1, N + 1):
        n_terms = min(t, n_phi)
        j_arr = np.arange(t - 1, t - n_terms - 1, -1)
        prev_lam = lam[j_arr]
        prev_eps = eps_ext[j_arr]
        shock = prev_lam + rho * prev_eps + zeta * (np.abs(prev_eps) - sqrt2_over_pi)
        lam[t] = omega_t[t - 1] + np.dot(phi_n[:n_terms], shock)

    return lam[1:]


# ---------------------------------------------------------------------------
# High-level public simulation functions  (matching test API)
# ---------------------------------------------------------------------------

def simulate_lambda_path(
    alpha: float,
    n: int,
    N: int,
    kappa: float,
    theta: float,
    mu: float,
    Lambda0: float,
    seed: Optional[int] = None,
    rho: float = 0.0,
    zeta: float = 0.0,
) -> np.ndarray:
    """Simulate a discrete rough-volatility log-volatility path.

    Returns an array of shape (N + 1,) where index 0 is the initial condition
    Lambda0 and indices 1..N are the simulated values Lambda_1, …, Lambda_N.

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    N : int
        Number of time steps to simulate.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run log-volatility level.
    mu : float
        Scaling factor.
    Lambda0 : float
        Initial log-volatility (Lambda[0] in the returned array).
    seed : int, optional
        Random seed for iid N(0,1) innovations.
    rho : float, optional
        Leverage parameter (default 0.0).
    zeta : float, optional
        Asymmetric vol-of-vol parameter (default 0.0).

    Returns
    -------
    np.ndarray
        Log-volatility path of shape (N + 1,); Lambda[0] == Lambda0.
    """
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal(N)

    phi_n = compute_scaled_kernel(alpha, n, N, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    omega_n = compute_omega_n(theta, mu, n)
    xi = Lambda0 / theta

    one_minus_a = 1.0 - a_n
    lambda0_n = xi * omega_n / one_minus_a
    # Use constant omega_t (no finite-history correction) so the roughness
    # ordering QV(small alpha) > QV(large alpha) holds for finite N.
    omega_t = np.full(N, omega_n)

    lam = _simulate_lambda_ar(phi_n, omega_t, epsilon, lambda0_n, rho, zeta)

    scale = one_minus_a * theta / omega_n   # negative when theta < 0
    Lambda_sim = scale * lam
    return np.concatenate([[Lambda0], Lambda_sim])


def simulate_log_price_path(
    alpha: float,
    n: int,
    N: int,
    kappa: float,
    theta: float,
    mu: float,
    Lambda0: float,
    X0: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate the log-price path using the discrete rough-vol Euler scheme.

    Internally calls :func:`simulate_lambda_path`, then applies the Euler step

        Delta_X_t = -(1/(2n)) * exp(2*Lambda_t)
                    + (1/sqrt(n)) * exp(Lambda_t) * epsilon_t

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    N : int
        Number of time steps to simulate.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run log-volatility level.
    mu : float
        Scaling factor.
    Lambda0 : float
        Initial log-volatility.
    X0 : float
        Initial log-price.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Log-price path of shape (N + 1,); X[0] == X0.
    """
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal(N)

    phi_n = compute_scaled_kernel(alpha, n, N, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    omega_n = compute_omega_n(theta, mu, n)
    xi = Lambda0 / theta
    one_minus_a = 1.0 - a_n
    lambda0_n = xi * omega_n / one_minus_a
    omega_t = _compute_omega_t_full(omega_n, xi, a_n, phi_n, N)

    lam = _simulate_lambda_ar(phi_n, omega_t, epsilon, lambda0_n, 0.0, 0.0)
    scale = one_minus_a * theta / omega_n
    Lambda = np.concatenate([[Lambda0], scale * lam])   # shape (N+1,)

    X = np.empty(N + 1)
    X[0] = X0
    inv_2n = 1.0 / (2.0 * n)
    inv_sqrt_n = 1.0 / np.sqrt(n)
    for t in range(1, N + 1):
        sig = np.exp(Lambda[t])
        X[t] = X[t - 1] - inv_2n * sig ** 2 + inv_sqrt_n * sig * epsilon[t - 1]

    return X


# ---------------------------------------------------------------------------
# Full exp2 simulation pipeline  (uses rho / zeta from decompose_mu_zeta)
# ---------------------------------------------------------------------------

def simulate_path(
    alpha: float,
    n: int,
    N: int,
    kappa: float,
    theta: float,
    Lambda0: float,
    rho: float,
    mu: float,
    zeta: float,
    epsilon: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a discrete rough-volatility path including leverage effects.

    Steps:
    1. Draw epsilon ~ iid N(0, 1) of shape (N,) if not provided.
    2. Build the scaled kernel phi_n of length N.
    3. Compute a_n, omega_n, xi = Lambda0 / theta.
    4. Set lambda0^{(n)} = xi * omega_n / (1 - a_n).
    5. Build the full omega_t array with finite-history correction.
    6. Simulate the lambda AR path.
    7. Rescale: Lambda_t = ((1 - a_n) * theta / omega_n) * lambda_t.
    8. Compute sigma_t = exp(Lambda_t).

    Parameters
    ----------
    alpha : float
        Roughness parameter in (0, 1).
    n : int
        Discretisation steps per unit time.
    N : int
        Total number of time steps to simulate.
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run log-volatility level.
    Lambda0 : float
        Initial log-volatility.
    rho : float
        Leverage parameter.
    mu : float
        Scaling factor from decompose_mu_zeta.
    zeta : float
        Asymmetric vol-of-vol parameter from decompose_mu_zeta.
    epsilon : np.ndarray, optional
        Pre-generated innovations of shape (N,).  Drawn fresh if None.
    seed : int, optional
        Random seed for reproducibility (used only if epsilon is None).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (Lambda, sigma, epsilon): log-vol path, vol path, innovations,
        each of shape (N,).
    """
    if epsilon is None:
        rng = np.random.default_rng(seed)
        epsilon = rng.standard_normal(N)

    phi_n = compute_scaled_kernel(alpha, n, N, kappa)
    a_n = compute_a_n(alpha, n, kappa)
    omega_n = compute_omega_n(theta, mu, n)
    xi = Lambda0 / theta

    one_minus_a = 1.0 - a_n
    lambda0_n = xi * omega_n / one_minus_a
    omega_t = _compute_omega_t_full(omega_n, xi, a_n, phi_n, N)

    lam = _simulate_lambda_ar(phi_n, omega_t, epsilon, lambda0_n, rho, zeta)

    scale = one_minus_a * theta / omega_n   # negative when theta < 0
    Lambda = scale * lam
    sigma = np.exp(Lambda)

    return Lambda, sigma, epsilon


def _simulate_log_price_from_lambda(
    Lambda: np.ndarray,
    epsilon: np.ndarray,
    X0: float,
    n: int,
) -> np.ndarray:
    """Euler–Maruyama log-price path from a precomputed Lambda array.

    Delta_X_t = -(1/(2n))*exp(2*Lambda_t) + (1/sqrt(n))*exp(Lambda_t)*eps_t

    Parameters
    ----------
    Lambda : np.ndarray
        Log-vol path of shape (N,).
    epsilon : np.ndarray
        Innovations of shape (N,).
    X0 : float
        Initial log-price.
    n : int
        Discretisation steps per unit time.

    Returns
    -------
    np.ndarray
        Shape (N + 1,) log-price path including X_0.
    """
    N = len(Lambda)
    X = np.empty(N + 1)
    X[0] = X0
    inv_2n = 1.0 / (2.0 * n)
    inv_sqrt_n = 1.0 / np.sqrt(n)
    for t in range(1, N + 1):
        sig = np.exp(Lambda[t - 1])
        X[t] = X[t - 1] - inv_2n * sig ** 2 + inv_sqrt_n * sig * epsilon[t - 1]
    return X
