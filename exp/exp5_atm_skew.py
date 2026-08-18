#!/usr/bin/env python3
"""Experiment 5: ATM implied-volatility skew power-law analysis.

Generates an option surface across maturities and strikes, inverts to
Black-Scholes implied volatility, estimates the ATM skew psi(tau) =
|d sigma_IV/dk|_{k=0}, and tests for power-law decay via log-log regression.

Methodology choices (undocumented in paper, fixed here):
    Maturity grid : tau in {1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 9/12, 1}
    Log-moneyness : k in {-0.05, -0.025, -0.01, 0.01, 0.025, 0.05}
    Skew estimator: local linear regression of IV(k) on k; slope = psi (signed)
    Implied vol   : Brent root-finder on Black-Scholes call formula, r=0
    MC paths      : M_half = 25,000 with antithetic variates → effective M = 50,000
    CRN           : Common random numbers across strikes at fixed maturity
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy import stats as scipy_stats

from src.kernel import compute_a_n, compute_phi_sequence
from src.simulation import (
    decompose_mu_zeta,
    compute_omega_n,
    _compute_omega_t_full,
)
from src.mc_engine import (
    _simulate_lambda_batch,
    MU, RHO, KAPPA, THETA, LAMBDA0, X0, S0,
)
from src.skew import implied_vol_call, powerlaw_regression

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

ALPHAS = [0.999, 0.62, 0.501]
# Maturity grid (in years)
TAU_GRID = np.array([1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0])
# Log-moneyness grid — exclude k=0 so every point contributes to the slope
K_GRID = np.array([-0.05, -0.025, -0.01, 0.01, 0.025, 0.05])
N_PER_YEAR = 500       # discretisation steps per year
M_HALF = 25_000        # half-paths; antithetic doubles to effective 50,000
BATCH_SIZE = 5_000
SEED_BASE = 7777


def _get_params(alpha: float) -> dict:
    """Return all model parameters for a given alpha."""
    _, zeta = decompose_mu_zeta(MU, RHO, 0.331)
    a_n = compute_a_n(alpha, N_PER_YEAR, KAPPA)
    omega_n = compute_omega_n(THETA, MU, N_PER_YEAR)
    xi = LAMBDA0 / THETA
    one_minus_a = 1.0 - a_n
    lambda0 = xi * omega_n / one_minus_a
    scale = one_minus_a * THETA / omega_n
    return {
        "zeta": zeta,
        "a_n": a_n,
        "omega_n": omega_n,
        "xi": xi,
        "lambda0": lambda0,
        "scale": scale,
    }


def _terminal_prices_antithetic(
    eps: np.ndarray,
    phi_n: np.ndarray,
    omega_t: np.ndarray,
    lambda0: float,
    scale: float,
    zeta: float,
    dt: float,
) -> np.ndarray:
    """Run eps and -eps through the recursion; return combined S_T array.

    Parameters
    ----------
    eps : (M_half, N_tau) standard-normal innovations.

    Returns
    -------
    S_T : (2 * M_half,) terminal stock prices.
    """
    inv_2n = dt / 2.0
    inv_sqrt_n = np.sqrt(dt)

    results = []
    for signs in (1.0, -1.0):
        eps_use = signs * eps
        lam = _simulate_lambda_batch(eps_use, phi_n, omega_t, lambda0, RHO, zeta)
        Lambda = scale * lam
        sig = np.exp(Lambda)
        dX = -inv_2n * sig**2 + inv_sqrt_n * sig * eps_use
        S_T = np.exp(X0 + dX.sum(axis=1))
        results.append(S_T)
    return np.concatenate(results)


def _simulate_surface_for_maturity(
    alpha: float,
    tau: float,
    params: dict,
    seed: int,
) -> dict:
    """Simulate call prices at all log-moneyness grid points for one maturity.

    Uses CRN across strikes (same eps → same S_T paths per batch) plus
    antithetic variates (also run -eps) for variance reduction.

    Returns
    -------
    dict mapping k → call_price (float).
    """
    N_tau = max(1, int(np.ceil(N_PER_YEAR * tau)))
    dt = tau / N_tau

    phi_seq = compute_phi_sequence(alpha, N_tau)
    phi_n = params["a_n"] * phi_seq
    omega_t = _compute_omega_t_full(
        params["omega_n"], params["xi"], params["a_n"], phi_n, N_tau
    )

    rng = np.random.default_rng(seed)
    # Accumulate payoffs per strike across batches
    payoffs: dict[float, list] = {k: [] for k in K_GRID}

    n_batches = (M_HALF + BATCH_SIZE - 1) // BATCH_SIZE
    for b_idx in range(n_batches):
        batch_m = min(BATCH_SIZE, M_HALF - b_idx * BATCH_SIZE)
        eps = rng.standard_normal((batch_m, N_tau))

        # CRN: same eps for both +eps and -eps directions
        S_T = _terminal_prices_antithetic(
            eps, phi_n, omega_t, params["lambda0"],
            params["scale"], params["zeta"], dt,
        )  # length 2 * batch_m

        for k in K_GRID:
            K = S0 * np.exp(k)
            payoffs[k].append(np.maximum(S_T - K, 0.0))

    return {k: float(np.concatenate(payoffs[k]).mean()) for k in K_GRID}


def _estimate_atm_skew_local_regression(
    tau: float,
    prices_by_k: dict,
) -> tuple[float | None, dict]:
    """Invert call prices to IV and estimate ATM skew via local OLS regression.

    Fits IV(k) = a + b*k over all valid K_GRID points; returns |b| as psi.

    Returns
    -------
    psi : absolute slope (None if fewer than 3 valid IV points)
    iv_map : {k: implied_vol} for diagnostics
    """
    iv_map: dict[float, float] = {}
    for k in K_GRID:
        price = prices_by_k[k]
        K = S0 * np.exp(k)
        iv = implied_vol_call(price, S0, K, tau, r=0.0)
        if iv is not None and 1e-4 < iv < 5.0:
            iv_map[k] = iv

    if len(iv_map) < 3:
        return None, iv_map

    ks = np.array(sorted(iv_map.keys()))
    ivs = np.array([iv_map[k] for k in ks])
    result = scipy_stats.linregress(ks, ivs)
    # slope is dIV/dk; psi = |slope|
    return float(abs(result.slope)), iv_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Experiment 5: ATM Implied-Vol Skew Power-Law Analysis")
    print(f"  M_half={M_HALF} (antithetic → effective {2*M_HALF}), N_per_year={N_PER_YEAR}")
    print(f"  Maturities: {TAU_GRID}")
    print(f"  K_GRID: {K_GRID}")
    print(f"  Skew estimator: local OLS regression of IV on k")
    print("=" * 70)

    all_records = []
    regression_results = {}

    for alpha in ALPHAS:
        print(f"\n>>> alpha = {alpha}")
        params = _get_params(alpha)
        psi_values = []
        valid_taus = []

        for tau in TAU_GRID:
            seed = SEED_BASE + int(alpha * 1000) + int(tau * 10000)
            t0 = time.perf_counter()

            prices_by_k = _simulate_surface_for_maturity(alpha, tau, params, seed)
            psi, iv_map = _estimate_atm_skew_local_regression(tau, prices_by_k)

            elapsed = time.perf_counter() - t0
            status = f"{psi:.4f}" if psi is not None else "N/A"
            n_valid_iv = len(iv_map)
            print(f"    tau={tau:.4f}  psi={status}  n_valid_iv={n_valid_iv}  [{elapsed:.1f}s]")

            record: dict = {"alpha": alpha, "tau": tau,
                            "psi": psi if psi is not None else float("nan")}
            for k in K_GRID:
                record[f"price_k{int(round(k*100)):+d}"] = prices_by_k[k]
            for k in K_GRID:
                record[f"iv_k{int(round(k*100)):+d}"] = iv_map.get(k, float("nan"))
            all_records.append(record)

            if psi is not None and psi > 0:
                psi_values.append(psi)
                valid_taus.append(tau)

        if len(valid_taus) >= 3:
            reg = powerlaw_regression(valid_taus, psi_values)
            regression_results[alpha] = reg
            print(f"  Power-law slope b={reg['slope']:.4f}, R²={reg['r_squared']:.4f}")
        else:
            regression_results[alpha] = None
            print("  Insufficient data for power-law regression.")

    # -----------------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / "exp5_atm_skew.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    reg_rows = []
    for alpha, reg in regression_results.items():
        if reg is not None:
            reg_rows.append({
                "alpha": alpha,
                "slope": reg["slope"],
                "intercept": reg["intercept"],
                "r_squared": reg["r_squared"],
                "p_value": reg["p_value"],
                "stderr_slope": reg["stderr_slope"],
            })
    if reg_rows:
        reg_df = pd.DataFrame(reg_rows)
        reg_csv = RESULTS_DIR / "exp5_powerlaw_regression.csv"
        reg_df.to_csv(reg_csv, index=False)
        print(f"Saved: {reg_csv}")

    # -----------------------------------------------------------------------
    # Log-log skew plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"0.999": "#2196F3", "0.62": "#FF5722", "0.501": "#4CAF50"}
    for alpha in ALPHAS:
        col = colors[str(alpha)]
        sub = df[df["alpha"] == alpha].dropna(subset=["psi"])
        if len(sub) == 0:
            continue
        taus = sub["tau"].values
        psis = sub["psi"].values
        axes[0].loglog(taus, psis, "o-", color=col, label=f"α={alpha}", linewidth=1.5)

        reg = regression_results.get(alpha)
        if reg is not None:
            t_fit = np.linspace(taus.min(), taus.max(), 100)
            psi_fit = np.exp(reg["intercept"]) * t_fit**reg["slope"]
            axes[0].loglog(t_fit, psi_fit, "--", color=col, alpha=0.6,
                          label=f"fit b={reg['slope']:.2f}")

    axes[0].set_xlabel("Maturity τ (years)")
    axes[0].set_ylabel("ATM skew ψ(τ)")
    axes[0].set_title("Log-log: ATM skew vs maturity")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # R² bar chart
    alphas_plot = [a for a in ALPHAS if regression_results.get(a)]
    r2_vals = [regression_results[a]["r_squared"] for a in alphas_plot]
    slopes = [regression_results[a]["slope"] for a in alphas_plot]

    x = np.arange(len(alphas_plot))
    bars = axes[1].bar(x, r2_vals, color=["#2196F3", "#FF5722", "#4CAF50"][:len(alphas_plot)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"α={a}" for a in alphas_plot])
    axes[1].set_ylabel("R² of log-log fit")
    axes[1].set_title("Power-law regression quality")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.85, color="gray", linestyle="--", label="R²=0.85 threshold")
    axes[1].legend()
    for bar, sl in zip(bars, slopes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"b={sl:.2f}", ha="center", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Exp 5: ATM Implied-Vol Skew Power-Law Analysis", fontsize=12)
    plt.tight_layout()
    out_fig = RESULTS_DIR / "exp5_atm_skew.png"
    plt.savefig(out_fig, dpi=120)
    plt.close()
    print(f"Saved: {out_fig}")

    # Summary
    print("\n--- Power-Law Regression Summary ---")
    for alpha in ALPHAS:
        reg = regression_results.get(alpha)
        if reg:
            print(f"  α={alpha}: slope={reg['slope']:.4f}, "
                  f"R²={reg['r_squared']:.4f}, p={reg['p_value']:.4f}")
        else:
            print(f"  α={alpha}: insufficient data")


if __name__ == "__main__":
    run_experiment()
