#!/usr/bin/env python3
"""Experiment 2: Volatility path visualisation under discrete rough-volatility.

Simulates log-volatility and log-price paths for three roughness parameters
alpha and visualises the qualitative differences in path roughness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.simulation import (
    simulate_path,
    _simulate_log_price_from_lambda,
    decompose_mu_zeta,
    compute_omega_n,
)


# ---------------------------------------------------------------------------
# Roughness diagnostics
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation of x at the given lag."""
    n = len(x)
    if lag >= n:
        return float("nan")
    return float(np.corrcoef(x[: n - lag], x[lag:])[0, 1])


def _hoelder_exponent(Lambda: np.ndarray, hs: list) -> float:
    """Estimate the Hölder exponent via log-log slope of std(increments at lag h).

    Parameters
    ----------
    Lambda : np.ndarray
        Log-volatility path.
    hs : list of int
        Lags to use (e.g. [1, 2, 4, 8]).

    Returns
    -------
    float
        Estimated Hölder exponent H (slope of log(std) vs log(h)).
    """
    log_h = np.log(hs)
    log_std = []
    for h in hs:
        diff = Lambda[h:] - Lambda[:-h]
        log_std.append(np.log(np.std(diff) + 1e-12))
    # Simple linear regression slope
    log_std = np.array(log_std)
    H = float(np.polyfit(log_h, log_std, 1)[0])
    return H


def compute_roughness_diagnostics(Lambda: np.ndarray, n: int) -> dict:
    """Compute roughness diagnostics for a log-volatility path.

    Parameters
    ----------
    Lambda : np.ndarray
        Log-volatility path of shape (N,).
    n : int
        Discretisation steps per unit time (used for labelling only).

    Returns
    -------
    dict
        Keys: 'incr_std', 'quad_variation', 'acf_1', 'acf_5', 'acf_10',
        'hoelder_exponent'.
    """
    increments = np.diff(Lambda)

    incr_std = float(np.std(increments))
    qv = float(np.sum(increments ** 2))
    acf1 = _acf(increments, 1)
    acf5 = _acf(increments, 5)
    acf10 = _acf(increments, 10)
    H = _hoelder_exponent(Lambda, [1, 2, 4, 8])

    return {
        "incr_std": incr_std,
        "quad_variation": qv,
        "acf_1": acf1,
        "acf_5": acf5,
        "acf_10": acf10,
        "hoelder_exponent": H,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment2(save_dir: Path) -> None:
    """Run volatility path experiment and save figures and CSV.

    Parameters
    ----------
    save_dir : Path
        Root results directory; figures go to save_dir/figures/.
    """
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Fixed parameters
    # ------------------------------------------------------------------ #
    X0 = np.log(100.0)
    Lambda0 = np.log(np.sqrt(0.0392))   # initial log-vol ≈ -1.617
    rho = -0.681
    kappa = 0.1
    theta = np.log(np.sqrt(0.3156))     # long-run log-vol ≈ -0.576
    alphas = [0.999, 0.62, 0.501]
    T = 5
    n = 1000
    N = n * T                            # 5 000

    # ------------------------------------------------------------------ #
    # Decompose (mu, zeta) — uses default mu=0.4
    # ------------------------------------------------------------------ #
    mu, zeta = decompose_mu_zeta(mu=0.4, rho=rho, product=0.331)
    print(f"Parameters: mu={mu:.4f}, zeta={zeta:.4f}")
    print(f"Check: sqrt(rho^2+zeta^2)*mu = "
          f"{np.sqrt(rho**2 + zeta**2) * mu:.4f}  (target 0.331)")

    # ------------------------------------------------------------------ #
    # Generate SAME epsilon for all alpha values (to isolate alpha effect)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(42)
    epsilon_shared = rng.standard_normal(N)

    time_axis = np.arange(1, N + 1) / n   # years, shape (N,)

    # ------------------------------------------------------------------ #
    # Simulate paths
    # ------------------------------------------------------------------ #
    paths = {}
    diags = {}
    print("\nSimulating paths …")
    for alpha in alphas:
        t0 = time.time()
        Lambda, sigma, eps = simulate_path(
            alpha=alpha,
            n=n,
            N=N,
            kappa=kappa,
            theta=theta,
            Lambda0=Lambda0,
            rho=rho,
            mu=mu,
            zeta=zeta,
            epsilon=epsilon_shared.copy(),
        )
        elapsed = time.time() - t0
        X_path = _simulate_log_price_from_lambda(Lambda, eps, X0, n)
        diag = compute_roughness_diagnostics(Lambda, n)

        paths[alpha] = {"Lambda": Lambda, "sigma": sigma, "X": X_path}
        diags[alpha] = diag
        print(
            f"  alpha={alpha}: H≈{diag['hoelder_exponent']:.3f}, "
            f"incr_std={diag['incr_std']:.4f}, "
            f"qv={diag['quad_variation']:.4f}  [{elapsed:.1f}s]"
        )

    # ------------------------------------------------------------------ #
    # Plot 1: sigma_t vs time (3 panels, one per alpha)
    # ------------------------------------------------------------------ #
    print("\nGenerating Figure 1 (sigma paths) …")
    alpha_colors = {0.999: "steelblue", 0.62: "darkorange", 0.501: "green"}
    alpha_labels = {0.999: "α = 0.999 (near-Brownian)", 0.62: "α = 0.62", 0.501: "α = 0.501 (rough)"}

    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, alpha in zip(axes1, alphas):
        sigma = paths[alpha]["sigma"]
        ax.plot(time_axis, sigma, color=alpha_colors[alpha], lw=0.8, alpha=0.9)
        ax.set_title(alpha_labels[alpha], fontsize=12)
        ax.set_xlabel("Time (years)", fontsize=10)
        ax.set_ylabel(r"$\sigma_t = e^{\Lambda_t}$", fontsize=10)
        ax.set_xlim(0, T)
        ax.grid(True, alpha=0.3)

    fig1.suptitle(
        "Instantaneous volatility paths $\\sigma_t$ for different $\\alpha$",
        fontsize=13,
    )
    fig1.tight_layout()
    path1 = fig_dir / "exp2_sigma_paths.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {path1}")

    # ------------------------------------------------------------------ #
    # Plot 2: Lambda_t paths on the same axes
    # ------------------------------------------------------------------ #
    print("Generating Figure 2 (log-vol paths) …")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for alpha in alphas:
        Lambda = paths[alpha]["Lambda"]
        ax2.plot(
            time_axis, Lambda,
            color=alpha_colors[alpha],
            lw=0.8, alpha=0.85,
            label=alpha_labels[alpha],
        )
    ax2.axhline(theta, color="k", lw=1.2, linestyle="--", label=r"$\theta$ (long-run)")
    ax2.set_xlabel("Time (years)", fontsize=11)
    ax2.set_ylabel(r"$\Lambda_t$ (log-vol)", fontsize=11)
    ax2.set_title(r"Log-volatility paths $\Lambda_t$ for different $\alpha$", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, T)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = fig_dir / "exp2_logvol_paths.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")

    # ------------------------------------------------------------------ #
    # Plot 3: Roughness diagnostics bar chart
    # ------------------------------------------------------------------ #
    print("Generating Figure 3 (roughness diagnostics) …")
    diag_keys = ["incr_std", "hoelder_exponent", "acf_1", "acf_5", "acf_10"]
    diag_titles = {
        "incr_std": "Increment Std",
        "hoelder_exponent": "Hölder exponent H",
        "acf_1": "ACF(lag=1)",
        "acf_5": "ACF(lag=5)",
        "acf_10": "ACF(lag=10)",
    }

    x_pos = np.arange(len(alphas))
    alpha_str = [str(a) for a in alphas]
    bar_colors = [alpha_colors[a] for a in alphas]

    fig3, axes3 = plt.subplots(1, len(diag_keys), figsize=(16, 4))
    for ax, key in zip(axes3, diag_keys):
        vals = [diags[a][key] for a in alphas]
        bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(alpha_str, fontsize=9)
        ax.set_title(diag_titles[key], fontsize=10)
        ax.set_xlabel("α", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig3.suptitle("Roughness diagnostics by alpha", fontsize=13)
    fig3.tight_layout()
    path3 = fig_dir / "exp2_roughness_diagnostics.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved {path3}")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    records = []
    for alpha in alphas:
        d = diags[alpha]
        records.append(
            {
                "alpha": alpha,
                "incr_std": d["incr_std"],
                "quad_variation": d["quad_variation"],
                "acf_1": d["acf_1"],
                "acf_5": d["acf_5"],
                "acf_10": d["acf_10"],
                "hoelder_exponent": d["hoelder_exponent"],
            }
        )
    df = pd.DataFrame(records)
    csv_path = save_dir / "exp2_volatility_paths.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved {csv_path}")

    print("\n=== Roughness Summary Table ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_experiment2(Path("results"))
