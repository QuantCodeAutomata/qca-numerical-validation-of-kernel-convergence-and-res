#!/usr/bin/env python3
"""Experiment 1: Numerical validation of kernel convergence.

Checks that the empirical kernel rho_n(x) converges to the Mittag-Leffler
kernel f^{alpha,kappa}(x) as n → ∞, for three roughness parameters alpha.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.kernel import (
    compute_phi_sequence,
    compute_a_n,
    compute_scaled_kernel,
    mittag_leffler_kernel,
    _psi_direct_from_phi,
    _psi_fft_from_phi,
    _compute_rho_from_psi,
    _errors_from_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psi_for_n(phi_n: np.ndarray, L: int, n: int) -> np.ndarray:
    """Choose direct or FFT inversion based on n.

    For small n (n <= 100) the FFT geometric series sum_{k>=1} phi_n^{*k}
    converges in O(100) iterations and provides a fast cross-check.
    For larger n, a_n is close to 1 so the geometric series converges very
    slowly; the vectorised direct (renewal-equation) method is used instead.
    """
    if n <= 100:
        return _psi_fft_from_phi(phi_n, L)
    return _psi_direct_from_phi(phi_n, L)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment1(save_dir: Path) -> None:
    """Run kernel convergence experiment and save figures and CSV.

    Parameters
    ----------
    save_dir : Path
        Root results directory; figures go to save_dir/figures/.
    """
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    alphas = [0.501, 0.62, 0.999]
    kappa = 0.1
    ns = [100, 250, 500, 1000, 2500, 5000]
    x_grid = np.linspace(0.01, 5.0, 200)

    # How many psi coefficients do we need for each n?
    # We need psi[floor(n * x_max) - 1], so L = floor(n * x_max) + 1.
    x_max = x_grid[-1]

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    # results[alpha][n] = {'rho_n': ..., 'rho_n_norm': ..., 'errors': ...}
    results = {}

    alpha_labels = {0.501: "α=0.501", 0.62: "α=0.62", 0.999: "α=0.999"}
    n_plot_subset = [100, 500, 1000, 5000]  # for trajectory plots

    # ------------------------------------------------------------------ #
    # Target kernels (computed once per alpha)
    # ------------------------------------------------------------------ #
    print("Computing Mittag-Leffler target kernels …")
    target_kernels = {}
    for alpha in alphas:
        print(f"  alpha={alpha} …", end=" ", flush=True)
        target_kernels[alpha] = mittag_leffler_kernel(x_grid, alpha, kappa)
        print("done")

    # ------------------------------------------------------------------ #
    # Phi sequence sum check (once per alpha, large L)
    # ------------------------------------------------------------------ #
    print("\nVerifying phi-sequence sums …")
    L_verify = max(ns) * 100  # 500 000
    for alpha in alphas:
        phi_big = compute_phi_sequence(alpha, L_verify)
        total = phi_big.sum()
        print(f"  alpha={alpha}: sum(phi, L={L_verify:,}) = {total:.8f}")

    # ------------------------------------------------------------------ #
    # Main loop: (alpha, n)
    # ------------------------------------------------------------------ #
    print("\nRunning convergence loop …")
    for alpha in alphas:
        results[alpha] = {}
        f_target = target_kernels[alpha]

        for n in ns:
            L = int(np.floor(n * x_max)) + 2  # psi truncation length

            phi_n = compute_scaled_kernel(alpha, n, L, kappa)
            a_n = compute_a_n(alpha, n, kappa)

            psi_n = _psi_for_n(phi_n, L, n)

            rn, rn_norm = _compute_rho_from_psi(psi_n, n, a_n, x_grid)

            errors = _errors_from_arrays(rn, f_target, x_grid)

            results[alpha][n] = {
                "rho_n": rn,
                "rho_n_norm": rn_norm,
                "errors": errors,
            }
            print(
                f"  alpha={alpha}, n={n:5d}: "
                f"sup={errors['sup_norm']:.4f}  "
                f"mae={errors['mae']:.4f}  "
                f"rmse={errors['rmse']:.4f}  "
                f"rel={errors['rel_error_away_from_zero']:.4f}"
            )

    # ------------------------------------------------------------------ #
    # Plot 1: rho_n(x) vs target for each alpha
    # ------------------------------------------------------------------ #
    print("\nGenerating Figure 1 …")
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in [0.1, 0.35, 0.6, 0.85]]

    for ax, alpha in zip(axes1, alphas):
        f_target = target_kernels[alpha]
        # Plot target kernel
        ax.plot(x_grid, f_target, "k-", lw=2.5, label="ML target", zorder=5)

        for ci, n in enumerate(n_plot_subset):
            if n in results[alpha]:
                rn = results[alpha][n]["rho_n"]
                ax.plot(x_grid, rn, color=colors[ci], lw=1.2,
                        linestyle="--", label=f"n={n}")

        ax.set_title(alpha_labels[alpha], fontsize=13)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel(r"$\rho_n(x)$", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 5)
        ax.grid(True, alpha=0.3)

    fig1.suptitle(
        r"Empirical kernel $\rho_n(x)$ vs Mittag-Leffler target "
        r"$f^{\alpha,\kappa}(x)$",
        fontsize=14,
    )
    fig1.tight_layout()
    path1 = fig_dir / "exp1_kernel_approximation.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {path1}")

    # ------------------------------------------------------------------ #
    # Plot 2: Log-log convergence of each metric vs n
    # ------------------------------------------------------------------ #
    print("Generating Figure 2 …")
    metric_keys = ["sup_norm", "mae", "rmse", "rel_error_away_from_zero"]
    metric_labels = {
        "sup_norm": "Sup-norm error",
        "mae": "MAE",
        "rmse": "RMSE",
        "rel_error_away_from_zero": "Relative error (x ≥ 0.1)",
    }
    alpha_colors = {0.501: "steelblue", 0.62: "darkorange", 0.999: "green"}
    alpha_markers = {0.501: "o", 0.62: "s", 0.999: "^"}

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.ravel()

    for ax, key in zip(axes2, metric_keys):
        for alpha in alphas:
            errs = [results[alpha][n]["errors"][key] for n in ns]
            ax.loglog(
                ns, errs,
                color=alpha_colors[alpha],
                marker=alpha_markers[alpha],
                linestyle="-",
                label=alpha_labels[alpha],
                lw=1.5,
                markersize=6,
            )
        ax.set_title(metric_labels[key], fontsize=12)
        ax.set_xlabel("n", fontsize=11)
        ax.set_ylabel("Error", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    fig2.suptitle("Kernel convergence: error vs n (log-log)", fontsize=14)
    fig2.tight_layout()
    path2 = fig_dir / "exp1_convergence_rates.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    records = []
    for alpha in alphas:
        for n in ns:
            errs = results[alpha][n]["errors"]
            records.append(
                {
                    "alpha": alpha,
                    "n": n,
                    "sup_norm": errs["sup_norm"],
                    "mae": errs["mae"],
                    "rmse": errs["rmse"],
                    "rel_error": errs["rel_error_away_from_zero"],
                }
            )
    df = pd.DataFrame(records)
    csv_path = save_dir / "exp1_kernel_convergence.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSaved {csv_path}")

    print("\n=== Summary Table ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_experiment1(Path("results"))
