#!/usr/bin/env python3
"""Experiment 3: Monte Carlo pricing of European, Asian, Lookback, and Barrier options.

Reproduces Table 1-style option prices under the discrete rough-volatility
approximation for alpha in {0.999, 0.62, 0.501}.

Key parameters (paper Section 4):
    S_0 = 100, X_0 = log(100), Lambda_0 = log(sqrt(0.0392))
    rho = -0.681, kappa = 0.1, theta = log(sqrt(0.3156))
    sqrt(rho^2+zeta^2)*mu = 0.331  -> mu=0.4, zeta≈0.470 (see src.mc_engine)
    T = 1 year, n = 500, N = 500, M = 50,000 paths (reduced for runtime)
    Strikes K in {80, 90, 100, 110, 120}
    Barriers: B_u = 110 (Up-In call), B_d = 90 (Down-Out put)
    r = 0 (no discounting)
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

from src.mc_engine import run_mc_pricing, MU, KAPPA, THETA, LAMBDA0, S0, RHO

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

ALPHAS = [0.999, 0.62, 0.501]
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
N_STEPS_PER_YEAR = 500
T_YEARS = 1
N = N_STEPS_PER_YEAR * T_YEARS      # = 500
M_PATHS = 50_000                     # reduced from 500k for runtime
BATCH_SIZE = 5_000
SEED = 12345
BARRIER_UP = 110.0
BARRIER_DOWN = 90.0
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPTION_LABELS = {
    "ec": "European Call",
    "ep": "European Put",
    "ac": "Asian Call",
    "ap": "Asian Put",
    "lc": "Lookback Call",
    "lp": "Lookback Put",
    "uic": "Up-In Call (B=110)",
    "dop": "Down-Out Put (B=90)",
}


def _fmt(price: float, se: float) -> str:
    """Format price with standard error."""
    return f"{price:.4f} ({se:.4f})"


def _validate_parity(results: dict, alpha: float) -> None:
    """Check call-put parity and barrier edge cases."""
    print(f"\n  Validation checks (alpha={alpha}):")
    for K in STRIKES:
        ec, ep = results[K]["ec"][0], results[K]["ep"][0]
        parity_err = abs((ec - ep) - (S0 - K))
        print(f"    K={K:.0f}: C-P={ec-ep:.4f}, S0-K={S0-K:.1f}, "
              f"parity_err={parity_err:.4f}")
    # Down-Out put at K <= B_d should be near zero
    print(f"    DOP at K=80: {results[80.0]['dop'][0]:.4f} (should be ~0)")
    print(f"    DOP at K=90: {results[90.0]['dop'][0]:.4f} (should be ~0)")


def _to_dataframe(results: dict, alpha: float) -> pd.DataFrame:
    """Convert results dict to a tidy DataFrame."""
    rows = []
    for K in STRIKES:
        row = {"alpha": alpha, "K": K}
        for key in ["ec", "ep", "ac", "ap", "lc", "lp", "uic", "dop"]:
            price, se = results[K][key]
            row[f"{key}_price"] = price
            row[f"{key}_se"] = se
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    all_dfs = []
    all_results = {}   # store for exp4 percentage-error computation

    print("=" * 70)
    print("Experiment 3: Monte Carlo Option Pricing (Rough Volatility)")
    print(f"  M={M_PATHS:,} paths, N={N} steps, T={T_YEARS}y, n={N_STEPS_PER_YEAR}")
    print(f"  mu={MU}, Barriers: B_u={BARRIER_UP}, B_d={BARRIER_DOWN}")
    print("=" * 70)

    for alpha in ALPHAS:
        print(f"\n>>> alpha = {alpha}")
        t0 = time.perf_counter()

        results = run_mc_pricing(
            alpha=alpha,
            n=N_STEPS_PER_YEAR,
            N=N,
            M=M_PATHS,
            strikes=STRIKES,
            barrier_up=BARRIER_UP,
            barrier_down=BARRIER_DOWN,
            batch_size=BATCH_SIZE,
            mu=MU,
            seed=SEED,
        )
        elapsed = time.perf_counter() - t0
        all_results[alpha] = results

        print(f"    Runtime: {elapsed:.1f}s")
        df = _to_dataframe(results, alpha)
        all_dfs.append(df)

        # Pretty-print table
        header = f"{'K':>5} | {'Eur Call':>14} | {'Eur Put':>14} | " \
                 f"{'Asian C':>14} | {'Asian P':>14} | " \
                 f"{'LB Call':>14} | {'LB Put':>14} | " \
                 f"{'UI Call':>14} | {'DO Put':>14}"
        print("\n  " + header)
        print("  " + "-" * len(header))
        for K in STRIKES:
            row_vals = " | ".join(
                _fmt(*results[K][key])
                for key in ["ec", "ep", "ac", "ap", "lc", "lp", "uic", "dop"]
            )
            print(f"  {K:>5.0f} | {row_vals}")

        _validate_parity(results, alpha)

    # -----------------------------------------------------------------------
    # Save combined CSV
    # -----------------------------------------------------------------------
    combined = pd.concat(all_dfs, ignore_index=True)
    out_csv = RESULTS_DIR / "exp3_mc_pricing.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Save alpha=0.999 reference for exp4
    ref_csv = RESULTS_DIR / "exp3_alpha_0999_reference.csv"
    ref_df = _to_dataframe(all_results[0.999], 0.999)
    ref_df.to_csv(ref_csv, index=False)
    print(f"Saved reference (alpha=0.999): {ref_csv}")

    # -----------------------------------------------------------------------
    # Visualisation: prices across alpha for each option type
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    opt_types = ["ec", "ep", "ac", "ap", "lc", "lp", "uic", "dop"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for ax, opt in zip(axes, opt_types):
        for alpha, col in zip(ALPHAS, colors):
            prices = [all_results[alpha][K][opt][0] for K in STRIKES]
            ses = [all_results[alpha][K][opt][1] for K in STRIKES]
            ax.errorbar(
                STRIKES, prices, yerr=ses,
                label=f"α={alpha}", color=col,
                marker="o", capsize=3, linewidth=1.5,
            )
        ax.set_title(OPTION_LABELS[opt], fontsize=10)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp 3: Option Prices vs Strike — Discrete Rough-Vol Approx "
        f"(M={M_PATHS:,}, N={N})",
        fontsize=12,
    )
    plt.tight_layout()
    out_fig = RESULTS_DIR / "exp3_mc_pricing.png"
    plt.savefig(out_fig, dpi=120)
    plt.close()
    print(f"Saved: {out_fig}")

    # -----------------------------------------------------------------------
    # Summary table for RESULTS.md
    # -----------------------------------------------------------------------
    print("\n--- Summary (European Call prices) ---")
    hdr = f"{'K':>5} | " + " | ".join(f"α={a}" for a in ALPHAS)
    print(hdr)
    for K in STRIKES:
        vals = " | ".join(
            f"{all_results[a][K]['ec'][0]:>8.4f}" for a in ALPHAS
        )
        print(f"{K:>5.0f} | {vals}")


if __name__ == "__main__":
    run_experiment()
