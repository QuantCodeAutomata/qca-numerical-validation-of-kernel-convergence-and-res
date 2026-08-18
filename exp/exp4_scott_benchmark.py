#!/usr/bin/env python3
"""Experiment 4: Benchmark comparison with the Scott lognormal SV model.

Simulates the Scott model (Markovian alpha=1 limit) via Euler-Maruyama and
prices the same contracts as exp_3.  Computes percentage errors relative to
the alpha=0.999 rough-model prices to reproduce Table 2.

Scott model:
    dX_t     = -(1/2) exp(2*Lambda_t) dt + exp(Lambda_t) dB_t
    dLambda_t = kappa*(theta - Lambda_t) dt + sigma * dW_t
    Cov(dB_t, dW_t) = rho * dt

Parameters (same as exp_3 for consistency):
    S_0=100, Lambda_0=log(sqrt(0.0392)), rho=-0.681
    kappa=0.1, theta=log(sqrt(0.3156)), sigma=0.331
    T=1, n=500, N=500, M=50,000 (reduced for runtime)
    Strikes {80,90,100,110,120}, B_u=110, B_d=90, r=0
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

from src.scott_model import simulate_scott_batch, compute_payoffs_scott
from src.mc_engine import LAMBDA0, THETA, RHO, KAPPA, X0, S0

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
SIGMA_SCOTT: float = 0.331           # effective vol-of-vol
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
N = 500
T = 1.0
DT = T / N
BARRIER_UP = 110.0
BARRIER_DOWN = 90.0
M_PATHS = 50_000
BATCH_SIZE = 5_000
SEED = 99999

OPTION_KEYS = ["ec", "ep", "ac", "ap", "lc", "lp", "uic", "dop"]
OPTION_LABELS = {
    "ec": "European Call",
    "ep": "European Put",
    "ac": "Asian Call",
    "ap": "Asian Put",
    "lc": "Lookback Call",
    "lp": "Lookback Put",
    "uic": "Up-In Call",
    "dop": "Down-Out Put",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scott_pricing() -> dict:
    """Run batched Scott-model MC pricing.

    Returns
    -------
    dict
        {K: {option_key: (price, se)}}
    """
    accum = {K: {ok: [] for ok in OPTION_KEYS} for K in STRIKES}
    rng = np.random.default_rng(SEED)
    n_batches = (M_PATHS + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        actual = min(BATCH_SIZE, M_PATHS - b * BATCH_SIZE)
        # Euler-Maruyama is fully vectorised — simulate all paths at once
        S, _ = simulate_scott_batch(
            n_paths=actual,
            N=N,
            dt=DT,
            kappa=KAPPA,
            theta=THETA,
            sigma=SIGMA_SCOTT,
            rho=RHO,
            Lambda0=LAMBDA0,
            X0=X0,
            seed=int(rng.integers(0, 2**31)),
        )
        for K in STRIKES:
            pf = compute_payoffs_scott(S, K, BARRIER_UP, BARRIER_DOWN)
            for ok in OPTION_KEYS:
                accum[K][ok].append(pf[ok])

    results = {}
    for K in STRIKES:
        results[K] = {}
        for ok in OPTION_KEYS:
            all_p = np.concatenate(accum[K][ok])
            results[K][ok] = (float(all_p.mean()), float(all_p.std() / np.sqrt(len(all_p))))
    return results


def _load_rough_reference() -> pd.DataFrame:
    """Load alpha=0.999 rough prices from exp3 output (if available)."""
    ref = RESULTS_DIR / "exp3_alpha_0999_reference.csv"
    if ref.exists():
        return pd.read_csv(ref)
    return None


def _pct_error(scott_price: float, rough_price: float) -> float:
    if rough_price == 0:
        return float("nan")
    return 100.0 * (scott_price - rough_price) / rough_price


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Experiment 4: Scott Lognormal SV Benchmark")
    print(f"  sigma={SIGMA_SCOTT}, kappa={KAPPA}, theta≈{THETA:.4f}")
    print(f"  M={M_PATHS:,} paths, N={N}, T={T}y, dt={DT:.4f}")
    print("=" * 70)

    t0 = time.perf_counter()
    scott_results = _run_scott_pricing()
    print(f"\nRuntime: {time.perf_counter()-t0:.1f}s")

    # Load reference rough-model prices
    ref_df = _load_rough_reference()

    rows = []
    for K in STRIKES:
        row = {"K": K}
        for ok in OPTION_KEYS:
            p_scott, se_scott = scott_results[K][ok]
            row[f"{ok}_scott"] = p_scott
            row[f"{ok}_scott_se"] = se_scott
            # Percentage error vs alpha=0.999 rough model
            if ref_df is not None:
                ref_row = ref_df[ref_df["K"] == K]
                if len(ref_row) > 0:
                    p_rough = float(ref_row[f"{ok}_price"].values[0])
                    row[f"{ok}_pct_err"] = _pct_error(p_scott, p_rough)
                else:
                    row[f"{ok}_pct_err"] = float("nan")
            else:
                row[f"{ok}_pct_err"] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "exp4_scott_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Pretty print Table 2-style
    print("\n--- Scott Model Prices (with % error vs alpha=0.999 rough) ---")
    print(f"{'K':>5} | {'Eur C':>18} | {'Eur P':>18} | {'Asian C':>18} | "
          f"{'Asian P':>18} | {'LB C':>18} | {'LB P':>18}")
    for r in rows:
        K = r["K"]
        parts = []
        for ok in ["ec", "ep", "ac", "ap", "lc", "lp"]:
            p = r[f"{ok}_scott"]
            pe = r.get(f"{ok}_pct_err", float("nan"))
            parts.append(f"{p:7.3f} ({pe:+5.1f}%)" if not np.isnan(pe) else f"{p:7.3f}")
        print(f"{K:>5.0f} | " + " | ".join(parts))

    # Validation
    print("\n--- Parity and barrier checks ---")
    for r in rows:
        K = r["K"]
        ec, ep = r["ec_scott"], r["ep_scott"]
        parity = abs(ec - ep - (S0 - K))
        print(f"  K={K:.0f}: C-P parity err={parity:.4f}  "
              f"DOP={r['dop_scott']:.4f}")

    # -----------------------------------------------------------------------
    # Plot: Scott prices vs rough alpha=0.999
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    plot_opts = ["ec", "ep", "ac", "lc", "uic", "dop"]

    if ref_df is not None:
        for ax, ok in zip(axes_flat, plot_opts):
            scott_prices = [scott_results[K][ok][0] for K in STRIKES]
            rough_prices = [
                float(ref_df[ref_df["K"] == K][f"{ok}_price"].values[0])
                if len(ref_df[ref_df["K"] == K]) > 0 else float("nan")
                for K in STRIKES
            ]
            ax.plot(STRIKES, rough_prices, "b-o", label="Rough α=0.999", linewidth=1.5)
            ax.plot(STRIKES, scott_prices, "r--s", label="Scott (α=1)", linewidth=1.5)
            ax.set_title(OPTION_LABELS[ok])
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Price")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    else:
        for ax, ok in zip(axes_flat, plot_opts):
            scott_prices = [scott_results[K][ok][0] for K in STRIKES]
            ax.plot(STRIKES, scott_prices, "r-o", linewidth=1.5)
            ax.set_title(OPTION_LABELS[ok] + " (Scott)")
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Exp 4: Scott vs Rough α=0.999 — Option Prices (M=50k, N=500)",
        fontsize=12,
    )
    plt.tight_layout()
    out_fig = RESULTS_DIR / "exp4_scott_benchmark.png"
    plt.savefig(out_fig, dpi=120)
    plt.close()
    print(f"\nSaved: {out_fig}")


if __name__ == "__main__":
    run_experiment()
