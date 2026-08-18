#!/usr/bin/env python3
"""Experiment 6: Runtime benchmark of FFT-accelerated vs direct simulation.

Benchmarks :func:`src.fft_benchmark.simulate_direct` against
:func:`src.fft_benchmark.simulate_fft_accelerated` for the paper's
long-memory AR kernel across multiple grid sizes.

Reproduces Table 3's qualitative pattern: increasing speedup as n grows.

Environment is logged to aid reproducibility of absolute timings.
"""

import sys
import platform
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fft_benchmark import benchmark_methods

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_VALUES = [100, 250, 500, 1000, 2500, 5000]
ALPHA = 0.62       # paper's main value (Table 3)
N_REPS = 5         # timing repetitions per (method, n)
SEED = 42


def _log_environment() -> dict:
    """Return a dict of hardware/software environment metadata."""
    import numpy as np
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": platform.os.cpu_count() if hasattr(platform, "os") else "unknown",
    }
    try:
        import os
        env["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return env


def run_experiment() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    env = _log_environment()
    print("=" * 70)
    print("Experiment 6: FFT-Accelerated vs Direct Simulation Benchmark")
    print(f"  alpha={ALPHA}, n_reps={N_REPS}, n_values={N_VALUES}")
    print(f"  Platform: {env['platform']}")
    print(f"  Python:   {env['python']}, NumPy: {env['numpy_version']}")
    print("=" * 70)

    t_total = time.perf_counter()
    df = benchmark_methods(N_VALUES, alpha=ALPHA, n_reps=N_REPS, seed=SEED)
    print(f"\nBenchmark completed in {time.perf_counter()-t_total:.1f}s\n")

    # Pretty-print table
    print(f"{'n':>6} | {'Direct (s)':>12} | {'FFT (s)':>12} | "
          f"{'Speedup':>8} | {'Max|diff|':>12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{int(row['n']):>6} | "
              f"{row['mean_direct_s']:>10.4f}±{row['std_direct_s']:.4f} | "
              f"{row['mean_fft_s']:>10.4f}±{row['std_fft_s']:.4f} | "
              f"{row['speedup']:>8.2f}× | "
              f"{row['max_abs_diff']:>12.2e}")

    # Save CSV
    out_csv = RESULTS_DIR / "exp6_fft_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Save environment metadata
    env_df = pd.DataFrame([env])
    env_df.to_csv(RESULTS_DIR / "exp6_environment.csv", index=False)

    # -----------------------------------------------------------------------
    # Plot: speedup and runtimes
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ns = df["n"].values
    ax1.loglog(ns, df["mean_direct_s"], "b-o", label="Direct O(n²)", linewidth=1.5)
    ax1.loglog(ns, df["mean_fft_s"], "r-s", label="FFT-Accelerated", linewidth=1.5)
    ax1.fill_between(ns,
                     df["mean_direct_s"] - df["std_direct_s"],
                     df["mean_direct_s"] + df["std_direct_s"],
                     alpha=0.15, color="blue")
    ax1.fill_between(ns,
                     df["mean_fft_s"] - df["std_fft_s"],
                     df["mean_fft_s"] + df["std_fft_s"],
                     alpha=0.15, color="red")
    ax1.set_xlabel("Grid size n")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Runtime vs grid size")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(ns, df["speedup"], "g-^", linewidth=1.5, markersize=8)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    for x, y in zip(ns, df["speedup"]):
        ax2.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax2.set_xlabel("Grid size n")
    ax2.set_ylabel("Speedup (Direct / FFT)")
    ax2.set_title("Speedup factor vs n")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Exp 6: FFT Benchmark (α={ALPHA}, {N_REPS} reps/method)",
        fontsize=12,
    )
    plt.tight_layout()
    out_fig = RESULTS_DIR / "exp6_fft_benchmark.png"
    plt.savefig(out_fig, dpi=120)
    plt.close()
    print(f"Saved: {out_fig}")

    # Summary
    max_speedup = df["speedup"].max()
    max_n = df.loc[df["speedup"].idxmax(), "n"]
    print(f"\nMax speedup: {max_speedup:.2f}× at n={max_n:.0f}")
    print("Numerical equivalence (max|direct − fft|):")
    for _, row in df.iterrows():
        label = "✓ OK" if row["max_abs_diff"] < 1e-8 else "✗ FAIL"
        print(f"  n={int(row['n'])}: {row['max_abs_diff']:.2e}  {label}")


if __name__ == "__main__":
    run_experiment()
