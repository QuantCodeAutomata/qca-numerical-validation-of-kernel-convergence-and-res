# Numerical Validation of Kernel Convergence and Rescaled Long-Memory Score-Driven Scaling Limit

This repository contains six numerical experiments that validate the theoretical convergence
of a rescaled long-memory score-driven model to a rough-volatility diffusion. Specifically,
the experiments verify that the discrete kernel φ^(n) converges—under appropriate time-scaling
n → ∞—to the Mittag-Leffler kernel, and that the resulting volatility dynamics reproduce the
stylised facts of rough volatility (steep ATM skew, power-law term structure, path roughness).

---

## Experiments

| # | Script | Description |
|---|--------|-------------|
| 1 | `exp/exp1_kernel_convergence.py` | Computes L¹ and L² errors between the rescaled discrete kernel φ^(n) and the continuous Mittag-Leffler kernel for increasing n; verifies the convergence rate. |
| 2 | `exp/exp2_volatility_paths.py` | Simulates log-volatility paths Λ^(n) for α ∈ {0.501, 0.62, 0.999} and visualises the roughness structure as a function of the Hurst index H = α − ½. |
| 3 | `exp/exp3_mc_pricing.py` | Prices European, down-and-out barrier, and Asian options via Monte Carlo under the rough-volatility model for several maturities and strikes. |
| 4 | `exp/exp4_scott_benchmark.py` | Benchmarks the rough-volatility model against Scott's (1987) square-root mean-reverting stochastic-volatility model at the Brownian limit (α → 1). |
| 5 | `exp/exp5_atm_skew.py` | Measures the ATM implied-volatility skew as a function of maturity T and fits the power-law exponent, confirming the theoretical prediction skew ∝ T^(H − ½). |
| 6 | `exp/exp6_fft_benchmark.py` | Compares runtime and numerical accuracy of direct O(N²) simulation against FFT-accelerated O(N log N) simulation for a range of path lengths N. |

---

## Installation

Install the required Python packages (Python 3.9 or later recommended):

```bash
pip install numpy scipy mpmath matplotlib seaborn pandas
```

---

## Usage

Run each experiment from the repository root. Every script writes its numerical results to
`results/` and (where applicable) saves figures to `results/figures/`.

```bash
# Experiment 1 – Kernel L¹/L² error vs n
python exp/exp1_kernel_convergence.py    # → results/exp1_kernel.csv

# Experiment 2 – Volatility path simulation and roughness visualisation
python exp/exp2_volatility_paths.py      # → results/exp2_paths.csv

# Experiment 3 – Monte Carlo option pricing (European, barrier, Asian)
python exp/exp3_mc_pricing.py            # → results/exp3_pricing.csv

# Experiment 4 – Scott (1987) benchmark comparison
python exp/exp4_scott_benchmark.py       # → results/exp4_scott.csv

# Experiment 5 – ATM implied-vol skew power-law fit
python exp/exp5_atm_skew.py              # → results/exp5_skew.csv

# Experiment 6 – FFT vs direct simulation runtime benchmark
python exp/exp6_fft_benchmark.py         # → results/exp6_fft.csv
```

### Running the test suite

```bash
pytest tests/ -v
```

---

## Results

Each experiment writes a CSV to `results/` and (where applicable) a PNG to `results/figures/`.
A human-readable summary is maintained in `results/RESULTS.md`.

| Experiment | Expected output |
|------------|-----------------|
| 1 | L¹ error decays as n^{−(1−α)}; the curve confirms the theoretical convergence rate. |
| 2 | For α close to 0.5, paths exhibit rough (fractal-like) texture; α → 1 gives smooth Ornstein-Uhlenbeck-like paths. |
| 3 | European call prices match Black-Scholes at long maturities; barrier and Asian prices are discounted relative to European. |
| 4 | The rough model reproduces Scott's prices at α → 1 but generates a steeper ATM skew for smaller α. |
| 5 | ATM skew ∝ T^{H − ½} with fitted H ≈ α − ½, consistent with the rough Bergomi / RFSV literature. |
| 6 | FFT simulation achieves ≥ 10× speedup for N ≥ 10 000 with max absolute error < 10^{−8} relative to direct simulation. |

---

## Repository Structure

```
rough-volatility/
├── data/                        # Placeholder for any input data
│   └── .gitkeep
├── exp/                         # Experiment entry-point scripts
│   ├── __init__.py
│   ├── exp1_kernel_convergence.py
│   ├── exp2_volatility_paths.py
│   ├── exp3_mc_pricing.py
│   ├── exp4_scott_benchmark.py
│   ├── exp5_atm_skew.py
│   └── exp6_fft_benchmark.py
├── results/
│   ├── figures/                 # Generated PNG figures (git-ignored)
│   │   └── .gitkeep
│   ├── RESULTS.md               # Auto-generated summary (tracked)
│   └── exp*.csv                 # Per-experiment numerical outputs (tracked)
├── src/                         # Core library modules
│   ├── __init__.py
│   ├── bs_utils.py              # Black-Scholes pricing and implied-vol utilities
│   ├── fft_benchmark.py         # FFT-accelerated path simulation
│   ├── kernel.py                # Kernel weights, Mittag-Leffler function, psi/rho
│   ├── pricing.py               # Monte Carlo option pricing routines
│   └── simulation.py            # Score-driven log-volatility path simulation
├── tests/                       # pytest test suite
│   ├── __init__.py
│   ├── test_benchmark.py        # Tests for FFT vs direct benchmark
│   ├── test_kernel.py           # Tests for kernel and Mittag-Leffler utilities
│   ├── test_pricing.py          # Tests for Black-Scholes and MC pricing
│   └── test_simulation.py       # Tests for path simulation
├── .gitignore
└── README.md
```

---

## Parameter Notes

### Roughness index α

The parameter α ∈ (0.5, 1) controls the Hurst index via

```
H = α − ½  ∈  (0, 0.5)
```

Small α yields rougher (lower H) volatility paths; α → 1 recovers standard (Brownian)
stochastic volatility. The three benchmark values used throughout the experiments are:

| α | H = α − ½ | Character |
|---|-----------|-----------|
| 0.501 | ≈ 0.001 | Maximally rough |
| 0.62 | 0.12 | Empirically calibrated |
| 0.999 | ≈ 0.499 | Near-Brownian |

### The μ = 0.4 and ζ ≈ 0.4706 decomposition

The model specifies the vol-of-vol amplitude through the triplet (μ, ρ, ζ) subject to the
identification constraint

```
√(ρ² + ζ²) · μ = 0.331
```

With the fixed leverage parameter ρ = −0.681 and μ = 0.4 this uniquely determines ζ:

```
ζ = √( (0.331 / μ)² − ρ² )
  = √( (0.8275)² − (0.681)² )
  = √( 0.6848 − 0.4638 )
  = √( 0.2210 )
  ≈ 0.4706
```

Parameter roles:

- **μ = 0.4** — overall vol-of-vol scaling that enters the score-driven update step.
- **ρ = −0.681** — leverage: negative correlation between log-price innovations and
  log-volatility innovations, producing the well-known volatility smile asymmetry.
- **ζ ≈ 0.4706** — residual (orthogonal) vol-of-vol component after projecting out the
  leverage direction; determined entirely by μ and ρ via the constraint above.
- **0.331** — total volatility-of-volatility entering the drift/score equation; calibrated
  to match empirical at-the-money skew levels.

**Feasibility condition.** If μ is chosen too large, no real ζ exists:

```
μ must satisfy  μ ≤ 0.331 / |ρ| = 0.331 / 0.681 ≈ 0.486
```

For μ > 0.486 (e.g. μ = 0.7) the parameterisation is invalid and `decompose_mu_zeta`
raises a `ValueError`.

### Other fixed calibration constants

| Symbol | Value | Meaning |
|--------|-------|---------|
| θ | log(√0.3156) ≈ −0.577 | Long-run log-volatility mean (exp(θ) = √0.3156 ≈ 0.5618) |
| Λ₀ | log(√0.0392) ≈ −1.618 | Initial log-volatility (exp(Λ₀) = √0.0392 ≈ 0.1980) |
| κ | calibrated per experiment | Mean-reversion speed of the score-driven update |
| n | varied | Time-scaling parameter; kernel convergence studied as n → ∞ |
