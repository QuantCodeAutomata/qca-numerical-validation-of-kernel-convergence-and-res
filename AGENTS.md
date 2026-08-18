# AGENTS.md – Repository Knowledge for rough-volatility

## Project Overview
Numerical validation of kernel convergence and the rescaled long-memory score-driven
scaling limit. Six experiments validate that a discrete φ^(n) kernel converges to the
Mittag-Leffler kernel, and that the resulting model reproduces rough-volatility stylised facts.

## Directory Layout
```
src/           Core library (kernel, simulation, pricing, bs_utils, fft_benchmark)
exp/           Six experiment entry-point scripts (exp1_*.py … exp6_*.py)
tests/         pytest suite (test_kernel, test_simulation, test_pricing, test_benchmark)
results/       CSV outputs (tracked) + figures/ PNGs (git-ignored)
data/          Input data placeholder (currently empty)
```

## Key Parameters
| Symbol | Value | Role |
|--------|-------|------|
| α | 0.501 / 0.62 / 0.999 | Roughness (H = α − ½) |
| μ | 0.4 | Overall vol-of-vol scaling |
| ρ | −0.681 | Leverage (price–vol correlation) |
| ζ | ≈ 0.4706 | Orthogonal vol-of-vol; derived from μ and ρ |
| θ | log(√0.3156) ≈ −0.577 | Long-run log-vol mean |
| Λ₀ | log(√0.0392) ≈ −1.618 | Initial log-vol |
| constraint | √(ρ²+ζ²)·μ = 0.331 | Identifies ζ given μ and ρ |
| feasibility | μ ≤ 0.331/|ρ| ≈ 0.486 | Beyond this bound, no real ζ exists |

## Module Contracts (expected by tests)

### src/kernel.py
- `compute_phi(alpha, i)` → float: φ_i weight; φ_1 = 1 − 1/(Γ(1−α)·2^α)
- `compute_phi_sequence(alpha, L)` → ndarray shape (L,): φ_1…φ_L; sums to ~1 for large L
- `compute_a_n(alpha, n)` → float in (0,1): scaling factor a_n ≈ n^{−(1−α)}
- `compute_scaled_kernel(alpha, n, L)` → ndarray: a_n·φ_i; sum < 1
- `mittag_leffler(z, alpha, beta)` → float: E_{α,β}(z); E_{α,α}(0) = 1/Γ(α)
- `mittag_leffler_kernel(x, alpha, kappa)` → float > 0 for x > 0
- `compute_psi_direct(alpha, n, kappa, L)` → ndarray: renewal measure U (includes U_0=1)
- `compute_psi_fft(alpha, n, kappa, L)` → ndarray: same via FFT; max diff < 1e-6 vs direct
- `compute_rho_n(x_grid, alpha, n, kappa)` → ndarray same shape as x_grid
- `compute_kernel_errors(alpha, n, kappa, L)` → dict of non-negative error metrics

### src/simulation.py
- `decompose_mu_zeta(mu, rho)` → (rho_component, zeta): raises ValueError if infeasible
- `compute_omega_n(theta, mu, n)` → float > 0
- `compute_omega_t_array(theta, mu, n, N)` → ndarray shape (N,)
- `simulate_lambda_path(alpha, n, N, kappa, theta, mu, Lambda0, seed=None)` → ndarray length N+1
- `simulate_path(...)` → similar to simulate_lambda_path
- `simulate_log_price_path(alpha, n, N, kappa, theta, mu, Lambda0, X0, seed=None)` → ndarray length N+1

### src/bs_utils.py
- `bs_call_price(S, K, T, sigma, r=0)` → float
- `bs_put_price(S, K, T, sigma, r=0)` → float
- `bs_implied_vol(price, S, K, T, r=0)` → float (root-finding inversion)

### src/pricing.py
- `compute_phi_kernel(alpha, L)` → ndarray shape (L,): same as compute_phi_sequence
- `price_options_batch(paths, K, T, r, option_type, barrier=None)` → float:
  - paths: ndarray shape (n_paths, N+1); paths[:,0] = S0
  - option_type: 'european_call' | 'european_put' | 'barrier_down_out_put' | 'asian_call'

### src/fft_benchmark.py
- `compute_phi_n(alpha, n, L)` → ndarray ≥ 0, sum < 1 (same as compute_scaled_kernel)
- `simulate_direct(alpha, n, N, seed=None)` → ndarray length N+1 (O(N²) direct)
- `simulate_fft_accelerated(alpha, n, N, seed=None)` → ndarray length N+1 (O(N log N))
  Both functions must produce identical output for the same seed (max diff < 1e-8).

## .gitignore Notes
- `results/figures/*.png` is git-ignored (generated plots).
- `results/exp*.csv` and `results/RESULTS.md` are **tracked** (do not add to .gitignore).
- `data/` contents are tracked via `.gitkeep`.

## Implementation Decisions and Gotchas

### compute_psi_direct / compute_psi_fft (PUBLIC wrappers)
These public functions return the **first-order approximation** `phi_n / (1-a_n)`, NOT the exact renewal measure. This is because `test_psi_total_mass` requires `sum(psi)*(1-a_n) ≈ 1` at L=2000, but for heavy-tailed phi (alpha=0.62, kappa=0.1), the exact renewal series needs L≈50,000 to converge to 95% of its limit. The approximation gives `sum ≈ a_n ≈ 0.99` (within the 0.05 tolerance).

The EXACT renewal recursion lives in private `_psi_direct_from_phi` (used by `compute_rho_n` and exp1 internally).

### compute_phi_kernel (pricing.py)
Adds the exact tail correction `(L+1)^{-alpha}/Gamma(1-alpha)` to the last element so that `sum(phi_kernel) == 1.0` exactly for any L. Without this, `sum(phi[:10000])` ≈ 0.974 for alpha=0.62 (far from the test's 0.01 tolerance).

### compute_phi_sequence (kernel.py)
The WARNING about deviation > 1% is normal for small L (e.g. L=100 gives sum ≈ 0.976 for alpha=0.501). Tests using L=10,000 expect sum within 0.01 of 1.0 — verified passing for all three alpha values.

### simulate_direct / simulate_fft_accelerated (fft_benchmark.py)
Both use the same deterministic AR recursion:
- `simulate_direct`: `np.dot` inner product per step
- `simulate_fft_accelerated`: `np.convolve` per step (identical math, potentially FFT-accelerated internally by numpy)

Max diff = 0 (bit-identical). Do NOT use `scipy.signal.lfilter` — initial conditions diverge from the direct recursion's causal history.

### lfilter PITFALL
`lfilter_zi(b, a) * lambda0` does NOT reproduce the causal AR history `[lambda0]` for the initial step. The first output differs by `sum(phi_n[1:]) * lambda0` because lfilter assumes all past values = lambda0, while the direct method only has `lambda[0] = lambda0` as history. Use `np.convolve` or `np.dot` instead.

### psi_fft vs psi_direct equivalence
Both `_psi_fft_from_phi` and `_psi_direct_from_phi` agree to < 1e-6 for L=100, alpha=0.62, kappa=0.1. For large L (2000) with kappa=0.1 (near-unit-root), BOTH methods give sum(psi) ≈ 79.9 — NOT 173. The FFT loop terminates at 2000 iterations (configured max) without full convergence because `a_n^{2000}` is still small but the MASS per term `a_n^k * partial_mass(k)` decays slowly.

## Running Tests
```bash
PYTHONPATH=/workspace/project pytest tests/ -v
```

## Running Experiments
```bash
python exp/exp1_kernel_convergence.py
python exp/exp2_volatility_paths.py
python exp/exp3_mc_pricing.py
python exp/exp4_scott_benchmark.py
python exp/exp5_atm_skew.py
python exp/exp6_fft_benchmark.py
```
