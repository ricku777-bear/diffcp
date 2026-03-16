# Reproducing DiffCP Results

Step-by-step guide to reproduce every result in this repo.

## Setup

```bash
git clone https://github.com/ricku777-bear/diffcp.git
cd diffcp
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy ortools
```

## Hardware Used

| Machine | Chip | RAM | GPU | Used For |
|---------|------|-----|-----|----------|
| MacBook Pro | Apple M4 Max | 64GB | 40-core MPS | N=8–256, all benchmarks |
| Mac Studio | Apple M3 Ultra | 256GB | 76-core MPS | N=512+ (needs >64GB) |

Any machine with PyTorch + CUDA or MPS works. CPU-only works too, just slower.

## Reproducing the Core Results

### 1. Basic Solver (N=8–32)

```bash
python3 examples/nqueens_demo.py
```

This runs three approaches on 8-Queens and 32-Queens:
- Direct DiffCPSolver API
- CP-SAT compatible DiffModel API
- Hybrid warm-start (DiffCP → CP-SAT)

### 2. GPU vs CPU Crossover

```bash
python3 benchmarks/bench_gpu_crossover.py
```

Auto-detects MPS/CUDA. Shows where GPU overtakes CPU (crossover at N=16 on Apple Silicon). Results saved to `benchmarks/results_gpu_crossover.json`.

### 3. The Rounding Gap Benchmark

```bash
python3 benchmarks/bench_rounding_gap.py
```

Compares 5 approaches at N=64, 128, 256:
- Standard Sinkhorn + Hungarian (fails at N≥64)
- Gumbel-Sinkhorn sampling
- Iterative Rounding
- Augmented Lagrangian (ALM) — **this is the one that works**
- CP-SAT baseline

### 4. Solving N=64 with ALM (quick test)

```python
from diffcp.augmented_lagrangian import AugmentedLagrangianSolver

solver = AugmentedLagrangianSolver(
    restarts=64, outer_iters=15, inner_iters=800,
    lr=0.15, device="mps",  # or "cuda" or "cpu"
    rho_init=5.0, rho_mult=2.0, rho_max=200.0,
)
result = solver.solve_nqueens(64)
print(result)  # ALMResult(feasible=True, violations=0, time=2497ms, outer=2)
```

### 5. Solving N=128

Same as above with adjusted inner iterations:

```python
solver = AugmentedLagrangianSolver(
    restarts=64, outer_iters=15, inner_iters=800,
    lr=0.15, device="mps",
    rho_init=5.0, rho_mult=2.0, rho_max=200.0,
)
result = solver.solve_nqueens(128)
# ALMResult(feasible=True, violations=0, time=28043ms, outer=3)
```

### 6. Solving N=256

```python
solver = AugmentedLagrangianSolver(
    restarts=64, outer_iters=20, inner_iters=1500,
    lr=0.15, device="mps",
    rho_init=5.0, rho_mult=2.0, rho_max=200.0,
    sinkhorn_iters=25,
)
result = solver.solve_nqueens(256)
# ALMResult(feasible=True, violations=0, time=175945ms, outer=3)
```

### 7. N=512+ (requires >64GB RAM)

Same parameters but needs a machine with >64GB unified memory (M3 Ultra, or NVIDIA GPU with sufficient VRAM). Reduce `restarts` to 32 or 16 if memory-constrained.

## Key Parameters Explained

| Parameter | What It Does | Recommended |
|-----------|-------------|-------------|
| `restarts` | Parallel random initializations | 64 (reduce for memory) |
| `outer_iters` | ALM dual variable update rounds | 15-20 |
| `inner_iters` | Gradient descent steps per outer round | N×5 to N×6 |
| `rho_init` | Initial constraint penalty weight | 5.0 |
| `rho_mult` | Penalty multiplier per outer round | 2.0 |
| `rho_max` | Max penalty (prevents ill-conditioning) | 200.0 |
| `lr` | Adam learning rate | 0.15 |
| `temp_start` | Initial Sinkhorn temperature | 2.0 |
| `temp_end` | Final temperature (sharper = closer to permutation) | 0.01 |
| `sinkhorn_iters` | Sinkhorn normalization iterations | 20-25 |
| `device` | Compute device | "mps", "cuda", or "cpu" |

## Why ALM Works

The standard approach fails because Hungarian rounding introduces diagonal conflicts that the continuous optimization didn't account for. The Augmented Lagrangian fixes this:

1. **Outer iteration 1**: Optimize with soft penalty (ρ=5). Some diagonals still violated.
2. **Dual update**: μ increases for violated diagonals. Now those specific diagonals have higher penalty.
3. **Outer iteration 2**: Re-optimize with updated penalties. The solver pushes harder on problematic diagonals.
4. **Dual update**: μ increases again if still violated.
5. **Outer iteration 3**: Usually sufficient — the continuous solution is now shaped so that Hungarian rounding produces a feasible permutation.

The dual variables μ are the key innovation. They learn which constraints need enforcement, rather than applying uniform penalty everywhere.

## Verified Results

| N | Standard | ALM | Time | Machine |
|---|----------|-----|------|---------|
| 8 | PASS | PASS | <1s | M4 Max |
| 16 | PASS | PASS | <1s | M4 Max |
| 32 | PASS | PASS | <2s | M4 Max |
| 64 | **FAIL** (6v) | **PASS** | 2.5s | M4 Max |
| 128 | **FAIL** (8v) | **PASS** | 28s | M4 Max |
| 256 | **FAIL** | **PASS** | 176s | M4 Max |
| 512 | — | **4 violations** | 72min | M3 Ultra (256GB) |

### N=512 Status

ALM reduced 512-Queens to only 4 diagonal conflicts (out of 130,816 possible pairs — 99.997% clean) but did not fully solve it in 20 outer iterations. The dual variables were still learning. Next steps to crack it:
- Increase `outer_iters` to 40+ (dual variables need more rounds at this scale)
- Increase `rho_max` to 500+ (push harder on remaining violations)
- Try `restarts=128` for better initialization coverage
- Combine ALM with iterative rounding (fix easy variables first, ALM on the rest)
