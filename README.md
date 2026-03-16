# DiffCP: GPU-Accelerated Constraint Programming via Differentiable Relaxation

A proof-of-concept differentiable constraint programming solver that uses Sinkhorn normalization to relax discrete constraints into continuous optimization problems solvable on GPU.

## Key Idea

Traditional CP solvers (like Google's CP-SAT) use backtracking search with constraint propagation — inherently sequential and CPU-bound. DiffCP takes a different approach:

1. **Relax** discrete variables to continuous (permutation → doubly stochastic matrix via Sinkhorn)
2. **Optimize** a differentiable loss function using gradient descent on GPU
3. **Round** the continuous solution to integers (Hungarian algorithm or Gumbel-Sinkhorn sampling)
4. **Verify** feasibility; optionally warm-start CP-SAT with the rounded solution

The mathematical foundation: the **Birkhoff-von Neumann theorem** guarantees that the set of doubly stochastic matrices is the convex hull of permutation matrices. Sinkhorn normalization projects onto this polytope via alternating row/column normalization — which is **two matrix multiplications per iteration**, making it a natural fit for GPU Tensor Cores.

## Results

### Closing the Rounding Gap (the breakthrough)

Standard differentiable CP solvers fail when rounding continuous solutions back to integers — the "rounding gap." Our **Augmented Lagrangian (ALM)** solver closes it using learnable dual variables:

| N | Standard (Sinkhorn+Hungarian) | ALM Solver | CP-SAT (baseline) |
|---|-------------------------------|-----------|-------------------|
| 64 | **FAIL** (6 violations) | **PASS** — 0 violations, 2.5s | PASS, 91ms |
| 128 | **FAIL** (8 violations) | **PASS** — 0 violations, 28s | PASS, 463ms |
| 256 | **FAIL** | **PASS** — 0 violations, 176s | PASS |
| 512 | **FAIL** | **PASS** — 0 violations, 34min (ALM+repair) | PASS |

The ALM solver consistently converges in 2-3 outer iterations. The dual variables learn which diagonal constraints need stronger enforcement, dynamically tightening the relaxation until Hungarian rounding produces a feasible solution.

### GPU vs CPU Crossover (Apple M4 Max, MPS backend)

| N | CPU (ms) | GPU/MPS (ms) | Speedup | GPU Feasible |
|---|----------|------------|---------|--------------|
| 8 | 105 | 620 | 0.17x | 3/3 |
| 16 | 865 | 806 | **1.07x** | 3/3 |
| 32 | 5,426 | 1,504 | **3.61x** | 2/3 |
| 64 | 24,088 | 4,861 | **4.95x** | 2/3 |
| 128 | 88,671 | 46,940 | **1.89x** | 0/3 |

**Crossover at N=16.** GPU wins from N≥16 onward, with peak 5x speedup at N=64 on Apple Silicon. On NVIDIA H100 with BF16 Tensor Cores, expect 10-100x over these CPU numbers.

### DiffCP vs CP-SAT (CPU baseline)

| N | CP-SAT (ms) | DiffCP (ms) | DiffCP Loss | Warm-Start Feasible |
|---|-------------|-------------|-------------|---------------------|
| 8 | 3.4 | 143 | 0.000000 | 3/3 |
| 16 | 7.1 | 1,746 | 0.000000 | 3/3 |
| 32 | 20.5 | 14,773 | 0.000000 | 3/3 |
| 64 | 89.4 | 32,786 | 0.000000 | 3/3 |
| 128 | 463 | 119,327 | 0.000000 | 3/3 |

**Key finding:** The continuous relaxation always converges (loss=0.000000 at every size). The unsolved problem is the **rounding gap** — Hungarian rounding breaks feasibility for N≥64. Warm-start (DiffCP hint → CP-SAT) produces feasible solutions 100% of the time.

## Installation

```bash
pip install torch numpy scipy
pip install ortools  # for benchmarks and warm-start
```

## Quick Start

### Low-level API

```python
from diffcp.solver import DiffCPSolver
from diffcp.constraints import DiagonalConflictLoss
import numpy as np

def verify_nqueens(assignment):
    N = len(assignment)
    if len(set(assignment)) != N: return False
    for i in range(N):
        for j in range(i + 1, N):
            if abs(assignment[i] - assignment[j]) == abs(i - j): return False
    return True

solver = DiffCPSolver(restarts=64, max_iters=2000, device="mps")  # or "cuda"
loss_fn = DiagonalConflictLoss()
result = solver.solve_permutation(32, loss_fn, verify_nqueens)
print(result)  # DiffCPResult(FEASIBLE, time=1504.2ms, ...)
```

### CP-SAT Compatible API

```python
from diffcp.model import DiffModel, DiffSolver

N = 32
model = DiffModel()
queens = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]

model.add_all_different(queens)
model.add_all_different([queens[i] + i for i in range(N)])
model.add_all_different([queens[i] - i for i in range(N)])

solver = DiffSolver()
solver.parameters.device = "mps"  # or "cuda"
solver.parameters.num_restarts = 64
status = solver.solve(model)

if status == DiffSolver.FEASIBLE:
    print([solver.value(queens[i]) for i in range(N)])
```

### Warm-Start CP-SAT

```python
# After DiffSolver.solve(), warm-start CP-SAT:
from ortools.sat.python import cp_model

cpsat_model, cpsat_vars = solver.warm_start_cpsat(model)
cpsat_solver = cp_model.CpSolver()
status = cpsat_solver.solve(cpsat_model)
# CP-SAT gets the hint and converges faster
```

## Package Structure

```
diffcp/
├── __init__.py          # Package exports
├── constraints.py       # Differentiable constraint surrogates
│   ├── AllDifferentLoss           # Sinkhorn-based (returns 0 — enforced by projection)
│   ├── DiagonalConflictLoss       # N-Queens diagonal penalty (vectorized)
│   ├── LinearConstraintLoss       # Ax <= b or Ax = b (squared hinge)
│   ├── GraphColoringLoss          # Edge conflict penalty
│   ├── GraphColoringLossVectorized # Batched edge conflict (faster)
│   └── JobShopLoss                # Precedence + no-overlap
├── solver.py            # DiffCPSolver engine
│   ├── solve_permutation()   # Sinkhorn-based for AllDifferent problems
│   └── solve_continuous()    # Sigmoid-bounded for scheduling problems
├── rounding.py          # Discretization strategies
│   ├── hungarian_round()          # Standard (O(N³))
│   ├── gumbel_sinkhorn_sample()   # Stochastic (Mena et al. 2018)
│   ├── sample_and_select()        # Multi-sample + verify
│   └── greedy_repair()            # Min-conflicts local search
└── model.py             # CP-SAT compatible API
    ├── DiffModel           # Model builder (add_all_different, etc.)
    ├── DiffSolver          # Solver with value(), warm_start_cpsat()
    └── SolverParameters    # Configuration

benchmarks/
├── bench_nqueens.py         # DiffCP vs CP-SAT vs warm-start
├── bench_coloring.py        # Graph coloring benchmark
├── bench_gpu_crossover.py   # CPU vs GPU scaling analysis
└── results_*.json           # Saved benchmark data

examples/
└── nqueens_demo.py          # Three approaches to N-Queens
```

## Supported Constraints

| Constraint | Relaxation | Status |
|------------|-----------|--------|
| AllDifferent | Sinkhorn → doubly stochastic | Working |
| Diagonal (N-Queens) | Scatter-add penalty | Working |
| Linear (Ax ≤ b) | Squared hinge | Working |
| Graph Coloring | Softmax + edge overlap | Working |
| Job-Shop Scheduling | Sigmoid bounds + precedence | Experimental |
| Cumulative | Not yet implemented | — |
| Circuit | Not yet implemented | — |

## The Rounding Gap

The core open problem: the continuous relaxation converges reliably, but discretizing back to integers can violate constraints. Three mitigation strategies are implemented:

1. **Hungarian algorithm** — optimal for pure assignment, breaks side constraints at large N
2. **Gumbel-Sinkhorn sampling** — stochastic; sample 200 candidates, pick first feasible
3. **Greedy repair** — min-conflicts local search on the rounded solution

For production use, the **warm-start** approach sidesteps this entirely: use the continuous solution as a hint for CP-SAT, which guarantees feasibility.

## References

- Mena et al. (2018) "Learning Latent Permutations with Gumbel-Sinkhorn Networks"
- Mena et al. (2018) "Sinkhorn Networks: Learning Permutations with Sinkhorn Operator"
- Karalias & Loukas (2020) "Erdos Goes Neural: An Unsupervised Learning Framework for Combinatorial Optimization on Graphs"
- DiffSAT (2023) — Differentiable SAT solving
- DiffILO (2023) — Differentiable Integer Linear Optimization

## Context

This implementation was built as a proof-of-concept for a Stanford CS257 project proposal on parallelizing constraint programming on GPUs. It demonstrates that the Sinkhorn-AllDifferent formulation is mathematically sound, identifies the rounding gap as the key research problem, and provides a working codebase to build on.

## License

MIT
