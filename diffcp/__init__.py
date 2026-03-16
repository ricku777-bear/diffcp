"""diffcp — Differentiable Constraint Programming.

A PyTorch-native solver for constraint satisfaction problems using:
  - Sinkhorn relaxation of permutation constraints (Birkhoff polytope)
  - Differentiable surrogate losses for domain constraints
  - Batch gradient descent with temperature annealing
  - Hungarian / Gumbel-Sinkhorn rounding to discrete solutions

Quick start::

    from diffcp import DiffCPSolver, DiagonalConflictLoss

    loss_fn = DiagonalConflictLoss()
    solver = DiffCPSolver(restarts=64, max_iters=2000, device="cpu")
    result = solver.solve_permutation(N=8, loss_fn=loss_fn)
    print(result)          # DiffCPResult(FEASIBLE, time=..., loss=0.0, ...)
    print(result.solution) # [col_0, col_1, ..., col_7]

Exported names
--------------
Solver
    DiffCPSolver    — main entry point for solving permutation / continuous CSPs
    DiffCPResult    — result container with solution, feasibility, and timing

Constraints (permutation / assignment)
    AllDifferentLoss            — Sinkhorn-based AllDifferent (zero loss; projection)
    DiagonalConflictLoss        — N-Queens diagonal constraint

Constraints (continuous / mixed)
    LinearConstraintLoss        — Ax <= b or Ax = b via squared hinge
    GraphColoringLoss           — per-edge color overlap (Python loop, small graphs)
    GraphColoringLossVectorized — per-edge color overlap (tensor ops, large graphs)
    JobShopLoss                 — job-shop precedence + no-overlap constraints

Rounding utilities
    hungarian_round             — O(N³) optimal assignment rounding
    gumbel_sinkhorn_sample      — stochastic Gumbel-Sinkhorn perturbation sampling
    sample_and_select           — sample + verify + fallback rounding
    greedy_repair               — min-conflicts local search post-processor
"""

from diffcp.constraints import (
    AllDifferentLoss,
    DiagonalConflictLoss,
    GraphColoringLoss,
    GraphColoringLossVectorized,
    JobShopLoss,
    LinearConstraintLoss,
)
from diffcp.rounding import (
    greedy_repair,
    gumbel_sinkhorn_sample,
    hungarian_round,
    sample_and_select,
)
from diffcp.solver import DiffCPResult, DiffCPSolver

__all__ = [
    # Core solver
    "DiffCPSolver",
    "DiffCPResult",
    # Constraints
    "AllDifferentLoss",
    "DiagonalConflictLoss",
    "LinearConstraintLoss",
    "GraphColoringLoss",
    "GraphColoringLossVectorized",
    "JobShopLoss",
    # Rounding
    "hungarian_round",
    "gumbel_sinkhorn_sample",
    "sample_and_select",
    "greedy_repair",
]

__version__ = "0.1.0"
