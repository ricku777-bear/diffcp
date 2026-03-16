"""
Augmented Lagrangian Solver — Tighter Constraint Enforcement.

Standard penalty method: loss = objective + λ * violation²
Problem: fixed λ might be too small (violations ignored) or too large (bad conditioning)

Augmented Lagrangian: loss = objective + μᵀ violation + (ρ/2) * violation²
Key insight: μ (dual variables) are LEARNED — they adapt to push harder on
constraints that keep being violated, while relaxing on satisfied ones.

Update rule after each outer iteration:
  μ_new = μ + ρ * violation
  ρ_new = min(ρ * β, ρ_max)

This gives us both:
1. Tighter constraint satisfaction than fixed penalty
2. Better conditioning than pure penalty with large λ
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import linear_sum_assignment


@dataclass
class ALMResult:
    solution: Optional[np.ndarray]
    feasible: bool
    time_ms: float
    outer_iters: int
    final_violation: float
    device: str


class AugmentedLagrangianSolver:
    """
    Solves permutation CSPs using Augmented Lagrangian Method.

    Outer loop: update dual variables μ and penalty ρ
    Inner loop: Sinkhorn + gradient descent with current μ, ρ
    """

    def __init__(
        self,
        restarts: int = 64,
        outer_iters: int = 10,
        inner_iters: int = 300,
        lr: float = 0.2,
        temp_start: float = 2.0,
        temp_end: float = 0.01,
        sinkhorn_iters: int = 20,
        rho_init: float = 1.0,
        rho_mult: float = 2.0,
        rho_max: float = 100.0,
        device: str = "cpu",
    ):
        self.restarts = restarts
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.sinkhorn_iters = sinkhorn_iters
        self.rho_init = rho_init
        self.rho_mult = rho_mult
        self.rho_max = rho_max
        self.device = device

    def _sinkhorn(self, log_alpha: torch.Tensor, temp: float) -> torch.Tensor:
        """Apply Sinkhorn normalization in log space.

        Args:
            log_alpha: (..., N, N) unnormalized log-assignment matrix
            temp: temperature for scaling

        Returns:
            (..., N, N) doubly stochastic matrix
        """
        scaled = log_alpha / temp
        for _ in range(self.sinkhorn_iters):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    def solve_nqueens(self, N: int) -> ALMResult:
        """Solve N-Queens via Augmented Lagrangian method.

        Args:
            N: board size

        Returns:
            ALMResult with solution array (assignment[row] = col) and metadata
        """
        if N == 1:
            return ALMResult(
                solution=np.array([0], dtype=int),
                feasible=True,
                time_ms=0.0,
                outer_iters=0,
                final_violation=0.0,
                device=self.device,
            )

        t0 = time.time()
        device = self.device

        # Initialize log-assignment matrix
        log_alpha = torch.randn(self.restarts, N, N, device=device) * 0.1
        log_alpha.requires_grad_(True)

        num_diags = 2 * N - 1

        # Dual variables for forward and backward diagonal constraints; shape (B, num_diags)
        mu_fwd = torch.zeros(self.restarts, num_diags, device=device)
        mu_bwd = torch.zeros(self.restarts, num_diags, device=device)
        rho = self.rho_init

        # Precompute flat diagonal index maps (shared across all outer iterations)
        rows = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
        cols = torch.arange(N, device=device).unsqueeze(0).expand(N, N)
        fwd_idx = (rows - cols + (N - 1)).reshape(-1)   # shape (N²,)
        bwd_idx = (rows + cols).reshape(-1)              # shape (N²,)

        best_assignment: Optional[np.ndarray] = None
        best_violation: float = float("inf")

        for outer in range(self.outer_iters):
            # --- Inner optimization with current dual variables ---
            optimizer = torch.optim.Adam([log_alpha], lr=self.lr)

            for inner in range(self.inner_iters):
                optimizer.zero_grad()

                # Temperature schedule: anneal within each outer iteration,
                # starting slightly lower each outer round
                frac = inner / max(self.inner_iters - 1, 1)
                outer_frac = outer / max(self.outer_iters - 1, 1)
                t_start = self.temp_start * (1.0 - 0.5 * outer_frac)
                t_end = self.temp_end
                temp = t_start * (t_end / t_start) ** frac

                P = self._sinkhorn(log_alpha, temp)

                # Compute per-diagonal sums
                B = P.shape[0]
                P_flat = P.reshape(B, -1)

                fwd_sums = torch.zeros(B, num_diags, device=device)
                fwd_sums.scatter_add_(
                    1, fwd_idx.unsqueeze(0).expand(B, -1), P_flat
                )
                bwd_sums = torch.zeros(B, num_diags, device=device)
                bwd_sums.scatter_add_(
                    1, bwd_idx.unsqueeze(0).expand(B, -1), P_flat
                )

                # Constraint violations: g(x) = max(0, diag_sum - 1)
                fwd_violation = F.relu(fwd_sums - 1.0)
                bwd_violation = F.relu(bwd_sums - 1.0)

                # Augmented Lagrangian terms: μᵀg + (ρ/2)||g||²
                al_fwd = (
                    (mu_fwd.detach() * fwd_violation).sum(dim=1)
                    + (rho / 2.0) * fwd_violation.pow(2).sum(dim=1)
                )
                al_bwd = (
                    (mu_bwd.detach() * bwd_violation).sum(dim=1)
                    + (rho / 2.0) * bwd_violation.pow(2).sum(dim=1)
                )

                total_loss = (al_fwd + al_bwd).sum()
                total_loss.backward()
                optimizer.step()

            # --- Dual variable update (after inner optimization converges) ---
            with torch.no_grad():
                P_current = self._sinkhorn(log_alpha, self.temp_end)
                P_flat = P_current.reshape(self.restarts, -1)

                fwd_sums = torch.zeros(self.restarts, num_diags, device=device)
                fwd_sums.scatter_add_(
                    1, fwd_idx.unsqueeze(0).expand(self.restarts, -1), P_flat
                )
                bwd_sums = torch.zeros(self.restarts, num_diags, device=device)
                bwd_sums.scatter_add_(
                    1, bwd_idx.unsqueeze(0).expand(self.restarts, -1), P_flat
                )

                fwd_violation = F.relu(fwd_sums - 1.0)
                bwd_violation = F.relu(bwd_sums - 1.0)

                # μ update: μ ← μ + ρ * g
                mu_fwd = mu_fwd + rho * fwd_violation
                mu_bwd = mu_bwd + rho * bwd_violation

                # ρ update: geometric increase, capped at ρ_max
                rho = min(rho * self.rho_mult, self.rho_max)

                # --- Attempt rounding: try best restarts first ---
                violations_scalar = (
                    fwd_violation.pow(2).sum(dim=1)
                    + bwd_violation.pow(2).sum(dim=1)
                )
                order = violations_scalar.argsort()

                for idx in order:
                    cost = -P_current[idx].cpu().numpy()
                    _, col_assignment = linear_sum_assignment(cost)
                    v = self._count_violations(col_assignment)
                    if v < best_violation:
                        best_violation = float(v)
                        best_assignment = col_assignment.copy()
                    if v == 0:
                        elapsed = (time.time() - t0) * 1000
                        return ALMResult(
                            solution=col_assignment,
                            feasible=True,
                            time_ms=elapsed,
                            outer_iters=outer + 1,
                            final_violation=0.0,
                            device=self.device,
                        )

        elapsed = (time.time() - t0) * 1000
        feasible = (
            best_assignment is not None
            and self._count_violations(best_assignment) == 0
        )
        return ALMResult(
            solution=best_assignment,
            feasible=feasible,
            time_ms=elapsed,
            outer_iters=self.outer_iters,
            final_violation=best_violation,
            device=self.device,
        )

    def _count_violations(self, assignment: np.ndarray) -> int:
        """Count diagonal conflicts in an N-Queens assignment.

        Args:
            assignment: array where assignment[row] = col

        Returns:
            Number of attacking queen pairs
        """
        N = len(assignment)
        v = 0
        for i in range(N):
            for j in range(i + 1, N):
                if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                    v += 1
        return v
