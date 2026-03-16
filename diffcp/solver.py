"""Core differentiable constraint programming solver."""
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class DiffCPResult:
    """Result from the differentiable solver.

    Attributes:
        solution: discrete solution array (permutation indices for solve_permutation,
                  float array for solve_continuous), or None if no solution found.
        feasible: True if the discrete solution satisfies all hard constraints.
        time_ms: wall-clock time in milliseconds.
        continuous_loss: best constraint violation loss achieved before rounding.
        iterations: number of gradient descent iterations executed.
        device: torch device string used for computation.
        restarts: number of parallel random restarts used.
    """

    solution: Optional[np.ndarray]
    feasible: bool
    time_ms: float
    continuous_loss: float
    iterations: int
    device: str
    restarts: int

    def __repr__(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (
            f"DiffCPResult({status}, time={self.time_ms:.1f}ms, "
            f"loss={self.continuous_loss:.6f}, iters={self.iterations}, "
            f"device={self.device})"
        )


def hungarian_round(P: torch.Tensor) -> np.ndarray:
    """Round a doubly stochastic matrix to a permutation via the Hungarian algorithm.

    Args:
        P: (N, N) doubly stochastic matrix (values in [0, 1], rows/cols sum to 1).

    Returns:
        col_ind: (N,) integer array where col_ind[i] is the column assigned to row i.
    """
    cost = -P.detach().cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


class DiffCPSolver:
    """Differentiable Constraint Programming Solver.

    Solves constraint satisfaction problems by:
      1. Relaxing discrete variables to continuous (Sinkhorn for permutations,
         sigmoid-bounded for continuous variables).
      2. Defining a differentiable loss for each constraint.
      3. Optimizing via Adam with temperature annealing (permutation problems)
         or direct gradient descent (continuous problems).
      4. Rounding to a discrete solution via the Hungarian algorithm.

    Supports batch solving — ``restarts`` independent random initializations run
    in a single vectorized forward/backward pass, not sequentially.

    Args:
        restarts: number of parallel random restarts (batch size).
        max_iters: maximum gradient descent iterations.
        lr: Adam learning rate.
        temp_start: initial Sinkhorn temperature for permutation relaxation.
        temp_end: final Sinkhorn temperature (lower = sharper, closer to discrete).
        sinkhorn_iters: inner Sinkhorn normalization iterations per forward pass.
        convergence_tol: stop early if best loss drops below this threshold.
        early_stop_check_interval: how often (in iterations) to attempt rounding
            and verify feasibility during training.
        device: torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.

    Example::

        from diffcp.solver import DiffCPSolver
        from diffcp.constraints import DiagonalConflictLoss

        diag = DiagonalConflictLoss()
        solver = DiffCPSolver(restarts=64, max_iters=2000)
        result = solver.solve_permutation(
            N=8,
            loss_fn=diag,
            verify_fn=lambda a: _is_valid_queens(a),
        )
        print(result)
    """

    def __init__(
        self,
        restarts: int = 64,
        max_iters: int = 2000,
        lr: float = 0.2,
        temp_start: float = 2.0,
        temp_end: float = 0.01,
        sinkhorn_iters: int = 20,
        convergence_tol: float = 1e-6,
        early_stop_check_interval: int = 50,
        device: str = "cpu",
    ):
        self.restarts = restarts
        self.max_iters = max_iters
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.sinkhorn_iters = sinkhorn_iters
        self.convergence_tol = convergence_tol
        self.early_stop_check_interval = early_stop_check_interval
        self.device = device

    def _sinkhorn(self, log_alpha: torch.Tensor, temp: float) -> torch.Tensor:
        """Batch Sinkhorn normalization: (B, N, N) log-space -> (B, N, N) doubly stochastic.

        Operates in log-space for numerical stability. The iteration alternates
        row- and column-wise log-sum-exp normalizations.

        Args:
            log_alpha: (B, N, N) unnormalized log assignment matrix.
            temp: positive temperature scalar. Lower values produce sharper
                  (more permutation-like) output matrices.

        Returns:
            (B, N, N) doubly stochastic matrices with values in (0, 1).
        """
        scaled = log_alpha / temp
        for _ in range(self.sinkhorn_iters):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    def solve_permutation(
        self,
        N: int,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        verify_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> DiffCPResult:
        """Solve a permutation-based constraint satisfaction problem.

        The decision variable is an N×N assignment matrix P where P[i, j] = 1
        iff item i is assigned to position j. Sinkhorn normalization relaxes
        this to the Birkhoff polytope of doubly stochastic matrices.

        Gradient descent minimizes ``loss_fn(P)`` over this relaxed space.
        After convergence, the Hungarian algorithm rounds P to a permutation
        and ``verify_fn`` confirms feasibility.

        Args:
            N: problem size — the assignment matrix is (N, N).
            loss_fn: callable mapping (B, N, N) doubly stochastic batch to
                     (B,) loss tensor. Lower is better; zero = no violations.
            verify_fn: optional callable mapping an (N,) integer assignment to
                       bool. If None, every rounded solution is accepted.

        Returns:
            DiffCPResult with the best found assignment and metadata.
        """
        t0 = time.time()
        device = self.device

        # Small random init keeps the early Sinkhorn iterations well-conditioned
        log_alpha = torch.randn(self.restarts, N, N, device=device) * 0.1
        log_alpha.requires_grad_(True)

        optimizer = torch.optim.Adam([log_alpha], lr=self.lr)
        best_loss = float("inf")
        final_losses: Optional[torch.Tensor] = None

        for it in range(self.max_iters):
            optimizer.zero_grad()

            # Exponential temperature schedule: temp_start -> temp_end
            frac = it / max(self.max_iters - 1, 1)
            temp = self.temp_start * (self.temp_end / self.temp_start) ** frac

            P = self._sinkhorn(log_alpha, temp)
            losses = loss_fn(P)   # (B,)
            total_loss = losses.sum()

            total_loss.backward()
            optimizer.step()

            min_loss = losses.min().item()
            if min_loss < best_loss:
                best_loss = min_loss
            final_losses = losses.detach()

            # Periodic feasibility check — avoids redundant rounding at every step
            if (
                best_loss < self.convergence_tol
                and it % self.early_stop_check_interval == 0
            ):
                result = self._try_round_all(
                    log_alpha, final_losses, N, verify_fn, t0, it + 1
                )
                if result is not None:
                    return result

        # Final rounding pass at the end of training
        result = self._try_round_all(
            log_alpha, final_losses, N, verify_fn, t0, self.max_iters
        )
        if result is not None:
            return result

        # Best effort — return lowest-loss rounding even if not verified feasible
        with torch.no_grad():
            P_final = self._sinkhorn(log_alpha, self.temp_end)
            best_k = final_losses.argmin().item()
            assignment = hungarian_round(P_final[best_k])

        elapsed = (time.time() - t0) * 1000
        return DiffCPResult(
            solution=assignment,
            feasible=False,
            time_ms=elapsed,
            continuous_loss=best_loss,
            iterations=self.max_iters,
            device=self.device,
            restarts=self.restarts,
        )

    def _try_round_all(
        self,
        log_alpha: torch.Tensor,
        losses: torch.Tensor,
        N: int,
        verify_fn: Optional[Callable[[np.ndarray], bool]],
        t0: float,
        iters: int,
    ) -> Optional[DiffCPResult]:
        """Attempt Hungarian rounding for all restarts in loss order.

        Iterates from lowest to highest loss, returning the first assignment
        that satisfies ``verify_fn`` (or the first one if verify_fn is None).

        Args:
            log_alpha: (B, N, N) current log-assignment parameters.
            losses: (B,) per-restart losses (detached).
            N: problem size.
            verify_fn: feasibility checker, or None.
            t0: start timestamp for elapsed time computation.
            iters: iteration count for the result metadata.

        Returns:
            DiffCPResult if a feasible solution is found, else None.
        """
        with torch.no_grad():
            P = self._sinkhorn(log_alpha, self.temp_end)
            order = losses.argsort()
            for idx in order:
                assignment = hungarian_round(P[idx])
                if verify_fn is None or verify_fn(assignment):
                    elapsed = (time.time() - t0) * 1000
                    return DiffCPResult(
                        solution=assignment,
                        feasible=True,
                        time_ms=elapsed,
                        continuous_loss=losses[idx].item(),
                        iterations=iters,
                        device=self.device,
                        restarts=self.restarts,
                    )
        return None

    def solve_continuous(
        self,
        num_vars: int,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        lb: float = 0.0,
        ub: float = 1.0,
    ) -> DiffCPResult:
        """Solve a continuous-variable constraint satisfaction problem.

        Variables are projected into [lb, ub] via a sigmoid transformation.
        No rounding step is needed — the returned solution is the continuous
        optimum from the best restart.

        Typical use: scheduling problems where start times are continuous and
        constraints are precedence / no-overlap penalties.

        Args:
            num_vars: total number of continuous decision variables per instance.
            loss_fn: callable mapping (B, num_vars) batch to (B,) loss tensor.
            lb: lower bound for all variables.
            ub: upper bound for all variables.

        Returns:
            DiffCPResult where ``solution`` is a (num_vars,) float array.
            ``feasible`` is True iff ``continuous_loss < 1e-4``.
        """
        t0 = time.time()
        device = self.device

        raw = torch.randn(self.restarts, num_vars, device=device) * 0.1
        raw.requires_grad_(True)

        optimizer = torch.optim.Adam([raw], lr=self.lr)
        best_loss = float("inf")
        final_it = 0

        for it in range(self.max_iters):
            optimizer.zero_grad()

            # Project to [lb, ub] through sigmoid — keeps gradients alive at bounds
            x = lb + (ub - lb) * torch.sigmoid(raw)
            losses = loss_fn(x)   # (B,)
            total_loss = losses.sum()

            total_loss.backward()
            optimizer.step()

            min_loss = losses.min().item()
            if min_loss < best_loss:
                best_loss = min_loss

            final_it = it
            if best_loss < self.convergence_tol:
                break

        with torch.no_grad():
            x_final = lb + (ub - lb) * torch.sigmoid(raw)
            best_k = losses.argmin().item()
            solution = x_final[best_k].cpu().numpy()

        elapsed = (time.time() - t0) * 1000
        return DiffCPResult(
            solution=solution,
            feasible=best_loss < 1e-4,
            time_ms=elapsed,
            continuous_loss=best_loss,
            iterations=final_it + 1,
            device=self.device,
            restarts=self.restarts,
        )
