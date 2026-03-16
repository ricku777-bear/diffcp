"""
Iterative Rounding Solver — The Missing Piece for Large N.

Standard approach (what fails):
  1. Optimize full N×N doubly stochastic matrix
  2. Round ALL variables at once via Hungarian
  → Fails at N≥64 because small continuous errors compound

Our approach:
  1. Optimize full N×N doubly stochastic matrix
  2. Find the MOST CONFIDENT variable (most peaked row)
  3. Fix that variable to its argmax
  4. Propagate: remove assigned row/column, adjust constraints
  5. Re-optimize the (N-1)×(N-1) reduced problem
  6. Repeat until all variables are assigned

This is polynomial: N rounds × (cost of Sinkhorn optimization on shrinking matrix).
GPU-friendly: each re-optimization is a smaller Sinkhorn problem.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class IterativeResult:
    solution: Optional[np.ndarray]
    feasible: bool
    time_ms: float
    rounds: int
    backtrack_count: int
    device: str


class IterativeRoundingSolver:
    """
    Solves N-Queens (and other permutation CSPs) by iteratively:
    1. Sinkhorn-optimize the full problem
    2. Fix the most confident variable
    3. Re-optimize remaining variables with updated constraints
    4. Repeat

    Optionally backtracks if a conflict is detected early.
    """

    def __init__(
        self,
        restarts: int = 32,
        max_iters_initial: int = 1000,  # iterations for first optimization
        max_iters_reopt: int = 200,     # iterations for re-optimization after each fix
        lr: float = 0.2,
        temp_start: float = 2.0,
        temp_end: float = 0.01,
        sinkhorn_iters: int = 20,
        device: str = "cpu",
        max_backtracks: int = 5,
    ):
        self.restarts = restarts
        self.max_iters_initial = max_iters_initial
        self.max_iters_reopt = max_iters_reopt
        self.lr = lr
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.sinkhorn_iters = sinkhorn_iters
        self.device = device
        self.max_backtracks = max_backtracks

    def _sinkhorn(self, log_alpha: torch.Tensor, temp: float, n_iters: Optional[int] = None) -> torch.Tensor:
        if n_iters is None:
            n_iters = self.sinkhorn_iters
        scaled = log_alpha / temp
        for _ in range(n_iters):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    def _optimize(self, N: int, loss_fn, max_iters: int, restarts: Optional[int] = None):
        """Run Sinkhorn optimization, return (log_alpha, P, losses)."""
        if restarts is None:
            restarts = self.restarts
        device = self.device

        log_alpha = torch.randn(restarts, N, N, device=device) * 0.1
        log_alpha.requires_grad_(True)
        optimizer = torch.optim.Adam([log_alpha], lr=self.lr)

        for it in range(max_iters):
            optimizer.zero_grad()
            frac = it / max(max_iters - 1, 1)
            temp = self.temp_start * (self.temp_end / self.temp_start) ** frac
            P = self._sinkhorn(log_alpha, temp)
            losses = loss_fn(P)
            losses.sum().backward()
            optimizer.step()

        with torch.no_grad():
            P_final = self._sinkhorn(log_alpha, self.temp_end, n_iters=40)
            final_losses = loss_fn(P_final)

        return log_alpha, P_final, final_losses

    def _confidence(self, P: torch.Tensor) -> torch.Tensor:
        """Measure confidence per row: max entry minus second-max.
        Higher = more certain about the assignment.

        Args:
            P: (B, N, N) doubly stochastic matrix

        Returns:
            (B, N) confidence scores
        """
        sorted_vals, _ = P.sort(dim=-1, descending=True)
        return sorted_vals[:, :, 0] - sorted_vals[:, :, 1]  # (B, N)

    def solve_nqueens(self, N: int) -> IterativeResult:
        """Solve N-Queens via iterative rounding.

        Args:
            N: board size

        Returns:
            IterativeResult with solution array (assignment[row] = col) and metadata
        """
        if N == 1:
            return IterativeResult(
                solution=np.array([0], dtype=int),
                feasible=True,
                time_ms=0.0,
                rounds=1,
                backtrack_count=0,
                device=self.device,
            )

        t0 = time.time()
        device = self.device

        # Track assignment: assignment[row] = col, -1 if unassigned
        assignment = np.full(N, -1, dtype=int)
        available_rows: List[int] = list(range(N))
        available_cols: List[int] = list(range(N))

        # Forbidden diagonals (forward: row-col, backward: row+col)
        forbidden_fwd: set = set()
        forbidden_bwd: set = set()

        backtrack_count = 0
        rounds = 0

        # Phase 1: Initial full optimization
        _, P_full, losses_full = self._optimize(N, self._diagonal_loss, self.max_iters_initial)

        # Pick best restart
        best_k = losses_full.argmin().item()
        P_best = P_full[best_k]  # (N, N)

        # Phase 2: Iterative rounding
        while len(available_rows) > 0:
            rounds += 1
            M = len(available_rows)

            if M == 1:
                # Last variable — just assign it
                row = available_rows[0]
                col = available_cols[0]
                assignment[row] = col
                available_rows.remove(row)
                available_cols.remove(col)
                break

            # Extract sub-matrix for available rows/cols
            row_idx = torch.tensor(available_rows, device=device)
            col_idx = torch.tensor(available_cols, device=device)

            # Find most confident assignment in current P
            sub_P = P_best[row_idx][:, col_idx]  # (M, M)

            # Build valid-move mask: exclude diagonals already occupied
            valid_mask = torch.ones(M, M, device=device, dtype=torch.bool)
            for i, row in enumerate(available_rows):
                for j, col in enumerate(available_cols):
                    if (row - col) in forbidden_fwd or (row + col) in forbidden_bwd:
                        valid_mask[i, j] = False

            # Confidence per row (max - second_max) using masked probabilities
            masked_P = sub_P * valid_mask.float()

            # Check for dead-end before proceeding
            valid_counts = valid_mask.sum(dim=1)  # (M,)
            if valid_counts.max().item() == 0:
                # Complete dead-end — no valid moves remain
                break

            # For rows with no valid moves, zero out their confidence so we skip them
            has_valid = (valid_counts > 0).float()

            sorted_masked, _ = masked_P.sort(dim=-1, descending=True)
            # confidence = top1 - top2 (or just top1 if only one valid col)
            confidence = sorted_masked[:, 0] - sorted_masked[:, 1]
            confidence = confidence * has_valid  # zero out rows with no valid moves

            # Pick the most confident row
            best_local_row = confidence.argmax().item()
            best_local_col = masked_P[best_local_row].argmax().item()

            row = available_rows[best_local_row]
            col = available_cols[best_local_col]

            # Sanity check: if still invalid (e.g., all zeros in masked row), backtrack
            if not valid_mask[best_local_row, best_local_col]:
                backtrack_count += 1
                if backtrack_count > self.max_backtracks:
                    break
                # Fall back to first row that has any valid column
                found = False
                for li in range(M):
                    if valid_counts[li].item() > 0:
                        best_local_row = li
                        masked_row = sub_P[li] * valid_mask[li].float()
                        best_local_col = masked_row.argmax().item()
                        row = available_rows[best_local_row]
                        col = available_cols[best_local_col]
                        found = True
                        break
                if not found:
                    break

            # Fix this assignment
            assignment[row] = col
            forbidden_fwd.add(row - col)
            forbidden_bwd.add(row + col)
            available_rows.remove(row)
            available_cols.remove(col)

            # Re-optimize remaining variables if more than 1 left
            if len(available_rows) > 1:
                M_new = len(available_rows)

                # Capture current state for the closure (by value via default args)
                remaining_rows = list(available_rows)
                remaining_cols = list(available_cols)
                fwd_snap = set(forbidden_fwd)
                bwd_snap = set(forbidden_bwd)

                def reduced_loss(
                    P_sub: torch.Tensor,
                    rows: List[int] = remaining_rows,
                    cols: List[int] = remaining_cols,
                    fwd: set = fwd_snap,
                    bwd: set = bwd_snap,
                ) -> torch.Tensor:
                    """Diagonal loss for reduced problem + forbidden diagonal penalty."""
                    return self._diagonal_loss_with_context(P_sub, rows, cols, fwd, bwd)

                _, P_reduced, losses_reduced = self._optimize(
                    M_new, reduced_loss, self.max_iters_reopt,
                    restarts=min(self.restarts, 16),
                )

                # Pick best restart from reduced optimization
                best_reduced_k = losses_reduced.argmin().item()
                best_reduced = P_reduced[best_reduced_k]  # (M_new, M_new)

                # Rebuild P_best: merge fixed assignments + new reduced solution
                P_new = torch.zeros(N, N, device=device)
                for r in range(N):
                    if assignment[r] >= 0:
                        P_new[r, assignment[r]] = 1.0
                for i, r in enumerate(remaining_rows):
                    for j, c in enumerate(remaining_cols):
                        P_new[r, c] = best_reduced[i, j]
                P_best = P_new

        # Verify solution
        feasible = self._verify_nqueens(assignment)
        elapsed = (time.time() - t0) * 1000

        return IterativeResult(
            solution=assignment,
            feasible=feasible,
            time_ms=elapsed,
            rounds=rounds,
            backtrack_count=backtrack_count,
            device=self.device,
        )

    def _diagonal_loss(self, P: torch.Tensor) -> torch.Tensor:
        """Standard vectorized diagonal loss for full N×N problem.

        Args:
            P: (B, N, N) doubly stochastic matrix

        Returns:
            (B,) loss per restart
        """
        B, N, _ = P.shape
        device = P.device
        rows = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
        cols = torch.arange(N, device=device).unsqueeze(0).expand(N, N)
        fwd_idx = (rows - cols + (N - 1)).reshape(-1)
        bwd_idx = (rows + cols).reshape(-1)
        P_flat = P.reshape(B, -1)
        num_diags = 2 * N - 1
        fwd_sums = torch.zeros(B, num_diags, device=device)
        fwd_sums.scatter_add_(1, fwd_idx.unsqueeze(0).expand(B, -1), P_flat)
        bwd_sums = torch.zeros(B, num_diags, device=device)
        bwd_sums.scatter_add_(1, bwd_idx.unsqueeze(0).expand(B, -1), P_flat)
        return (
            F.relu(fwd_sums - 1.0).pow(2).sum(dim=1)
            + F.relu(bwd_sums - 1.0).pow(2).sum(dim=1)
        )

    def _diagonal_loss_with_context(
        self,
        P_sub: torch.Tensor,
        rows: List[int],
        cols: List[int],
        forbidden_fwd: set,
        forbidden_bwd: set,
    ) -> torch.Tensor:
        """Diagonal loss for a sub-problem with already-fixed constraints.

        Args:
            P_sub: (B, M, M) doubly stochastic matrix for remaining variables
            rows: list of original row indices (length M)
            cols: list of original column indices (length M)
            forbidden_fwd: set of (row-col) diagonals already occupied
            forbidden_bwd: set of (row+col) anti-diagonals already occupied

        Returns:
            (B,) loss per restart
        """
        B, M, _ = P_sub.shape
        device = P_sub.device

        if M == 0:
            return torch.zeros(B, device=device)

        # --- Penalty for placing queens on already-occupied diagonals ---
        forbidden_loss = torch.zeros(B, device=device)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if (r - c) in forbidden_fwd:
                    forbidden_loss = forbidden_loss + P_sub[:, i, j].pow(2) * 100.0
                if (r + c) in forbidden_bwd:
                    forbidden_loss = forbidden_loss + P_sub[:, i, j].pow(2) * 100.0

        # --- Internal diagonal conflicts among remaining variables ---
        # Use original-scale indices so diagonal identities are preserved
        N_orig = max(max(rows), max(cols)) + 1 if rows and cols else 1
        num_fwd = 2 * N_orig - 1
        num_bwd = 2 * N_orig - 1

        r_t = torch.tensor(rows, dtype=torch.long, device=device).unsqueeze(1).expand(M, M)
        c_t = torch.tensor(cols, dtype=torch.long, device=device).unsqueeze(0).expand(M, M)

        fwd_idx = (r_t - c_t + N_orig - 1).reshape(-1)  # offset to make non-negative
        bwd_idx = (r_t + c_t).reshape(-1)

        P_flat = P_sub.reshape(B, -1)

        fwd_sums = torch.zeros(B, num_fwd, device=device)
        fwd_sums.scatter_add_(1, fwd_idx.unsqueeze(0).expand(B, -1), P_flat)
        bwd_sums = torch.zeros(B, num_bwd, device=device)
        bwd_sums.scatter_add_(1, bwd_idx.unsqueeze(0).expand(B, -1), P_flat)

        internal_loss = (
            F.relu(fwd_sums - 1.0).pow(2).sum(dim=1)
            + F.relu(bwd_sums - 1.0).pow(2).sum(dim=1)
        )

        return forbidden_loss + internal_loss

    def _verify_nqueens(self, assignment: np.ndarray) -> bool:
        """Verify that the assignment is a valid N-Queens solution.

        Args:
            assignment: array where assignment[row] = col

        Returns:
            True if valid (no two queens attack each other)
        """
        N = len(assignment)
        if -1 in assignment:
            return False
        if len(set(assignment)) != N:
            return False
        for i in range(N):
            for j in range(i + 1, N):
                if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                    return False
        return True
