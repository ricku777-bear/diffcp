"""
DiffCP v2: Fully Vectorized Differentiable N-Queens Solver
Uses batch Sinkhorn + vectorized diagonal loss for real performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Result:
    solution: Optional[np.ndarray]
    feasible: bool
    time_ms: float
    loss: float
    iters: int


def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Batch Sinkhorn: (B, N, N) -> (B, N, N) doubly stochastic."""
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def diagonal_loss_fast(P: torch.Tensor) -> torch.Tensor:
    """
    Vectorized diagonal conflict loss for N-Queens.
    P: (B, N, N) batch of soft assignment matrices.
    Returns: (B,) loss per instance.

    Key insight: for each diagonal, the constraint is that at most 1 queen
    can be on it. We penalize sum > 1 via ReLU(sum - 1)^2.

    We use the trick: P[i,j] is on forward diagonal (i-j) and backward diagonal (i+j).
    We scatter P values into diagonal bins and compute per-diagonal sums.
    """
    B, N, _ = P.shape
    device = P.device

    # Create index tensors for diagonal assignments
    rows = torch.arange(N, device=device).unsqueeze(1).expand(N, N)  # (N, N)
    cols = torch.arange(N, device=device).unsqueeze(0).expand(N, N)  # (N, N)

    # Forward diagonals: i - j + (N-1) maps to [0, 2N-2]
    fwd_idx = (rows - cols + (N - 1)).reshape(-1)  # (N*N,)
    # Backward diagonals: i + j maps to [0, 2N-2]
    bwd_idx = (rows + cols).reshape(-1)  # (N*N,)

    P_flat = P.reshape(B, -1)  # (B, N*N)

    # Scatter-add to compute diagonal sums
    num_diags = 2 * N - 1

    fwd_sums = torch.zeros(B, num_diags, device=device)
    fwd_idx_expanded = fwd_idx.unsqueeze(0).expand(B, -1)
    fwd_sums.scatter_add_(1, fwd_idx_expanded, P_flat)

    bwd_sums = torch.zeros(B, num_diags, device=device)
    bwd_idx_expanded = bwd_idx.unsqueeze(0).expand(B, -1)
    bwd_sums.scatter_add_(1, bwd_idx_expanded, P_flat)

    # Penalize diagonals with sum > 1
    fwd_penalty = F.relu(fwd_sums - 1.0).pow(2).sum(dim=1)
    bwd_penalty = F.relu(bwd_sums - 1.0).pow(2).sum(dim=1)

    return fwd_penalty + bwd_penalty


def verify(assignment: np.ndarray) -> bool:
    N = len(assignment)
    if len(set(assignment)) != N:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            if abs(assignment[i] - assignment[j]) == abs(i - j):
                return False
    return True


def solve_diffcp(
    N: int,
    restarts: int = 64,
    max_iters: int = 2000,
    lr: float = 0.2,
    temp_start: float = 2.0,
    temp_end: float = 0.01,
    sinkhorn_iters: int = 20,
    device: str = "cpu",
) -> Result:
    t0 = time.time()

    log_alpha = torch.randn(restarts, N, N, device=device) * 0.1
    log_alpha.requires_grad_(True)

    optimizer = torch.optim.Adam([log_alpha], lr=lr)

    best_loss = float('inf')

    for it in range(max_iters):
        optimizer.zero_grad()

        # Temperature schedule
        frac = it / max(max_iters - 1, 1)
        temp = temp_start * (temp_end / temp_start) ** frac

        # Batch Sinkhorn
        P = sinkhorn(log_alpha / temp, n_iters=sinkhorn_iters)

        # Diagonal loss
        losses = diagonal_loss_fast(P)  # (B,)
        total_loss = losses.sum()

        total_loss.backward()
        optimizer.step()

        min_loss = losses.min().item()
        if min_loss < best_loss:
            best_loss = min_loss

        # Early stop if any restart found zero-loss
        if best_loss < 1e-6:
            # Check every 50 iters if we have a feasible rounding
            if it % 50 == 0:
                with torch.no_grad():
                    P_sharp = sinkhorn(log_alpha / temp_end, n_iters=40)
                    for k in range(restarts):
                        if losses[k].item() < 1e-4:
                            cost = -P_sharp[k].cpu().numpy()
                            _, col = linear_sum_assignment(cost)
                            if verify(col):
                                elapsed = (time.time() - t0) * 1000
                                return Result(col, True, elapsed, 0.0, it + 1)

    # Final rounding — try all restarts
    with torch.no_grad():
        P_final = sinkhorn(log_alpha / temp_end, n_iters=50)
        # Sort by loss to try best first
        order = losses.argsort()
        for idx in order:
            cost = -P_final[idx].cpu().numpy()
            _, col = linear_sum_assignment(cost)
            if verify(col):
                elapsed = (time.time() - t0) * 1000
                return Result(col, True, elapsed, best_loss, max_iters)

    # Best effort — return lowest-loss solution even if infeasible
    best_k = order[0].item()
    cost = -P_final[best_k].cpu().numpy()
    _, col = linear_sum_assignment(cost)
    elapsed = (time.time() - t0) * 1000
    return Result(col, False, elapsed, best_loss, max_iters)


def solve_cpsat(N: int) -> Result:
    from ortools.sat.python import cp_model
    t0 = time.time()
    model = cp_model.CpModel()
    q = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]
    model.add_all_different(q)
    model.add_all_different([q[i] + i for i in range(N)])
    model.add_all_different([q[i] - i for i in range(N)])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.solve(model)
    elapsed = (time.time() - t0) * 1000
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = np.array([solver.value(q[i]) for i in range(N)])
        return Result(sol, True, elapsed, 0.0, 0)
    return Result(None, False, elapsed, float('inf'), 0)


def solve_warmstart(N: int, hint: np.ndarray) -> Result:
    from ortools.sat.python import cp_model
    t0 = time.time()
    model = cp_model.CpModel()
    q = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]
    model.add_all_different(q)
    model.add_all_different([q[i] + i for i in range(N)])
    model.add_all_different([q[i] - i for i in range(N)])
    for i in range(N):
        model.add_hint(q[i], int(hint[i]))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.solve(model)
    elapsed = (time.time() - t0) * 1000
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = np.array([solver.value(q[i]) for i in range(N)])
        return Result(sol, True, elapsed, 0.0, 0)
    return Result(None, False, elapsed, float('inf'), 0)


if __name__ == "__main__":
    import json

    sizes = [8, 16, 32, 64, 128]
    trials = 3
    all_results = []

    for N in sizes:
        print(f"\n{'='*70}")
        print(f"  N-Queens: N={N}")
        print(f"{'='*70}")

        restarts = min(128, max(32, N * 2))
        max_iters = min(3000, max(800, N * 20))

        cpsat_times, diffcp_times, warm_times = [], [], []
        cpsat_feas, diffcp_feas, warm_feas = 0, 0, 0
        diffcp_sols = []

        for t in range(trials):
            # CP-SAT
            r1 = solve_cpsat(N)
            cpsat_times.append(r1.time_ms)
            if r1.feasible: cpsat_feas += 1
            print(f"  CP-SAT  t{t+1}: {r1.time_ms:>8.1f}ms  feas={r1.feasible}")

            # DiffCP
            r2 = solve_diffcp(N, restarts=restarts, max_iters=max_iters, device="cpu")
            diffcp_times.append(r2.time_ms)
            if r2.feasible: diffcp_feas += 1
            diffcp_sols.append(r2)
            print(f"  DiffCP  t{t+1}: {r2.time_ms:>8.1f}ms  feas={r2.feasible}  loss={r2.loss:.6f}  iters={r2.iters}")

            # Warm-start
            if r2.solution is not None:
                r3 = solve_warmstart(N, r2.solution)
                total = r2.time_ms + r3.time_ms
                warm_times.append(total)
                if r3.feasible: warm_feas += 1
                print(f"  Warm    t{t+1}: {total:>8.1f}ms  (diff={r2.time_ms:.0f}+cpsat={r3.time_ms:.0f})  feas={r3.feasible}")

        row = {
            "N": N,
            "cpsat_median": float(np.median(cpsat_times)),
            "cpsat_feas": f"{cpsat_feas}/{trials}",
            "diffcp_median": float(np.median(diffcp_times)),
            "diffcp_feas": f"{diffcp_feas}/{trials}",
            "warm_median": float(np.median(warm_times)) if warm_times else None,
            "warm_feas": f"{warm_feas}/{trials}",
        }
        all_results.append(row)

    # Final table
    print(f"\n{'='*90}")
    print(f"  BENCHMARK RESULTS: DiffCP vs CP-SAT vs Warm-Start (N-Queens)")
    print(f"  Platform: Apple M4 Max, CPU only (no CUDA — students would use NVIDIA GPU)")
    print(f"  Trials per size: {trials}")
    print(f"{'='*90}")
    print(f"{'N':>5} | {'CP-SAT':>12} | {'DiffCP':>12} | {'Warm-Start':>12} | "
          f"{'CP feas':>7} | {'Diff feas':>9} | {'W feas':>6}")
    print("-" * 90)
    for r in all_results:
        w = f"{r['warm_median']:.1f}ms" if r['warm_median'] else "N/A"
        print(f"{r['N']:>5} | {r['cpsat_median']:>10.1f}ms | {r['diffcp_median']:>10.1f}ms | "
              f"{w:>12} | {r['cpsat_feas']:>7} | {r['diffcp_feas']:>9} | {r['warm_feas']:>6}")

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to benchmark_results.json")
