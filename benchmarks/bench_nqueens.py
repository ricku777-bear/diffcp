"""N-Queens Benchmark: DiffCP vs CP-SAT vs Warm-Start

Compares three solving approaches across sizes N = 8, 16, 32, 64, 128:

  1. CP-SAT (OR-Tools)          — pure constraint propagation baseline
  2. DiffCP standalone          — Sinkhorn + diagonal loss + Hungarian rounding
  3. DiffCP → warm-start CP-SAT — DiffCP solution used as AddHint for CP-SAT

Also tracks Gumbel-Sinkhorn vs plain Hungarian rounding feasibility rate
for each problem size using the DiffCP continuous optimum.

Run:
    python benchmarks/bench_nqueens.py

Results are saved to benchmarks/results_nqueens.json.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

# Make the package importable when run directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcp.constraints import DiagonalConflictLoss
from diffcp.rounding import gumbel_sinkhorn_sample, hungarian_round
from diffcp.solver import DiffCPResult, DiffCPSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [8, 16, 32, 64, 128]
TRIALS = 3          # independent runs per (size, method)
GUMBEL_SAMPLES = 200  # stochastic rounding candidates per trial

# Solver hyperparameters tuned for feasibility across all sizes
RESTARTS = 128
MAX_ITERS = 3000
LR = 0.15
TEMP_START = 2.0
TEMP_END = 0.005
SINKHORN_ITERS = 30

# Auto-detect best available device
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helper: N-Queens feasibility check
# ---------------------------------------------------------------------------

def is_valid_queens(assignment: np.ndarray) -> bool:
    """Return True iff the assignment is a valid N-Queens solution.

    The assignment encodes the column of the queen in each row.
    Validity requires:
      - All columns distinct (AllDifferent — guaranteed by permutation encoding)
      - No two queens on the same forward or backward diagonal
    """
    N = len(assignment)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                return False
    return True


def count_diagonal_conflicts(assignment: np.ndarray) -> int:
    """Count the number of diagonal conflict pairs."""
    N = len(assignment)
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                conflicts += 1
    return conflicts


# ---------------------------------------------------------------------------
# Solver: CP-SAT baseline
# ---------------------------------------------------------------------------

def solve_cpsat(N: int, time_limit: float = 60.0) -> Dict:
    """Solve N-Queens with OR-Tools CP-SAT (no hints).

    Returns a dict with keys: feasible, time_ms, solution.
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    model = cp_model.CpModel()
    queens = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]

    # Column uniqueness (row encoding already gives each row exactly one queen)
    model.add_all_different(queens)
    # Forward diagonal uniqueness: q[i] - i must all differ
    model.add_all_different([queens[i] - i for i in range(N)])
    # Backward diagonal uniqueness: q[i] + i must all differ
    model.add_all_different([queens[i] + i for i in range(N)])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)

    elapsed = (time.time() - t0) * 1000
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution = np.array([solver.value(queens[i]) for i in range(N)]) if feasible else None

    return {"feasible": feasible, "time_ms": elapsed, "solution": solution}


# ---------------------------------------------------------------------------
# Solver: DiffCP standalone
# ---------------------------------------------------------------------------

def solve_diffcp(N: int) -> Dict:
    """Solve N-Queens with DiffCP (Sinkhorn + diagonal loss + Hungarian rounding).

    Returns a dict with keys: feasible, time_ms, continuous_loss, solution, log_alpha.
    The log_alpha is retained for the warm-start benchmark.
    """
    loss_fn = DiagonalConflictLoss()
    solver = DiffCPSolver(
        restarts=RESTARTS,
        max_iters=MAX_ITERS,
        lr=LR,
        temp_start=TEMP_START,
        temp_end=TEMP_END,
        sinkhorn_iters=SINKHORN_ITERS,
        device=DEVICE,
    )

    result: DiffCPResult = solver.solve_permutation(
        N=N,
        loss_fn=loss_fn,
        verify_fn=is_valid_queens,
    )

    # Re-run the final sinkhorn pass to get log_alpha for warm-start / Gumbel
    # The solver result already contains the best solution — we just need the
    # internal log_alpha for Gumbel sampling. Re-solve briefly to extract it.
    # For simplicity, we use the returned solution directly.
    return {
        "feasible": result.feasible,
        "time_ms": result.time_ms,
        "continuous_loss": result.continuous_loss,
        "solution": result.solution,
        "iterations": result.iterations,
    }


# ---------------------------------------------------------------------------
# Solver: DiffCP → warm-start CP-SAT
# ---------------------------------------------------------------------------

def solve_warmstart(N: int, diffcp_solution: Optional[np.ndarray], time_limit: float = 60.0) -> Dict:
    """Solve N-Queens with CP-SAT, warm-started by a DiffCP hint.

    Even an infeasible DiffCP solution is useful: it biases CP-SAT's search
    toward regions of the search space that are close to constraint satisfaction.

    Args:
        N:               Problem size.
        diffcp_solution: (N,) integer array from DiffCP (may be infeasible).
        time_limit:      CP-SAT wall-clock limit in seconds.

    Returns a dict with keys: feasible, time_ms, solution.
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    model = cp_model.CpModel()
    queens = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]

    model.add_all_different(queens)
    model.add_all_different([queens[i] - i for i in range(N)])
    model.add_all_different([queens[i] + i for i in range(N)])

    # Inject DiffCP solution as warm-start hints
    if diffcp_solution is not None:
        for i, col in enumerate(diffcp_solution):
            # Clamp to valid range in case of any numerical oddity
            hint_val = int(np.clip(col, 0, N - 1))
            model.add_hint(queens[i], hint_val)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)

    elapsed = (time.time() - t0) * 1000
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution = np.array([solver.value(queens[i]) for i in range(N)]) if feasible else None

    return {"feasible": feasible, "time_ms": elapsed, "solution": solution}


# ---------------------------------------------------------------------------
# Rounding comparison: Hungarian vs Gumbel-Sinkhorn
# ---------------------------------------------------------------------------

def compare_rounding(N: int) -> Dict:
    """Compare Hungarian and Gumbel-Sinkhorn rounding feasibility on N-Queens.

    Runs the DiffCP optimizer once to convergence, then applies each rounding
    strategy and records how often each produces a feasible solution.

    Returns dict with keys: hungarian_feasible, gumbel_feasible.
    """
    loss_fn = DiagonalConflictLoss()

    # Run a short internal solve to get the trained log_alpha
    # We tap into the solver internals by running gradient descent manually
    device = DEVICE
    log_alpha = torch.randn(RESTARTS, N, N, device=device) * 0.1
    log_alpha.requires_grad_(True)
    optimizer = torch.optim.Adam([log_alpha], lr=LR)

    def sinkhorn(la: torch.Tensor, temp: float) -> torch.Tensor:
        scaled = la / temp
        for _ in range(SINKHORN_ITERS):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    best_loss = float("inf")
    best_k = 0

    for it in range(MAX_ITERS):
        optimizer.zero_grad()
        frac = it / max(MAX_ITERS - 1, 1)
        temp = TEMP_START * (TEMP_END / TEMP_START) ** frac
        P = sinkhorn(log_alpha, temp)
        losses = loss_fn(P)
        losses.sum().backward()
        optimizer.step()
        min_loss = losses.min().item()
        if min_loss < best_loss:
            best_loss = min_loss
            best_k = int(losses.argmin().item())

    # --- Hungarian rounding ---
    with torch.no_grad():
        P_final = sinkhorn(log_alpha, TEMP_END)
        hungarian_sol = hungarian_round(P_final[best_k])
    hungarian_feasible = is_valid_queens(hungarian_sol)

    # --- Gumbel-Sinkhorn rounding ---
    # Sample many candidates and pick the first feasible one
    gumbel_feasible = False
    with torch.no_grad():
        samples = gumbel_sinkhorn_sample(
            log_alpha[best_k],
            temp=0.05,
            n_samples=GUMBEL_SAMPLES,
            sinkhorn_iters=SINKHORN_ITERS,
        )
        for i in range(GUMBEL_SAMPLES):
            candidate = hungarian_round(samples[i])
            if is_valid_queens(candidate):
                gumbel_feasible = True
                break

    return {
        "hungarian_feasible": hungarian_feasible,
        "gumbel_feasible": gumbel_feasible,
        "continuous_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks() -> Dict:
    """Run all benchmarks and return a nested results dict."""
    results = {}

    print(f"\nDiffCP N-Queens Benchmark")
    print(f"Device: {DEVICE}")
    print(f"Config: restarts={RESTARTS}, max_iters={MAX_ITERS}, trials={TRIALS}")
    print("=" * 80)

    for N in SIZES:
        print(f"\nN = {N}")
        print("-" * 60)

        cpsat_times, cpsat_feasible = [], []
        diffcp_times, diffcp_feasible, diffcp_losses = [], [], []
        warm_times, warm_feasible = [], []

        for trial in range(TRIALS):
            # --- CP-SAT ---
            r = solve_cpsat(N)
            cpsat_times.append(r["time_ms"])
            cpsat_feasible.append(r["feasible"])
            print(f"  [T{trial+1}] CP-SAT:      {'OK' if r['feasible'] else 'FAIL'} | {r['time_ms']:.1f}ms")

            # --- DiffCP standalone ---
            r = solve_diffcp(N)
            diffcp_times.append(r["time_ms"])
            diffcp_feasible.append(r["feasible"])
            diffcp_losses.append(r["continuous_loss"])
            diffcp_solution = r["solution"]
            print(f"  [T{trial+1}] DiffCP:      {'OK' if r['feasible'] else 'FAIL'} | {r['time_ms']:.1f}ms | loss={r['continuous_loss']:.4f}")

            # --- Warm-start CP-SAT ---
            r = solve_warmstart(N, diffcp_solution)
            warm_times.append(r["time_ms"])
            warm_feasible.append(r["feasible"])
            print(f"  [T{trial+1}] Warm-start:  {'OK' if r['feasible'] else 'FAIL'} | {r['time_ms']:.1f}ms")

        results[N] = {
            "cpsat": {
                "times_ms": cpsat_times,
                "mean_ms": float(np.mean(cpsat_times)),
                "feasible_rate": float(np.mean(cpsat_feasible)),
            },
            "diffcp": {
                "times_ms": diffcp_times,
                "mean_ms": float(np.mean(diffcp_times)),
                "feasible_rate": float(np.mean(diffcp_feasible)),
                "mean_continuous_loss": float(np.mean(diffcp_losses)),
            },
            "warmstart": {
                "times_ms": warm_times,
                "mean_ms": float(np.mean(warm_times)),
                "feasible_rate": float(np.mean(warm_feasible)),
            },
        }

    # --- Rounding comparison (one run per size) ---
    print("\nRounding Comparison: Hungarian vs Gumbel-Sinkhorn")
    print("-" * 60)
    rounding_results = {}
    for N in SIZES:
        r = compare_rounding(N)
        rounding_results[N] = r
        print(
            f"  N={N:3d}: Hungarian={'FEASIBLE' if r['hungarian_feasible'] else 'INFEASIBLE'} | "
            f"Gumbel={'FEASIBLE' if r['gumbel_feasible'] else 'INFEASIBLE'} | "
            f"loss={r['continuous_loss']:.4f}"
        )

    results["rounding_comparison"] = {str(N): rounding_results[N] for N in SIZES}
    return results


def print_summary_table(results: Dict) -> None:
    """Print a formatted summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'N':>6} | {'CP-SAT ms':>10} {'Feas':>6} | {'DiffCP ms':>10} {'Feas':>6} {'Loss':>10} | {'Warm ms':>10} {'Feas':>6}"
    print(header)
    print("-" * 80)
    for N in SIZES:
        r = results[N]
        cs = r["cpsat"]
        dc = r["diffcp"]
        ws = r["warmstart"]
        print(
            f"{N:>6} | {cs['mean_ms']:>10.1f} {cs['feasible_rate']:>5.0%}  | "
            f"{dc['mean_ms']:>10.1f} {dc['feasible_rate']:>5.0%}  {dc['mean_continuous_loss']:>10.4f} | "
            f"{ws['mean_ms']:>10.1f} {ws['feasible_rate']:>5.0%}"
        )
    print("=" * 80)


if __name__ == "__main__":
    results = run_benchmarks()
    print_summary_table(results)

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_nqueens.json")
    # Convert int keys to strings for JSON serialization
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
