"""DiffCP Demo: Solving N-Queens Three Ways

Demonstrates the full DiffCP API on the N-Queens problem:

  1. Direct DiffCPSolver API
     - Low-level entry point: creates the constraint loss manually, calls
       solve_permutation(), and inspects the DiffCPResult directly.

  2. CP-SAT compatible DiffModel API
     - High-level entry point: builds a declarative model with add_all_different()
       (mirroring the OR-Tools CP-SAT Python API), then calls DiffSolver.solve().

  3. Hybrid: DiffCP → warm-start CP-SAT
     - Runs DiffCP first to find a good hint, then feeds that hint to OR-Tools
       CP-SAT via AddHint. Combines DiffCP's gradient-based exploration with
       CP-SAT's complete search guarantee.

Each approach is timed. A visual board is printed for the 8-Queens case.

Run:
    python examples/nqueens_demo.py
"""

import os
import sys
import time

import numpy as np
import torch

# Make the package importable when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcp.constraints import DiagonalConflictLoss
from diffcp.model import DiffModel, DiffSolver
from diffcp.rounding import greedy_repair, gumbel_sinkhorn_sample, hungarian_round
from diffcp.solver import DiffCPResult, DiffCPSolver

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_N = 32          # size for the three-way comparison
VISUAL_N = 8         # size for the visual board
QUEEN_SYMBOL = "♛"   # Unicode queen

# Auto-detect device
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Shared utility: feasibility checker
# ---------------------------------------------------------------------------

def is_valid_queens(assignment: np.ndarray) -> bool:
    """Return True iff the assignment is a valid N-Queens solution.

    The assignment is a permutation where assignment[row] = column.
    Validity requires no two queens share a forward or backward diagonal.
    """
    N = len(assignment)
    for i in range(N):
        for j in range(i + 1, N):
            if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                return False
    return True


def count_conflicts(assignment: np.ndarray) -> int:
    """Count total diagonal conflicts in an assignment."""
    N = len(assignment)
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                conflicts += 1
    return conflicts


# ---------------------------------------------------------------------------
# Visual board printer
# ---------------------------------------------------------------------------

def print_board(assignment: np.ndarray, title: str = "") -> None:
    """Print a Unicode chess board for an N-Queens solution.

    Args:
        assignment: (N,) array where assignment[row] = queen_column.
        title:      Optional title line printed above the board.
    """
    N = len(assignment)
    if title:
        print(f"\n  {title}")

    # Top border
    print("  ┌" + "───┬" * (N - 1) + "───┐")

    for row in range(N):
        row_str = "  │"
        for col in range(N):
            if assignment[row] == col:
                # Alternate square shading based on (row + col) parity
                cell = f" {QUEEN_SYMBOL} "
            else:
                cell = " · " if (row + col) % 2 == 0 else " ░ "
            row_str += cell + "│"
        print(row_str)

        # Row separator or bottom border
        if row < N - 1:
            print("  ├" + "───┼" * (N - 1) + "───┤")
        else:
            print("  └" + "───┴" * (N - 1) + "───┘")

    conflicts = count_conflicts(assignment)
    status = "Valid solution" if conflicts == 0 else f"{conflicts} diagonal conflict(s)"
    print(f"  {status}\n")


# ---------------------------------------------------------------------------
# Approach 1: Direct DiffCPSolver API
# ---------------------------------------------------------------------------

def demo_approach_1(N: int) -> None:
    """Solve N-Queens using the low-level DiffCPSolver API.

    Steps:
      1. Instantiate DiagonalConflictLoss — a differentiable surrogate for
         the N-Queens diagonal uniqueness constraint.
      2. Create a DiffCPSolver with parallel restarts and temperature schedule.
      3. Call solve_permutation() — returns a DiffCPResult with the solution,
         feasibility flag, timing, and residual continuous loss.

    The solver maintains a batch of (B, N, N) log-assignment matrices and
    applies Sinkhorn normalization to project each one onto the Birkhoff
    polytope (set of doubly stochastic matrices). Gradient descent minimizes
    the diagonal conflict loss over this relaxed space. After convergence,
    the Hungarian algorithm rounds each restart to a hard permutation.
    """
    print("=" * 60)
    print(f"APPROACH 1: Direct DiffCPSolver API  (N={N})")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Building DiagonalConflictLoss + DiffCPSolver...")

    # The diagonal loss penalizes any two queens on the same forward or
    # backward diagonal using scatter_add over O(N²) tensor ops.
    loss_fn = DiagonalConflictLoss()

    solver = DiffCPSolver(
        restarts=128,           # 128 parallel random initializations
        max_iters=2000,         # gradient descent steps
        lr=0.15,                # Adam learning rate
        temp_start=2.0,         # initial Sinkhorn temperature (soft assignments)
        temp_end=0.005,         # final temperature (nearly hard assignments)
        sinkhorn_iters=20,      # inner Sinkhorn normalization iterations
        device=DEVICE,
    )

    print(f"  Solving {N}-Queens (128 restarts × 2000 iters)...")
    t0 = time.time()
    result: DiffCPResult = solver.solve_permutation(
        N=N,
        loss_fn=loss_fn,
        verify_fn=is_valid_queens,  # called each time a candidate is rounded
    )
    wall_ms = (time.time() - t0) * 1000

    print(f"\n  Result: {result}")
    print(f"  Wall time: {wall_ms:.1f}ms")

    if result.solution is not None:
        conflicts = count_conflicts(result.solution)
        print(f"  Diagonal conflicts (discrete): {conflicts}")

        if N == VISUAL_N:
            print_board(result.solution, title=f"Approach 1: {N}-Queens (DiffCPSolver)")

    if not result.feasible:
        print(f"  DiffCP returned INFEASIBLE — trying greedy_repair...")
        repaired = greedy_repair(
            result.solution,
            conflict_fn=count_conflicts,
            max_iters=10000,
        )
        if count_conflicts(repaired) == 0:
            print(f"  greedy_repair succeeded! Conflicts: 0")
            if N == VISUAL_N:
                print_board(repaired, title=f"Approach 1: {N}-Queens (after repair)")
        else:
            remaining = count_conflicts(repaired)
            print(f"  greedy_repair reduced to {remaining} conflict(s)")


# ---------------------------------------------------------------------------
# Approach 2: CP-SAT compatible DiffModel API
# ---------------------------------------------------------------------------

def demo_approach_2(N: int) -> None:
    """Solve N-Queens using the high-level DiffModel / DiffSolver API.

    This API mirrors the OR-Tools CP-SAT Python interface. The same model
    definition that works with CP-SAT also works with DiffSolver — enabling
    a drop-in experiment with the differentiable backend.

    Steps:
      1. Create a DiffModel.
      2. Add N integer variables queen[0..N-1] ∈ [0, N-1].
      3. Declare three AllDifferent constraints:
           - queen[i] all different           (column uniqueness)
           - queen[i] + i all different       (forward diagonal uniqueness)
           - queen[i] - i all different       (backward diagonal uniqueness)
      4. Call DiffSolver.solve(model).
      5. Read values back with solver.value() / solver.value_list().
    """
    print("=" * 60)
    print(f"APPROACH 2: CP-SAT compatible DiffModel API  (N={N})")
    print("=" * 60)

    # Build the model declaratively — same syntax as CP-SAT
    model = DiffModel()
    queens = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]

    # Standard N-Queens three-constraint encoding
    model.add_all_different(queens)
    model.add_all_different([queens[i] + i for i in range(N)])  # forward diagonals
    model.add_all_different([queens[i] - i for i in range(N)])  # backward diagonals

    solver = DiffSolver()
    solver.parameters.num_restarts = 128
    solver.parameters.max_iterations = 2000
    solver.parameters.device = DEVICE

    print(f"  Solving {N}-Queens via DiffModel...")
    t0 = time.time()
    status = solver.solve(model)
    wall_ms = (time.time() - t0) * 1000

    status_name = {0: "OPTIMAL", 1: "FEASIBLE", 2: "INFEASIBLE", 3: "UNKNOWN"}
    print(f"\n  Status: {status_name.get(status, status)}")
    print(f"  Wall time: {wall_ms:.1f}ms")
    print(f"  Continuous loss: {solver.continuous_loss:.6f}")

    if status in (DiffSolver.OPTIMAL, DiffSolver.FEASIBLE):
        board = solver.value_list(queens)
        solution = np.array(board)
        conflicts = count_conflicts(solution)
        print(f"  Solution: {board[:min(8, N)]}{'...' if N > 8 else ''}")
        print(f"  Diagonal conflicts: {conflicts}")

        if N == VISUAL_N:
            print_board(solution, title=f"Approach 2: {N}-Queens (DiffModel)")
    else:
        print(f"  No feasible solution found within budget.")


# ---------------------------------------------------------------------------
# Approach 3: Hybrid — DiffCP → warm-start CP-SAT
# ---------------------------------------------------------------------------

def demo_approach_3(N: int) -> None:
    """Solve N-Queens via DiffCP warm-starting CP-SAT.

    Strategy:
      1. Run DiffCP to find a high-quality (possibly infeasible) solution.
         The continuous optimum reflects the gradient landscape and is a
         better starting point than a random hint.
      2. Feed the DiffCP solution as AddHint to OR-Tools CP-SAT.
         CP-SAT uses the hint to guide its VSIDS/LCG search — it tries
         the hinted values first before branching, which can dramatically
         reduce the time to find a feasible solution for large N.
      3. Compare timings: pure CP-SAT vs DiffCP-warm-started CP-SAT.

    This is particularly effective when:
      - DiffCP finds a near-feasible solution (low continuous loss)
      - CP-SAT struggles with large N due to branching factor
    """
    print("=" * 60)
    print(f"APPROACH 3: DiffCP → Warm-Start CP-SAT  (N={N})")
    print("=" * 60)

    # --- Step 1: Run DiffCP ---
    print(f"  Step 1: Running DiffCP for {N}-Queens...")
    loss_fn = DiagonalConflictLoss()
    solver = DiffCPSolver(
        restarts=64,
        max_iters=1500,
        lr=0.15,
        temp_start=2.0,
        temp_end=0.005,
        sinkhorn_iters=20,
        device=DEVICE,
    )
    t0 = time.time()
    result = solver.solve_permutation(N=N, loss_fn=loss_fn, verify_fn=is_valid_queens)
    diffcp_ms = (time.time() - t0) * 1000
    print(f"  DiffCP: {'FEASIBLE' if result.feasible else 'INFEASIBLE'} | {diffcp_ms:.1f}ms | loss={result.continuous_loss:.4f}")

    # --- Step 2: Pure CP-SAT baseline ---
    from ortools.sat.python import cp_model

    print(f"\n  Step 2: Pure CP-SAT (no hints)...")
    t0 = time.time()
    cp_baseline = cp_model.CpModel()
    q_baseline = [cp_baseline.new_int_var(0, N - 1, f"q{i}") for i in range(N)]
    cp_baseline.add_all_different(q_baseline)
    cp_baseline.add_all_different([q_baseline[i] - i for i in range(N)])
    cp_baseline.add_all_different([q_baseline[i] + i for i in range(N)])
    baseline_solver = cp_model.CpSolver()
    baseline_solver.parameters.max_time_in_seconds = 30.0
    baseline_status = baseline_solver.solve(cp_baseline)
    cpsat_ms = (time.time() - t0) * 1000
    cpsat_feasible = baseline_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print(f"  CP-SAT: {'FEASIBLE' if cpsat_feasible else 'FAIL/TIMEOUT'} | {cpsat_ms:.1f}ms")

    # --- Step 3: DiffCP-warm-started CP-SAT ---
    print(f"\n  Step 3: Warm-started CP-SAT (DiffCP hint)...")
    t0 = time.time()
    cp_warm = cp_model.CpModel()
    q_warm = [cp_warm.new_int_var(0, N - 1, f"q{i}") for i in range(N)]
    cp_warm.add_all_different(q_warm)
    cp_warm.add_all_different([q_warm[i] - i for i in range(N)])
    cp_warm.add_all_different([q_warm[i] + i for i in range(N)])

    # Inject DiffCP solution as hints — clamp to valid column range
    if result.solution is not None:
        for i, col in enumerate(result.solution):
            hint_val = int(np.clip(col, 0, N - 1))
            cp_warm.add_hint(q_warm[i], hint_val)

    warm_solver = cp_model.CpSolver()
    warm_solver.parameters.max_time_in_seconds = 30.0
    warm_status = warm_solver.solve(cp_warm)
    warm_ms = (time.time() - t0) * 1000
    warm_feasible = warm_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    print(f"  Warm CP-SAT: {'FEASIBLE' if warm_feasible else 'FAIL/TIMEOUT'} | {warm_ms:.1f}ms")

    # --- Summary ---
    print(f"\n  COMPARISON (N={N})")
    print(f"  {'Method':<25} {'Time':>10} {'Status':>12}")
    print(f"  {'-'*50}")
    print(f"  {'DiffCP standalone':<25} {diffcp_ms:>9.1f}ms {'FEASIBLE' if result.feasible else 'INFEASIBLE':>12}")
    print(f"  {'CP-SAT (no hint)':<25} {cpsat_ms:>9.1f}ms {'FEASIBLE' if cpsat_feasible else 'FAIL':>12}")
    print(f"  {'DiffCP → warm CP-SAT':<25} {warm_ms:>9.1f}ms {'FEASIBLE' if warm_feasible else 'FAIL':>12}")

    if warm_feasible and cpsat_feasible and cpsat_ms > 0:
        speedup = cpsat_ms / warm_ms
        print(f"\n  Warm-start speedup vs pure CP-SAT: {speedup:.1f}x")
        print(f"  (This varies by N, device, and CP-SAT's random seed)")

    # Print board if small enough to be readable
    if warm_feasible and N == VISUAL_N:
        warm_sol = np.array([warm_solver.value(q_warm[i]) for i in range(N)])
        print_board(warm_sol, title=f"Approach 3: {N}-Queens (warm-start CP-SAT)")


# ---------------------------------------------------------------------------
# Bonus: Gumbel-Sinkhorn rounding demonstration
# ---------------------------------------------------------------------------

def demo_gumbel_rounding(N: int) -> None:
    """Show the difference between Hungarian and Gumbel-Sinkhorn rounding.

    After the DiffCP optimizer converges, the trained log_alpha encodes a
    near-permutation distribution. Two rounding strategies differ in how
    they convert this to a discrete solution:

      - Hungarian: O(N³) optimal assignment. Deterministic but may miss
        feasible solutions that are nearby in permutation space.

      - Gumbel-Sinkhorn: Stochastic. Adds Gumbel noise before Sinkhorn
        normalization, generating diverse near-permutation samples. The
        first feasible one is returned. More likely to succeed for hard
        problems where the continuous optimum is close to a feasible integer
        solution but Hungarian rounds to the wrong one.
    """
    print("=" * 60)
    print(f"BONUS: Gumbel-Sinkhorn vs Hungarian Rounding  (N={N})")
    print("=" * 60)

    # Run the internal optimizer to get a trained log_alpha
    loss_fn = DiagonalConflictLoss()
    device = DEVICE

    log_alpha = torch.randn(64, N, N, device=device) * 0.1
    log_alpha.requires_grad_(True)
    optimizer = torch.optim.Adam([log_alpha], lr=0.15)

    def sinkhorn(la, temp):
        scaled = la / temp
        for _ in range(20):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    best_k = 0
    best_loss = float("inf")
    for it in range(2000):
        optimizer.zero_grad()
        frac = it / 1999
        temp = 2.0 * (0.005 / 2.0) ** frac
        P = sinkhorn(log_alpha, temp)
        losses = loss_fn(P)
        losses.sum().backward()
        optimizer.step()
        min_l = losses.min().item()
        if min_l < best_loss:
            best_loss = min_l
            best_k = int(losses.argmin().item())

    print(f"  Continuous loss after {2000} iters: {best_loss:.4f}")

    with torch.no_grad():
        P_final = sinkhorn(log_alpha, 0.005)

        # Hungarian rounding on best restart
        hungarian_sol = hungarian_round(P_final[best_k])
        h_feasible = is_valid_queens(hungarian_sol)
        h_conflicts = count_conflicts(hungarian_sol)
        print(f"  Hungarian rounding: {'FEASIBLE' if h_feasible else 'INFEASIBLE'} | conflicts={h_conflicts}")

        # Gumbel-Sinkhorn: 200 stochastic samples
        n_samples = 200
        samples = gumbel_sinkhorn_sample(
            log_alpha[best_k],
            temp=0.05,
            n_samples=n_samples,
            sinkhorn_iters=20,
        )
        gumbel_feasible = False
        gumbel_at = None
        for i in range(n_samples):
            cand = hungarian_round(samples[i])
            if is_valid_queens(cand):
                gumbel_feasible = True
                gumbel_at = i + 1
                gumbel_sol = cand
                break

        if gumbel_feasible:
            print(f"  Gumbel rounding: FEASIBLE (sample #{gumbel_at} of {n_samples})")
        else:
            print(f"  Gumbel rounding: INFEASIBLE after {n_samples} samples")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DiffCP Demo: Solving N-Queens Three Ways")
    print("=" * 60)
    print(f"Primary demo size: {DEMO_N}-Queens")
    print(f"Visual demo size:  {VISUAL_N}-Queens (boards printed for this)")
    print(f"Device: {DEVICE}")

    # Approach 1: Low-level DiffCPSolver
    print("\n")
    demo_approach_1(DEMO_N)

    # Visual board on 8-Queens
    if DEMO_N != VISUAL_N:
        print(f"\n--- Visual board for {VISUAL_N}-Queens ---")
        demo_approach_1(VISUAL_N)

    # Approach 2: High-level DiffModel
    print("\n")
    demo_approach_2(DEMO_N)

    # Visual board on 8-Queens
    if DEMO_N != VISUAL_N:
        demo_approach_2(VISUAL_N)

    # Approach 3: Hybrid warm-start
    print("\n")
    demo_approach_3(DEMO_N)

    # Bonus: rounding comparison on 8-Queens (fast to run)
    print("\n")
    demo_gumbel_rounding(VISUAL_N)

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)
