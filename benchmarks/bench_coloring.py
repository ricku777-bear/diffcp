"""Graph Coloring Benchmark: DiffCP vs CP-SAT

Tests graph coloring on Erdos-Renyi random graphs at:
  - N = 20 nodes
  - C = 3, 4, 5 colors
  - Edge probability p = 0.3

DiffCP approach:
  - Optimize (B, N, C) soft color assignment matrices
  - Row-normalize via softmax (each node's color distribution sums to 1)
  - Temperature anneal to sharpen assignments toward hard colors
  - Round via argmax per node
  - Verify: no adjacent nodes share the same color

CP-SAT approach:
  - Standard graph coloring model (channel encoding: x[node] in [0, C-1])

Run:
    python benchmarks/bench_coloring.py

Results are saved to benchmarks/results_coloring.json.
"""

import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make the package importable when run directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffcp.constraints import GraphColoringLossVectorized

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_NODES = 20
COLORS_LIST = [3, 4, 5]
EDGE_PROB = 0.3
TRIALS = 3           # independent graph instances per (C,) configuration
NUM_SEEDS = 3        # random seeds for reproducibility

# DiffCP hyperparameters
RESTARTS = 64
MAX_ITERS = 2000
LR = 0.15
TEMP_START = 2.0     # initial softmax temperature (soft assignments)
TEMP_END = 0.05      # final temperature (nearly hard assignments)

# Auto-detect best available device
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def generate_erdos_renyi(N: int, p: float, seed: int) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
    """Generate an Erdos-Renyi random graph G(N, p).

    Args:
        N:    Number of nodes.
        p:    Edge probability (each undirected edge exists independently with prob p).
        seed: Random seed for reproducibility.

    Returns:
        edges:      List of (u, v) undirected edge tuples (u < v).
        edge_index: (2, E) integer tensor in PyG convention (both directions).
    """
    rng = random.Random(seed)
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < p:
                edges.append((u, v))

    if len(edges) == 0:
        # Degenerate graph — add one edge to avoid empty edge_index issues
        edges.append((0, 1))

    # Build bidirectional edge index for the vectorized loss
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return edges, edge_index


def is_valid_coloring(assignment: np.ndarray, edges: List[Tuple[int, int]]) -> bool:
    """Return True iff no two adjacent nodes share the same color."""
    for u, v in edges:
        if assignment[u] == assignment[v]:
            return False
    return True


# ---------------------------------------------------------------------------
# Solver: DiffCP for graph coloring
# ---------------------------------------------------------------------------

def solve_diffcp_coloring(
    N: int,
    C: int,
    edges: List[Tuple[int, int]],
    edge_index: torch.Tensor,
) -> Dict:
    """Solve graph coloring with DiffCP.

    Maintains a (B, N, C) soft color assignment matrix where each row is a
    probability distribution over colors (via softmax). Optimizes the
    GraphColoringLossVectorized via Adam with temperature annealing to push
    assignments toward hard color choices.

    Args:
        N:          Number of nodes.
        C:          Number of colors.
        edges:      List of (u, v) tuples for feasibility verification.
        edge_index: (2, E) tensor for the vectorized loss.

    Returns:
        Dict with keys: feasible, time_ms, continuous_loss, solution, iterations.
    """
    t0 = time.time()
    device = DEVICE

    # Move edge_index to target device
    edge_index = edge_index.to(device)
    loss_fn = GraphColoringLossVectorized(edge_index, num_nodes=N)

    # Initialize: (B, N, C) raw logits — small random values keep softmax nearly uniform
    logits = torch.randn(RESTARTS, N, C, device=device) * 0.1
    logits.requires_grad_(True)

    optimizer = torch.optim.Adam([logits], lr=LR)

    best_loss = float("inf")
    best_k = 0
    final_logits = None

    for it in range(MAX_ITERS):
        optimizer.zero_grad()

        # Exponential temperature schedule: TEMP_START -> TEMP_END
        frac = it / max(MAX_ITERS - 1, 1)
        temp = TEMP_START * (TEMP_END / TEMP_START) ** frac

        # Soft color assignments: each node's distribution sums to 1
        # Divide logits by temp before softmax to sharpen over training
        P = F.softmax(logits / temp, dim=-1)  # (B, N, C)

        losses = loss_fn(P)  # (B,)
        losses.sum().backward()
        optimizer.step()

        min_loss = losses.min().item()
        if min_loss < best_loss:
            best_loss = min_loss
            best_k = int(losses.argmin().item())

        # Early exit if loss is near zero (all edges satisfied)
        if best_loss < 1e-6 and it % 50 == 0:
            # Round and verify
            with torch.no_grad():
                P_sharp = F.softmax(logits / TEMP_END, dim=-1)
                sol = P_sharp[best_k].argmax(dim=-1).cpu().numpy()
            if is_valid_coloring(sol, edges):
                elapsed = (time.time() - t0) * 1000
                return {
                    "feasible": True,
                    "time_ms": elapsed,
                    "continuous_loss": best_loss,
                    "solution": sol,
                    "iterations": it + 1,
                }

    # Final rounding: argmax over best restart
    with torch.no_grad():
        P_sharp = F.softmax(logits / TEMP_END, dim=-1)
        # Try all restarts in loss order
        losses_detached = torch.tensor(
            [loss_fn(P_sharp[k:k+1]).item() for k in range(RESTARTS)]
        )
        order = losses_detached.argsort()
        for k in order:
            sol = P_sharp[k].argmax(dim=-1).cpu().numpy()
            if is_valid_coloring(sol, edges):
                elapsed = (time.time() - t0) * 1000
                return {
                    "feasible": True,
                    "time_ms": elapsed,
                    "continuous_loss": best_loss,
                    "solution": sol,
                    "iterations": MAX_ITERS,
                }

    # Best-effort: return argmax of best-loss restart even if not feasible
    sol = P_sharp[best_k].argmax(dim=-1).cpu().numpy()
    elapsed = (time.time() - t0) * 1000
    return {
        "feasible": False,
        "time_ms": elapsed,
        "continuous_loss": best_loss,
        "solution": sol,
        "iterations": MAX_ITERS,
    }


# ---------------------------------------------------------------------------
# Solver: CP-SAT baseline
# ---------------------------------------------------------------------------

def solve_cpsat_coloring(
    N: int,
    C: int,
    edges: List[Tuple[int, int]],
    time_limit: float = 60.0,
) -> Dict:
    """Solve graph coloring with OR-Tools CP-SAT.

    Encodes each node as an integer variable in [0, C-1] and adds NotEqual
    constraints for every edge.

    Args:
        N:          Number of nodes.
        C:          Number of colors.
        edges:      List of (u, v) undirected edge tuples.
        time_limit: Wall-clock limit in seconds.

    Returns:
        Dict with keys: feasible, time_ms, solution.
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    model = cp_model.CpModel()

    # Each node gets a color in [0, C-1]
    color = [model.new_int_var(0, C - 1, f"c{i}") for i in range(N)]

    # No two adjacent nodes may share a color
    for u, v in edges:
        model.add(color[u] != color[v])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)

    elapsed = (time.time() - t0) * 1000
    feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution = np.array([solver.value(color[i]) for i in range(N)]) if feasible else None

    return {"feasible": feasible, "time_ms": elapsed, "solution": solution}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks() -> Dict:
    """Run all benchmarks and return a nested results dict."""
    results = {}

    print(f"\nDiffCP Graph Coloring Benchmark")
    print(f"Device: {DEVICE}")
    print(f"Config: N={NUM_NODES}, p={EDGE_PROB}, restarts={RESTARTS}, iters={MAX_ITERS}, trials={TRIALS}")
    print("=" * 80)

    for C in COLORS_LIST:
        print(f"\nC = {C} colors")
        print("-" * 60)

        cpsat_times, cpsat_feasible = [], []
        diffcp_times, diffcp_feasible, diffcp_losses = [], [], []
        edge_counts = []

        for trial in range(TRIALS):
            seed = trial * 13 + C * 7  # deterministic but varied seeds
            edges, edge_index = generate_erdos_renyi(NUM_NODES, EDGE_PROB, seed)
            edge_counts.append(len(edges))

            # --- CP-SAT ---
            r = solve_cpsat_coloring(NUM_NODES, C, edges)
            cpsat_times.append(r["time_ms"])
            cpsat_feasible.append(r["feasible"])
            print(f"  [T{trial+1}] CP-SAT:  {'OK' if r['feasible'] else 'FAIL'} | {r['time_ms']:.1f}ms | edges={len(edges)}")

            # --- DiffCP ---
            r = solve_diffcp_coloring(NUM_NODES, C, edges, edge_index)
            diffcp_times.append(r["time_ms"])
            diffcp_feasible.append(r["feasible"])
            diffcp_losses.append(r["continuous_loss"])
            print(f"  [T{trial+1}] DiffCP: {'OK' if r['feasible'] else 'FAIL'} | {r['time_ms']:.1f}ms | loss={r['continuous_loss']:.4f} | iters={r['iterations']}")

        results[C] = {
            "num_colors": C,
            "num_nodes": NUM_NODES,
            "edge_prob": EDGE_PROB,
            "mean_edges": float(np.mean(edge_counts)),
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
        }

    return results


def print_summary_table(results: Dict) -> None:
    """Print a formatted summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (N=20 nodes, p=0.3)")
    print("=" * 80)
    header = (
        f"{'Colors':>8} {'Edges':>7} | "
        f"{'CP-SAT ms':>10} {'Feas':>6} | "
        f"{'DiffCP ms':>10} {'Feas':>6} {'Loss':>10}"
    )
    print(header)
    print("-" * 80)
    for C in COLORS_LIST:
        r = results[C]
        cs = r["cpsat"]
        dc = r["diffcp"]
        print(
            f"{C:>8} {r['mean_edges']:>7.1f} | "
            f"{cs['mean_ms']:>10.1f} {cs['feasible_rate']:>5.0%}  | "
            f"{dc['mean_ms']:>10.1f} {dc['feasible_rate']:>5.0%}  {dc['mean_continuous_loss']:>10.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    results = run_benchmarks()
    print_summary_table(results)

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_coloring.json")
    serializable = {str(k): v for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
