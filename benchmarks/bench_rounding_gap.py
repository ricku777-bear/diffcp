"""
THE BENCHMARK: Can we close the rounding gap?

Compares 5 approaches to N-Queens at N=64, 128, 256:
1. Standard DiffCP (Sinkhorn + Hungarian) — baseline, fails at N≥64
2. DiffCP + Gumbel-Sinkhorn sampling — stochastic rounding
3. Iterative Rounding — fix one variable at a time, re-optimize
4. Augmented Lagrangian — learnable dual variables
5. CP-SAT — the gold standard (for timing comparison)

This is the experiment that determines if the rounding gap can be closed.
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from diffcp.solver import DiffCPSolver
from diffcp.constraints import DiagonalConflictLoss
from diffcp.rounding import gumbel_sinkhorn_sample, hungarian_round
from diffcp.iterative_solver import IterativeRoundingSolver
from diffcp.augmented_lagrangian import AugmentedLagrangianSolver


def verify(assignment):
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


def count_violations(assignment):
    N = len(assignment)
    v = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                v += 1
    return v


def solve_standard(N, device):
    """Standard Sinkhorn + Hungarian."""
    solver = DiffCPSolver(
        restarts=64, max_iters=min(3000, N * 20),
        lr=0.2, temp_start=2.0, temp_end=0.01,
        sinkhorn_iters=20, device=device,
    )
    loss_fn = DiagonalConflictLoss()
    result = solver.solve_permutation(N, loss_fn, verify)
    return {
        "method": "standard",
        "feasible": result.feasible,
        "time_ms": result.time_ms,
        "violations": count_violations(result.solution) if result.solution is not None else -1,
    }


def solve_gumbel(N, device):
    """Sinkhorn optimize + Gumbel-Sinkhorn rounding (500 samples)."""
    t0 = time.time()
    restarts = 64
    max_iters = min(3000, N * 20)

    log_alpha = torch.randn(restarts, N, N, device=device) * 0.1
    log_alpha.requires_grad_(True)
    optimizer = torch.optim.Adam([log_alpha], lr=0.2)

    for it in range(max_iters):
        optimizer.zero_grad()
        frac = it / max(max_iters - 1, 1)
        temp = 2.0 * (0.01 / 2.0) ** frac
        scaled = log_alpha / temp
        for _ in range(20):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        P = torch.exp(scaled)

        # Diagonal loss
        B_, N_, _ = P.shape
        rows = torch.arange(N_, device=device).unsqueeze(1).expand(N_, N_)
        cols = torch.arange(N_, device=device).unsqueeze(0).expand(N_, N_)
        fwd_idx = (rows - cols + (N_ - 1)).reshape(-1)
        bwd_idx = (rows + cols).reshape(-1)
        P_flat = P.reshape(B_, -1)
        nd = 2 * N_ - 1
        fwd_s = torch.zeros(B_, nd, device=device)
        fwd_s.scatter_add_(1, fwd_idx.unsqueeze(0).expand(B_, -1), P_flat)
        bwd_s = torch.zeros(B_, nd, device=device)
        bwd_s.scatter_add_(1, bwd_idx.unsqueeze(0).expand(B_, -1), P_flat)
        losses = torch.relu(fwd_s - 1).pow(2).sum(1) + torch.relu(bwd_s - 1).pow(2).sum(1)
        losses.sum().backward()
        optimizer.step()

    # Gumbel-Sinkhorn sampling on best restart
    with torch.no_grad():
        best_k = losses.argmin().item()
        best_log_alpha = log_alpha[best_k]

        samples = gumbel_sinkhorn_sample(best_log_alpha, temp=0.05, n_samples=500, sinkhorn_iters=30)
        for i in range(500):
            col = hungarian_round(samples[i])
            if verify(col):
                elapsed = (time.time() - t0) * 1000
                return {
                    "method": "gumbel",
                    "feasible": True,
                    "time_ms": elapsed,
                    "violations": 0,
                    "samples_tried": i + 1,
                }

        # Fallback
        col = hungarian_round(samples[0])
        elapsed = (time.time() - t0) * 1000
        return {
            "method": "gumbel",
            "feasible": False,
            "time_ms": elapsed,
            "violations": count_violations(col),
            "samples_tried": 500,
        }


def solve_iterative(N, device):
    """Iterative rounding solver."""
    solver = IterativeRoundingSolver(
        restarts=32,
        max_iters_initial=min(2000, N * 15),
        max_iters_reopt=min(300, max(100, N * 3)),
        lr=0.2, device=device,
        max_backtracks=10,
    )
    result = solver.solve_nqueens(N)
    return {
        "method": "iterative",
        "feasible": result.feasible,
        "time_ms": result.time_ms,
        "violations": count_violations(result.solution) if result.solution is not None else -1,
        "rounds": result.rounds,
        "backtracks": result.backtrack_count,
    }


def solve_alm(N, device):
    """Augmented Lagrangian solver."""
    solver = AugmentedLagrangianSolver(
        restarts=64,
        outer_iters=15,
        inner_iters=min(500, max(200, N * 5)),
        lr=0.2, device=device,
        rho_init=1.0, rho_mult=1.5, rho_max=50.0,
    )
    result = solver.solve_nqueens(N)
    return {
        "method": "alm",
        "feasible": result.feasible,
        "time_ms": result.time_ms,
        "violations": int(result.final_violation) if result.solution is not None else -1,
        "outer_iters": result.outer_iters,
    }


def solve_cpsat(N):
    """CP-SAT baseline."""
    from ortools.sat.python import cp_model
    t0 = time.time()
    model = cp_model.CpModel()
    q = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]
    model.add_all_different(q)
    model.add_all_different([q[i] + i for i in range(N)])
    model.add_all_different([q[i] - i for i in range(N)])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120.0
    status = solver.solve(model)
    elapsed = (time.time() - t0) * 1000
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"method": "cpsat", "feasible": True, "time_ms": elapsed, "violations": 0}
    return {"method": "cpsat", "feasible": False, "time_ms": elapsed, "violations": -1}


def run():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print(f"Device: {device}\n")

    sizes = [64, 128, 256]
    all_results = []

    for N in sizes:
        print(f"\n{'='*70}")
        print(f"  N-QUEENS: N={N}")
        print(f"{'='*70}")

        results_for_n = {"N": N}

        # CP-SAT baseline
        print(f"\n  CP-SAT...")
        r = solve_cpsat(N)
        print(f"    {r['time_ms']:.0f}ms  feas={r['feasible']}")
        results_for_n["cpsat"] = r

        # Standard
        print(f"\n  Standard (Sinkhorn + Hungarian)...")
        r = solve_standard(N, device)
        print(f"    {r['time_ms']:.0f}ms  feas={r['feasible']}  violations={r['violations']}")
        results_for_n["standard"] = r

        # Gumbel-Sinkhorn (skip for N=256 — too slow for sampling)
        if N <= 128:
            print(f"\n  Gumbel-Sinkhorn (500 samples)...")
            r = solve_gumbel(N, device)
            print(f"    {r['time_ms']:.0f}ms  feas={r['feasible']}  violations={r['violations']}  samples={r.get('samples_tried', '?')}")
            results_for_n["gumbel"] = r

        # Iterative Rounding
        print(f"\n  Iterative Rounding...")
        r = solve_iterative(N, device)
        print(f"    {r['time_ms']:.0f}ms  feas={r['feasible']}  violations={r['violations']}  rounds={r['rounds']}  backtracks={r['backtracks']}")
        results_for_n["iterative"] = r

        # Augmented Lagrangian
        print(f"\n  Augmented Lagrangian...")
        r = solve_alm(N, device)
        print(f"    {r['time_ms']:.0f}ms  feas={r['feasible']}  violations={r['violations']}  outer_iters={r.get('outer_iters', '?')}")
        results_for_n["alm"] = r

        all_results.append(results_for_n)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  ROUNDING GAP BENCHMARK: Can We Close It?")
    print(f"{'='*90}")
    print(f"{'N':>5} | {'CP-SAT':>10} | {'Standard':>10} | {'Gumbel':>10} | {'Iterative':>10} | {'ALM':>10}")
    print(f"{'':>5} | {'(baseline)':>10} | {'(fails)':>10} | {'(sample)':>10} | {'(ours)':>10} | {'(ours)':>10}")
    print("-" * 90)

    for r in all_results:
        N = r["N"]
        cpsat = f"{r['cpsat']['time_ms']:.0f}ms" if r['cpsat']['feasible'] else "FAIL"
        std = "PASS" if r['standard']['feasible'] else f"{r['standard']['violations']}v"
        gumbel = "PASS" if r.get('gumbel', {}).get('feasible') else f"{r.get('gumbel', {}).get('violations', 'N/A')}v" if 'gumbel' in r else "skip"
        iterative = "PASS" if r['iterative']['feasible'] else f"{r['iterative']['violations']}v"
        alm = "PASS" if r['alm']['feasible'] else f"{r['alm']['violations']}v"

        print(f"{N:>5} | {cpsat:>10} | {std:>10} | {gumbel:>10} | {iterative:>10} | {alm:>10}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rounding_gap.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
