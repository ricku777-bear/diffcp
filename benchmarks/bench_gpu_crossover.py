"""GPU vs CPU Crossover Analysis.

Measures where GPU (MPS/CUDA) overtakes CPU for the Sinkhorn solver.
This is the key result: at what N does the differentiable approach
become competitive when properly accelerated?
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from diffcp.solver import DiffCPSolver
from diffcp.constraints import DiagonalConflictLoss


def verify_nqueens(assignment):
    N = len(assignment)
    if len(set(assignment)) != N:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            if abs(assignment[i] - assignment[j]) == abs(i - j):
                return False
    return True


def bench_device(N, device, restarts=64, max_iters=1000):
    """Run solver on given device, return time in ms."""
    loss_fn = DiagonalConflictLoss()
    solver = DiffCPSolver(
        restarts=restarts,
        max_iters=max_iters,
        lr=0.2,
        temp_start=2.0,
        temp_end=0.01,
        sinkhorn_iters=20,
        device=device,
    )
    # Warm-up run (for GPU kernel compilation)
    _ = solver.solve_permutation(N, loss_fn, verify_nqueens)

    # Timed run
    result = solver.solve_permutation(N, loss_fn, verify_nqueens)
    return result.time_ms, result.feasible, result.iterations


def run():
    sizes = [8, 16, 32, 64, 128]
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print(f"Devices: {devices}")
    print(f"Sizes: {sizes}")
    print()

    results = []

    for N in sizes:
        restarts = min(128, max(32, N * 2))
        max_iters = min(2000, max(500, N * 15))

        row = {"N": N, "restarts": restarts, "max_iters": max_iters}

        print(f"N={N} (restarts={restarts}, iters={max_iters})")

        for device in devices:
            times = []
            feas_count = 0
            for trial in range(3):
                t, f, iters = bench_device(N, device, restarts, max_iters)
                times.append(t)
                if f:
                    feas_count += 1
                print(f"  {device:>4} trial {trial+1}: {t:>9.1f}ms  feas={f}  iters={iters}")

            row[f"{device}_median_ms"] = float(np.median(times))
            row[f"{device}_feas"] = f"{feas_count}/3"

        # Speedup
        if "mps" in devices:
            cpu_t = row["cpu_median_ms"]
            mps_t = row["mps_median_ms"]
            row["mps_speedup"] = round(cpu_t / mps_t, 2) if mps_t > 0 else 0
            print(f"  → MPS speedup: {row['mps_speedup']}x")

        results.append(row)
        print()

    # Summary table
    print(f"\n{'='*80}")
    print(f"GPU vs CPU Crossover Analysis")
    print(f"{'='*80}")

    header = f"{'N':>5} |"
    for device in devices:
        header += f" {device:>12} |"
    if "mps" in devices:
        header += f" {'MPS speedup':>12} |"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['N']:>5} |"
        for device in devices:
            line += f" {r[f'{device}_median_ms']:>10.1f}ms |"
        if "mps" in devices:
            su = r.get("mps_speedup", "N/A")
            line += f" {su:>11}x |"
        print(line)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gpu_crossover.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
