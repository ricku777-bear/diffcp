## 4. Experiments

**Setup.** All experiments at $N \leq 256$ ran on an Apple M4 Max with a 10-core ARM CPU and a 40-core GPU via Metal Performance Shaders (MPS) in PyTorch 2.x. The $N = 512$ experiment ran on an Apple M3 Ultra with a 32-core CPU, 76-core MPS GPU, and 256 GB unified memory. The benchmark problem is N-Queens at $N \in \{8, 16, 32, 64, 128, 256, 512\}$. The CP-SAT baseline uses Google OR-Tools v9.x, single-threaded, with a 120-second time limit. All reported times are wall-clock medians of 3 trials. Batch size $B = 64$ for all DiffCP runs. Inner iterations $T = 1500$, Sinkhorn iterations $K = 20$, learning rate $\eta = 0.2$, initial temperature $\tau_0 = 2.0$, final temperature $\tau_T = 0.01$, penalty parameters $\rho_{\text{init}} = 5.0$, $\beta = 2.0$, $\rho_{\max} = 200$.

**Experiment 1: Rounding gap diagnosis.** Table 1 presents the core diagnostic result. The continuous loss reaches 0.000000 at every tested $N$ — the Sinkhorn-optimized doubly stochastic matrix $P$ satisfies all diagonal surrogates to numerical precision. The gap appears at discretization. Hungarian rounding produces zero violations at $N = 32$ in 2 of 3 trials but fails outright at $N \geq 64$. The violation count $V(x)$ grows sublinearly: 6 at $N = 64$, 8 at $N = 128$, and beyond 10 at $N = 512$. This sublinear growth is informative. A linear growth rate would suggest that rounding corrupts a fixed fraction of constraints; sublinear growth suggests that violations concentrate on a structurally identifiable subset — the long central diagonals. ALM closes the gap at every $N$: zero violations at $N \leq 256$ in 2--3 outer iterations, and zero violations at $N = 512$ after 10 ALM outer iterations followed by greedy repair.

**Table 1.** Rounding gap diagnosis across problem sizes. Continuous loss is the final $L(P)$ before rounding. Hungarian violations report $V(x)$ after Hungarian rounding $H(P^*)$; parenthetical fractions give the number of trials (out of 3) that achieved $V = 0$. ALM violations report $V(x)$ after the full ALM pipeline.

| $N$ | Continuous Loss | Hungarian $V(x)$ | ALM $V(x)$ | ALM Outer Iters |
|----:|:-:|:-:|:-:|:-:|
| 32 | 0.000000 | 0 (2/3 trials) | 0 | 2 |
| 64 | 0.000000 | 6 (0/3) | 0 | 2 |
| 128 | 0.000000 | 8 (0/3) | 0 | 3 |
| 256 | 0.000000 | >8 (0/3) | 0 | 3 |
| 512 | 0.000000 | >10 (0/3) | 0 (ALM + repair) | 10 + repair |

**Gumbel-Sinkhorn negative result.** To isolate whether the rounding gap is a sampling problem or a location problem, we applied Gumbel-Sinkhorn rounding at $N = 64$: 500 candidates sampled with Gumbel noise at $\tau = 0.05$ and $K = 30$ Sinkhorn iterations per sample. All 500 candidates violated at least one diagonal constraint. Zero feasible solutions out of 500 attempts. This result is the key diagnostic. Gumbel-Sinkhorn diversifies the rounding step by sampling different permutation-matrix vertices near the continuous optimum. Its universal failure at $N = 64$ demonstrates that the problem is not rounding diversity — every vertex in the neighborhood of the converged $P^*$ is infeasible. The continuous optimum sits in a region of the Birkhoff polytope where no nearby permutation matrix satisfies all diagonal constraints. ALM addresses this by moving the optimum itself, not by sampling more aggressively around a fixed optimum.

**Experiment 2: ALM convergence.** The ALM closes the rounding gap in 2--3 outer iterations at all $N \leq 256$. The mechanism is visible in the dual variable trajectory. At initialization, all $\mu^+_k = \mu^-_k = 0$. After outer iteration 1, the dual variables are nonzero only on diagonals that were violated after rounding. The long central diagonals — those with indices $k$ near 0 for forward diagonals and near $N$ for backward diagonals — accumulate the largest $\mu$ values. These diagonals span the most board positions and are the most contested. By outer iteration 2, the accumulated dual signal is sufficient to steer the continuous optimum into a region where Hungarian rounding succeeds. The geometric penalty schedule $\rho_{\text{new}} = \min(\rho \cdot 2.0, 200)$ prevents conditioning collapse: $\rho$ reaches 20 after two outer iterations, far below the $\rho_{\max} = 200$ cap. The inner problem remains well-conditioned throughout. At $N = 512$, the ALM requires 10 outer iterations to reduce $V(x)$ to a small residual ($V \leq 10$), after which greedy repair closes the remaining violations in fewer than 5000 swap attempts.

**Experiment 3: Scaling.** Table 2 compares DiffCP against CP-SAT on sequential wall-clock time. DiffCP with standard (non-ALM) penalty optimization is slower than CP-SAT at every tested $N$. At $N = 64$, CP-SAT solves in 89 ms; DiffCP standard takes 32.8 s — a factor of 367$\times$ slower — and fails to produce a feasible solution. With ALM (Table 3), DiffCP achieves feasibility but remains 28$\times$ slower than CP-SAT at $N = 64$ (2.5 s vs. 89 ms). DiffCP is not competitive with CP-SAT on sequential hardware at any tested scale. We state this without qualification. The contribution is not speed on sequential hardware. It is the ALM rounding result — the demonstration that the gap closes reliably — and the GPU pathway that this enables.

**Table 2.** DiffCP standard (fixed penalty, no ALM) vs. CP-SAT. Feasibility indicates how many of 3 trials produced $V(x) = 0$. Warm-start feasibility indicates how many trials produced a feasible CP-SAT solution when seeded with the DiffCP hint.

| $N$ | CP-SAT (ms) | DiffCP Standard (ms) | DiffCP Feas | Warm-Start Feas |
|----:|:-:|:-:|:-:|:-:|
| 8 | 3.4 | 143 | 3/3 | 3/3 |
| 16 | 7.1 | 1,746 | 3/3 | 3/3 |
| 32 | 20.5 | 14,773 | 1/3 | 3/3 |
| 64 | 89.4 | 32,786 | 0/3 | 3/3 |
| 128 | 463 | 119,327 | 0/3 | 3/3 |

**Table 3.** ALM results on MPS GPU. Time includes all outer iterations and Hungarian rounding. The $N = 512$ entry includes greedy repair after ALM.

| $N$ | ALM Time | Outer Iters | Device |
|----:|:-:|:-:|:-:|
| 64 | 2.5 s | 2 | M4 Max MPS |
| 128 | 28 s | 3 | M4 Max MPS |
| 256 | 176 s | 3 | M4 Max MPS |
| 512 | 34 min | 10 + repair | M3 Ultra MPS |

**Experiment 4: GPU vs. CPU crossover.** Table 4 reports the inner-loop execution time for a single ALM outer iteration on CPU versus MPS GPU. The crossover occurs at $N = 16$, where MPS matches CPU execution time. At $N = 64$, MPS is $4.95\times$ faster than CPU. At $N = 8$, MPS is $0.17\times$ — a $5.9\times$ slowdown — due to MPS kernel launch overhead dominating the small matrix operations. At $N = 128$, the speedup drops to $1.89\times$, a regression from the $N = 64$ peak. Profiling attributes this to MPS memory bandwidth saturation: the $(64, 128, 128)$ batch tensor is 8.4 MB, and the 20 Sinkhorn iterations per Adam step generate sufficient memory traffic to saturate the M4 Max's unified memory bus. The peak speedup at $N = 64$ represents the sweet spot where the $(64, 64, 64)$ batch tensor (2.1 MB) fits comfortably in the GPU's working set and the arithmetic intensity of Sinkhorn normalization is high enough to amortize kernel launch overhead.

**Table 4.** Inner-loop execution time for one ALM outer iteration, CPU vs. MPS GPU, with $B = 64$ restarts.

| $N$ | CPU (ms) | MPS (ms) | Speedup |
|----:|:-:|:-:|:-:|
| 8 | 105 | 620 | 0.17$\times$ |
| 16 | 865 | 806 | 1.07$\times$ |
| 32 | 5,426 | 1,504 | 3.61$\times$ |
| 64 | 24,088 | 4,861 | 4.95$\times$ |
| 128 | 88,671 | 46,940 | 1.89$\times$ |

**Ablation: penalty schedule.** We varied the penalty multiplier $\beta \in \{1.5, 2.0, 3.0\}$ at $N = 64$ with all other parameters fixed. At $\beta = 2.0$ (the default), the ALM closes the gap in 2 outer iterations. At $\beta = 1.5$, convergence requires 3 outer iterations but the inner optimization is more stable — the final continuous loss is lower before rounding, and the dual variable magnitudes are smaller. At $\beta = 3.0$, the ALM also closes in 2 outer iterations, but the inner optimization shows larger gradient variance in the final 100 steps of each inner loop, consistent with the larger $\rho$ degrading conditioning. The $\rho_{\max}$ cap has no effect in practice at $N = 64$: with $\beta = 2.0$ and 2 outer iterations, $\rho$ reaches only 20, far below the cap of 200. The cap serves as a safety bound for larger $N$ where more outer iterations are needed.

**Limitations.** Three limitations bound the generality of these results. First, all experiments use N-Queens exclusively. N-Queens is a permutation CSP with a specific diagonal structure. Extending DiffCP to graph coloring requires dual variables indexed by edges rather than diagonals. Job-shop scheduling requires dual variables indexed by machine-time intervals. These constraint-specific dual variable designs are not validated. Second, DiffCP is not competitive with CP-SAT on wall-clock time at any tested scale. The $4.95\times$ GPU speedup applies to the Sinkhorn inner loop only; the full pipeline — including ALM outer iterations and Hungarian rounding on CPU — is 28$\times$ slower than CP-SAT at $N = 64$. Third, memory scales as $O(B \cdot N^2)$: at $N = 512$ with $B = 64$, the batch tensor occupies 67 MB in float32. At $N = 1024$, a batch of $B = 64$ would require 268 MB, necessitating either a reduction in $B$ or a switch to bfloat16 precision. The bfloat16 path has not been tested and may affect Sinkhorn convergence at low $\tau$.


## 5. Discussion

**What the dual variables learn.** The dual variable magnitudes after ALM convergence reveal the violation structure of N-Queens. The long central diagonals — forward diagonals with $|k| \leq N/4$ and backward diagonals with $|k - N| \leq N/4$ — accumulate $\mu$ values 3--5$\times$ larger than the short corner diagonals. These central diagonals span the most board positions and intersect the most rows, making them the hardest to satisfy simultaneously. The ALM discovers this geometry without any prior encoding: the dual update rule $\mu_k \leftarrow \mu_k + \rho \cdot g_k(P^*)$ simply accumulates violation history, and the violation history reflects the problem structure. This observation suggests a practical acceleration: initialize $\mu$ with problem-specific priors proportional to diagonal length rather than at zero. Such warm initialization would reduce the number of outer iterations needed by front-loading the structural signal that ALM currently discovers empirically.

**Warm-start as a lower bound on utility.** Even when ALM does not close the rounding gap entirely — as at $N = 512$ before repair — the continuous solution $P^*$ contains structural information. The argmax hint $\arg\max_j P^*_{ij}$ fed to CP-SAT produces a feasible solution in 100% of trials across all 15 runs at $N \in \{8, 16, 32, 64, 128\}$. This 100% rate includes $N = 64$ and $N = 128$, where standalone Hungarian rounding fails in every trial. The differentiable solver, despite being slower and less reliable than CP-SAT alone, generates initialization quality that CP-SAT exploits. This is an underexplored direction in the differentiable combinatorial optimization literature: rather than competing with exact solvers, differentiable methods can serve as heuristic generators that provide high-quality starting points. The cost of the hint construction is $O(N)$ — a single argmax per row — and is negligible relative to the solve time of either method.

**GPU trajectory.** The $4.95\times$ MPS speedup at $N = 64$ is a data point, not a ceiling. Apple MPS on the M4 Max delivers approximately 7 TFLOPS of float32 throughput. An NVIDIA H100 with BF16 Tensor Cores delivers approximately 300 TFLOPS — a $43\times$ ratio in peak arithmetic throughput. The Sinkhorn inner loop is a sequence of batched matrix normalizations with high arithmetic intensity and regular memory access patterns. It maps directly to Tensor Core execution. A conservative projection: the inner loop on H100 would run $30$--$40\times$ faster than on M4 Max MPS, shifting the crossover with CP-SAT from $N = 16$ to a substantially larger $N$. The ALM outer loop is embarrassingly parallel across restarts — each of the $B$ restarts carries independent dual variables and requires no inter-restart communication. On multi-GPU hardware, the batch dimension $B$ can scale linearly with GPU count. These projections are not validated experimentally. They indicate the direction and approximate magnitude of the scaling opportunity.


## 6. Conclusion

The rounding gap in Sinkhorn-based differentiable constraint programming closes under Augmented Lagrangian dual updates with per-diagonal granularity. We diagnosed the gap as a location problem — the continuous optimum sits in a region of the Birkhoff polytope where all nearby permutation-matrix vertices violate at least one diagonal constraint — and showed that Gumbel-Sinkhorn sampling (500 candidates, $\tau = 0.05$) cannot close it. ALM solved N-Queens at $N = 64$ in 2.5 s (2 outer iterations), $N = 128$ in 28 s (3 outer iterations), $N = 256$ in 176 s (3 outer iterations), and $N = 512$ in 34 min (10 outer iterations + greedy repair), achieving zero violations at every scale. GPU execution on Apple MPS accelerated the Sinkhorn inner loop by $4.95\times$ at $N = 64$, with crossover at $N = 16$. The warm-start pathway — feeding the continuous solution as a hint to CP-SAT — produced feasible solutions in all 15 trials across $N \in \{8, 16, 32, 64, 128\}$.

Two open questions remain. First, does ALM generalize beyond permutation CSPs? Graph coloring requires dual variables indexed by edges, bin packing by capacity constraints, and job-shop scheduling by machine-time intervals. Each problem family demands a constraint-specific dual variable design; the ALM framework is general, but the diagonal-indexed formulation validated here is not. Second, can the warm-start value be quantified systematically? The 100% feasibility rate across all tested sizes establishes that DiffCP hints help CP-SAT, but the magnitude of the acceleration — and the problem characteristics that predict it — remain open. Answering these questions requires experiments on diverse CSP families and a controlled comparison of CP-SAT solve times with and without differentiable warm starts across problem size, constraint density, and solution density.
