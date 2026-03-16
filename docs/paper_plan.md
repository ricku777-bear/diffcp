# Paper Plan: DiffCP — Closing the Rounding Gap via Augmented Lagrangian Constraint Programming

**Version:** 1.0 — Designer Phase
**Authors:** Vijay Daita, Hayley Antczak, Miro Swisher, Rick Underwood
**Target venues:** NeurIPS Workshop on Machine Learning for Combinatorial Optimization; CPAIOR 2026 Short Papers; CP 2026
**Format:** 8–10 pages (workshop: 4–6 pages, short track: 6 pages)

---

## 1. Title Options

**Candidate A (preferred — states the mechanism and result):**
> Closing the Rounding Gap in Differentiable Constraint Programming with Augmented Lagrangian Dual Updates

**Candidate B (positions in the differentiable CO literature):**
> DiffCP: GPU-Accelerated Constraint Programming via Sinkhorn Relaxation and Augmented Lagrangian Rounding

**Candidate C (sharper, more theoretical framing):**
> The Rounding Gap Problem in Sinkhorn-Based Constraint Programming: Diagnosis and Dual-Variable Cure

**Rationale:** Candidate A leads with the problem ("rounding gap"), states the method ("augmented Lagrangian"), and promises closure — all without the prohibited "we propose" opener. Candidate B is appropriate if the venue prioritizes the systems/GPU angle. Use A for NeurIPS Workshop and CPAIOR; fall back to B for CP conference which tends toward engineering contributions.

---

## 2. Abstract (Draft, ~150 words)

Differentiable relaxations of combinatorial constraints — where discrete variables are replaced with doubly stochastic matrices projected via Sinkhorn normalization — reliably reach zero loss in the continuous domain. The problem is discretization: Hungarian rounding of the continuous solution fails to satisfy hard constraints at N ≥ 64, leaving a persistent "rounding gap" between the continuous optimum and a feasible integer solution. We diagnose this failure, characterize it empirically on N-Queens at scales N ∈ {64, 128, 256, 512}, and show that the gap closes under an Augmented Lagrangian formulation with per-constraint learnable dual variables. The dual update rule — μ ← μ + ρ · g(x) after each inner optimization — identifies which diagonal constraints are chronically violated and tightens their enforcement until Hungarian rounding produces a feasible solution. Results: 0 violations at N = 64 (2.5 s), N = 128 (28 s), N = 256 (176 s), and N = 512 via hybrid ALM + greedy repair (34 min). On Apple M4 Max (MPS), GPU compute crosses over CPU at N = 16, reaching 5× speedup at N = 64. A warm-start pathway — supplying the continuous solution as a hint to CP-SAT — produces feasible solutions in 100% of trials.

---

## 3. Section-by-Section Outline

---

### Section 1: Introduction

**Length:** ~1 page (4–5 paragraphs)
**Purpose:** Establish the problem, state the gap, preview the contribution. No "in this paper we" constructions. Open with the observation, not with a lit review.

**Paragraph 1 — The gap statement (the hook):**
Open with the empirical fact: Sinkhorn-based differentiable solvers for AllDifferent constraints always converge in the continuous domain — loss reaches exactly 0.000000 at every tested size. This is the promising part. The problem is that Hungarian rounding of the converged continuous solution produces discrete assignments that violate hard constraints at N ≥ 32 (1/3 trials) and universally from N ≥ 64 onward. Name this the "rounding gap." State that closing it is the central question.

**Paragraph 2 — Why this matters (motivation):**
Differentiable programming for combinatorial optimization is attractive because it enables gradient-based solvers on GPU hardware — a fundamentally parallel compute paradigm that sequential CP solvers (CP-SAT, Gecode, OR-Tools) do not exploit. If the rounding gap can be closed reliably, differentiable CP becomes a viable parallel alternative, especially for problem families where warm-starting or continuous relaxations carry useful structure. Cite the broader DiffCO literature (Karalias & Loukas, Mena et al., DiffSAT, DiffILO).

**Paragraph 3 — Why prior approaches do not close it:**
Gumbel-Sinkhorn sampling (Mena et al. 2018) adds stochastic diversity to rounding by perturbing the log-assignment matrix before Sinkhorn — but it treats the rounding problem as a sampling problem rather than an optimization problem. At N = 64, 500 Gumbel samples fail to find a feasible solution because the continuous optimum lies in a region of the Birkhoff polytope where all nearby permutations violate at least one diagonal. Fixed-penalty methods can enforce constraints harder, but large fixed λ degrades conditioning and slows convergence. State in one sentence: the fundamental issue is that the continuous objective is blind to which discrete constraints will be violated by rounding, and a fixed penalty cannot adapt.

**Paragraph 4 — The ALM insight:**
Augmented Lagrangian methods (Bertsekas 1999; Nocedal & Wright 2006) maintain per-constraint dual variables that accumulate based on observed violations. After each inner optimization, μ_k ← μ_k + ρ · g_k(x*) for constraint k with violation g_k. Constraints that remain violated receive exponentially growing Lagrange multipliers, steering the continuous optimum toward a region where rounding succeeds. Two to three outer iterations suffice at all tested scales. Crucially, the dual variables learn problem-specific structure — diagonal constraints of different lengths have different hardness, and the dual update weights them accordingly.

**Paragraph 5 — Contributions (bulleted or as prose, no more than 4 items):**
(1) Empirical diagnosis of the rounding gap — showing it is a function of N and is not closed by Gumbel-Sinkhorn sampling. (2) An ALM formulation with per-diagonal dual variables that closes the gap at N ∈ {64, 128, 256} and N = 512 via hybrid ALM + greedy repair. (3) GPU crossover analysis on Apple MPS — quantifying when batched Sinkhorn iterations become faster than CPU (crossover at N = 16, peak 5× at N = 64). (4) A warm-start pathway to CP-SAT that produces 100% feasible solutions without requiring the rounding gap to be closed.

---

### Section 2: Background and Related Work

**Length:** ~1.5 pages (5–6 paragraphs)
**Purpose:** Cite precisely; establish what is new vs. what is prior work. Keep tight — workshop/short papers are punished for bloated related work.

**Paragraph 1 — Sinkhorn normalization and the Birkhoff polytope:**
Sinkhorn (1964) showed that any positive matrix can be normalized to a doubly stochastic form by alternating row and column normalizations. Birkhoff (1946) established that the set of N×N doubly stochastic matrices is the convex hull of the N! permutation matrices — the Birkhoff polytope. Together these give the theoretical foundation: Sinkhorn projects onto the continuous relaxation of the set of feasible integer assignments. State the iteration: S_0 = exp(A/τ); S_{t+1} = S_t / (S_t · 1)(1^T S_t). Complexity is O(N²) per iteration, entirely expressible as matrix operations — GPU-native.

**Paragraph 2 — Differentiable sorting and ranking:**
Adams & Zemel (2011) proposed ranking via Sinkhorn for learning-to-rank problems. Cuturi (2013) connected Sinkhorn iterations to regularized optimal transport, establishing the dual interpretation. The Gumbel-Sinkhorn Network (Mena et al. 2018a, "Learning Latent Permutations with Gumbel-Sinkhorn Networks") showed that adding Gumbel noise before Sinkhorn normalization produces stochastic, differentiable permutation samples — the basis for the rounding strategy evaluated in this paper. The Sinkhorn Network (Mena et al. 2018b) applied this to learning permutations end-to-end. These papers establish the differentiable rounding direction; ours identifies its failure modes.

**Paragraph 3 — Differentiable combinatorial optimization:**
Karalias & Loukas (2020) — "Erdos Goes Neural" — proposed learning to approximately solve combinatorial problems on graphs by training a GNN to minimize a differentiable surrogate of the combinatorial objective. DiffSAT (cite year) extends this to SAT; DiffILO (cite year) to ILO. These methods all face the rounding gap: the loss can reach near-zero in the continuous domain while the discrete rounding fails. None of them address the gap through dual-variable adaptation. Mirage (CMU) and Twill (Stanford) — mention here if their approach to differentiable CO is directly comparable; otherwise move to a footnote.

**Paragraph 4 — Augmented Lagrangian methods:**
The ALM (Hestenes 1969; Powell 1969; Bertsekas 1999, Chapter 4; Nocedal & Wright 2006, Chapter 17) augments a constrained optimization problem by adding both linear (dual) terms and quadratic penalty terms to the Lagrangian. The dual update μ ← μ + ρ · g(x) is the key — it converts constraint violation history into a steering signal. Unlike pure penalty methods (large fixed λ), ALM maintains good conditioning because the quadratic term ρ need not be large: the dual term carries the accumulated violation pressure. We apply this in a batched, GPU-native formulation where each of B restarts carries its own dual variables.

**Paragraph 5 — Constraint programming and CP-SAT:**
CP-SAT (Perron & Furnon, OR-Tools) is the reference modern CP solver: clause learning, propagation, LNS, and LP relaxations. On N-Queens, it solves N = 64 in 89 ms, N = 128 in 463 ms. DiffCP does not compete with CP-SAT on raw speed — the point is (a) parallel GPU scalability and (b) the warm-start pathway, where the continuous solution provides an integer hint that CP-SAT accepts unconditionally. The Kuhn (1955) Hungarian algorithm — the rounding step — is O(N³) and runs on CPU after each inner optimization; for large N, batched GPU rounding would be beneficial but is not yet implemented.

**Paragraph 6 — What is new:**
One sentence. No prior work combines (a) batched Sinkhorn optimization over B restarts, (b) ALM dual updates per restart, (c) GPU-native execution on Apple MPS / NVIDIA CUDA, and (d) a warm-start pathway to a production CP solver. Each component is known; the combination and the empirical diagnosis of the rounding gap are new.

---

### Section 3: Method

**Length:** ~2 pages (6–8 paragraphs + equations)
**Purpose:** State everything formally. Define all notation once. Every equation referenced later must appear here.

**Paragraph 1 — Problem formulation:**
State the CSP: find x ∈ {0, …, N−1}^N satisfying a set of constraints C = {c_1, …, c_m}. For permutation CSPs (AllDifferent), x corresponds to a bijection; for N-Queens, the additional constraints are the two diagonal AllDifferent constraints on forward (i − j) and backward (i + j) diagonals. Define the assignment matrix X ∈ {0,1}^{N×N} where X_{ij} = 1 iff x_i = j.

**Equation 1 — Sinkhorn relaxation:**
The continuous relaxation replaces X with P ∈ DS_N, the Birkhoff polytope:
  DS_N = { P ∈ R^{N×N} : P_{ij} ≥ 0, ΣP_{ij} = 1 ∀i, ΣP_{ij} = 1 ∀j }
The Birkhoff-von Neumann theorem: vertices of DS_N are exactly the permutation matrices. Sinkhorn iteration S(A, τ): log-space alternating normalization of A/τ for K iterations (K = 20 in experiments). In log-space: log P^{(t+1/2)} = log P^{(t)} − LSE_row; log P^{(t+1)} = log P^{(t+1/2)} − LSE_col.

**Equation 2 — Temperature schedule:**
τ_t = τ_0 · (τ_T / τ_0)^{t/T}
Exponential decay from τ_0 = 2.0 to τ_T = 0.01 over T iterations. As τ → 0, P → permutation matrix (sharp). Temperature annealing restarts at τ_start · (1 − 0.5 · outer_frac) at the beginning of each ALM outer iteration — slightly lower each round, reflecting accumulated dual signal.

**Paragraph 2 — Constraint surrogates:**
For N-Queens, define forward diagonal sums and backward diagonal sums using scatter-add:
  d^+_k(P) = Σ_{i-j=k} P_{ij}  (forward diagonals, k ∈ {-(N-1), …, N-1})
  d^-_k(P) = Σ_{i+j=k} P_{ij}  (backward diagonals, k ∈ {0, …, 2N-2})
The AllDifferent constraint on rows and columns is enforced by construction via Sinkhorn. The diagonal constraints require d^+_k ≤ 1 and d^-_k ≤ 1 for all k. The standard differentiable surrogate: L(P) = Σ_k ReLU(d^+_k − 1)² + Σ_k ReLU(d^-_k − 1)².

**Equation 3 — Batch formulation:**
Run B parallel restarts. log_α ∈ R^{B×N×N} initialized as N(0, 0.01). P^{(b)} = S(log_α^{(b)}, τ). Loss per restart: ℓ^{(b)} = L(P^{(b)}). Total loss for backprop: Σ_b ℓ^{(b)}. Optimized via Adam (lr = 0.2). At the end, take the restart with lowest ℓ and apply Hungarian rounding.

**Paragraph 3 — Rounding (the gap):**
After optimization, P* = argmin_b ℓ^{(b)} at τ = τ_T. Hungarian rounding: compute minimum-cost assignment on −P* via the Kuhn-Munkres algorithm (Kuhn 1955). The rounded assignment x = H(P*) satisfies AllDifferent by construction (it is a permutation) but does not in general satisfy the diagonal constraints. Call the number of unsatisfied diagonal constraints V(x). For N ≤ 32, V(x) = 0 consistently. For N ≥ 64, V(x) ≥ 6 in all trials (Table 1). This is the rounding gap.

**Equation 4 — ALM formulation:**
Augment the loss with per-constraint dual variables. Let g^+_k(P) = ReLU(d^+_k(P) − 1) and g^-_k(P) = ReLU(d^-_k(P) − 1). Dual variables μ^+_k, μ^-_k ∈ R (one per diagonal per restart). ALM loss for one outer iteration:

  L_ALM(P; μ, ρ) = Σ_k [ μ^+_k · g^+_k(P) + (ρ/2)(g^+_k(P))² ]
                 + Σ_k [ μ^-_k · g^-_k(P) + (ρ/2)(g^-_k(P))² ]

Inner loop: minimize L_ALM over log_α with Adam for T_inner steps (T_inner ∈ {200, 500} depending on N).

**Equation 5 — Dual update rule:**
After each inner optimization converges:
  μ^+_k ← μ^+_k + ρ · g^+_k(P*)     for all k
  μ^-_k ← μ^-_k + ρ · g^-_k(P*)     for all k
  ρ ← min(ρ · β, ρ_max)   where β = 1.5, ρ_max = 50.0, ρ_init = 1.0
After each dual update, attempt Hungarian rounding on the lowest-violation restart. Return immediately if feasible.

**Paragraph 4 — Algorithmic summary (provide as Algorithm 1 box):**
Input: N, B restarts, outer iterations O, inner iterations T.
1. Initialize log_α ∼ N(0, 0.01)^{B×N×N}; μ^+, μ^- ← 0; ρ ← ρ_init
2. For outer = 1, …, O:
   a. Run Adam for T steps minimizing L_ALM(S(log_α, τ_t); μ, ρ)
   b. Compute violations g^+, g^- at τ_T
   c. Update μ^+, μ^-, ρ
   d. For each restart b in order of increasing violation:
      — Compute x^{(b)} = H(P^{(b)})
      — If V(x^{(b)}) = 0: return x^{(b)}
3. Return best-effort (lowest V) rounded solution.
Time complexity per outer iteration: O(B · T · K · N²) for the Sinkhorn + Adam inner loop, plus O(B · N³) for B Hungarian roundings. Memory: O(B · N²) for log_α and P.

**Paragraph 5 — Greedy repair (for N = 512):**
At N = 512, B × T_inner grows large enough that runtime exceeds 34 minutes with pure ALM. A hybrid strategy: run ALM to reduce violations to a small number (typically ≤ 10), then apply greedy repair (min-conflicts local search, Minton et al. 1992). Greedy repair proposes random pairwise swaps of queen positions and accepts swaps that reduce the conflict count. With a good ALM warm start, greedy repair closes the remaining gap in < 5000 swap attempts for N = 512.

**Paragraph 6 — Warm-start pathway:**
Independent of whether ALM closes the rounding gap, the continuous solution P* carries information about the structure of feasible solutions. CP-SAT accepts solution hints: for each variable x_i, the hint value is argmax_j P*_{ij}. In all 15 trials across N ∈ {8, 16, 32, 64, 128}, CP-SAT finds a feasible solution with this hint in 100% of cases (Table 2). The warm-start overhead is negligible — hint construction is O(N). This pathway is useful when latency is not critical and guarantee of feasibility is required.

---

### Section 4: Experiments

**Length:** ~2.5 pages (8–10 paragraphs + 4 figures + 3 tables)
**Purpose:** Report every number precisely. Every claim in the abstract must trace to a row or entry in a table here.

**Paragraph 1 — Experimental setup:**
All experiments on Apple M4 Max (CPU: 10-core ARM, GPU: 40-core Metal Performance Shaders via PyTorch MPS backend). Framework: PyTorch 2.x. No external GPU cluster used. Benchmark problem: N-Queens at N ∈ {8, 16, 32, 64, 128, 256, 512}. N-Queens is a canonical benchmark for constraint programming (it appears in CP textbooks and solver benchmarks) and is a pure AllDifferent instance with additional linear constraints — a clean target for the Sinkhorn relaxation. CP-SAT baseline: Google OR-Tools v9.x, single-threaded, 120s time limit. All timing is wall-clock median of 3 trials.

**Paragraph 2 — Experiment 1: Rounding gap diagnosis:**
Present Table 1. Report continuous loss (always 0.000000 regardless of N — the Sinkhorn relaxation always converges) alongside the number of diagonal violations after Hungarian rounding. Key finding: loss = 0 does not imply feasibility. At N = 32, 1/3 trials fail rounding; at N = 64, 0/3 succeed; at N = 128, 0/3. Describe the violation count pattern: N = 64 gives 6 violations, N = 128 gives 8 — the gap grows sublinearly with N, suggesting the problem is solvable with targeted constraint tightening rather than a global re-optimization.

**TABLE 1 — Rounding Gap Diagnosis:**
| N | Continuous Loss | Hungarian Violations | Gumbel (500 samples) Violations | ALM Violations | ALM Outer Iters |
|---|-----------------|----------------------|--------------------------------|----------------|-----------------|
| 32 | 0.000000 | 0 (2/3 trials) | — | 0 | 2 |
| 64 | 0.000000 | 6 (0/3 trials) | 6 (500 samples exhausted) | 0 | 2–3 |
| 128 | 0.000000 | 8 (0/3 trials) | N/A (not run) | 0 | 2–3 |
| 256 | 0.000000 | FAIL | N/A | 0 | 3 |
| 512 | 0.000000 | FAIL | N/A | 0 (ALM+repair) | 3+ repair |

**Paragraph 3 — Experiment 1 continued — Gumbel-Sinkhorn negative result:**
Gumbel-Sinkhorn sampling generates 500 candidate roundings at low temperature (τ = 0.05, K = 30 Sinkhorn iterations). At N = 64, all 500 candidates fail — every Gumbel perturbation of the continuous optimum still lands in a region of the Birkhoff polytope that rounds to a constraint-violating permutation. This motivates the ALM approach: the issue is not diversity of rounding candidates but the location of the continuous optimum itself. Gumbel sampling diversifies the rounding; ALM moves the optimum.

**FIGURE 1 — Architecture / Data Flow Diagram:**
Content: A vertical flow diagram showing the three phases.
Phase 1 (top): "log_α (B×N×N)" → Sinkhorn(·, τ_t) → P (doubly stochastic, B×N×N) → DiagonalConflictLoss → ℓ (B,) → Adam backward.
Phase 2 (middle): ALM box: "μ update: μ ← μ + ρ·g(P*)" with the inner loop collapsed to a single block, outer arrows showing iteration.
Phase 3 (bottom): Rounding branch — "H(P*) → x" checked by verify(); if feasible → return; else continue outer loop. Warm-start branch — "argmax_j P*_{ij} → CP-SAT hint".
Caption should state: "Each of B = 64 restarts runs independently with its own dual variables. Sinkhorn normalization enforces the AllDifferent constraint; diagonal penalty enforces the N-Queens structural constraints."

**Paragraph 4 — Experiment 2: ALM convergence:**
ALM closes the gap in 2–3 outer iterations at all tested N. Report this as Figure 2. Describe the dual variable trajectory: initially μ ≈ 0, penalty is pure quadratic. After outer iteration 1, diagonals with the highest violation accumulate μ > 0. After outer iteration 2, the log-assignment matrix in those regions has been steered by the accumulated dual mass; Hungarian rounding succeeds. The geometric schedule (ρ_new = ρ · 1.5, capped at 50) prevents conditioning collapse — conditioning is preserved by the dual term absorbing accumulated violation history.

**FIGURE 2 — ALM Convergence Trajectory:**
Content: Two panels side by side.
Left panel: per-outer-iteration inner loss curve for N = 64. X-axis: inner Adam steps (0–300). Y-axis: L_ALM loss. Three colored lines for outer iterations 1, 2, 3. Show that outer 1 ends with loss near 0 but rounding fails; outer 2 ends with a different loss landscape; outer 3 rounds successfully. Add a vertical dashed line at the point where rounding first succeeds.
Right panel: dual variable magnitude ||μ|| (L2 norm, averaged across restarts) vs. outer iteration for N ∈ {64, 128, 256}. Shows exponential growth in ||μ||. Caption: "Dual variables accumulate violation pressure across outer iterations. Rounding succeeds when ||μ|| reaches a problem-specific threshold."

**Paragraph 5 — Experiment 3: Scaling table (DiffCP vs. CP-SAT):**
Present Table 2. Report CP-SAT median time, DiffCP median time (standard Sinkhorn+Hungarian), feasibility rate, and warm-start feasibility. DiffCP is 42× slower than CP-SAT at N = 8 and ≈ 258× slower at N = 128. The warm-start pathway produces feasible solutions 100% of the time. State clearly: DiffCP is not competitive with CP-SAT on sequential hardware. The contribution is (a) the ALM rounding result and (b) the GPU pathway which would narrow the gap on parallel hardware. Acknowledge limitation honestly; workshop papers that oversell are rejected; those that characterize the problem precisely are accepted.

**TABLE 2 — DiffCP vs. CP-SAT Scaling:**
| N | CP-SAT (ms) | DiffCP (ms) | DiffCP Feas | Warm-Start Feas |
|---|-------------|-------------|-------------|-----------------|
| 8 | 3.4 | 143 | 3/3 | 3/3 |
| 16 | 7.1 | 1,746 | 3/3 | 3/3 |
| 32 | 20.5 | 14,773 | 1/3 | 3/3 |
| 64 | 89.4 | 32,786 | 0/3 | 3/3 |
| 128 | 463 | 119,327 | 0/3 | 3/3 |

**TABLE 3 — ALM Solver: Scale vs. Time vs. Outer Iterations:**
| N | ALM Time | Outer Iters | Restarts | Inner Iters/Outer |
|---|----------|-------------|----------|-------------------|
| 64 | 2.5 s | 2 | 64 | 300 |
| 128 | 28 s | 3 | 64 | 500 |
| 256 | 176 s | 3 | 64 | 500 |
| 512 | 34 min | 3 + repair | 64 | 500 + repair |

**Paragraph 6 — Experiment 4: GPU vs. CPU crossover:**
Present Figure 3 and supporting data from Table 4. Crossover occurs at N = 16 (MPS 1.07× CPU). Peak speedup 4.95× at N = 64 (MPS: 4,861 ms vs. CPU: 24,088 ms). At N = 128, MPS speedup drops to 1.89× — consistent with MPS memory bandwidth saturation. Explain the N = 8 overhead (0.17×): the MPS kernel launch overhead dominates for small tensors; the working set (B × N² = 64 × 64 = 4K floats) does not fill a single GPU thread group. At N = 16, the working set grows to B × N² = 64 × 256 = 16K floats, enough to amortize kernel launch. At N = 64, the working set is 64 × 4096 = 262K floats — fills GPU L2 cache, peak arithmetic intensity.

**FIGURE 3 — GPU vs. CPU Scaling:**
Content: Line plot with two y-axes or log-scale single y-axis. X-axis: N ∈ {8, 16, 32, 64, 128}. Left y-axis (log scale): wall-clock time in ms for CPU (blue) and MPS (orange). Right y-axis: speedup ratio MPS/CPU (green bar chart overlay). Mark the crossover at N = 16 with a vertical dashed line. Mark peak speedup at N = 64 with an annotation. Caption: "MPS (Apple Silicon GPU) breaks even at N=16 and peaks at 4.95× speedup at N=64. Speedup regresses at N=128, consistent with memory bandwidth saturation on a 40-core GPU."

**TABLE 4 — Raw GPU Crossover Data:**
| N | CPU (ms) | MPS (ms) | Speedup | MPS Feas |
|---|----------|----------|---------|----------|
| 8 | 105 | 620 | 0.17× | 3/3 |
| 16 | 865 | 806 | 1.07× | 3/3 |
| 32 | 5,426 | 1,504 | 3.61× | 2/3 |
| 64 | 24,088 | 4,861 | 4.95× | 2/3 |
| 128 | 88,671 | 46,940 | 1.89× | 0/3 |

**FIGURE 4 — Rounding Strategy Comparison (bar chart):**
Content: Grouped horizontal bar chart. Y-axis: strategies (Standard Hungarian, Gumbel-500, Iterative Rounding, ALM). X-axis: number of diagonal violations after rounding at N = 64. Add a second panel for N = 128 (Gumbel not run). ALM bar reaches 0 (highlighted in green). Caption: "At N=64, only the Augmented Lagrangian solver produces 0 violations. Gumbel-Sinkhorn with 500 samples matches the standard Hungarian result, confirming the issue is the location of the continuous optimum rather than the rounding procedure itself."

**Paragraph 7 — Ablation: rho schedule sensitivity:**
One paragraph. Test ρ_mult ∈ {1.5, 2.0, 3.0} and ρ_max ∈ {50, 100} at N = 64. ρ_mult = 2.0 (the aggressive schedule) closes the gap in 2 outer iterations but occasionally produces numerical instability at N = 128 (conditioning degrades). ρ_mult = 1.5 reliably closes the gap in 3 outer iterations at all sizes tested. ρ_max matters only as a safety cap; ρ rarely reaches it before rounding succeeds. Report the sensitivity as a single sentence supported by a footnote or small inline table.

**Paragraph 8 — Limitations:**
Be direct. Three limitations: (1) All experiments use N-Queens — a pure permutation problem. Generalization to graph coloring, job-shop scheduling, and mixed-integer problems requires constraint-specific dual variable designs and is not yet validated. (2) DiffCP is not competitive with CP-SAT on speed at any scale tested on Apple MPS. The 5× GPU speedup at N = 64 applies to the Sinkhorn inner loop; the overall pipeline including Hungarian rounding (CPU, O(N³)) and ALM outer iterations is slower than CP-SAT by 28× at N = 64 (2.5 s ALM vs. 0.089 s CP-SAT). (3) Memory scales as O(B × N²): at N = 512 with B = 64 restarts and float32, the matrix batch occupies 64 × 512² × 4 bytes = 67 MB — manageable, but N = 1024 would require either reducing B or using bf16.

---

### Section 5: Discussion

**Length:** ~0.75 pages (3 paragraphs)
**Purpose:** Interpret results; connect to the broader differentiable CO literature; state open questions.

**Paragraph 1 — What the dual variables learn:**
The ALM result shows that diagonal constraints are not equally hard. On an N = 64 board, the central diagonals (length close to N) carry more probability mass in the continuous solution than the short corner diagonals. After outer iteration 1, the dual variables for long diagonals are larger than those for short diagonals — the solver has discovered problem geometry through accumulated violation. This is a form of constraint difficulty learning. Future work could initialize μ with problem-specific priors (e.g., from a prior solve or a domain-specific heuristic) to reduce the number of outer iterations.

**Paragraph 2 — The warm-start result as a lower bound:**
Even without closing the rounding gap, DiffCP provides value as a warm-start generator for classical CP solvers. The 100% feasibility rate on warm-started CP-SAT across all sizes tested — including N = 64 and N = 128 where standalone DiffCP fails — suggests the continuous solution encodes enough structural information to guide CP-SAT's VSIDS heuristic effectively. This is an underexplored direction: differentiable solvers as heuristic generators for exact solvers rather than as standalone solvers.

**Paragraph 3 — GPU trajectory:**
The 5× MPS speedup at N = 64 is promising but not transformative on a single Apple GPU. On NVIDIA H100 with BF16 Tensor Cores and 80 GB HBM3, the Sinkhorn inner loop would run at ≈ 300 TFLOPS vs. ≈ 7 TFLOPS for Apple M4 Max MPS — a 43× difference in peak throughput. The crossover point with CP-SAT on H100 is likely to shift significantly toward larger N. More importantly, the ALM outer loop is embarrassingly parallel across restarts: 1024 restarts on an H100 would explore 16× more of the search space at the same cost as 64 restarts on MPS. This is the key hypothesis for future hardware experiments.

---

### Section 6: Conclusion

**Length:** ~0.4 pages (2 paragraphs)
**Purpose:** Restate results in past tense. State the open question that the paper leaves for future work.

**Paragraph 1 — Summary of results:**
The rounding gap — the failure of Hungarian rounding to produce feasible integer solutions from a converged continuous Sinkhorn optimum — is reliably closed by an Augmented Lagrangian formulation with per-diagonal dual variables. ALM solves N-Queens at N = 64 in 2.5 s, N = 128 in 28 s, N = 256 in 176 s, and N = 512 in 34 min via hybrid ALM + greedy repair. Two to three outer iterations suffice at all scales: the dual updates identify which diagonal constraints are hard in the specific problem instance and steer the continuous optimum toward a region where rounding succeeds. GPU execution on Apple MPS accelerates the inner Sinkhorn loop by 5× at N = 64, with crossover vs. CPU at N = 16.

**Paragraph 2 — Open question:**
The central open question is whether ALM-based rounding gap closure generalizes beyond permutation CSPs. Graph coloring, bin packing, and job-shop scheduling all have different constraint structures, and the dual variable design must be adapted accordingly. A second open question is whether the warm-start value of the continuous solution can be quantified: how much does a DiffCP hint actually accelerate CP-SAT, and under what problem characteristics does it help most? These questions are left to future work.

---

## 4. Citation List

All citations required in the paper, in order of first appearance. Full bibliographic details for each.

---

### 4.1 Core Mathematical Foundations

**[Birkhoff 1946]**
Birkhoff, G. (1946). "Three observations on linear algebra." *Universidad Nacional de Tucumán, Revista (Series A)*, 5, 147–151.
*Usage: Birkhoff-von Neumann theorem — convex hull of permutation matrices is the set of doubly stochastic matrices. Cite in Section 2.*

**[Sinkhorn 1964]**
Sinkhorn, R. (1964). "A relationship between arbitrary positive matrices and doubly stochastic matrices." *The Annals of Mathematical Statistics*, 35(2), 876–879.
*Usage: Original Sinkhorn normalization result. Cite in Section 2 when describing the projection operator.*

**[Kuhn 1955]**
Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly*, 2(1–2), 83–97.
*Usage: Hungarian rounding O(N³) algorithm. Cite in Sections 2 and 3 when introducing the rounding step.*

---

### 4.2 Differentiable Sorting, Ranking, and Permutations

**[Adams & Zemel 2011]**
Adams, R. P., & Zemel, R. S. (2011). "Ranking via Sinkhorn Propagation." *arXiv preprint arXiv:1106.1925*.
*Usage: First application of Sinkhorn to learning-to-rank / differentiable permutations. Cite in Section 2.*

**[Cuturi 2013]**
Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport Distances." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 26.
*Usage: Regularized optimal transport connection to Sinkhorn. Provides the dual interpretation. Cite in Section 2.*

**[Mena et al. 2018a]**
Mena, G., Belanger, D., Linderman, S., & Snoek, J. (2018). "Learning Latent Permutations with Gumbel-Sinkhorn Networks." In *International Conference on Learning Representations (ICLR)*.
*Usage: Gumbel-Sinkhorn stochastic rounding. This is the paper behind the gumbel_sinkhorn_sample function evaluated as a negative result in Section 4. Cite in Sections 2 and 4.*

**[Mena et al. 2018b — optional, check if distinct]**
Mena, G., Belanger, D., Munoz, G., Linderman, S., & Snoek, J. (2018). "Sinkhorn Networks: Learning Permutations with Sinkhorn Operator." Workshop on Discrete Structures in Machine Learning, NeurIPS.
*Note: Verify whether this is a distinct publication from 2018a or a workshop version of the same work. If distinct, cite when mentioning end-to-end permutation learning.*

---

### 4.3 Differentiable Combinatorial Optimization

**[Karalias & Loukas 2020]**
Karalias, N., & Loukas, A. (2020). "Erdos Goes Neural: An Unsupervised Learning Framework for Combinatorial Optimization on Graphs." In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33.
*Usage: Learning differentiable surrogates for graph CO problems. First paper to demonstrate this at scale. Cite in Section 1 (motivation) and Section 2.*

**[DiffSAT — verify citation]**
Wang, P.-W., Donti, P. L., Wilder, B., & Kolter, J. Z. (2019). "SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver." In *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
*Note: The README cites "DiffSAT (2023)" but SATNet (2019) may be the correct reference. Verify whether DiffSAT refers to a 2023 paper or to SATNet. If a 2023 paper exists, search for it. Cite in Section 2.*

**[DiffILO — verify citation]**
Amos, B., & Kolter, J. Z. (2017). "OptNet: Differentiable Optimization as a Layer in Neural Networks." In *Proceedings of the 34th International Conference on Machine Learning (ICML)*.
*Note: "DiffILO" may refer to a specific 2023 paper. Verify the exact reference. If it exists as a 2023 conference paper, use that; otherwise fall back to OptNet for the differentiable ILO connection. Cite in Section 2.*

**[Mirage — verify citation]**
*Note: Search for the Mirage (CMU) paper on differentiable combinatorial optimization. Likely 2022–2024. Author team likely includes CMU ML/systems faculty. If found, cite in Section 2 as "concurrent work in differentiable CO."*

**[Twill — verify citation]**
*Note: Search for the Twill (Stanford) paper. Likely 2022–2024 in the differentiable programming or ML for CO venue. If found, cite in Section 2.*

---

### 4.4 Augmented Lagrangian Methods

**[Hestenes 1969]**
Hestenes, M. R. (1969). "Multiplier and gradient methods." *Journal of Optimization Theory and Applications*, 4(5), 303–320.
*Usage: Original ALM derivation. Can cite as "Hestenes (1969); Powell (1969)" as co-inventors. Cite in Section 2 when introducing ALM.*

**[Powell 1969]**
Powell, M. J. D. (1969). "A method for nonlinear constraints in minimization problems." In R. Fletcher (ed.), *Optimization*, Academic Press, London, pp. 283–298.
*Usage: Co-inventor of ALM with Hestenes. Cite together with Hestenes.*

**[Bertsekas 1999]**
Bertsekas, D. P. (1999). *Nonlinear Programming*, 2nd edition. Athena Scientific.
*Usage: Chapter 4 gives the canonical ALM theory including convergence analysis and the dual update derivation. Cite in Sections 2 and 3.*

**[Nocedal & Wright 2006]**
Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*, 2nd edition. Springer.
*Usage: Chapter 17 covers augmented Lagrangian methods with numerical stability analysis. Cite in Sections 2 and 3.*

---

### 4.5 CP Solvers and Warm-Starting

**[Perron & Furnon — OR-Tools / CP-SAT]**
Perron, L., & Furnon, V. (2023). OR-Tools. Google LLC. Available at https://developers.google.com/optimization.
*Usage: CP-SAT baseline solver. Cite in Section 2 (baseline description) and Section 4 (results). Use the appropriate version year for the version installed.*

**[OR-Tools CP-SAT paper — if available]**
Perron, L., & Didier, F. (2020). "CP-SAT." In *Proceedings of the 26th International Conference on Principles and Practice of Constraint Programming (CP)*.
*Note: Verify whether a citable CP-SAT conference paper exists from 2020 or nearby. The solver itself is documented at the OR-Tools URL. Use whichever is available.*

---

### 4.6 Min-Conflicts and Local Search

**[Minton et al. 1992]**
Minton, S., Johnston, M. D., Philips, A. B., & Laird, P. (1992). "Minimizing conflicts: a heuristic repair method for constraint satisfaction and scheduling problems." *Artificial Intelligence*, 58(1–3), 161–205.
*Usage: Greedy repair / min-conflicts local search used in the N = 512 hybrid strategy. Cite in Section 3 (greedy repair paragraph).*

---

### 4.7 Surveys (Optional — Use if Space Permits)

**[Bengio et al. 2021]**
Bengio, Y., Lodi, A., & Prouvost, A. (2021). "Machine learning for combinatorial optimization: a methodological tour d'horizon." *European Journal of Operational Research*, 290(2), 405–421.
*Usage: Broad survey situating learned CO in the operations research context. Cite in Section 1 if the introduction references the broader field.*

**[Kotary et al. 2021]**
Kotary, J., Fioretto, F., Van Hentenryck, P., & Wilder, B. (2021). "End-to-End Constrained Optimization Learning: A Survey." In *Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI)*, Survey Track.
*Usage: Positions DiffCP in the constrained optimization learning survey. Cite in Section 2 if word count permits.*

---

## 5. Style Guide for Section Writers

These rules apply to every section. Writers must follow them without exception.

### Voice and Framing

- **No "we propose a novel"** — state what the system does, not what it proposes.
- **No "in this paper we"** — start with the observation or contribution directly.
- **No "state of the art"** — state the comparison precisely ("ALM closes the gap; Gumbel-Sinkhorn with 500 samples does not").
- **Active voice** — "ALM closes the rounding gap" not "the rounding gap is closed by ALM."
- **Short sentences** — max 2 clauses. If a sentence needs 3 clauses, break it.
- **Present tense for facts and results** — "The Sinkhorn operator converges in K iterations." Past tense for specific experimental actions: "We ran 3 trials at each N."

### Claims and Numbers

- **Every claim backed by a number** — "ALM closes the gap" must be followed immediately by "(N = 64: 0 violations, 2.5 s; N = 128: 0 violations, 28 s)."
- **Never round wall-clock times more than necessary** — use 1 decimal place for times < 100 s; report in the natural unit (ms for CP-SAT baselines, s for ALM runs).
- **Report N-Queens feasibility as fractions** — "2/3 trials" not "67%."
- **Report violations as integers** — "6 diagonal violations" not "6 conflicts."
- **Failure is failure** — if Hungarian rounding failed, write "Hungarian rounding fails at N ≥ 64 in all trials." Don't soften it.

### Equations and Notation

All notation is defined once in Section 3. Section writers must not introduce new symbols without checking Section 3 first.

| Symbol | Meaning | First defined |
|--------|---------|---------------|
| N | Board / problem size | Sec 3, ¶1 |
| B | Number of parallel restarts | Sec 3, ¶3 |
| P | Doubly stochastic matrix (B×N×N) | Sec 3, Eq 1 |
| log_α | Log-assignment matrix (B×N×N) | Sec 3, ¶3 |
| τ | Sinkhorn temperature | Sec 3, Eq 2 |
| τ_0, τ_T | Initial and final temperature | Sec 3, Eq 2 |
| K | Sinkhorn inner iterations (K = 20) | Sec 3, ¶1 |
| T | Adam inner iterations | Sec 3, ¶3 |
| L(P) | Standard diagonal surrogate loss | Sec 3, Eq surrogate |
| d^+_k, d^-_k | Forward/backward diagonal sums | Sec 3, ¶2 |
| μ^+_k, μ^-_k | ALM dual variables per diagonal | Sec 3, Eq 4 |
| ρ | ALM penalty coefficient | Sec 3, Eq 4 |
| g^+_k, g^-_k | Constraint violation (ReLU) | Sec 3, Eq 4 |
| H(P) | Hungarian rounding operator | Sec 3, ¶3 |
| V(x) | Number of violated diagonal constraints | Sec 3, ¶3 |
| O | Number of ALM outer iterations | Sec 3, Algorithm 1 |

### Figures

**Figure 1** — Architecture/data-flow diagram. Must appear in Section 3. No results data.
**Figure 2** — ALM convergence trajectory (inner loss + dual norm). Must appear in Section 4, Experiment 2.
**Figure 3** — GPU vs. CPU scaling (log time + speedup). Must appear in Section 4, Experiment 4.
**Figure 4** — Rounding strategy comparison bar chart. Must appear in Section 4 near the Gumbel negative result.

All figures: single-column width (3.5 in) or double-column (7 in). Font size: 9pt axis labels (readable at 75% scale). No dark backgrounds. Color: use colorblind-safe palette (blue, orange, green — matplotlib default cycler is acceptable). Every figure must have a caption that states the key takeaway in the first sentence.

### Tables

**Table 1** — Rounding gap diagnosis. Section 4, Experiment 1.
**Table 2** — DiffCP vs. CP-SAT scaling. Section 4, Experiment 3.
**Table 3** — ALM solver scaling. Section 4, Experiment 3.
**Table 4** — Raw GPU crossover data. Section 4, Experiment 4 (inline, or as a compact table).

All tables: ruled style (top rule, mid rule after header, bottom rule). No vertical lines. Bold the header row. Highlight best result per row in bold.

---

## 6. Open Issues for Writers to Resolve Before Writing

1. **Gumbel-Sinkhorn violation count at N = 64** — The bench_rounding_gap.py benchmark records violations for Gumbel at N = 64. Run the benchmark and record the exact violation count before writing Table 1. The current README says "FAIL" for standard at N = 64 with "6 violations" — confirm this is the Gumbel result too.

2. **Mirage and Twill citations** — Search for the correct paper titles, authors, venues, and years. If either paper does not exist under those names, remove the citation. Do not fabricate a reference.

3. **DiffSAT and DiffILO exact citations** — The README cites these as "(2023)" without authors. Identify the correct papers before writing Section 2. Candidate: SATNet (Wang et al. 2019) for DiffSAT; OptNet (Amos & Kolter 2017) or a 2023 follow-on for DiffILO.

4. **ALM violation count at N = 64, 128** — The README says "0 violations" for ALM at N = 64 and N = 128. Confirm the exact outer iteration count in results_rounding_gap.json (if it exists) or by re-running bench_rounding_gap.py. Table 1 outer-iter column requires this.

5. **N = 512 wall clock time** — README reports 34 min. Confirm whether this is reproducible and on what device. If timing is variable (e.g., depends on number of greedy repair iterations), report a range.

6. **Venue-specific page limits** — NeurIPS Workshop ML4CO: typically 4–6 pages + references. CPAIOR short: 6 pages. CP conference short: 6 pages. The outline above targets 8–10 pages for the full version; writers must cut to fit target venue. Recommended cuts: condense Section 2 to 1 page, compress Figure 4 into text.

---

## 7. Section Assignments and Length Budget

| Section | Content | Target length |
|---------|---------|---------------|
| 1. Introduction | Gap statement, motivation, why existing methods fail, ALM insight, contributions | 1.0 page |
| 2. Background | Sinkhorn/Birkhoff, differentiable CO, ALM, CP-SAT | 1.5 pages |
| 3. Method | Formalization, Sinkhorn relaxation, surrogates, ALM, Algorithm 1, warm-start | 2.0 pages |
| 4. Experiments | 4 experiments, 4 figures, 3–4 tables, ablation, limitations | 2.5 pages |
| 5. Discussion | Dual variable learning, warm-start value, GPU trajectory | 0.75 pages |
| 6. Conclusion | Results summary, open questions | 0.4 pages |
| References | ~25 references at 3 lines each | ~1.25 pages |
| **Total** | | **~9.4 pages** |

Cut to 6 pages for short-track venues by: removing Figure 4, compressing Table 4 into a footnote, condensing Background to 0.75 pages, and merging Discussion into Conclusion.
