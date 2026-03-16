## 3. Method

**Problem formulation.**
The goal is to find an assignment $x \in \{0, \ldots, N-1\}^N$ satisfying a set of constraints $C = \{c_1, \ldots, c_m\}$. For permutation constraint satisfaction problems, the AllDifferent constraint requires $x$ to be a bijection: each value appears exactly once. N-Queens imposes two additional families of constraints beyond AllDifferent. The forward diagonal constraint requires all values $i - x_i$ to be distinct for $i = 0, \ldots, N-1$. The backward diagonal constraint requires all values $i + x_i$ to be distinct. Define the assignment matrix $X \in \{0,1\}^{N \times N}$, where $X_{ij} = 1$ if and only if $x_i = j$. A feasible N-Queens solution corresponds to a permutation matrix $X$ in which no two entries equal to 1 share a forward or backward diagonal.

**Sinkhorn relaxation.**
The discrete optimization over permutation matrices is relaxed to a continuous optimization over the Birkhoff polytope $DS_N$:

$$DS_N = \{P \in \mathbb{R}^{N \times N} : P_{ij} \geq 0,\; \sum_{j} P_{ij} = 1 \;\forall\, i,\; \sum_{i} P_{ij} = 1 \;\forall\, j\}. \tag{1}$$

The Birkhoff--von Neumann theorem states that the vertices of $DS_N$ are exactly the $N!$ permutation matrices. Any doubly stochastic matrix is therefore a convex combination of permutations. The Sinkhorn operator $S(\log\alpha, \tau)$ projects an arbitrary log-assignment matrix $\log\alpha \in \mathbb{R}^{N \times N}$ onto $DS_N$ via $K$ iterations of alternating log-space row and column normalization at temperature $\tau$:

$$\log P^{(t+1/2)} = \log P^{(t)} - \mathrm{LSE}_{\mathrm{row}}(\log P^{(t)}), \qquad \log P^{(t+1)} = \log P^{(t+1/2)} - \mathrm{LSE}_{\mathrm{col}}(\log P^{(t+1/2)}),$$

where $\log P^{(0)} = \log\alpha / \tau$ and $\mathrm{LSE}_{\mathrm{row}}$ denotes the log-sum-exp computed along each row (and analogously for columns). Each iteration costs $O(N^2)$ and consists entirely of element-wise and reduction operations that map directly to GPU execution. We use $K = 20$ Sinkhorn iterations throughout.

**Temperature schedule.**
The temperature $\tau$ controls the sharpness of $P$. At high $\tau$, the Sinkhorn output is a uniform doubly stochastic matrix. As $\tau \to 0$, $P$ approaches a permutation matrix. The temperature follows an exponential decay schedule:

$$\tau_t = \tau_0 \cdot \left(\frac{\tau_T}{\tau_0}\right)^{t/T}, \tag{2}$$

where $\tau_0 = 2.0$ is the initial temperature, $\tau_T = 0.01$ is the final temperature, and $T$ is the number of Adam inner iterations. At the beginning of each ALM outer iteration, the starting temperature resets to $\tau_0 \cdot (1 - 0.5 \cdot o/O)$, where $o$ is the current outer iteration index and $O$ is the total number of outer iterations. This ensures that each successive outer round begins at a slightly lower temperature, reflecting the accumulated dual signal.

**Constraint surrogates.**
Sinkhorn normalization enforces AllDifferent on rows and columns by construction: every output of the Sinkhorn operator is doubly stochastic, and the rounded permutation assigns exactly one queen per row and per column. The diagonal constraints require separate enforcement. Define the forward and backward diagonal sums via scatter-add:

$$d^+_k(P) = \sum_{\substack{i,j \\ i - j = k}} P_{ij}, \qquad d^-_k(P) = \sum_{\substack{i,j \\ i + j = k}} P_{ij},$$

where $k \in \{-(N-1), \ldots, N-1\}$ for forward diagonals and $k \in \{0, \ldots, 2N-2\}$ for backward diagonals. For a feasible N-Queens assignment, each diagonal sum equals either 0 or 1. The constraints are $d^+_k \leq 1$ and $d^-_k \leq 1$ for all $k$. The standard differentiable surrogate penalizes violations quadratically:

$$L(P) = \sum_k \mathrm{ReLU}(d^+_k(P) - 1)^2 + \sum_k \mathrm{ReLU}(d^-_k(P) - 1)^2.$$

**Batch formulation.**
The solver runs $B$ parallel restarts to increase the probability of finding a feasible rounding. Initialize $\log\alpha \in \mathbb{R}^{B \times N \times N}$ with entries drawn from $\mathcal{N}(0, 0.01)$. Each restart $b \in \{1, \ldots, B\}$ produces a doubly stochastic matrix $P^{(b)} = S(\log\alpha^{(b)}, \tau_t)$ and incurs a per-restart loss $\ell^{(b)} = L(P^{(b)})$. The total loss $\sum_b \ell^{(b)}$ is minimized by the Adam optimizer [Kingma & Ba 2015] with learning rate 0.2. All $B$ restarts share a single parameter tensor and a single backward pass. At termination, the restart with the lowest loss is selected and rounded. This batch formulation maps to a single $(B, N, N)$ tensor operation on GPU, with no inter-restart communication.

$$P^{(b)} = S(\log\alpha^{(b)}, \tau_t), \qquad \ell^{(b)} = L(P^{(b)}), \qquad \mathcal{L}_{\text{total}} = \sum_{b=1}^{B} \ell^{(b)}. \tag{3}$$

**Rounding and the gap.**
After inner optimization completes, the best restart is identified: $P^* = \arg\min_b \ell^{(b)}$, evaluated at $\tau = \tau_T$. The Hungarian rounding operator $H$ computes the minimum-cost assignment on the negated matrix $-P^*$ via the Kuhn--Munkres algorithm [Kuhn 1955], yielding a discrete assignment $x = H(P^*)$. This assignment is a permutation by construction and therefore satisfies AllDifferent. It does not, in general, satisfy the diagonal constraints. Define the violation count $V(x)$ as the number of diagonal constraints $d^+_k \leq 1$ or $d^-_k \leq 1$ that the discrete solution violates. For $N \leq 32$, $V(x) = 0$ consistently across all trials. For $N \geq 64$, $V(x) \geq 6$ in every trial. This is the rounding gap: the continuous loss is zero, but the discrete solution is infeasible. The gap arises because the continuous optimum lies in a region of the Birkhoff polytope where all nearby permutation-matrix vertices violate at least one diagonal constraint.

**Augmented Lagrangian formulation.**
The ALM replaces the fixed-penalty surrogate with an adaptive loss that maintains per-constraint dual variables. Define the constraint violation functions:

$$g^+_k(P) = \mathrm{ReLU}(d^+_k(P) - 1), \qquad g^-_k(P) = \mathrm{ReLU}(d^-_k(P) - 1).$$

Introduce dual variables $\mu^+_k, \mu^-_k \in \mathbb{R}$ for each diagonal $k$ and each restart (suppressing the restart index for clarity). The ALM loss for one outer iteration is:

$$L_{\mathrm{ALM}}(P;\, \mu, \rho) = \sum_k \left[\mu^+_k \cdot g^+_k(P) + \frac{\rho}{2}\,(g^+_k(P))^2\right] + \sum_k \left[\mu^-_k \cdot g^-_k(P) + \frac{\rho}{2}\,(g^-_k(P))^2\right]. \tag{4}$$

The inner loop minimizes $L_{\mathrm{ALM}}$ over $\log\alpha$ with Adam for $T_{\text{inner}}$ steps at the current values of $\mu$ and $\rho$. The linear terms $\mu^+_k \cdot g^+_k$ bias the gradient toward reducing violations on diagonals that were violated in previous outer iterations. The quadratic terms $(\rho/2)(g^+_k)^2$ stabilize the optimization. Because $\mu$ carries the accumulated violation history, $\rho$ need not grow large to enforce feasibility — this preserves the conditioning of the inner problem, unlike a pure penalty method.

**Dual update rule.**
After each inner optimization converges, the dual variables and penalty coefficient are updated:

$$\mu^+_k \leftarrow \mu^+_k + \rho \cdot g^+_k(P^*), \qquad \mu^-_k \leftarrow \mu^-_k + \rho \cdot g^-_k(P^*), \tag{5}$$

$$\rho \leftarrow \min(\rho \cdot \beta,\; \rho_{\max}),$$

where $\beta = 2.0$, $\rho_{\max} = 200$, and $\rho_{\text{init}} = 5.0$. Diagonals with zero violation after the inner solve receive no update to their dual variable. Diagonals with persistent violation accumulate increasing $\mu$ values across outer iterations, which focuses the gradient signal on the hardest constraints. After each dual update, Hungarian rounding is attempted on all $B$ restarts in order of increasing violation magnitude. If any restart yields $V(x) = 0$, the algorithm returns immediately with a feasible solution.

**Algorithm 1: ALM Solver for Permutation CSPs**

> **Input:** Board size $N$, restarts $B$, outer iterations $O$, inner iterations $T$, Sinkhorn iterations $K$, learning rate $\eta$, temperatures $\tau_0$, $\tau_T$, penalty schedule $\rho_{\text{init}}$, $\beta$, $\rho_{\max}$.
>
> **Output:** Feasible assignment $x \in \{0, \ldots, N-1\}^N$ or best-effort solution.
>
> 1. Initialize $\log\alpha \sim \mathcal{N}(0, 0.01)^{B \times N \times N}$; set $\mu^+_k \leftarrow 0$, $\mu^-_k \leftarrow 0$ for all diagonals $k$; set $\rho \leftarrow \rho_{\text{init}}$.
> 2. **For** $o = 1, \ldots, O$:
>     1. Set $\tau_{\text{start}} \leftarrow \tau_0 \cdot (1 - 0.5 \cdot o / O)$.
>     2. Initialize Adam optimizer on $\log\alpha$ with learning rate $\eta$.
>     3. **For** $t = 1, \ldots, T$:
>         1. Compute temperature: $\tau_t \leftarrow \tau_{\text{start}} \cdot (\tau_T / \tau_{\text{start}})^{t/T}$.
>         2. Compute $P^{(b)} \leftarrow S(\log\alpha^{(b)}, \tau_t)$ for all $b$ via $K$ Sinkhorn iterations.
>         3. Compute diagonal violations $g^+_k(P^{(b)})$, $g^-_k(P^{(b)})$ via scatter-add.
>         4. Compute $L_{\mathrm{ALM}}(P;\, \mu, \rho)$ per Equation (4).
>         5. Backpropagate and update $\log\alpha$ via Adam.
>     4. Evaluate $P^{(b)} \leftarrow S(\log\alpha^{(b)}, \tau_T)$ at final temperature for all $b$.
>     5. Compute violations $g^+_k(P^{(b)})$, $g^-_k(P^{(b)})$ at $\tau_T$.
>     6. Update dual variables: $\mu^+_k \leftarrow \mu^+_k + \rho \cdot g^+_k$, $\mu^-_k \leftarrow \mu^-_k + \rho \cdot g^-_k$.
>     7. Update penalty: $\rho \leftarrow \min(\rho \cdot \beta, \rho_{\max})$.
>     8. **For** each restart $b$ in order of increasing $\sum_k (g^+_k)^2 + (g^-_k)^2$:
>         1. Compute $x^{(b)} \leftarrow H(P^{(b)})$ via Hungarian algorithm.
>         2. **If** $V(x^{(b)}) = 0$: **return** $x^{(b)}$.
> 3. **Return** $x^{(b^*)}$ where $b^* = \arg\min_b V(x^{(b)})$ (best-effort).

**Greedy repair for large $N$.**
At $N = 512$, pure ALM reduces the violation count to a small residual (typically $V(x) \leq 10$) but does not close to zero within a practical number of outer iterations. A hybrid strategy combines ALM with greedy min-conflicts repair [Minton et al. 1992]. After ALM terminates with its best assignment $x$, the repair phase proposes random pairwise queen swaps: select two rows $i, j$ uniformly at random and swap $x_i \leftrightarrow x_j$. Accept the swap if and only if it reduces the total conflict count. Reject otherwise. The swap preserves the permutation structure, so AllDifferent remains satisfied throughout. With the ALM warm start — which places the solution within a small number of violations of feasibility — greedy repair closes the remaining gap in fewer than 5000 swap attempts for $N = 512$.

**Warm-start pathway.**
Independent of whether ALM closes the rounding gap entirely, the continuous solution $P^*$ encodes structural information about feasible solutions. CP-SAT [Perron & Furnon 2023] accepts solution hints: for each variable $x_i$, the hint value is set to $\arg\max_j P^*_{ij}$. The hint construction is $O(N)$ — a single argmax per row of the $N \times N$ matrix. In all 15 trials across $N \in \{8, 16, 32, 64, 128\}$, CP-SAT finds a feasible solution when initialized with this hint, achieving a 100\% success rate. The warm-start pathway is useful when feasibility guarantees are required and the latency overhead of running a CP solver after the differentiable phase is acceptable.

**Complexity.**
Each ALM outer iteration performs $T$ inner Adam steps, each requiring $K$ Sinkhorn iterations on a $(B, N, N)$ tensor. The per-step cost is $O(B \cdot K \cdot N^2)$ for Sinkhorn normalization plus $O(B \cdot N^2)$ for the scatter-add diagonal computation and backward pass. The total inner loop cost per outer iteration is $O(B \cdot T \cdot K \cdot N^2)$. Hungarian rounding after each outer iteration costs $O(B \cdot N^3)$ for $B$ independent $N \times N$ assignments. Memory is $O(B \cdot N^2)$ for the log-assignment tensor and the doubly stochastic matrices, plus $O(B \cdot N)$ for the dual variables.
