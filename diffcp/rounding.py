"""Rounding strategies for converting continuous solutions to discrete.

Three complementary approaches, ordered from cheapest to most powerful:

1. ``hungarian_round``        — O(N³) deterministic, optimal for pure assignment.
2. ``gumbel_sinkhorn_sample`` — Stochastic; samples many near-permutations by
                                adding Gumbel noise before Sinkhorn. From Mena et al. (2018).
3. ``sample_and_select``      — Wraps Gumbel sampling: returns first feasible rounded
                                solution or falls back to Hungarian on the mean.
4. ``greedy_repair``          — Min-conflicts local search on a discrete assignment.
                                Use as a post-processing step when rounding alone fails.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def hungarian_round(P: torch.Tensor) -> np.ndarray:
    """Round a doubly stochastic matrix to a permutation via the Hungarian algorithm.

    Finds the minimum-cost perfect matching in the bipartite graph defined by -P.
    Optimal in the sense that it maximises the sum of matched P[i, col_ind[i]] entries,
    but it can break side constraints (e.g., diagonal conflicts) because it is unaware
    of problem structure beyond the assignment matrix.

    Args:
        P: (N, N) doubly stochastic matrix with values in [0, 1].
           Detached from the autograd graph before the call if necessary.

    Returns:
        col_ind: (N,) integer array where col_ind[i] is the column assigned to row i.
    """
    cost = -P.detach().cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def gumbel_sinkhorn_sample(
    log_alpha: torch.Tensor,
    temp: float = 0.1,
    n_samples: int = 100,
    sinkhorn_iters: int = 20,
) -> torch.Tensor:
    """Sample near-permutation matrices via Gumbel-Sinkhorn perturbation.

    The Gumbel-Sinkhorn trick (Mena et al., 2018) generates stochastic
    rounding candidates by:
      1. Adding i.i.d. Gumbel(0,1) noise to the log-assignment parameters.
      2. Sinkhorn-normalizing the noisy matrix at a low temperature.

    At very low temperature and many iterations, each sample is close to a
    hard permutation matrix. Sampling many candidates diversifies the search
    over the discrete neighbourhood of the continuous optimum.

    Args:
        log_alpha: (N, N) log-assignment matrix from the optimizer (single restart).
        temp: sampling temperature. Lower = sharper samples, closer to hard
              permutations. Values in [0.01, 0.5] work well in practice.
        n_samples: number of noisy perturbations to generate.
        sinkhorn_iters: Sinkhorn iterations per sample. 20 is usually sufficient.

    Returns:
        (n_samples, N, N) batch of approximate permutation matrices.
    """
    device = log_alpha.device

    # (n_samples, N, N)
    log_alpha_expanded = log_alpha.unsqueeze(0).expand(n_samples, -1, -1)

    # Gumbel(0, 1) noise: -log(-log(U)) for U ~ Uniform(0, 1)
    # Small epsilon guards against log(0)
    U = torch.rand_like(log_alpha_expanded)
    gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
    perturbed = (log_alpha_expanded + gumbel_noise) / temp

    # Sinkhorn in log-space for numerical stability
    for _ in range(sinkhorn_iters):
        perturbed = perturbed - torch.logsumexp(perturbed, dim=-1, keepdim=True)
        perturbed = perturbed - torch.logsumexp(perturbed, dim=-2, keepdim=True)

    return torch.exp(perturbed)  # (n_samples, N, N)


def sample_and_select(
    log_alpha: torch.Tensor,
    verify_fn,
    temp: float = 0.1,
    n_samples: int = 200,
    sinkhorn_iters: int = 20,
) -> tuple:
    """Sample many Gumbel-Sinkhorn roundings and return the first feasible one.

    This is the recommended rounding strategy when the problem has side constraints
    (e.g., N-Queens diagonals, graph coloring edges) that ``hungarian_round`` alone
    may not satisfy. The discrete neighbour space is explored stochastically — each
    Gumbel perturbation gives a different rounding candidate.

    Complexity: O(n_samples · N² · sinkhorn_iters) in tensor ops, fully batched.

    Args:
        log_alpha: (N, N) log-assignment matrix from the optimizer (single restart).
                   Should be from the best-loss restart at the end of training.
        verify_fn: callable(assignment: np.ndarray) -> bool.
                   Returns True iff the discrete assignment satisfies all constraints.
        temp: Gumbel-Sinkhorn sampling temperature (see ``gumbel_sinkhorn_sample``).
        n_samples: number of candidates to try before giving up.
        sinkhorn_iters: inner Sinkhorn iterations per sample.

    Returns:
        (assignment, feasible): tuple of (np.ndarray, bool).
        If a feasible assignment is found, feasible=True.
        Otherwise returns Hungarian rounding of the mean sample with feasible=False.
    """
    samples = gumbel_sinkhorn_sample(log_alpha, temp, n_samples, sinkhorn_iters)

    for i in range(n_samples):
        col_ind = hungarian_round(samples[i])
        if verify_fn(col_ind):
            return col_ind, True

    # Fallback: Hungarian on the mean of all samples (reduces variance)
    mean_P = samples.mean(dim=0)
    return hungarian_round(mean_P), False


def greedy_repair(
    assignment: np.ndarray,
    conflict_fn,
    max_iters: int = 5000,
) -> np.ndarray:
    """Min-conflicts local search: repair a discrete assignment by random swaps.

    Repeatedly proposes random pairwise swaps and accepts them only if they
    reduce the total number of conflicts. This is a hill-climbing variant of
    the min-conflicts heuristic (Minton et al., 1992).

    Use as a post-processing step after ``hungarian_round`` or ``sample_and_select``
    when the rounded solution has a small residual constraint violation that local
    search can close quickly.

    Args:
        assignment: (N,) integer permutation array to repair in-place copy.
        conflict_fn: callable(assignment: np.ndarray) -> int.
                     Returns the total number of constraint violations.
                     Must be O(N) or O(N²) — called O(max_iters) times.
        max_iters: maximum swap attempts. For N=20 problems, 5000 is usually
                   sufficient; scale up for N > 100.

    Returns:
        Repaired (N,) integer array. May still have conflicts if max_iters is
        exhausted before reaching a feasible state.
    """
    assignment = assignment.copy()
    N = len(assignment)
    rng = np.random.default_rng()

    current_conflicts = conflict_fn(assignment)

    for _ in range(max_iters):
        if current_conflicts == 0:
            break

        # Propose a random swap of two positions
        i, j = rng.choice(N, size=2, replace=False)
        assignment[i], assignment[j] = assignment[j], assignment[i]
        new_conflicts = conflict_fn(assignment)

        if new_conflicts < current_conflicts:
            # Accept: improvement found
            current_conflicts = new_conflicts
        else:
            # Reject: restore original
            assignment[i], assignment[j] = assignment[j], assignment[i]

    return assignment
