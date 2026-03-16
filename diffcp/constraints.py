"""Differentiable constraint surrogates for constraint programming."""
import torch
import torch.nn.functional as F


class AllDifferentLoss:
    """Sinkhorn-based AllDifferent constraint.

    Maps AllDifferent to the doubly stochastic polytope via Sinkhorn normalization.
    The Birkhoff-von Neumann theorem guarantees this is the convex relaxation.
    """

    def __init__(self, sinkhorn_iters=20):
        self.sinkhorn_iters = sinkhorn_iters

    def sinkhorn(self, log_alpha, temp=1.0):
        """Batch Sinkhorn: (B, N, N) -> (B, N, N) doubly stochastic."""
        scaled = log_alpha / temp
        for _ in range(self.sinkhorn_iters):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        return torch.exp(scaled)

    def __call__(self, P):
        """Returns 0 — Sinkhorn projection already enforces AllDifferent.
        The loss comes from problem-specific constraints (diagonals, etc.)."""
        return torch.zeros(P.shape[0], device=P.device)


class DiagonalConflictLoss:
    """N-Queens diagonal constraint: at most one queen per diagonal.

    Uses scatter_add for fully vectorized diagonal sum computation.
    No Python loops over N — O(N²) tensor ops only.
    """

    def __call__(self, P):
        """
        Args:
            P: (B, N, N) doubly stochastic matrix — P[b, i, j] = probability
               of placing queen in row i, column j for instance b.

        Returns:
            (B,) loss tensor. Zero means all diagonal constraints satisfied.
        """
        B, N, _ = P.shape
        device = P.device

        rows = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
        cols = torch.arange(N, device=device).unsqueeze(0).expand(N, N)

        # Forward diagonals: row - col + (N-1) in [0, 2N-2]
        fwd_idx = (rows - cols + (N - 1)).reshape(-1)
        # Backward diagonals: row + col in [0, 2N-2]
        bwd_idx = (rows + cols).reshape(-1)

        P_flat = P.reshape(B, -1)
        num_diags = 2 * N - 1

        fwd_sums = torch.zeros(B, num_diags, device=device)
        fwd_sums.scatter_add_(1, fwd_idx.unsqueeze(0).expand(B, -1), P_flat)

        bwd_sums = torch.zeros(B, num_diags, device=device)
        bwd_sums.scatter_add_(1, bwd_idx.unsqueeze(0).expand(B, -1), P_flat)

        # Squared hinge: penalize each diagonal that has more than one queen
        fwd_loss = F.relu(fwd_sums - 1.0).pow(2).sum(dim=1)
        bwd_loss = F.relu(bwd_sums - 1.0).pow(2).sum(dim=1)
        return fwd_loss + bwd_loss


class LinearConstraintLoss:
    """Differentiable linear constraint: Ax <= b.

    Penalizes violations via squared hinge loss: sum(relu(Ax - b)^2).
    For equality constraints Ax = b, use equality=True to penalize both
    directions: sum((Ax - b)^2).
    """

    def __init__(self, A: torch.Tensor, b: torch.Tensor, equality: bool = False):
        """
        Args:
            A: (M, N) constraint matrix
            b: (M,) right-hand side
            equality: if True, enforce Ax = b; otherwise Ax <= b
        """
        self.A = A
        self.b = b
        self.equality = equality

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N) batch of variable assignments

        Returns:
            (B,) loss tensor
        """
        # (B, M) = (B, N) @ (N, M)
        Ax = x @ self.A.T
        violation = Ax - self.b.unsqueeze(0)  # (B, M)
        if self.equality:
            return violation.pow(2).sum(dim=1)
        else:
            return F.relu(violation).pow(2).sum(dim=1)


class GraphColoringLoss:
    """Differentiable graph coloring constraint.

    For each edge (u, v), penalize if u and v are assigned the same color.
    Uses soft assignment matrices where P[b, i, c] = probability node i has color c.

    Prefer GraphColoringLossVectorized for large graphs — this version iterates
    over edges in Python and is O(E) in loop overhead.
    """

    def __init__(self, edges):
        """
        Args:
            edges: list of (u, v) integer tuples representing undirected edges
        """
        self.edges = edges

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P: (B, N, C) soft color assignments (each row sums to 1 via softmax/Sinkhorn)

        Returns:
            (B,) loss tensor
        """
        B = P.shape[0]
        loss = torch.zeros(B, device=P.device)
        for u, v in self.edges:
            # Inner product of color distributions = probability of same color
            overlap = (P[:, u, :] * P[:, v, :]).sum(dim=1)  # (B,)
            loss = loss + overlap.pow(2)
        return loss


class GraphColoringLossVectorized:
    """Vectorized graph coloring loss using edge index tensors.

    All edge operations run as batch tensor ops — no Python loop over edges.
    Requires pre-built edge_index tensor (compatible with PyG convention).
    """

    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Args:
            edge_index: (2, E) integer tensor of edge endpoints
            num_nodes: N (used for shape validation)
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P: (B, N, C) soft color assignments

        Returns:
            (B,) loss tensor
        """
        src = self.edge_index[0]  # (E,)
        dst = self.edge_index[1]  # (E,)

        src_colors = P[:, src, :]   # (B, E, C)
        dst_colors = P[:, dst, :]   # (B, E, C)

        # Per-edge overlap: sum element-wise product over colors
        overlap = (src_colors * dst_colors).sum(dim=2)  # (B, E)
        return overlap.pow(2).sum(dim=1)                # (B,)


class JobShopLoss:
    """Differentiable job-shop scheduling constraints.

    Variables: start_times[j, o] for each operation o of job j, flattened
    into a single (B, total_ops) tensor.

    Two constraint types:
    1. Precedence — within a job, each operation must finish before the next starts.
    2. No-overlap — on each machine, operations cannot run simultaneously.

    No-overlap uses a soft min-of-relu relaxation so the gradient signal is
    informative on both sides of each pairwise ordering decision.
    """

    def __init__(self, jobs, machines_count: int):
        """
        Args:
            jobs: list of lists of (machine, duration) tuples.
                  jobs[j][o] = (machine_id, processing_time) for operation o of job j.
            machines_count: total number of machines M
        """
        self.jobs = jobs
        self.machines_count = machines_count

    def precedence_loss(self, start_times: torch.Tensor) -> torch.Tensor:
        """Penalize if operation o+1 starts before operation o finishes.

        Args:
            start_times: (B, total_ops) flattened start times

        Returns:
            (B,) loss tensor
        """
        B = start_times.shape[0]
        loss = torch.zeros(B, device=start_times.device)
        idx = 0
        for job in self.jobs:
            for o in range(len(job) - 1):
                _, duration = job[o]
                # Constraint: start[o+1] >= start[o] + duration
                violation = start_times[:, idx] + duration - start_times[:, idx + 1]
                loss = loss + F.relu(violation).pow(2)
                idx += 1
            idx += 1  # advance past last operation of this job
        return loss

    def no_overlap_loss(self, start_times: torch.Tensor) -> torch.Tensor:
        """Penalize overlapping operations on the same machine.

        For each pair of operations on the same machine, at least one of the
        two orderings must hold. The loss penalizes the minimum of the two
        possible overlap amounts — zero only when one ordering is respected.

        Args:
            start_times: (B, total_ops) flattened start times

        Returns:
            (B,) loss tensor
        """
        B = start_times.shape[0]
        loss = torch.zeros(B, device=start_times.device)

        # Group operations by machine
        machine_ops = [[] for _ in range(self.machines_count)]
        idx = 0
        for job in self.jobs:
            for machine, duration in job:
                machine_ops[machine].append((idx, duration))
                idx += 1

        # For each machine, penalize all overlapping pairs
        for ops in machine_ops:
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    idx_i, dur_i = ops[i]
                    idx_j, dur_j = ops[j]
                    # Either i ends before j starts, or j ends before i starts
                    gap_ij = F.relu(start_times[:, idx_i] + dur_i - start_times[:, idx_j])
                    gap_ji = F.relu(start_times[:, idx_j] + dur_j - start_times[:, idx_i])
                    loss = loss + torch.min(gap_ij, gap_ji).pow(2)

        return loss

    def __call__(self, start_times: torch.Tensor) -> torch.Tensor:
        """
        Args:
            start_times: (B, total_ops) flattened start times

        Returns:
            (B,) combined constraint loss
        """
        return self.precedence_loss(start_times) + self.no_overlap_loss(start_times)
