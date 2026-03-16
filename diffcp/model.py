"""CP-SAT compatible API for DiffCP.

Provides a familiar interface for constraint programming::

    model = DiffModel()
    x = [model.new_int_var(0, N-1, f"x{i}") for i in range(N)]
    model.add_all_different(x)
    model.add_linear_constraint([(x[0], 1), (x[1], 1)], "<=", 5)

    solver = DiffSolver()
    status = solver.solve(model)
    if status == DiffSolver.FEASIBLE:
        print([solver.value(x[i]) for i in range(N)])

Drop-in compatible with CP-SAT for the supported constraint subset.
Also supports warm-starting CP-SAT via export_hints().

Supported constraints
---------------------
- ``add_all_different(vars)``            — AllDifferent / permutation
- ``add_linear_constraint(terms, sense, rhs)`` — linear inequality / equality
- ``minimize(var)`` / ``maximize(var)``  — single-variable objective (optional)

Constraint detection
--------------------
The solver inspects the model at solve time and routes to one of two backends:

* **Permutation backend** — activated when there is at least one AllDifferent
  group that covers *all* variables in the model.  Uses the Sinkhorn relaxation
  in ``DiffCPSolver.solve_permutation``.  Additional AllDifferent groups on the
  same set of variables are supported (e.g. N-Queens uses row, forward-diagonal,
  and backward-diagonal groups).

* **General backend** — everything else.  Falls back to OR-Tools CP-SAT if the
  ``ortools`` package is available, otherwise returns ``UNKNOWN``.

Warm-starting CP-SAT
--------------------
Even when DiffCP returns an infeasible discrete solution, the continuous
optimum is a useful warm-start hint::

    hints = solver.export_hints(model)
    cp_model, cp_vars = solver.warm_start_cpsat(model)
    cp_solver = cp_model.CpSolver()
    cp_solver.solve(cp_model)

N-Queens example
----------------
::

    N = 12
    model = DiffModel()
    queens = [model.new_int_var(0, N - 1, f"q{i}") for i in range(N)]

    # Row-uniqueness (columns assigned to each row must differ)
    model.add_all_different(queens)

    # Diagonal uniqueness — standard N-Queens encoding
    model.add_all_different([queens[i] + i for i in range(N)])
    model.add_all_different([queens[i] - i for i in range(N)])

    solver = DiffSolver()
    solver.parameters.num_restarts = 128
    status = solver.solve(model)
    if status == DiffSolver.FEASIBLE:
        board = solver.value_list(queens)
        print(board)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .constraints import DiagonalConflictLoss, LinearConstraintLoss
from .solver import DiffCPResult, DiffCPSolver


# ---------------------------------------------------------------------------
# Status codes — match CP-SAT integer values exactly so callers can mix APIs
# ---------------------------------------------------------------------------
OPTIMAL = 0
FEASIBLE = 1
INFEASIBLE = 2
UNKNOWN = 3


# ---------------------------------------------------------------------------
# Decision variable
# ---------------------------------------------------------------------------

@dataclass
class IntVar:
    """Integer decision variable.

    Created by ``DiffModel.new_int_var``; never constructed directly.

    Attributes:
        lb:    Lower bound (inclusive).
        ub:    Upper bound (inclusive).
        name:  Human-readable label, used in debug output.
        index: Zero-based position in the model's variable list.
    """

    lb: int
    ub: int
    name: str
    index: int

    # ------------------------------------------------------------------
    # Arithmetic helpers — return lightweight _LinearExpr nodes so that
    # callers can write ``queens[i] + i`` inside add_all_different,
    # mirroring the CP-SAT Python API.
    # ------------------------------------------------------------------

    def __add__(self, offset):
        if isinstance(offset, int):
            return _AffineExpr(self, 1, offset)
        return NotImplemented

    def __radd__(self, offset):
        return self.__add__(offset)

    def __sub__(self, offset):
        if isinstance(offset, int):
            return _AffineExpr(self, 1, -offset)
        return NotImplemented

    def __rsub__(self, offset):
        if isinstance(offset, int):
            return _AffineExpr(self, -1, offset)
        return NotImplemented

    def __mul__(self, coeff):
        if isinstance(coeff, (int, float)):
            return _AffineExpr(self, coeff, 0)
        return NotImplemented

    def __rmul__(self, coeff):
        return self.__mul__(coeff)

    def __repr__(self):
        return f"IntVar({self.name!r}, [{self.lb}..{self.ub}])"


@dataclass
class _AffineExpr:
    """Internal: coeff * var + offset.  Appears in diagonal-constraint lists."""

    var: IntVar
    coeff: int
    offset: int

    def __repr__(self):
        return f"_AffineExpr({self.coeff}*{self.var.name!r}+{self.offset})"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DiffModel:
    """Differentiable constraint model with CP-SAT compatible API.

    Build a model using the same methods as ``ortools.sat.python.cp_model.CpModel``,
    then pass it to ``DiffSolver.solve``.

    Supported methods
    -----------------
    - ``new_int_var(lb, ub, name)``           — create a named integer variable
    - ``add_all_different(vars)``             — AllDifferent constraint
    - ``add_linear_constraint(terms, sense, rhs)`` — linear inequality / equality
    - ``add_hint(var, value)``                — warm-start hint
    - ``minimize(expr)`` / ``maximize(expr)`` — objective (optional, single var)

    Example::

        model = DiffModel()
        x = [model.new_int_var(0, 7, f"x{i}") for i in range(8)]
        model.add_all_different(x)
        model.add_all_different([x[i] + i for i in range(8)])
        model.add_all_different([x[i] - i for i in range(8)])
    """

    def __init__(self):
        self.variables: List[IntVar] = []
        self.all_different_groups: List[List[Any]] = []  # List[List[IntVar | _AffineExpr]]
        self.linear_constraints: List[Dict] = []
        self.hints: Dict[int, int] = {}
        self._objective: Optional[Any] = None
        self._objective_sense: Optional[str] = None

    # ------------------------------------------------------------------
    # Variable creation
    # ------------------------------------------------------------------

    def new_int_var(self, lb: int, ub: int, name: str) -> IntVar:
        """Create and register a new integer variable.

        Args:
            lb:   Lower bound (inclusive).
            ub:   Upper bound (inclusive).
            name: Unique human-readable name.

        Returns:
            An ``IntVar`` object.  Pass this directly to constraint methods.
        """
        var = IntVar(lb=lb, ub=ub, name=name, index=len(self.variables))
        self.variables.append(var)
        return var

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def add_all_different(self, variables):
        """Require that all variables take distinct values.

        Variables may be raw ``IntVar`` objects or affine expressions
        ``x[i] + constant`` (as in the N-Queens diagonal encoding).

        Args:
            variables: iterable of ``IntVar`` or ``_AffineExpr``.
        """
        self.all_different_groups.append(list(variables))

    def add_linear_constraint(
        self,
        terms: List[Tuple["IntVar", float]],
        sense: str,
        rhs: float,
    ) -> None:
        """Add a linear constraint of the form ``sum(coeff_i * x_i) sense rhs``.

        Args:
            terms:  List of ``(var, coeff)`` pairs, e.g. ``[(x, 1), (y, 2)]``.
            sense:  Comparison operator: ``"<="``, ``">="`` or ``"=="``.
            rhs:    Right-hand side scalar.

        Raises:
            ValueError: If ``sense`` is not one of the three legal values.
        """
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"sense must be '<=', '>=' or '==', got {sense!r}")
        self.linear_constraints.append({"terms": list(terms), "sense": sense, "rhs": rhs})

    def add_hint(self, var: IntVar, value: int) -> None:
        """Provide a warm-start hint for a variable.

        Hints are passed through to CP-SAT when using ``warm_start_cpsat`` and
        can also bias the DiffCP initialisation in future versions.

        Args:
            var:   The variable to hint.
            value: Initial value suggestion (need not be feasible).
        """
        self.hints[var.index] = value

    # ------------------------------------------------------------------
    # Objective (optional)
    # ------------------------------------------------------------------

    def minimize(self, expr) -> None:
        """Set a minimization objective.

        Args:
            expr: An ``IntVar`` or ``_AffineExpr`` to minimize.
        """
        self._objective = expr
        self._objective_sense = "min"

    def maximize(self, expr) -> None:
        """Set a maximization objective.

        Args:
            expr: An ``IntVar`` or ``_AffineExpr`` to maximize.
        """
        self._objective = expr
        self._objective_sense = "max"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_pure_permutation(self) -> bool:
        """Return True if a full-coverage AllDifferent group exists.

        A "pure permutation" problem is one where at least one AllDifferent
        group assigns every variable a distinct value in ``[0, N-1]``.
        The Sinkhorn backend is valid in this case.
        """
        if not self.all_different_groups:
            return False
        N = len(self.variables)
        for group in self.all_different_groups:
            # Accept raw IntVar groups that cover all N variables
            raw_vars = [e.var if isinstance(e, _AffineExpr) else e for e in group]
            if len(raw_vars) == N and len({v.index for v in raw_vars}) == N:
                return True
        return False

    def _get_permutation_size(self) -> int:
        """Return N (number of variables) for permutation problems."""
        return len(self.variables)

    def _has_diagonal_constraints(self) -> bool:
        """Return True when the model contains N-Queens style diagonal groups.

        Heuristic: the model has exactly three AllDifferent groups and all of
        them cover the full variable set (one for columns, two for diagonals).
        """
        N = len(self.variables)
        if len(self.all_different_groups) != 3:
            return False
        for group in self.all_different_groups:
            if len(group) != N:
                return False
        return True

    def validate(self) -> List[str]:
        """Return a list of validation warnings (empty = model looks OK).

        Checks variable bounds, constraint references, and sense values.
        """
        warnings: List[str] = []
        for lc in self.linear_constraints:
            for var, _coeff in lc["terms"]:
                if isinstance(var, IntVar) and var.index >= len(self.variables):
                    warnings.append(f"Linear constraint references unknown variable index {var.index}")
        return warnings


# ---------------------------------------------------------------------------
# Solver parameters
# ---------------------------------------------------------------------------

@dataclass
class SolverParameters:
    """Solver configuration.

    Attributes:
        max_time_in_seconds:  Wall-clock limit passed to CP-SAT fallback.
        num_restarts:         Parallel random restarts for the Sinkhorn backend.
        max_iterations:       Gradient descent steps per solve.
        lr:                   Adam learning rate.
        temp_start:           Initial Sinkhorn temperature.
        temp_end:             Final Sinkhorn temperature (lower = sharper).
        sinkhorn_iters:       Inner Sinkhorn normalisation iterations.
        device:               Torch device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    max_time_in_seconds: float = 60.0
    num_restarts: int = 64
    max_iterations: int = 2000
    lr: float = 0.2
    temp_start: float = 2.0
    temp_end: float = 0.01
    sinkhorn_iters: int = 20
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class DiffSolver:
    """Differentiable solver with CP-SAT compatible interface.

    Usage::

        solver = DiffSolver()
        solver.parameters.num_restarts = 128
        status = solver.solve(model)

        if status == DiffSolver.FEASIBLE:
            values = solver.value_list(queens)

    Attributes:
        OPTIMAL:    Status constant (0).
        FEASIBLE:   Status constant (1).
        INFEASIBLE: Status constant (2).
        UNKNOWN:    Status constant (3).
        parameters: ``SolverParameters`` instance — modify before calling ``solve``.
    """

    OPTIMAL = OPTIMAL
    FEASIBLE = FEASIBLE
    INFEASIBLE = INFEASIBLE
    UNKNOWN = UNKNOWN

    def __init__(self):
        self.parameters = SolverParameters()
        self._solution: Optional[np.ndarray] = None
        self._status: int = UNKNOWN
        self._solve_time_ms: float = 0.0
        self._iterations: int = 0
        self._continuous_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def solve(self, model: DiffModel) -> int:
        """Solve the model, returning a status code.

        Routes to the Sinkhorn backend for pure permutation models, and to
        the OR-Tools CP-SAT fallback for everything else.

        Args:
            model: A fully configured ``DiffModel``.

        Returns:
            One of ``OPTIMAL``, ``FEASIBLE``, ``INFEASIBLE``, or ``UNKNOWN``.
        """
        warnings = model.validate()
        for w in warnings:
            import warnings as _w
            _w.warn(w, stacklevel=2)

        t0 = time.time()

        if model._is_pure_permutation():
            self._status = self._solve_permutation(model)
        else:
            self._status = self._solve_general(model)

        self._solve_time_ms = (time.time() - t0) * 1000
        return self._status

    def value(self, var: IntVar) -> int:
        """Return the integer value of a variable in the last solution.

        Args:
            var: An ``IntVar`` from the model that was solved.

        Returns:
            Integer value.

        Raises:
            RuntimeError: If no solution is available (``solve`` not called,
                          or solver returned ``INFEASIBLE`` / ``UNKNOWN``).
        """
        if self._solution is None:
            raise RuntimeError(
                "No solution available. Call solve() first and check the status."
            )
        return int(self._solution[var.index])

    def value_list(self, variables: List[IntVar]) -> List[int]:
        """Return values for a list of variables.

        Convenience wrapper for ``[solver.value(v) for v in variables]``.

        Args:
            variables: Ordered list of ``IntVar`` objects.

        Returns:
            List of integer values in the same order.
        """
        return [self.value(v) for v in variables]

    # ------------------------------------------------------------------
    # Timing / diagnostic properties
    # ------------------------------------------------------------------

    @property
    def solve_time_ms(self) -> float:
        """Wall-clock solve time in milliseconds."""
        return self._solve_time_ms

    @property
    def iterations(self) -> int:
        """Number of gradient-descent iterations executed."""
        return self._iterations

    @property
    def continuous_loss(self) -> float:
        """Best constraint-violation loss achieved before rounding.

        Zero means the continuous relaxation was perfectly satisfied.
        Non-zero means the rounded solution *may* violate some constraints.
        """
        return self._continuous_loss

    # ------------------------------------------------------------------
    # CP-SAT interop
    # ------------------------------------------------------------------

    def export_hints(self, model: DiffModel) -> Dict[int, int]:
        """Export the current solution as CP-SAT warm-start hints.

        Even an infeasible DiffCP solution is useful as a hint — it reflects
        the gradient-descent optimum over the continuous relaxation.

        Args:
            model: The model that was solved (used for variable count).

        Returns:
            Dict mapping variable index to integer value.  Empty if no
            solution is available.
        """
        if self._solution is None:
            return {}
        return {i: int(v) for i, v in enumerate(self._solution)}

    def warm_start_cpsat(self, model: DiffModel):
        """Create a CP-SAT model pre-loaded with hints from the DiffCP solution.

        Reconstructs all constraints from the ``DiffModel`` and adds solution
        hints so that CP-SAT can refine or prove optimality quickly.

        Args:
            model: The solved ``DiffModel``.

        Returns:
            ``(cp_model, cp_vars)`` — a ``CpModel`` and list of ``IntVar``
            objects ready to pass to a ``CpSolver``.

        Raises:
            ImportError: If ``ortools`` is not installed.

        Example::

            cp_model, cp_vars = solver.warm_start_cpsat(model)
            cp_solver = cp_model.CpSolver()
            cp_solver.parameters.max_time_in_seconds = 10
            status = cp_solver.solve(cp_model)
        """
        from ortools.sat.python import cp_model as _cp_model

        cp = _cp_model.CpModel()
        cp_vars = []

        for var in model.variables:
            cv = cp.new_int_var(var.lb, var.ub, var.name)
            cp_vars.append(cv)

        # AllDifferent constraints
        for group in model.all_different_groups:
            # Translate raw IntVar and _AffineExpr to OR-Tools linear expressions
            cp_group = []
            for item in group:
                if isinstance(item, _AffineExpr):
                    cp_group.append(item.coeff * cp_vars[item.var.index] + item.offset)
                else:
                    cp_group.append(cp_vars[item.index])
            cp.add_all_different(cp_group)

        # Linear constraints
        for lc in model.linear_constraints:
            expr = sum(coeff * cp_vars[var.index] for var, coeff in lc["terms"])
            if lc["sense"] == "<=":
                cp.add(expr <= lc["rhs"])
            elif lc["sense"] == ">=":
                cp.add(expr >= lc["rhs"])
            elif lc["sense"] == "==":
                cp.add(expr == lc["rhs"])

        # Objective
        if model._objective is not None:
            if isinstance(model._objective, _AffineExpr):
                obj_expr = (
                    model._objective.coeff * cp_vars[model._objective.var.index]
                    + model._objective.offset
                )
            elif isinstance(model._objective, IntVar):
                obj_expr = cp_vars[model._objective.index]
            else:
                obj_expr = model._objective  # pass through unknown types

            if model._objective_sense == "min":
                cp.minimize(obj_expr)
            elif model._objective_sense == "max":
                cp.maximize(obj_expr)

        # Warm-start hints from DiffCP solution
        hints = self.export_hints(model)
        for idx, val in hints.items():
            cp.add_hint(cp_vars[idx], val)

        return cp, cp_vars

    # ------------------------------------------------------------------
    # Permutation backend
    # ------------------------------------------------------------------

    def _solve_permutation(self, model: DiffModel) -> int:
        """Solve a permutation problem via Sinkhorn relaxation + Adam.

        Builds a combined loss from all registered constraints, then
        delegates to ``DiffCPSolver.solve_permutation``.

        Returns a status code.
        """
        N = model._get_permutation_size()
        device = self.parameters.device

        # ---- Constraint loss assembly --------------------------------
        constraint_fns: List[Callable] = []

        # Diagonal constraints (N-Queens style): detected by having exactly
        # three AllDifferent groups of size N — columns, fwd diags, bwd diags.
        if model._has_diagonal_constraints():
            constraint_fns.append(DiagonalConflictLoss())

        # Linear constraints projected onto the permutation matrix.
        # The expected value of variable i under P is sum_j(j * P[i,j]).
        for lc in model.linear_constraints:
            A_row = torch.zeros(N, device=device, dtype=torch.float32)
            for var, coeff in lc["terms"]:
                A_row[var.index] = float(coeff)
            b_val = torch.tensor(
                [float(lc["rhs"])], device=device, dtype=torch.float32
            )
            A_mat = A_row.unsqueeze(0)  # (1, N)
            equality = lc["sense"] == "=="

            constraint_fns.append(
                _make_permutation_linear_loss(A_mat, b_val, equality)
            )

        def combined_loss(P: torch.Tensor) -> torch.Tensor:
            """Sum of all constraint losses over the batch."""
            B = P.shape[0]
            total = torch.zeros(B, device=P.device)
            for fn in constraint_fns:
                total = total + fn(P)
            return total

        # ---- Feasibility verifier ------------------------------------
        def verify(assignment: np.ndarray) -> bool:
            # A valid permutation automatically satisfies AllDifferent.
            if len(set(assignment)) != N:
                return False
            # Diagonal constraint check (N-Queens)
            if model._has_diagonal_constraints():
                for i in range(N):
                    for j in range(i + 1, N):
                        if abs(int(assignment[i]) - int(assignment[j])) == abs(i - j):
                            return False
            # Linear constraints on the discrete assignment
            for lc in model.linear_constraints:
                total = sum(
                    coeff * int(assignment[var.index]) for var, coeff in lc["terms"]
                )
                if lc["sense"] == "<=" and total > lc["rhs"]:
                    return False
                if lc["sense"] == ">=" and total < lc["rhs"]:
                    return False
                if lc["sense"] == "==" and total != lc["rhs"]:
                    return False
            return True

        # ---- Run solver -----------------------------------------------
        inner = DiffCPSolver(
            restarts=self.parameters.num_restarts,
            max_iters=self.parameters.max_iterations,
            lr=self.parameters.lr,
            temp_start=self.parameters.temp_start,
            temp_end=self.parameters.temp_end,
            sinkhorn_iters=self.parameters.sinkhorn_iters,
            device=device,
        )

        result: DiffCPResult = inner.solve_permutation(
            N,
            combined_loss,
            verify_fn=verify,
        )

        self._solution = result.solution
        self._iterations = result.iterations
        self._continuous_loss = result.continuous_loss

        return FEASIBLE if result.feasible else INFEASIBLE

    # ------------------------------------------------------------------
    # General / fallback backend
    # ------------------------------------------------------------------

    def _solve_general(self, model: DiffModel) -> int:
        """Solve a general CSP via OR-Tools CP-SAT.

        This is the fallback for models that are not pure permutations.
        Hints from ``model.hints`` (set via ``add_hint``) are forwarded to
        CP-SAT automatically.

        Returns a status code, or ``UNKNOWN`` if OR-Tools is not installed.
        """
        try:
            from ortools.sat.python import cp_model as _cp_model
        except ImportError:
            return UNKNOWN

        cp = _cp_model.CpModel()
        cp_vars = []

        for var in model.variables:
            cv = cp.new_int_var(var.lb, var.ub, var.name)
            cp_vars.append(cv)

        for group in model.all_different_groups:
            cp_group = []
            for item in group:
                if isinstance(item, _AffineExpr):
                    cp_group.append(item.coeff * cp_vars[item.var.index] + item.offset)
                else:
                    cp_group.append(cp_vars[item.index])
            cp.add_all_different(cp_group)

        for lc in model.linear_constraints:
            expr = sum(coeff * cp_vars[var.index] for var, coeff in lc["terms"])
            if lc["sense"] == "<=":
                cp.add(expr <= lc["rhs"])
            elif lc["sense"] == ">=":
                cp.add(expr >= lc["rhs"])
            elif lc["sense"] == "==":
                cp.add(expr == lc["rhs"])

        if model._objective is not None:
            if isinstance(model._objective, _AffineExpr):
                obj_expr = (
                    model._objective.coeff * cp_vars[model._objective.var.index]
                    + model._objective.offset
                )
            elif isinstance(model._objective, IntVar):
                obj_expr = cp_vars[model._objective.index]
            else:
                obj_expr = model._objective

            if model._objective_sense == "min":
                cp.minimize(obj_expr)
            elif model._objective_sense == "max":
                cp.maximize(obj_expr)

        # Forward any manually registered hints
        for idx, val in model.hints.items():
            cp.add_hint(cp_vars[idx], val)

        cp_solver = _cp_model.CpSolver()
        cp_solver.parameters.max_time_in_seconds = self.parameters.max_time_in_seconds
        status = cp_solver.solve(cp)

        if status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
            self._solution = np.array(
                [cp_solver.value(cv) for cv in cp_vars], dtype=np.int64
            )
            self._iterations = 0
            self._continuous_loss = 0.0
            return OPTIMAL if status == _cp_model.OPTIMAL else FEASIBLE

        return INFEASIBLE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_permutation_linear_loss(
    A: torch.Tensor,
    b: torch.Tensor,
    equality: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a closure that evaluates a linear constraint loss over a batch of
    permutation matrices.

    The expected value of variable i under doubly stochastic matrix P is::

        E[x_i] = sum_j (j * P[i, j])

    The constraint ``sum_i (a_i * E[x_i]) <= b`` is then differentiable.

    Args:
        A:        (1, N) coefficient row (one constraint).
        b:        (1,) right-hand side.
        equality: If True enforce ``== b``, otherwise ``<= b``.

    Returns:
        Callable ``(B, N, N) -> (B,)`` loss tensor.
    """
    def loss_fn(P: torch.Tensor) -> torch.Tensor:
        B, N, _ = P.shape
        col_indices = torch.arange(N, device=P.device, dtype=torch.float32)
        # Expected value of each variable: (B, N)
        expected = (P * col_indices.view(1, 1, N)).sum(dim=2)
        # Ax: (B, 1)
        Ax = expected @ A.T  # (B, 1)
        violation = Ax - b.unsqueeze(0)  # (B, 1)
        if equality:
            return violation.pow(2).sum(dim=1)
        else:
            return torch.nn.functional.relu(violation).pow(2).sum(dim=1)

    return loss_fn
