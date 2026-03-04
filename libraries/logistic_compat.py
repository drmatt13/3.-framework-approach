from __future__ import annotations

from typing import Iterable

NON_AUTO_LOGISTIC_SOLVERS: tuple[str, ...] = (
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
)

LOGISTIC_SOLVER_PENALTY_COMPAT: dict[str, tuple[str, ...]] = {
    "lbfgs": ("none", "l2"),
    "liblinear": ("l1", "l2"),
    "newton-cg": ("none", "l2"),
    "newton-cholesky": ("none", "l2"),
    "sag": ("none", "l2"),
    "saga": ("none", "l1", "l2", "elasticnet"),
}

_PENALTY_DISPLAY_ORDER: tuple[str, ...] = ("none", "l1", "l2", "elasticnet")


def compatible_penalties_for_solver(solver: str, *, include_none: bool = True) -> list[str]:
    penalties = list(LOGISTIC_SOLVER_PENALTY_COMPAT.get(solver, ()))
    if not include_none:
        penalties = [penalty for penalty in penalties if penalty != "none"]
    return [penalty for penalty in _PENALTY_DISPLAY_ORDER if penalty in penalties]


def compatible_solvers_for_penalty(penalty: str) -> list[str]:
    return [
        solver
        for solver in NON_AUTO_LOGISTIC_SOLVERS
        if penalty in LOGISTIC_SOLVER_PENALTY_COMPAT.get(solver, ())
    ]


def all_supported_penalties() -> list[str]:
    unique: set[str] = set()
    for values in LOGISTIC_SOLVER_PENALTY_COMPAT.values():
        unique.update(values)
    return [penalty for penalty in _PENALTY_DISPLAY_ORDER if penalty in unique]
