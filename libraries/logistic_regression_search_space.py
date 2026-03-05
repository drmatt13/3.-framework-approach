from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from libraries.logistic_compat import (
    LOGISTIC_SOLVER_PENALTY_COMPAT,
    NON_AUTO_LOGISTIC_SOLVERS,
)


@dataclass(frozen=True)
class LogisticRegressionSearchGridConfig:
    c_grid: Sequence[float]
    max_iter: Sequence[int]
    class_weight: Sequence[Any]
    elasticnet_l1_ratio: Sequence[float]
    solver_penalty_compat: Mapping[str, Sequence[str]] | None = None
    solver_order: Sequence[str] | None = None


def _as_non_empty_list(values: Sequence[Any], *, field_name: str) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence.")
    result = list(values)
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    return result


def _resolve_solver_order(config: LogisticRegressionSearchGridConfig, compat: Mapping[str, Sequence[str]]) -> list[str]:
    if config.solver_order is None:
        return [solver for solver in NON_AUTO_LOGISTIC_SOLVERS if solver in compat]
    return _as_non_empty_list(config.solver_order, field_name="solver_order")


def _resolve_compat(config: LogisticRegressionSearchGridConfig) -> dict[str, list[str]]:
    source = config.solver_penalty_compat or LOGISTIC_SOLVER_PENALTY_COMPAT
    resolved: dict[str, list[str]] = {}
    for solver, penalties in source.items():
        if solver == "auto":
            continue
        resolved[solver] = _as_non_empty_list(penalties, field_name=f"solver_penalty_compat[{solver}]")
    if not resolved:
        raise ValueError("solver_penalty_compat must contain at least one non-auto solver.")
    return resolved


def _validate_config(config: LogisticRegressionSearchGridConfig) -> None:
    _as_non_empty_list(config.c_grid, field_name="c_grid")
    _as_non_empty_list(config.max_iter, field_name="max_iter")
    _as_non_empty_list(config.class_weight, field_name="class_weight")
    _as_non_empty_list(config.elasticnet_l1_ratio, field_name="elasticnet_l1_ratio")
    compat = _resolve_compat(config)
    order = _resolve_solver_order(config, compat)
    if not any(solver in compat for solver in order):
        raise ValueError("solver_order must include at least one solver present in solver_penalty_compat.")


def build_logistic_regression_search_space(
    solver: str,
    penalty: str,
    random_state: int,
    config: LogisticRegressionSearchGridConfig,
) -> list[dict[str, list]]:
    _ = random_state
    _validate_config(config)

    c_grid = _as_non_empty_list(config.c_grid, field_name="c_grid")
    max_iter = _as_non_empty_list(config.max_iter, field_name="max_iter")
    class_weight = _as_non_empty_list(config.class_weight, field_name="class_weight")
    elasticnet_l1_ratio = _as_non_empty_list(config.elasticnet_l1_ratio, field_name="elasticnet_l1_ratio")

    compat = _resolve_compat(config)
    solver_order = _resolve_solver_order(config, compat)
    ordered_solvers = [candidate for candidate in solver_order if candidate in compat]
    unordered_remainder = [candidate for candidate in compat if candidate not in ordered_solvers]
    all_solvers = ordered_solvers + unordered_remainder

    grouped_non_elasticnet: dict[tuple[str, ...], list[str]] = {}
    elasticnet_solvers: list[str] = []

    for candidate_solver in all_solvers:
        penalties = compat[candidate_solver]
        if "elasticnet" in penalties:
            elasticnet_solvers.append(candidate_solver)

        non_elasticnet_penalties = tuple(
            valid_penalty
            for valid_penalty in ("none", "l1", "l2")
            if valid_penalty in penalties
        )
        if non_elasticnet_penalties:
            grouped_non_elasticnet.setdefault(non_elasticnet_penalties, []).append(candidate_solver)

    all_grids: list[dict[str, list]] = []
    for supported_penalties, grouped_solvers in grouped_non_elasticnet.items():
        all_grids.append(
            {
                "classifier__solver": grouped_solvers,
                "classifier__penalty": list(supported_penalties),
                "classifier__C": c_grid,
                "classifier__class_weight": class_weight,
                "classifier__max_iter": max_iter,
            }
        )

    if elasticnet_solvers:
        all_grids.append(
            {
                "classifier__solver": elasticnet_solvers,
                "classifier__penalty": ["elasticnet"],
                "classifier__l1_ratio": elasticnet_l1_ratio,
                "classifier__C": c_grid,
                "classifier__class_weight": class_weight,
                "classifier__max_iter": max_iter,
            }
        )

    filtered: list[dict[str, list]] = []
    for grid in all_grids:
        grid_solvers = grid["classifier__solver"]
        grid_penalties = grid["classifier__penalty"]

        if solver != "auto":
            grid_solvers = [candidate for candidate in grid_solvers if candidate == solver]
            if not grid_solvers:
                continue

        if penalty != "auto":
            grid_penalties = [candidate for candidate in grid_penalties if candidate == penalty]
            if not grid_penalties:
                continue

        new_grid = dict(grid)
        new_grid["classifier__solver"] = grid_solvers
        new_grid["classifier__penalty"] = grid_penalties
        filtered.append(new_grid)

    if not filtered:
        raise ValueError(
            f"No valid search space for solver='{solver}', penalty='{penalty}'. "
            "Check solver/penalty compatibility."
        )

    return filtered
