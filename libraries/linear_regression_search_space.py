from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


@dataclass(frozen=True)
class LinearRegressionSearchGridConfig:
    fit_intercept_grid: Sequence[bool]
    ridge_alpha_grid: Sequence[float]
    lasso_alpha_grid: Sequence[float]
    lasso_max_iter_grid: Sequence[int]
    elasticnet_alpha_grid: Sequence[float]
    elasticnet_l1_ratio_grid: Sequence[float]
    elasticnet_max_iter_grid: Sequence[int]


def _as_non_empty_list(values: Sequence[Any], *, field_name: str) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence.")
    result = list(values)
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    return result


def _validate_config(config: LinearRegressionSearchGridConfig) -> None:
    _as_non_empty_list(config.fit_intercept_grid, field_name="fit_intercept_grid")
    _as_non_empty_list(config.ridge_alpha_grid, field_name="ridge_alpha_grid")
    _as_non_empty_list(config.lasso_alpha_grid, field_name="lasso_alpha_grid")
    _as_non_empty_list(config.lasso_max_iter_grid, field_name="lasso_max_iter_grid")
    _as_non_empty_list(config.elasticnet_alpha_grid, field_name="elasticnet_alpha_grid")
    _as_non_empty_list(config.elasticnet_l1_ratio_grid, field_name="elasticnet_l1_ratio_grid")
    _as_non_empty_list(config.elasticnet_max_iter_grid, field_name="elasticnet_max_iter_grid")


def build_linear_regression_search_space(
    penalty: str,
    random_state: int,
    config: LinearRegressionSearchGridConfig,
) -> list[dict[str, list]]:
    _validate_config(config)

    fit_intercept_grid = _as_non_empty_list(config.fit_intercept_grid, field_name="fit_intercept_grid")
    ridge_alpha_grid = _as_non_empty_list(config.ridge_alpha_grid, field_name="ridge_alpha_grid")
    lasso_alpha_grid = _as_non_empty_list(config.lasso_alpha_grid, field_name="lasso_alpha_grid")
    lasso_max_iter_grid = _as_non_empty_list(config.lasso_max_iter_grid, field_name="lasso_max_iter_grid")
    elasticnet_alpha_grid = _as_non_empty_list(config.elasticnet_alpha_grid, field_name="elasticnet_alpha_grid")
    elasticnet_l1_ratio_grid = _as_non_empty_list(config.elasticnet_l1_ratio_grid, field_name="elasticnet_l1_ratio_grid")
    elasticnet_max_iter_grid = _as_non_empty_list(config.elasticnet_max_iter_grid, field_name="elasticnet_max_iter_grid")

    if penalty == "auto":
        return [
            {
                "regressor": [LinearRegression()],
                "regressor__fit_intercept": fit_intercept_grid,
            },
            {
                "regressor": [Ridge(random_state=random_state)],
                "regressor__alpha": ridge_alpha_grid,
                "regressor__fit_intercept": fit_intercept_grid,
            },
            {
                "regressor": [Lasso(random_state=random_state)],
                "regressor__alpha": lasso_alpha_grid,
                "regressor__fit_intercept": fit_intercept_grid,
                "regressor__max_iter": lasso_max_iter_grid,
            },
            {
                "regressor": [ElasticNet(random_state=random_state)],
                "regressor__alpha": elasticnet_alpha_grid,
                "regressor__l1_ratio": elasticnet_l1_ratio_grid,
                "regressor__fit_intercept": fit_intercept_grid,
                "regressor__max_iter": elasticnet_max_iter_grid,
            },
        ]
    if penalty == "none":
        return [
            {
                "regressor": [LinearRegression()],
                "regressor__fit_intercept": fit_intercept_grid,
            }
        ]
    if penalty == "l2":
        return [
            {
                "regressor": [Ridge(random_state=random_state)],
                "regressor__alpha": ridge_alpha_grid,
                "regressor__fit_intercept": fit_intercept_grid,
            }
        ]
    if penalty == "l1":
        return [
            {
                "regressor": [Lasso(random_state=random_state)],
                "regressor__alpha": lasso_alpha_grid,
                "regressor__fit_intercept": fit_intercept_grid,
                "regressor__max_iter": lasso_max_iter_grid,
            }
        ]
    if penalty == "elasticnet":
        return [
            {
                "regressor": [ElasticNet(random_state=random_state)],
                "regressor__alpha": elasticnet_alpha_grid,
                "regressor__l1_ratio": elasticnet_l1_ratio_grid,
                "regressor__fit_intercept": fit_intercept_grid,
                "regressor__max_iter": elasticnet_max_iter_grid,
            }
        ]
    raise ValueError(f"Unsupported penalty '{penalty}'")