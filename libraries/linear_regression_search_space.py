from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


@dataclass(frozen=True)
class LinearRegressionSearchGridConfig:
    fit_intercept: Sequence[bool]
    ridge_alpha: Sequence[float]
    lasso_alpha: Sequence[float]
    lasso_max_iter: Sequence[int]
    elasticnet_alpha: Sequence[float]
    elasticnet_l1_ratio: Sequence[float]
    elasticnet_max_iter: Sequence[int]


def _as_non_empty_list(values: Sequence[Any], *, field_name: str) -> list[Any]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence.")
    result = list(values)
    if not result:
        raise ValueError(f"{field_name} must not be empty.")
    return result


def _validate_config(config: LinearRegressionSearchGridConfig) -> None:
    _as_non_empty_list(config.fit_intercept, field_name="fit_intercept")
    _as_non_empty_list(config.ridge_alpha, field_name="ridge_alpha")
    _as_non_empty_list(config.lasso_alpha, field_name="lasso_alpha")
    _as_non_empty_list(config.lasso_max_iter, field_name="lasso_max_iter")
    _as_non_empty_list(config.elasticnet_alpha, field_name="elasticnet_alpha")
    _as_non_empty_list(config.elasticnet_l1_ratio, field_name="elasticnet_l1_ratio")
    _as_non_empty_list(config.elasticnet_max_iter, field_name="elasticnet_max_iter")


def build_linear_regression_search_space(
    penalty: str,
    random_state: int,
    config: LinearRegressionSearchGridConfig,
) -> list[dict[str, list]]:
    _validate_config(config)

    fit_intercept = _as_non_empty_list(config.fit_intercept, field_name="fit_intercept")
    ridge_alpha = _as_non_empty_list(config.ridge_alpha, field_name="ridge_alpha")
    lasso_alpha = _as_non_empty_list(config.lasso_alpha, field_name="lasso_alpha")
    lasso_max_iter = _as_non_empty_list(config.lasso_max_iter, field_name="lasso_max_iter")
    elasticnet_alpha = _as_non_empty_list(config.elasticnet_alpha, field_name="elasticnet_alpha")
    elasticnet_l1_ratio = _as_non_empty_list(config.elasticnet_l1_ratio, field_name="elasticnet_l1_ratio")
    elasticnet_max_iter = _as_non_empty_list(config.elasticnet_max_iter, field_name="elasticnet_max_iter")

    if penalty == "auto":
        return [
            {
                "regressor": [LinearRegression()],
                "regressor__fit_intercept": fit_intercept,
            },
            {
                "regressor": [Ridge(random_state=random_state)],
                "regressor__alpha": ridge_alpha,
                "regressor__fit_intercept": fit_intercept,
            },
            {
                "regressor": [Lasso(random_state=random_state)],
                "regressor__alpha": lasso_alpha,
                "regressor__fit_intercept": fit_intercept,
                "regressor__max_iter": lasso_max_iter,
            },
            {
                "regressor": [ElasticNet(random_state=random_state)],
                "regressor__alpha": elasticnet_alpha,
                "regressor__l1_ratio": elasticnet_l1_ratio,
                "regressor__fit_intercept": fit_intercept,
                "regressor__max_iter": elasticnet_max_iter,
            },
        ]
    if penalty == "none":
        return [
            {
                "regressor": [LinearRegression()],
                "regressor__fit_intercept": fit_intercept,
            }
        ]
    if penalty == "l2":
        return [
            {
                "regressor": [Ridge(random_state=random_state)],
                "regressor__alpha": ridge_alpha,
                "regressor__fit_intercept": fit_intercept,
            }
        ]
    if penalty == "l1":
        return [
            {
                "regressor": [Lasso(random_state=random_state)],
                "regressor__alpha": lasso_alpha,
                "regressor__fit_intercept": fit_intercept,
                "regressor__max_iter": lasso_max_iter,
            }
        ]
    if penalty == "elasticnet":
        return [
            {
                "regressor": [ElasticNet(random_state=random_state)],
                "regressor__alpha": elasticnet_alpha,
                "regressor__l1_ratio": elasticnet_l1_ratio,
                "regressor__fit_intercept": fit_intercept,
                "regressor__max_iter": elasticnet_max_iter,
            }
        ]
    raise ValueError(f"Unsupported penalty '{penalty}'")