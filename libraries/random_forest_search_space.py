from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RandomForestSearchGridConfig:
	n_estimators: list[int] = field(default_factory=lambda: [100, 200, 300, 500])
	max_depth: list[int | None] = field(default_factory=lambda: [None, 8, 16, 32])
	max_leaf_nodes: list[int | None] = field(default_factory=lambda: [None, 64, 128])
	max_features: list[str | float | None] = field(default_factory=lambda: [None, "sqrt", "log2", 1.0])
	max_samples_when_bootstrap: list[int | float | None] = field(default_factory=lambda: [None, 0.7, 1.0])
	min_weight_fraction_leaf: list[float] = field(default_factory=lambda: [0.0, 0.01])
	min_impurity_decrease: list[float] = field(default_factory=lambda: [0.0, 1e-6, 1e-4])
	ccp_alpha: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-3])


def _unique_preserve_order(values: list[Any]) -> list[Any]:
	seen = set()
	result: list[Any] = []
	for value in values:
		key = repr(value)
		if key in seen:
			continue
		seen.add(key)
		result.append(value)
	return result


def build_random_forest_search_space(
	*,
	step_name: str,
	n_estimators: int,
	max_depth: int | None,
	min_samples_split: int,
	min_samples_leaf: int,
	min_weight_fraction_leaf: float,
	max_leaf_nodes: int | None,
	min_impurity_decrease: float,
	max_features: str | float | None,
	bootstrap: bool,
	max_samples: int | float | None,
	ccp_alpha: float,
	random_state: int,
	config: RandomForestSearchGridConfig,
) -> list[dict[str, list[Any]]]:
	_ = random_state

	if max_depth is None:
		max_depth_candidates = list(config.max_depth)
	else:
		max_depth_candidates = _unique_preserve_order([max_depth, max(2, int(max_depth) // 2), int(max_depth) * 2])

	min_samples_split_candidates = _unique_preserve_order([2, int(min_samples_split), max(2, int(min_samples_split) * 2)])
	min_samples_leaf_candidates = _unique_preserve_order([1, int(min_samples_leaf), max(1, int(min_samples_leaf) * 2)])
	_ = min_weight_fraction_leaf
	min_weight_fraction_leaf_candidates = list(config.min_weight_fraction_leaf)

	if max_leaf_nodes is None:
		max_leaf_nodes_candidates = list(config.max_leaf_nodes)
	else:
		max_leaf_nodes_candidates = _unique_preserve_order([None, int(max_leaf_nodes), max(2, int(max_leaf_nodes) * 2)])

	_ = min_impurity_decrease
	min_impurity_decrease_candidates = list(config.min_impurity_decrease)

	if max_features is None:
		max_features_candidates = list(config.max_features)
	else:
		max_features_candidates = _unique_preserve_order([max_features, "sqrt", "log2", 1.0])

	_ = ccp_alpha
	ccp_alpha_candidates = list(config.ccp_alpha)

	prefix = f"{step_name}__"
	base: dict[str, list[Any]] = {
		f"{prefix}n_estimators": list(config.n_estimators),
		f"{prefix}max_depth": max_depth_candidates,
		f"{prefix}min_samples_split": min_samples_split_candidates,
		f"{prefix}min_samples_leaf": min_samples_leaf_candidates,
		f"{prefix}min_weight_fraction_leaf": min_weight_fraction_leaf_candidates,
		f"{prefix}max_leaf_nodes": max_leaf_nodes_candidates,
		f"{prefix}min_impurity_decrease": min_impurity_decrease_candidates,
		f"{prefix}max_features": max_features_candidates,
		f"{prefix}ccp_alpha": ccp_alpha_candidates,
		f"{prefix}bootstrap": [bool(bootstrap)],
	}
	if bootstrap:
		base[f"{prefix}max_samples"] = [max_samples] if max_samples is not None else list(config.max_samples_when_bootstrap)

	return [base]
