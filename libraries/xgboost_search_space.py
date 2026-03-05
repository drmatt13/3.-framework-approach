from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class XGBoostSearchGridConfig:
	n_estimators: list[int] = field(default_factory=lambda: [100, 200])
	learning_rate: list[float] = field(default_factory=lambda: [0.05, 0.1])
	reg_lambda: list[float] = field(default_factory=lambda: [1.0])
	reg_alpha: list[float] = field(default_factory=lambda: [0.0])
	booster_when_auto: list[str] = field(default_factory=lambda: ["gbtree", "gblinear"])
	max_depth: list[int] = field(default_factory=lambda: [4, 6])
	subsample: list[float] = field(default_factory=lambda: [0.8, 1.0])
	colsample_bytree: list[float] = field(default_factory=lambda: [0.8, 1.0])
	min_child_weight: list[float] = field(default_factory=lambda: [1.0, 3.0])


_TREE_BOOSTERS = {"gbtree", "dart"}


def build_xgboost_search_space(
	*,
	step_name: str,
	booster: str,
	config: XGBoostSearchGridConfig,
) -> dict[str, list]:
	space: dict[str, list] = {
		f"{step_name}__n_estimators": list(config.n_estimators),
		f"{step_name}__learning_rate": list(config.learning_rate),
		f"{step_name}__reg_lambda": list(config.reg_lambda),
		f"{step_name}__reg_alpha": list(config.reg_alpha),
	}

	if booster == "auto":
		space[f"{step_name}__booster"] = list(config.booster_when_auto)
		space[f"{step_name}__max_depth"] = list(config.max_depth)
		space[f"{step_name}__subsample"] = list(config.subsample)
		space[f"{step_name}__colsample_bytree"] = list(config.colsample_bytree)
		space[f"{step_name}__min_child_weight"] = list(config.min_child_weight)
	elif booster in _TREE_BOOSTERS:
		space[f"{step_name}__max_depth"] = list(config.max_depth)
		space[f"{step_name}__subsample"] = list(config.subsample)
		space[f"{step_name}__colsample_bytree"] = list(config.colsample_bytree)
		space[f"{step_name}__min_child_weight"] = list(config.min_child_weight)

	return space
