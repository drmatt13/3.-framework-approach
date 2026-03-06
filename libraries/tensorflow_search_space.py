from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseNNSearchGridConfig:
	hidden_units: list[list[int]]
	activation: list[str]
	optimizer: list[str]
	learning_rate: list[float]
	batch_size: list[int]
	dropout: list[float]
	l1: list[float]
	l2: list[float]


def _regularization_matches(l1: float, l2: float, mode: str) -> bool:
	mode_normalized = str(mode).strip().lower()
	if mode_normalized == "auto":
		return True
	if mode_normalized == "none":
		return float(l1) == 0.0 and float(l2) == 0.0
	if mode_normalized == "l1":
		return float(l1) > 0.0 and float(l2) == 0.0
	if mode_normalized == "l2":
		return float(l1) == 0.0 and float(l2) > 0.0
	if mode_normalized == "l1_l2":
		return float(l1) > 0.0 and float(l2) > 0.0
	raise ValueError(f"Unsupported regularization mode '{mode}'.")


def build_dense_nn_search_candidates(
	*,
	config: DenseNNSearchGridConfig,
	optimizer: str = "auto",
	activation: str = "auto",
	regularization: str = "auto",
) -> list[dict[str, object]]:
	optimizer_normalized = str(optimizer).strip().lower()
	activation_normalized = str(activation).strip().lower()

	optimizer_values = [str(v).strip().lower() for v in config.optimizer]
	activation_values = [str(v).strip().lower() for v in config.activation]

	if optimizer_normalized == "auto":
		selected_optimizers = optimizer_values
	else:
		if optimizer_normalized not in optimizer_values:
			raise ValueError(f"Unsupported optimizer filter '{optimizer}'.")
		selected_optimizers = [optimizer_normalized]

	if activation_normalized == "auto":
		selected_activations = activation_values
	else:
		if activation_normalized not in activation_values:
			raise ValueError(f"Unsupported activation filter '{activation}'.")
		selected_activations = [activation_normalized]

	candidates: list[dict[str, object]] = []
	seen: set[tuple[object, ...]] = set()
	for opt in selected_optimizers:
		for act in selected_activations:
			for lr in config.learning_rate:
				for batch_size in config.batch_size:
					for hidden in config.hidden_units:
						for dropout in config.dropout:
							for l1 in config.l1:
								for l2 in config.l2:
									if not _regularization_matches(float(l1), float(l2), regularization):
										continue
									key = (
										opt,
										act,
										float(lr),
										int(batch_size),
										tuple(int(v) for v in hidden),
										float(dropout),
										float(l1),
										float(l2),
									)
									if key in seen:
										continue
									seen.add(key)
									candidates.append(
										{
											"optimizer": opt,
											"activation": act,
											"learning_rate": float(lr),
											"batch_size": int(batch_size),
											"hidden_layers": [int(v) for v in hidden],
											"dropout": float(dropout),
											"l1": float(l1),
											"l2": float(l2),
										}
									)
	return candidates