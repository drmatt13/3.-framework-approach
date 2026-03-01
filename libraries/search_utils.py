import numpy as np


def cv_scoring_name(name: str, mapping: dict[str, str], option_label: str = "--cv-scoring") -> str:
	if name not in mapping:
		allowed = ", ".join(mapping.keys())
		raise ValueError(f"Unsupported {option_label} '{name}'. Choose from: {allowed}")
	return mapping[name]


def search_space_size(search_space: dict[str, list] | list[dict[str, list]]) -> int:
	if isinstance(search_space, dict):
		lengths = [len(values) for values in search_space.values() if isinstance(values, list)]
		return int(np.prod(lengths)) if lengths else 0

	total = 0
	for space in search_space:
		lengths = [len(values) for values in space.values() if isinstance(values, list)]
		total += int(np.prod(lengths)) if lengths else 0
	return total
