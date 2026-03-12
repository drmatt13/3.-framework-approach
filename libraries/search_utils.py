import numpy as np
from itertools import product


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


def enumerate_search_candidates(search_space: dict[str, list] | list[dict[str, list]]) -> list[dict[str, object]]:
	spaces = [search_space] if isinstance(search_space, dict) else list(search_space)
	results: list[dict[str, object]] = []
	for space in spaces:
		keys = [key for key, values in space.items() if isinstance(values, list)]
		if not keys:
			continue
		value_lists = [list(space[key]) for key in keys]
		for combo in product(*value_lists):
			results.append({key: value for key, value in zip(keys, combo)})
	return results
