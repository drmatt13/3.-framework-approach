import numpy as np


def json_safe_param_value(value):
	if value is None:
		return None
	if isinstance(value, (bool, int, float, str)):
		if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
			return None
		return value
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.floating,)):
		numeric = float(value)
		if np.isnan(numeric) or np.isinf(numeric):
			return None
		return numeric
	if isinstance(value, (np.bool_,)):
		return bool(value)
	if isinstance(value, (list, tuple)):
		return [json_safe_param_value(item) for item in value]
	if isinstance(value, dict):
		return {str(key): json_safe_param_value(item) for key, item in value.items()}
	if hasattr(value, "get_params"):
		return type(value).__name__
	return str(value)


def json_safe_best_params(params: dict[str, object]) -> dict[str, object]:
	return {str(key): json_safe_param_value(value) for key, value in params.items()}
