import argparse


def parse_optional_int(value: str | int | None) -> int | None:
	if value is None:
		return None
	if isinstance(value, int):
		return value
	text = str(value).strip().lower()
	if text in {"none", "null", ""}:
		return None
	return int(text)


def parse_max_features(value: str) -> str | float | None:
	text = str(value).strip().lower()
	if text in {"none", "null", ""}:
		return None
	if text in {"auto", "sqrt", "log2"}:
		return text
	try:
		fval = float(text)
		if 0.0 < fval <= 1.0:
			return fval
		return text
	except ValueError:
		return text


def parse_optional_max_samples(value: str | int | float | None) -> int | float | None:
	if value is None:
		return None
	v = str(value).strip().lower()
	if v in {"none", "null", ""}:
		return None
	try:
		if "." in v:
			parsed_float = float(v)
			if not (0.0 < parsed_float <= 1.0):
				raise ValueError("Float max_samples must be in range (0, 1].")
			return parsed_float
		parsed_int = int(v)
		if parsed_int <= 0:
			raise ValueError("Integer max_samples must be > 0.")
		return parsed_int
	except ValueError:
		raise
	except Exception as exc:
		raise argparse.ArgumentTypeError("--max-samples must be int, float in (0,1], or none") from exc


def resolved_n_iter(model_step) -> int | None:
	n_iter_value = getattr(model_step, "n_iter_", None)
	if n_iter_value is None:
		return None
	if hasattr(n_iter_value, "tolist"):
		n_iter_value = n_iter_value.tolist()
	if isinstance(n_iter_value, (list, tuple)):
		if not n_iter_value:
			return None
		return int(max(n_iter_value))
	return int(n_iter_value)
