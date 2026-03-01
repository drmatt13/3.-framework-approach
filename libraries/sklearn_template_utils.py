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
