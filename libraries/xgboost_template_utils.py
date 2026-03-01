def resolve_xgboost_device(device_flag: str) -> tuple[str, str | None]:
	if device_flag == "gpu":
		message = (
			"Requested --device=gpu, but this template preprocesses features on CPU via scikit-learn Pipeline. "
			"Using CPU for XGBoost to avoid repeated device-mismatch fallback warnings during CV and prediction."
		)
		return "cpu", message
	return "cpu", None
