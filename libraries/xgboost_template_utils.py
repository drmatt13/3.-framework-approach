from functools import lru_cache

import numpy as np
import xgboost as xgb


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
	"""Return True when a CUDA-capable XGBoost runtime is available."""
	try:
		probe = xgb.XGBRegressor(
			device="cuda",
			n_estimators=1,
			max_depth=1,
			learning_rate=0.1,
			objective="reg:squarederror",
			verbosity=0,
		)
		x_probe = np.array([[0.0], [1.0]], dtype=np.float32)
		y_probe = np.array([0.0, 1.0], dtype=np.float32)
		probe.fit(x_probe, y_probe)
		return True
	except Exception:
		return False


def resolve_xgboost_device(device_flag: str) -> tuple[str, str | None]:
	normalized = str(device_flag or "auto").strip().lower()

	if normalized == "cpu":
		return "cpu", None

	if normalized == "gpu":
		if _cuda_available():
			return "cuda", None
		return "cpu", "Requested --device=gpu, but CUDA-capable XGBoost runtime is unavailable; using CPU instead."

	if normalized == "auto":
		if _cuda_available():
			return "cuda", "Auto device selected CUDA for XGBoost."
		return "cpu", None

	return "cpu", f"Unknown --device value '{device_flag}'. Falling back to CPU."
