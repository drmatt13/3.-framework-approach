import argparse
import hashlib
import json
from functools import partial
import numpy as np
import pickle
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure project root is importable so generated templates can load shared helpers.
_current_file = Path(__file__).resolve()
for _candidate in [_current_file.parent, *_current_file.parents]:
	if (_candidate / "libraries" / "__init__.py").exists():
		if str(_candidate) not in sys.path:
			sys.path.insert(0, str(_candidate))
		break

from libraries.model_template_helpers import (
	artifact_map as _artifact_map,
	compact_metadata as _compact_metadata,
	find_project_root as _project_root,
	parse_bool_flag as _parse_bool,
	post_transform_feature_count as _post_transform_feature_count,
	round_metric as _round_metric_base,
	select_estimator_params as _select_estimator_params,
	validate_etl_outputs as _validate_etl_outputs,
	write_model_schemas as _write_model_schemas,
)

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (common usage)
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
#   --verbose 0|1|2|auto
#   --metric-decimals <int>
#   --penalty none|l1|l2|elasticnet
#   --alpha <float>
#   --fit-intercept true|false
#   --l1-ratio <float>
#   --enable-tuning true|false
#   --tuning-method grid|random
#   --cv-folds <int>
#   --cv-scoring rmse|mae|r2
#   --cv-n-iter <int>
#   --cv-n-jobs <int>
# ---------------------------------------------------------------------

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4
DEFAULT_PENALTY = "none"
DEFAULT_ALPHA = 1.0 # type: ignore
DEFAULT_FIT_INTERCEPT = "True" == "True"
DEFAULT_L1_RATIO = float("0.5")
DEFAULT_ENABLE_TUNING = "False" == "True"
DEFAULT_TUNING_METHOD = "grid"
DEFAULT_CV_FOLDS = int("5")
DEFAULT_CV_SCORING = "rmse"
DEFAULT_CV_N_ITER = int("20")
DEFAULT_CV_N_JOBS = int("-1")

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Linear Regression with optional hyperparameter tuning")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
parser.add_argument("--penalty", choices=["none", "l1", "l2", "elasticnet"], default=DEFAULT_PENALTY)
parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
parser.add_argument("--fit-intercept", type=_parse_bool, default=DEFAULT_FIT_INTERCEPT)
parser.add_argument("--l1-ratio", type=float, default=DEFAULT_L1_RATIO)
parser.add_argument("--enable-tuning", type=_parse_bool, default=DEFAULT_ENABLE_TUNING)
parser.add_argument("--tuning-method", choices=["grid", "random"], default=DEFAULT_TUNING_METHOD)
parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
parser.add_argument("--cv-scoring", choices=["rmse", "mae", "r2"], default=DEFAULT_CV_SCORING)
parser.add_argument("--cv-n-iter", type=int, default=DEFAULT_CV_N_ITER)
parser.add_argument("--cv-n-jobs", type=int, default=DEFAULT_CV_N_JOBS)
args = parser.parse_args()
SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
METRIC_DECIMALS = int(args.metric_decimals)
_round_metric = partial(_round_metric_base, decimals=METRIC_DECIMALS)

# =============================================================
# ================== MODEL CODE STARTS HERE ===================
# =============================================================
# This section contains model definition, training, evaluation,
# and artifact generation logic.

# =============================================================
# ======================= DATA ETL ============================
# =============================================================

# ---------------------------------------------------------------------
# DATA ETL OUTPUT CONTRACT (required by all downstream sections)
# At end of ETL, you MUST define:
#   data_path: Path
#   df: pd.DataFrame
#   X: pd.DataFrame
#   y: pd.Series
#   target_column_name: str
#
# This block is intentionally swappable. Downstream code relies ONLY on
# this contract — not on dataset-specific assumptions.
# ---------------------------------------------------------------------

# ---------------------------------------------------------
# Load dataset and create initial dataframe
# ---------------------------------------------------------

# Template injection points:
#   - DATA_TASK_DIR / DATA_FILE
#   - READ_CSV_STATEMENT / POST_READ_DATASET_SETUP
data_path = _project_root() / "data" / "template_data" / "regression" / "ames_housing.csv"
df = pd.read_csv(data_path)


# Drop common CSV index artifacts (e.g., "Unnamed: 0") so they never leak into features.
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# ---------------------------------------------------------
# Normalize raw dataframe (dataset-level cleanup)
# ---------------------------------------------------------

# Goal: reduce category fragmentation and standardize missingness.
#   - trim whitespace in string-like columns
#   - convert empty strings to NaN
#   - normalize pd.NA -> np.nan for consistent downstream behavior
for column in df.select_dtypes(include=["object", "string"]).columns:
	series = df[column].astype("string").str.strip()
	series = series.replace("", np.nan)
	df[column] = series.astype("object")

df = df.replace({pd.NA: np.nan})

# ---------------------------------------------------------
# Define target + features (semantic boundary)
# ---------------------------------------------------------

# Template injection points:
#   - TARGET_COLUMN
#   - FEATURE_DROP_COLUMNS
TARGET_COLUMN = "SalePrice"
COLUMNS_TO_DROP = ["SalePrice", "Order", "PID"]

if TARGET_COLUMN not in df.columns:
	raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

# y is the supervised target; X is the feature space (minus optional drops).
y = df[TARGET_COLUMN]
y_original = y.copy()
X = df.drop(columns=[TARGET_COLUMN])

# ---------------------------------------------------------
# Drop unwanted feature columns (feature-level filtering)
# ---------------------------------------------------------

# Used for: IDs, leaky columns, high-missingness columns, or dataset-specific exclusions.
X = X.drop(columns=COLUMNS_TO_DROP, errors="ignore")

# ---------------------------------------------------------
# Remove rows with missing target
# ---------------------------------------------------------

# Training assumes y is defined. We drop rows where y is missing,
# while leaving missing values in X for downstream imputers to handle.
valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()
y_original = y_original.loc[valid_target_mask].copy()

target_column_name = str(TARGET_COLUMN)

if len(y) == 0:
	raise ValueError("No rows remain after target filtering. Check selected dataset and target column.")

# ---------------------------------------------------------
# Final ETL contract validation
# ---------------------------------------------------------

# Sanity-check the contract so downstream sections can run without defensive checks.
_validate_etl_outputs(
	data_path=data_path,
	df=df,
	X=X,
	y=y,
	target_column_name=target_column_name,
)

# =============================================================
# ================= PREPROCESSING / SPLIT =====================
# =============================================================

# Split BEFORE fitting transformers to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=args.test_size,
	random_state=args.random_state,
)

# Define column groups from training data only.
# Include "str" explicitly for pandas 3 compatibility.
categorical_cols = X_train.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

# OneHotEncoder compatibility: sparse_output (new) vs sparse (old).
try:
	one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
	one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Preprocess: impute missing values, then scale numeric and one-hot encode categorical features.
numeric_transformer = Pipeline(
	steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	]
)
categorical_transformer = Pipeline(
	steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", one_hot_encoder),
	]
)

preprocessor = ColumnTransformer(
	transformers=[
		("num", numeric_transformer, numerical_cols),
		("cat", categorical_transformer, categorical_cols),
	],
	remainder="drop",
)

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================


def _base_regressor_from_flags() -> LinearRegression | Ridge | Lasso | ElasticNet:
	if args.penalty == "none":
		return LinearRegression(fit_intercept=args.fit_intercept)
	if args.penalty == "l2":
		return Ridge(alpha=args.alpha, fit_intercept=args.fit_intercept, random_state=args.random_state)
	if args.penalty == "l1":
		return Lasso(
			alpha=args.alpha,
			fit_intercept=args.fit_intercept,
			random_state=args.random_state,
			max_iter=10_000,
		)
	if args.penalty == "elasticnet":
		return ElasticNet(
			alpha=args.alpha,
			l1_ratio=args.l1_ratio,
			fit_intercept=args.fit_intercept,
			random_state=args.random_state,
			max_iter=10_000,
		)
	raise ValueError(f"Unsupported --penalty '{args.penalty}'. Choose from: none, l1, l2, elasticnet")


def _cv_scoring_name(name: str) -> str:
	mapping = {
		"rmse": "neg_root_mean_squared_error",
		"mae": "neg_mean_absolute_error",
		"r2": "r2",
	}
	if name not in mapping:
		raise ValueError(f"Unsupported --cv-scoring '{name}'. Choose from: rmse, mae, r2")
	return mapping[name]


def _build_search_space(penalty: str, random_state: int) -> list[dict[str, list]]:
	if penalty == "none":
		return [
			{
				"regressor": [LinearRegression()],
				"regressor__fit_intercept": [True, False],
			}
		]
	if penalty == "l2":
		return [
			{
				"regressor": [Ridge(random_state=random_state)],
				"regressor__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
				"regressor__fit_intercept": [True, False],
			}
		]
	if penalty == "l1":
		return [
			{
				"regressor": [Lasso(random_state=random_state)],
				"regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
				"regressor__fit_intercept": [True, False],
				"regressor__max_iter": [5_000, 10_000, 20_000],
			}
		]
	if penalty == "elasticnet":
		return [
			{
				"regressor": [ElasticNet(random_state=random_state)],
				"regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
				"regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
				"regressor__fit_intercept": [True, False],
				"regressor__max_iter": [5_000, 10_000, 20_000],
			}
		]
	raise ValueError(f"Unsupported penalty '{penalty}'")


def _grid_size(search_space: list[dict[str, list]]) -> int:
	total = 0
	for space in search_space:
		lengths = [len(values) for values in space.values() if isinstance(values, list)]
		total += int(np.prod(lengths)) if lengths else 0
	return total


def _json_safe_param_value(value):
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
		return [_json_safe_param_value(item) for item in value]
	if isinstance(value, dict):
		return {str(key): _json_safe_param_value(item) for key, item in value.items()}
	if hasattr(value, "get_params"):
		return type(value).__name__
	return str(value)


def _json_safe_best_params(params: dict[str, object]) -> dict[str, object]:
	return {str(key): _json_safe_param_value(value) for key, value in params.items()}


# Bundle preprocessing + model into one inference-ready pipeline.
model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("regressor", _base_regressor_from_flags()),
	]
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================

selected_cv_scoring = _cv_scoring_name(args.cv_scoring)
tuning_summary = {
	"enabled": False,
	"method": None,
	"cv_folds": None,
	"scoring": None,
	"scoring_sklearn": None,
	"n_iter": None,
	"n_candidates": None,
	"best_score": None,
	"best_score_std": None,
	"best_params": None,
}

fit_started_at = time.perf_counter()
if training_verbose > 0:
	if args.enable_tuning:
		print(
			f"Training started with tuning: method={args.tuning_method}, penalty={args.penalty}, "
			f"cv={args.cv_folds}, scoring={args.cv_scoring}"
		)
	else:
		print(f"Training started: {type(model.named_steps['regressor']).__name__} (penalty={args.penalty}, alpha={args.alpha})")

if args.enable_tuning:
	search_space = _build_search_space(args.penalty, int(args.random_state))
	search = None
	if args.tuning_method == "grid":
		search = GridSearchCV(
			estimator=model,
			param_grid=search_space,
			scoring=selected_cv_scoring,
			cv=int(args.cv_folds),
			n_jobs=int(args.cv_n_jobs),
			refit=False,
		)
	else:
		n_iter = int(args.cv_n_iter)
		n_candidates_upper = _grid_size(search_space)
		if n_candidates_upper > 0:
			n_iter = min(n_iter, n_candidates_upper)
		search = RandomizedSearchCV(
			estimator=model,
			param_distributions=search_space,
			n_iter=int(n_iter),
			scoring=selected_cv_scoring,
			cv=int(args.cv_folds),
			n_jobs=int(args.cv_n_jobs),
			refit=False,
			random_state=int(args.random_state),
		)

	search.fit(X_train, y_train)
	best_params = dict(search.best_params_)
	best_params_for_artifacts = _json_safe_best_params(best_params)
	model.set_params(**best_params)
	model.fit(X_train, y_train)

	best_score = float(search.best_score_)
	if args.cv_scoring in ("rmse", "mae"):
		best_score = -best_score
	best_std = None
	if hasattr(search, "cv_results_") and "std_test_score" in search.cv_results_:
		best_std = float(search.cv_results_["std_test_score"][search.best_index_])
		if args.cv_scoring in ("rmse", "mae"):
			best_std = abs(best_std)

	n_candidates = int(len(search.cv_results_["params"])) if hasattr(search, "cv_results_") else None
	tuning_summary = {
		"enabled": True,
		"method": args.tuning_method,
		"cv_folds": int(args.cv_folds),
		"scoring": args.cv_scoring,
		"scoring_sklearn": selected_cv_scoring,
			"n_iter": int(search.n_iter) if args.tuning_method == "random" else None,
		"n_candidates": n_candidates,
		"best_score": _round_metric(best_score),
		"best_score_std": _round_metric(best_std) if best_std is not None else None,
		"best_params": _compact_metadata(best_params_for_artifacts),
	}
else:
	model.fit(X_train, y_train)

fit_time_seconds = float(time.perf_counter() - fit_started_at)
regressor_name = type(model.named_steps["regressor"]).__name__
if training_verbose > 0:
	print(f"Training completed in {fit_time_seconds:.3f}s: {regressor_name}")

training_control = {
	"enabled": bool(tuning_summary["enabled"]),
	"type": f"{args.tuning_method}_search_cv" if tuning_summary["enabled"] else None,
	"max_steps_configured": tuning_summary["n_candidates"] if args.tuning_method == "grid" else (int(tuning_summary["n_iter"]) if tuning_summary["enabled"] and tuning_summary["n_iter"] is not None else None),
	"steps_completed": int(args.cv_folds) * int(tuning_summary["n_candidates"]) if tuning_summary["enabled"] and tuning_summary["n_candidates"] is not None else None,
	"patience": None,
	"monitor_metric": f"cv_{args.cv_scoring}" if tuning_summary["enabled"] else None,
	"monitor_split": "cv" if tuning_summary["enabled"] else None,
	"monitor_direction": ("min" if args.cv_scoring in ("rmse", "mae") else "max") if tuning_summary["enabled"] else None,
	"best_step": None,
	"best_score": tuning_summary["best_score"],
	"stopped_early": False,
}

# =============================================================
# ==================== EVALUATE MODEL =========================
# =============================================================

# Evaluate model on train/test splits.
predict_started_at = time.perf_counter()
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)
predict_time_seconds = float(time.perf_counter() - predict_started_at)
train_mse = mean_squared_error(y_train, train_predictions)
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = root_mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)
train_max_error = max_error(y_train, train_predictions)
train_residuals = y_train - train_predictions
train_residual_mean = float(train_residuals.mean())
train_residual_std = float(train_residuals.std())

# Test metrics
test_mse = mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_rmse = root_mean_squared_error(y_test, predictions)
test_r2 = r2_score(y_test, predictions)
test_max_error = max_error(y_test, predictions)
target_mean_train = float(y_train.mean())
target_std_train = float(y_train.std())

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model fit on data it learned from) ----
print("Train MSE:", _round_metric(train_mse))  # Mean Squared Error on training set (average squared residuals; penalizes large errors heavily)
print("Train MAE:", _round_metric(train_mae))  # Mean Absolute Error on training set (average absolute prediction error)
print("Train RMSE:", _round_metric(train_rmse))  # Root Mean Squared Error on training set (error in original target units)
print("Train R2:", _round_metric(train_r2))  # R² on training set (proportion of variance explained by model)
print("Train Max Error:", _round_metric(train_max_error))  # Largest single absolute prediction error on training set
print("Train Residual Mean:", _round_metric(train_residual_mean))  # Mean of residuals (should be ~0 for unbiased linear regression)
print("Train Residual Std:", _round_metric(train_residual_std))  # Standard deviation of residuals (spread of prediction errors)

# ---- Test Metrics (model performance on unseen data) ----
print("Test MSE:", _round_metric(test_mse))  # Mean Squared Error on test set (average squared prediction errors)
print("Test MAE:", _round_metric(test_mae))  # Mean Absolute Error on test set (average absolute difference from true values)
print("Test RMSE:", _round_metric(test_rmse))  # Root Mean Squared Error on test set (interpretable error in target units)
print("Test R2:", _round_metric(test_r2))  # R² score on test set (generalization performance)
print("Test Max Error:", _round_metric(test_max_error))  # Largest single absolute prediction error on test set (worst-case mistake)
print("Target Mean (Train):", _round_metric(target_mean_train))  # Train-split target mean for leakage-safe summary
print("Target Std (Train):", _round_metric(target_std_train))  # Train-split target standard deviation for leakage-safe summary
print("Training Control Enabled:", training_control["enabled"])  # Whether iterative training control / early stopping was used

# ---- Sanity Checks ----
print("First 5 predictions:", predictions[:5])  # Sample predictions for quick sanity check
print("First 5 true values:", y_test.iloc[:5].tolist())  # Corresponding true values for sanity check

# =============================================================
# ========= EXPORT ARTIFACTS & MODEL REGISTRY ================
# =============================================================
	
# Artifact export and registry logging.
if SAVE_MODEL:
	model_name = args.name.strip() or Path(__file__).stem
	model_root_dir = _project_root() / "artifacts" / "models" / model_name
	timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
	run_id = str(uuid.uuid4())
	data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
	data_rows = int(len(df))
	data_columns = int(df.shape[1])
	if args.artifact_name_mode == "short":
		run_label = f"{timestamp}_{model_name[:24]}_{run_id.split('-')[0]}"
	else:
		run_label = f"{timestamp}_{model_name}"
	run_dir = model_root_dir / run_label

	model_dir = run_dir / "model"
	preprocess_dir = run_dir / "preprocess"
	eval_dir = run_dir / "eval"
	data_dir = run_dir / "data"
	inference_dir = run_dir / "inference"

	for directory in (model_dir, preprocess_dir, eval_dir, data_dir, inference_dir):
		directory.mkdir(parents=True, exist_ok=True)

	with (model_dir / "model.pkl").open("wb") as model_file:
		# Saves full inference-ready pipeline: preprocess + regressor.
		pickle.dump(model, model_file)

	with (preprocess_dir / "preprocessor.pkl").open("wb") as preprocess_file:
		pickle.dump(model.named_steps["preprocess"], preprocess_file)

	metrics = {
		"train": {
			"mse": _round_metric(train_mse),
			"mae": _round_metric(train_mae),
			"rmse": _round_metric(train_rmse),
			"r2": _round_metric(train_r2),
			"max_error": _round_metric(train_max_error),
			"residual_mean": _round_metric(train_residual_mean),
			"residual_std": _round_metric(train_residual_std),
		},
		"test": {
			"mse": _round_metric(test_mse),
			"mae": _round_metric(test_mae),
			"rmse": _round_metric(test_rmse),
			"r2": _round_metric(test_r2),
			"max_error": _round_metric(test_max_error),
		},
		"target_summary": {
			"split": "train",
			"mean": _round_metric(target_mean_train),
			"std": _round_metric(target_std_train),
		},
		"data_sizes": {
			"n_train": int(len(X_train)),
			"n_val": 0,
			"n_test": int(len(X_test)),
		},
		"primary_metric": {
			"name": "rmse",
			"split": "test",
			"direction": "min",
			"value": _round_metric(test_rmse),
		},
		"tuning": tuning_summary,
	}
	metrics["training_control"] = training_control
	metrics["selection"] = training_control
	metrics["calibration"] = {"source": None, "calibrated": None, "calibration_method": None}
	metrics["timing"] = {"fit_seconds": _round_metric(fit_time_seconds), "predict_seconds": _round_metric(predict_time_seconds)}
	with (eval_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
		json.dump(metrics, metrics_file, indent=2)

	predictions_preview = pd.DataFrame(
		{
			"y_true": y_test.iloc[:50].tolist(),
			"y_pred": predictions[:50].tolist(),
		}
	)
	predictions_preview.to_csv(eval_dir / "predictions_preview.csv", index=False)

	# Example inference rows are kept in raw feature format on purpose.
	# Do NOT pre-one-hot-encode these rows; model.pkl handles preprocessing.
	inference_rows = X_test.iloc[:5].to_dict(orient="records")
	expected_values = y_test.iloc[:5].tolist()
	sample_rows_literal = json.dumps(inference_rows, indent=2)
	sample_rows_literal = sample_rows_literal.replace(": NaN", ": np.nan")
	sample_rows_literal = sample_rows_literal.replace(": Infinity", ": np.inf")
	sample_rows_literal = sample_rows_literal.replace(": -Infinity", ": -np.inf")
	inference_script = f'''import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

sample_rows = {sample_rows_literal}
expected_y = {json.dumps(expected_values, indent=2)}

with MODEL_PATH.open("rb") as model_file:
	model = pickle.load(model_file)

features = pd.DataFrame(sample_rows)
predictions = model.predict(features)

print("Inference Example")
print("Input Rows:", len(features))
print("Predictions:", predictions.tolist())
print("Expected:", expected_y)

results = features.copy()
results["y_expected"] = expected_y
results["y_pred"] = predictions
print(results)
'''
	with (inference_dir / "inference_example.py").open("w", encoding="utf-8") as inference_file:
		inference_file.write(inference_script)

	schema_artifacts = _write_model_schemas(
		schema_dir=data_dir,
		X_raw=X,
		y_model=y,
		target_column_name=target_column_name,
		transformed_features=model.named_steps["preprocess"].transform(X_train.iloc[:1]),
		preprocessor=model.named_steps["preprocess"],
		y_original=y_original,
	)

	post_transform_feature_count = _post_transform_feature_count(model.named_steps["preprocess"], X_train.iloc[:1])
	regressor_params = model.named_steps["regressor"].get_params()
	_lr_param_keys = [
		"fit_intercept",
		"copy_X",
		"tol",
		"positive",
		"n_jobs",
	]
	# Include regularization-specific params when a penalty is active.
	if args.penalty != "none":
		_lr_param_keys.extend(["alpha", "max_iter"])
	if args.penalty == "elasticnet":
		_lr_param_keys.append("l1_ratio")
	estimator_params_compact = _select_estimator_params(regressor_params, _lr_param_keys)

	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": "scikit-learn",
		"task": "regression",
		"algorithm": "linear_regression",
		"estimator_class": type(model.named_steps["regressor"]).__name__,
		"model_id": f"sklearn.{type(model.named_steps['regressor']).__name__.lower()}",
		"dataset": {
			"path": str(data_path.relative_to(_project_root())),
			"sha256": data_hash,
			"rows": data_rows,
			"columns": data_columns,
		},
		"data_split": {
			"strategy": "train_test_split",
			"test_size": float(args.test_size),
			"random_state": int(args.random_state),
			"stratify": None,
			"validation": {
				"enabled": bool(tuning_summary["enabled"]),
				"strategy": f"{args.tuning_method}_search_cv" if tuning_summary["enabled"] else None,
				"validation_fraction": None,
				"random_state": int(args.random_state) if tuning_summary["enabled"] else None,
			},
			"sizes": {
				"n_rows": data_rows,
				"n_train": int(len(X_train)),
				"n_val": 0,
				"n_test": int(len(X_test)),
			},
		},
		"preprocessing": {
			"pipeline_class": "Pipeline",
			"scaler": "StandardScaler",
			"encoder": "OneHotEncoder",
			"one_hot_handle_unknown": "ignore",
			"imputer": {
				"numeric": "median",
				"categorical": "most_frequent",
			},
			"feature_count": {
				"raw": int(X.shape[1]),
				"post_transform": post_transform_feature_count,
			},
		},
		"params": {
			"estimator_params": _compact_metadata(estimator_params_compact),
			"test_size": float(args.test_size),
			"random_state": int(args.random_state),
			"penalty": args.penalty,
			"alpha": float(args.alpha),
			"fit_intercept": bool(args.fit_intercept),
			"l1_ratio": float(args.l1_ratio) if args.penalty == "elasticnet" else None,
			"enable_tuning": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_folds": int(args.cv_folds) if tuning_summary["enabled"] else None,
			"cv_scoring": args.cv_scoring if tuning_summary["enabled"] else None,
			"cv_n_iter": int(args.cv_n_iter) if tuning_summary["enabled"] and args.tuning_method == "random" else None,
			"cv_n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"tuning": tuning_summary,
		"training_control": training_control,
		"selection": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"artifacts": _artifact_map(
			run_dir,
			{
				"model": model_dir / "model.pkl",
				"preprocess": preprocess_dir / "preprocessor.pkl",
				"eval_metrics": eval_dir / "metrics.json",
				"eval_predictions_preview": eval_dir / "predictions_preview.csv",
				**schema_artifacts,
				"inference_example": inference_dir / "inference_example.py",
			},
		),
		"versions": {
			"python": platform.python_version(),
			"pandas": pd.__version__,
			"scikit-learn": sklearn.__version__,
		},
	}
	run_metadata = _compact_metadata(run_metadata)
	with (run_dir / "run.json").open("w", encoding="utf-8") as run_file:
		json.dump(run_metadata, run_file, indent=2)

	registry_path = model_root_dir / "model_registry.csv"
	if registry_path.exists():
		registry_df = pd.read_csv(registry_path)
		if "model_id" in registry_df.columns and not registry_df.empty:
			model_id = int(registry_df["model_id"].max()) + 1
		else:
			model_id = 1
	else:
		registry_df = pd.DataFrame()
		model_id = 1

	registry_row = pd.DataFrame(
		[
			{
				"model_id": model_id,
				"run_id": run_id,
				"name": model_name,
				"timestamp": timestamp,
				"dataset_sha256": data_hash,
				"dataset_rows": data_rows,
				"dataset_columns": data_columns,
				"random_state": int(args.random_state),
				"tuning_enabled": bool(tuning_summary["enabled"]),
				"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
				"cv_best_score": tuning_summary["best_score"],
				"mse": _round_metric(test_mse),
				"mae": _round_metric(test_mae),
				"rmse": _round_metric(test_rmse),
				"r2": _round_metric(test_r2),
				"n_train": int(len(X_train)),
				"n_test": int(len(X_test)),
			}
		]
	)
	registry_df = pd.concat([registry_df, registry_row], ignore_index=True)
	registry_df.to_csv(registry_path, index=False)

	print(f"Artifacts exported to: {run_dir}")
