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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
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
#   --library scikit-learn
#   --model linear_regression
#   --task regression
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
#   --verbose 0|1|2|auto
#   --metric-decimals <int>
# ---------------------------------------------------------------------

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Linear Regression baseline")
parser.add_argument("--library", choices=["scikit-learn"], default="scikit-learn")
parser.add_argument("--model", choices=["linear_regression"], default="linear_regression")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
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
#   project_root: Path
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
# Load dataset
# ---------------------------------------------------------

# Template injection points:
#   - DATA_TASK_DIR / DATA_FILE
#   - READ_CSV_STATEMENT / POST_READ_DATASET_SETUP
project_root = _project_root()
data_path = project_root / "data" / "template_data" / "{{DATA_TASK_DIR}}" / "{{DATA_FILE}}"
{{READ_CSV_STATEMENT}}
{{POST_READ_DATASET_SETUP}}

# Drop common CSV index artifacts (e.g., "Unnamed: 0") so they never leak into features.
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# Optional metadata for run logs / reverse-mapping (mainly for classification).
target_dtype_original = None
target_label_map = None

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
#   - COLUMNS_TO_DROP
TARGET_COLUMN = "{{TARGET_COLUMN}}"
COLUMNS_TO_DROP = {{COLUMNS_TO_DROP}}

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

# ---------------------------------------------------------
# Target normalization (regression vs classification fallback)
# ---------------------------------------------------------

# Normalize y into a downstream-friendly numeric representation:
#   - numeric y: keep numeric; cast float->int if values are integer-like
#   - non-numeric y: category-encode to stable integer class IDs (with a label map)
if pd.api.types.is_numeric_dtype(y):
	target_dtype_original = str(y.dtype)
	if pd.api.types.is_float_dtype(y):
		y_without_na = y.dropna()
		if not y_without_na.empty and (y_without_na % 1 == 0).all():
			y = y.round().astype("int64")
else:
	target_dtype_original = str(y.dtype)
	target_categories = y.astype("category")
	target_label_map = {
		str(category): int(code)
		for code, category in enumerate(target_categories.cat.categories.tolist())
	}
	y = target_categories.cat.codes.astype("int64")

target_column_name = str(TARGET_COLUMN)

if len(y) == 0:
	raise ValueError("No rows remain after target filtering. Check selected dataset and target column.")

# ---------------------------------------------------------
# Final ETL contract validation
# ---------------------------------------------------------

# Sanity-check the contract so downstream sections can run without defensive checks.
_validate_etl_outputs(
	project_root=project_root,
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

# Bundle preprocessing + model into one inference-ready pipeline.
model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("regressor", LinearRegression()),
	]
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================

# Fit on training data (pipeline fits preprocessors + model).
fit_started_at = time.perf_counter()
if training_verbose > 0:
	print("Training started: LinearRegression")
model.fit(X_train, y_train)
fit_time_seconds = float(time.perf_counter() - fit_started_at)
if training_verbose > 0:
	print(f"Training completed in {fit_time_seconds:.3f}s: LinearRegression")

training_control = {
	"enabled": False,
	"strategy": None,
	"monitor_name": None,
	"monitor_mode": None,
	"max_steps_configured": None,
	"steps_completed": None,
	"best_step": None,
	"best_score": None,
	"n_iter_no_change": None,
	"validation_fraction": None,
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
# ========= EXPORT ARTIFACTS & MODEL REGISTRY =================
# =============================================================
	
# Artifact export and registry logging.
if SAVE_MODEL:
	project_root = _project_root()
	model_name = args.name.strip() or Path(__file__).stem
	model_root_dir = project_root / "artifacts" / "models" / model_name
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
	estimator_params_compact = _select_estimator_params(
		regressor_params,
		[
			"fit_intercept",
			"copy_X",
			"tol",
			"positive",
			"n_jobs",
		],
	)

	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": args.library,
		"task": args.task,
		"algorithm": "linear_regression",
		"estimator_class": "LinearRegression",
		"model_id": "sklearn.linearregression",
		"dataset": {
			"path": str(data_path.relative_to(project_root)),
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
				"enabled": False,
				"strategy": None,
				"validation_fraction": None,
				"random_state": None,
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
			"target_encoding": {
				"strategy": "category_codes" if target_label_map is not None else None,
				"mapping": target_label_map,
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
		},
		"training_control": training_control,
		"selection": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": None,
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
