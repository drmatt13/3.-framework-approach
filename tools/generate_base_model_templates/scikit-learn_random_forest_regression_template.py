import argparse
import hashlib
import json
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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
	round_metric as _shared_round_metric,
	select_estimator_params as _select_estimator_params,
	validate_etl_outputs as _validate_etl_outputs,
)

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (common usage)
#   --library scikit-learn
#   --model random_forest
#   --task regression
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
# ---------------------------------------------------------------------

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
METRIC_DECIMALS = 4

# Helper function: round metrics for cleaner output.
def _round_metric(value):
	return _shared_round_metric(value, METRIC_DECIMALS)

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Random Forest Regressor baseline")
parser.add_argument("--library", choices=["scikit-learn"], default="scikit-learn")
parser.add_argument("--model", choices=["random_forest"], default="random_forest")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
args = parser.parse_args()
SAVE_MODEL = args.save_model

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
# Keep custom changes inside DATA ETL; everything below assumes this contract.
# ---------------------------------------------------------------------

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
# Edit points for custom datasets:
# - DATA_TASK_DIR / DATA_FILE
# - READ_CSV_STATEMENT / POST_READ_DATASET_SETUP
project_root = _project_root()
data_path = project_root / "data" / "template_data" / "{{DATA_TASK_DIR}}" / "{{DATA_FILE}}"
{{READ_CSV_STATEMENT}}
{{POST_READ_DATASET_SETUP}}
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# ---------------------------------------------------------
# Normalize raw dataframe (dataset-level cleanup)
# ---------------------------------------------------------
for column in df.select_dtypes(include=["object", "string"]).columns:
	series = df[column].astype("string").str.strip()
	series = series.replace("", np.nan)
	df[column] = series.astype("object")

df = df.replace({pd.NA: np.nan})

# ---------------------------------------------------------
# Define target + features (semantic boundary)
# ---------------------------------------------------------
# Edit points for target/feature mapping:
# - TARGET_COLUMN / FEATURE_DROP_COLUMNS
# - TARGET_PREPROCESS
TARGET_COLUMN = "{{TARGET_COLUMN}}"
FEATURE_DROP_COLUMNS = {{FEATURE_DROP_COLUMNS}}

if TARGET_COLUMN not in df.columns:
	raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

y = df[TARGET_COLUMN]
{{TARGET_PREPROCESS}} # type: ignore
X = df.drop(columns=FEATURE_DROP_COLUMNS, errors="ignore")

# ---------------------------------------------------------
# Remove rows with missing target
# ---------------------------------------------------------
valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].copy()

target_column_name = str(TARGET_COLUMN)

if len(y) == 0:
	raise ValueError("No rows remain after target filtering. Check selected dataset and target column.")

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
		("regressor", RandomForestRegressor(random_state=args.random_state)),
	]
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================

# Fit on training data (pipeline fits preprocessors + model).
fit_started_at = time.perf_counter()
model.fit(X_train, y_train)
fit_time_seconds = float(time.perf_counter() - fit_started_at)

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

# ---- Test Metrics (generalization to unseen data) ----
test_mse = mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_rmse = root_mean_squared_error(y_test, predictions)
test_r2 = r2_score(y_test, predictions)
test_max_error = max_error(y_test, predictions)

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model performance on training data) ----
print("Train MSE:", _round_metric(train_mse))  # Mean Squared Error on training set (average squared residuals)
print("Train MAE:", _round_metric(train_mae))  # Mean Absolute Error on training set (average absolute prediction error)
print("Train RMSE:", _round_metric(train_rmse))  # Root Mean Squared Error on training set (error in original target units)
print("Train R2:", _round_metric(train_r2))  # R² on training set (variance explained by model)
print("Train Max Error:", _round_metric(train_max_error))  # Largest absolute prediction error on training data
print("Train Residual Mean:", _round_metric(train_residual_mean))  # Mean of residuals (should be near 0 if unbiased)
print("Train Residual Std:", _round_metric(train_residual_std))  # Standard deviation of residuals (spread of errors)

# ---- Test Metrics (model generalization to unseen data) ----
print("Test MSE:", _round_metric(test_mse))  # Mean Squared Error on test set (average squared prediction errors)
print("Test MAE:", _round_metric(test_mae))  # Mean Absolute Error on test set (average absolute difference from true values)
print("Test RMSE:", _round_metric(test_rmse))  # Root Mean Squared Error on test set (interpretable error magnitude)
print("Test R2:", _round_metric(test_r2))  # R² on test set (generalization performance)

print("Test Max Error:", _round_metric(test_max_error))  # Worst-case absolute prediction error on test set

print("Target Mean:", _round_metric(y.mean()))  # Overall target mean (prefer y_train.mean() to avoid leakage)
print("Target Std:", _round_metric(y.std()))  # Overall target standard deviation (prefer y_train.std() for clean separation)

print("First 5 predictions:", predictions[:5])  # Quick sanity check of predicted values and scale

# =============================================================
# ========= EXPORT ARTIFACTS & MODEL REGISTRY =================
# =============================================================
# Artifact export and registry logging.
if SAVE_MODEL:
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
			"mean": _round_metric(y.mean()),
			"std": _round_metric(y.std()),
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
	metrics["selection"] = None
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

	feature_schema = {
		"feature_columns": {col: str(dtype) for col, dtype in X.dtypes.items()},
		"target": {target_column_name: str(y.dtype)},
	}
	with (data_dir / "feature_schema.json").open("w", encoding="utf-8") as schema_file:
		json.dump(feature_schema, schema_file, indent=2)

	post_transform_feature_count = _post_transform_feature_count(model.named_steps["preprocess"], X_train.iloc[:1])
	regressor_params = model.named_steps["regressor"].get_params()
	estimator_params_compact = _select_estimator_params(
		regressor_params,
		[
			"n_estimators",
			"criterion",
			"max_depth",
			"max_features",
			"min_samples_split",
			"min_samples_leaf",
			"bootstrap",
			"max_samples",
			"random_state",
			"n_jobs",
		],
	)

	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": args.library,
		"task": args.task,
		"algorithm": "random_forest",
		"estimator_class": "RandomForestRegressor",
		"model_id": "sklearn.randomforestregressor",
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
		"selection": None,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": model.named_steps["regressor"].get_params().get("n_jobs"),
		},
		"artifacts": _artifact_map(
			run_dir,
			{
				"model": model_dir / "model.pkl",
				"preprocess": preprocess_dir / "preprocessor.pkl",
				"eval_metrics": eval_dir / "metrics.json",
				"eval_predictions_preview": eval_dir / "predictions_preview.csv",
				"feature_schema": data_dir / "feature_schema.json",
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
				"mse": float(test_mse),
				"mae": float(test_mae),
				"rmse": float(test_rmse),
				"r2": float(test_r2),
				"n_train": int(len(X_train)),
				"n_test": int(len(X_test)),
			}
		]
	)
	registry_df = pd.concat([registry_df, registry_row], ignore_index=True)
	registry_df.to_csv(registry_path, index=False)

	print(f"Artifacts exported to: {run_dir}")
