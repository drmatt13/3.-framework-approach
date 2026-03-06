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
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Ensure project root is importable so generated templates can load shared helpers.
_current_file = Path(__file__).resolve()
for _candidate in [_current_file.parent, *_current_file.parents]:
	if (_candidate / "libraries").is_dir():
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
	transformed_feature_names as _transformed_feature_names,
	validate_artifact_contract as _validate_artifact_contract,
	validate_etl_outputs as _validate_etl_outputs,
	write_unified_registry_sqlite as _write_unified_registry_sqlite,
	write_model_schemas as _write_model_schemas,
)
from libraries.linear_regression_search_space import (
	LinearRegressionSearchGridConfig,
	build_linear_regression_search_space,
)
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.search_utils import cv_scoring_name as _cv_scoring_name, search_space_size as _search_space_size
from libraries.serialization_utils import json_safe_best_params as _json_safe_best_params

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (Linear Regression)
#
# Core run options (auto-configured for all ML models generated)
#   --name <model_name>                             (model name used for registry and artifact folder; default: script filename)
#   --artifact-name-mode full|short                 (full = timestamp + UUID for unique runs; short = readable name but may overwrite previous runs)
#   --save-model true|false                         (save trained model and artifacts; false logs metrics only)
#   --verbose 0|1|2|auto                            (0=silent, 1=training progress, 2=training + tuning progress, auto=adaptive verbosity)
#   --metric-decimals <int>                         (decimal precision for logged metrics and artifacts)
#
# Data split / reproducibility
#   --random-state <int>                            (random seed for reproducibility)
#   --test-size <float>                             (test set fraction; e.g., 0.2 = 80/20 split)
#
# Hyperparameter tuning configuration
#   --enable-tuning true|false                  		(enable hyperparameter tuning with cross-validation)
# 
# Direct-fit hyperparameters 										(used when --enable-tuning=false)
#   --penalty none|l1|l2|elasticnet             		(linear model variant / regularization type to use)
#   --l1-ratio <float>                              (elasticnet mixing parameter; only used when penalty=elasticnet)
#   --alpha <float>                                 (regularization strength; used for l1, l2, and elasticnet)
#   --fit-intercept true|false                      (whether to learn an intercept term)
#
# Hyperparameter tuning 												(used when --enable-tuning=true)
#   --penalty auto|l1|l2|elasticnet	               	(linear model variant / regularization type to use)
#   --tuning-method grid|random                     (grid = exhaustive search over grid; random = randomized search over iterations)
#   --cv-n-iter <int>                               (random search iterations; only used when --tuning-method=random)
#   --cv-folds <int>                                (number of cross-validation folds)
#   --cv-scoring rmse|mae|r2                        (metric used during CV tuning)
#   --cv-n-jobs <int>                               (CV search parallelism; -1 uses all cores)
# ---------------------------------------------------------------------

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Linear Regression with optional hyperparameter tuning")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=False)
parser.add_argument("--random-state", type=int, default=1)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default="1")
parser.add_argument("--metric-decimals", type=int, default=4)
parser.add_argument("--penalty", choices=["auto", "none", "l1", "l2", "elasticnet"], default="{{LR_PENALTY_DEFAULT}}")
parser.add_argument("--alpha", type=float, default={{LR_ALPHA_DEFAULT}})  # type: ignore
parser.add_argument("--fit-intercept", type=_parse_bool, default="{{LR_FIT_INTERCEPT_DEFAULT}}" == "True")
parser.add_argument("--l1-ratio", type=float, default=float("{{LR_L1_RATIO_DEFAULT}}"))
parser.add_argument("--enable-tuning", type=_parse_bool, default="{{LR_ENABLE_TUNING_DEFAULT}}" == "True")
parser.add_argument("--tuning-method", choices=["grid", "random"], default="{{LR_TUNING_METHOD_DEFAULT}}")
parser.add_argument("--cv-folds", type=int, default=int("{{LR_CV_FOLDS_DEFAULT}}"))
parser.add_argument("--cv-scoring", choices=["rmse", "mae", "r2"], default="{{LR_CV_SCORING_DEFAULT}}")
parser.add_argument("--cv-n-iter", type=int, default=int("{{LR_CV_N_ITER_DEFAULT}}"))
parser.add_argument("--cv-n-jobs", type=int, default=int("{{LR_CV_N_JOBS_DEFAULT}}"))
args = parser.parse_args()
if args.penalty == "auto" and not args.enable_tuning:
	raise ValueError("--penalty=auto requires --enable-tuning=true.")
SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
cv_verbose = 0 if training_verbose <= 1 else 2
METRIC_DECIMALS = int(args.metric_decimals)
_round_metric = partial(_round_metric_base, decimals=METRIC_DECIMALS)

# NOTE: Adjust these grids to customize search breadth for tuning.
LINEAR_REGRESSION_SEARCH_GRID_CONFIG = LinearRegressionSearchGridConfig(
	fit_intercept=[True, False],  # whether to learn an intercept term (bias)
	ridge_alpha=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],  # Ridge (L2) regularization strength
	lasso_alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],  # Lasso (L1) regularization strength
	lasso_max_iter=[5_000, 10_000, 20_000],  # max optimization iterations for Lasso convergence
	elasticnet_alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],  # ElasticNet overall regularization strength
	elasticnet_l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # ElasticNet mixing parameter (0≈Ridge/L2, 1≈Lasso/L1)
	elasticnet_max_iter=[5_000, 10_000, 20_000],  # max optimization iterations for ElasticNet convergence
)

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
data_path = _project_root() / "data" / "template_data" / "{{DATA_TASK_DIR}}" / "{{DATA_FILE}}"
{{READ_CSV_STATEMENT}}
{{POST_READ_DATASET_SETUP}}

# Drop common CSV index artifacts (e.g., "Unnamed: 0") so they never leak into features.
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# ---------------------------------------------------------
# Normalize raw dataframe (dataset-level cleanup)
# ---------------------------------------------------------

# Goal: reduce category fragmentation and standardize missingness.
#   - trim whitespace in string-like columns
#   - convert empty strings to NaN
#   - normalize pd.NA -> np.nan for consistent downstream behavior
df = _normalize_string_columns(df)

# ---------------------------------------------------------
# Define target + features (semantic boundary)
# ---------------------------------------------------------

# Template injection points:
#   - TARGET_COLUMN
#   - FEATURE_DROP_COLUMNS
TARGET_COLUMN = "{{TARGET_COLUMN}}"
COLUMNS_TO_DROP = {{FEATURE_DROP_COLUMNS}}

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

preprocessor = _build_preprocessor(X_train)

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Helper to construct the appropriate regressor based on CLI flags.
def _base_regressor_from_flags() -> LinearRegression | Ridge | Lasso | ElasticNet:
	if args.penalty == "auto":
		return Ridge(alpha=max(float(args.alpha), 1e-3), fit_intercept=args.fit_intercept, random_state=args.random_state)
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

selected_cv_scoring = _cv_scoring_name(
	args.cv_scoring,
	{"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
)
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
	search_space = build_linear_regression_search_space(
		penalty=args.penalty,
		random_state=int(args.random_state),
		config=LINEAR_REGRESSION_SEARCH_GRID_CONFIG,
	)
	search = None
	if args.tuning_method == "grid":
		search = GridSearchCV(
			estimator=model,
			param_grid=search_space,
			scoring=selected_cv_scoring,
			cv=int(args.cv_folds),
			n_jobs=int(args.cv_n_jobs),
			verbose=cv_verbose,
			refit=False,
		)
	else:
		n_iter = int(args.cv_n_iter)
		n_candidates_upper = _search_space_size(search_space)
		if n_candidates_upper > 0:
			n_iter = min(n_iter, n_candidates_upper)
		search = RandomizedSearchCV(
			estimator=model,
			param_distributions=search_space,
			n_iter=int(n_iter),
			scoring=selected_cv_scoring,
			cv=int(args.cv_folds),
			n_jobs=int(args.cv_n_jobs),
			verbose=cv_verbose,
			refit=False,
			random_state=int(args.random_state),
		)

	# Each candidate model is trained only inside CV folds.
	search.fit(X_train, y_train)
	best_params = dict(search.best_params_)
	best_params_for_artifacts = _json_safe_best_params(best_params)
	model.set_params(**best_params)

	# Train the final model
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
print("First 5 predictions:", [_round_metric(x, decimals=4) for x in predictions[:5].tolist()])  # Sample predictions for quick sanity check
print("First 5 true values:", [_round_metric(x, decimals=4) for x in y_test.iloc[:5].tolist()])  # Corresponding true values for sanity check

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
	coefficient_artifacts: dict[str, Path] = {}
	feature_names = _transformed_feature_names(model.named_steps["preprocess"], X_train)
	regressor_step = model.named_steps["regressor"]
	if hasattr(regressor_step, "coef_"):
		coef_array = np.asarray(getattr(regressor_step, "coef_"), dtype=float).reshape(-1)
		usable_length = min(len(feature_names), int(coef_array.shape[0]))
		coefficient_table = pd.DataFrame(
			{
				"feature": feature_names[:usable_length],
				"coefficient": coef_array[:usable_length].tolist(),
			}
		)
		coefficients_csv_path = eval_dir / "coefficients.csv"
		coefficient_table.to_csv(coefficients_csv_path, index=False)
		intercept_value = None
		if hasattr(regressor_step, "intercept_"):
			intercept_raw = np.asarray(getattr(regressor_step, "intercept_"), dtype=float).reshape(-1)
			if intercept_raw.size > 0:
				intercept_value = float(intercept_raw[0])
		coefficients_summary_path = eval_dir / "coefficients_summary.json"
		coefficients_summary_path.write_text(
			json.dumps(
				{
					"intercept": _round_metric(intercept_value) if intercept_value is not None else None,
					"n_coefficients": int(usable_length),
					"feature_source": "preprocessor.get_feature_names_out",
				},
				indent=2,
			),
			encoding="utf-8",
		)
		coefficient_artifacts = {
			"eval_coefficients": coefficients_csv_path,
			"eval_coefficients_summary": coefficients_summary_path,
		}

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

	artifacts_for_map = {
		"model": model_dir / "model.pkl",
		"preprocess": preprocess_dir / "preprocessor.pkl",
		"eval_metrics": eval_dir / "metrics.json",
		"eval_predictions_preview": eval_dir / "predictions_preview.csv",
		**schema_artifacts,
		**coefficient_artifacts,
		"inference_example": inference_dir / "inference_example.py",
	}
	run_metadata = {
		"run_id": run_id,
		"model_name": model_name,
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
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"artifacts": _artifact_map(run_dir, artifacts_for_map),
		"versions": {
			"python": platform.python_version(),
			"pandas": pd.__version__,
			"scikit-learn": sklearn.__version__,
		},
	}
	run_metadata = _compact_metadata(run_metadata)
	with (run_dir / "run.json").open("w", encoding="utf-8") as run_file:
		json.dump(run_metadata, run_file, indent=2)
	artifact_warnings = _validate_artifact_contract(
		run_dir=run_dir,
		artifact_files=artifacts_for_map,
		run_metadata=run_metadata,
		metrics=metrics,
		required_artifact_keys=[
			"model",
			"preprocess",
			"eval_metrics",
			"eval_predictions_preview",
			"input_schema",
			"target_mapping_schema",
			"inference_example",
		],
		warn_only=True,
	)
	if artifact_warnings:
		print(f"Artifact validation warnings: {artifact_warnings}")

	registry_path = model_root_dir / "registry.csv"
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
				"model_name": model_name,
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
	_write_unified_registry_sqlite(
		project_root=_project_root(),
		run_dir=run_dir,
		run_metadata=run_metadata,
		metrics=metrics,
	)

	print(f"Artifacts exported to: {run_dir}")
