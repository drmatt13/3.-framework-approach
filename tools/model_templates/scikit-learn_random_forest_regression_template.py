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
from sklearn.ensemble import RandomForestRegressor
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
	validate_artifact_contract as _validate_artifact_contract,
	validate_etl_outputs as _validate_etl_outputs,
	write_unified_registry_sqlite as _write_unified_registry_sqlite,
	write_model_schemas as _write_model_schemas,
)
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.random_forest_search_space import RandomForestSearchGridConfig, build_random_forest_search_space
from libraries.search_utils import cv_scoring_name as _cv_scoring_name, search_space_size as _search_space_size
from libraries.serialization_utils import json_safe_best_params as _json_safe_best_params
from libraries.sklearn_template_utils import parse_max_features as _parse_max_features, parse_optional_int as _parse_optional_int

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (Random Forest Regression)
#
# Core run options (auto configured for all ML models generated)
#   --name <model_name>                             (model name used for registry and artifact folder; default: script filename)
#   --artifact-name-mode full|short                 (full = timestamp + UUID for unique runs; short = readable name but may overwrite previous runs)
#   --save-model true|false                         (save trained model and artifacts; false logs metrics only)
#   --verbose 0|1|2|auto                            (0=silent, 1=training progress, 2=training + tuning progress, auto=adaptive verbosity)
#   --metric-decimals <int>                         (decimal precision for logged metrics and artifacts)
#
# Task + reproducibility
#   --random-state <int>                           	(random seed for reproducibility)
#   --test-size <float>                            	(test set fraction; e.g., 0.2 = 80/20 split)
#
# Hyperparameter tuning configuration
#   --enable-tuning true|false                  		(enable hyperparameter tuning with cross-validation)
# 
# Direct-fit model settings 										(used when --enable-tuning=false)
#   --n-estimators <int>                           	(number of trees in the forest)
#   --max-depth <int>                         			(maximum depth of each tree)
#   --max-leaf-nodes <int|none>                    	(maximum number of leaf nodes)
#   --min-samples-split <int>                      	(minimum samples required to split an internal node)
#   --min-samples-leaf <int>                       	(minimum samples required at a leaf node)
#   --min-impurity-decrease <float>                	(minimum impurity decrease required to split a node)
#   --min-weight-fraction-leaf <float>             	(minimum weighted fraction of samples required at a leaf node)
#   --max-features sqrt|log2|auto|none|custom       (number of features considered when selecting each split)
#   --bootstrap true|false                         	(use bootstrap samples when building trees)
#   --max-samples <int|float>                 			(requires --bootstrap=true; 1.0 uses entire dataset)
#   --ccp-alpha <float>                            	(cost-complexity pruning strength)
#   --n-jobs <int>                            			(estimator parallelism; -1 uses all cores)
#
# Hyperparameter tuning 												(used when --enable-tuning=true)
#   --enable-tuning true|false                     	(enable hyperparameter tuning with cross-validation)
#   --tuning-method grid|random                    	(grid = exhaustive search over grid; random = randomized search over iterations)
#   --cv-n-iter <int>                              	(random search iterations; only used when --tuning-method=random)
#   --cv-folds <int>                               	(number of cross-validation folds)
#   --cv-scoring rmse|mae|r2									     	(metric used during CV tuning)
#   --cv-n-jobs <int>                              	(CV search parallelism; -1 uses all cores)
# ---------------------------------------------------------------------

# NOTE: Adjust these grids to customize search breadth for tuning.
RANDOM_FOREST_SEARCH_GRID_CONFIG = RandomForestSearchGridConfig(
	n_estimators_grid=[100, 200, 300, 500],  # number of trees in the forest
	max_depth_grid=[None, 4, 8, 16, 32],  # None = unlimited depth (trees expand until other stopping rules)
	max_leaf_nodes_grid=[None, 32, 64, 128],  # None = no limit on number of leaf nodes
	max_features_grid=[1.0, "sqrt", "log2"],  # number of features considered at each split (1.0 = all features)
	max_samples_when_bootstrap_grid=[0.5, 0.7, 1.0],  # fraction of rows sampled per tree when bootstrap=True
	min_weight_fraction_leaf_grid=[0.0, 0.01],  # minimum weighted fraction of samples required at a leaf node
	min_impurity_decrease_grid=[0.0, 1e-6, 1e-4],  # minimum impurity reduction required to split a node
	ccp_alpha_grid=[0.0, 1e-5, 1e-4, 1e-3],  # cost-complexity pruning strength (larger values prune more)
)

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = Fals
DEFAULT_RANDOM_STATE = 1
DEFAULT_N_ESTIMATORS = int("{{RF_N_ESTIMATORS_DEFAULT}}")
DEFAULT_MAX_DEPTH = _parse_optional_int("{{RF_MAX_DEPTH_DEFAULT}}")
DEFAULT_MIN_SAMPLES_SPLIT = int("{{RF_MIN_SAMPLES_SPLIT_DEFAULT}}")
DEFAULT_MIN_SAMPLES_LEAF = int("{{RF_MIN_SAMPLES_LEAF_DEFAULT}}")
DEFAULT_MIN_WEIGHT_FRACTION_LEAF = float("{{RF_MIN_WEIGHT_FRACTION_LEAF_DEFAULT}}")
DEFAULT_MAX_LEAF_NODES = _parse_optional_int("{{RF_MAX_LEAF_NODES_DEFAULT}}")
DEFAULT_MIN_IMPURITY_DECREASE = float("{{RF_MIN_IMPURITY_DECREASE_DEFAULT}}")
DEFAULT_MAX_FEATURES = "{{RF_MAX_FEATURES_DEFAULT}}"
DEFAULT_BOOTSTRAP = "{{RF_BOOTSTRAP_DEFAULT}}" == "True"
DEFAULT_MAX_SAMPLES_TOKEN = "{{RF_MAX_SAMPLES_DEFAULT}}"
DEFAULT_CCP_ALPHA = float("{{RF_CCP_ALPHA_DEFAULT}}")
DEFAULT_N_JOBS = _parse_optional_int("{{RF_N_JOBS_DEFAULT}}")
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4
DEFAULT_ENABLE_TUNING = "{{RF_ENABLE_TUNING_DEFAULT}}" == "True"
DEFAULT_TUNING_METHOD = "{{RF_TUNING_METHOD_DEFAULT}}"
DEFAULT_CV_FOLDS = int("{{RF_CV_FOLDS_DEFAULT}}")
DEFAULT_CV_SCORING = "{{RF_CV_SCORING_DEFAULT}}"
DEFAULT_CV_N_ITER = int("{{RF_CV_N_ITER_DEFAULT}}")
DEFAULT_CV_N_JOBS = int("{{RF_CV_N_JOBS_DEFAULT}}")


def _parse_optional_max_samples(value: str) -> int | float | None:
	v = str(value).strip().lower()
	if v in {"none", "null"}:
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

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Random Forest Regressor baseline")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
parser.add_argument("--max-depth", type=_parse_optional_int, default=DEFAULT_MAX_DEPTH)
parser.add_argument("--min-samples-split", type=int, default=DEFAULT_MIN_SAMPLES_SPLIT)
parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
parser.add_argument("--min-weight-fraction-leaf", type=float, default=DEFAULT_MIN_WEIGHT_FRACTION_LEAF)
parser.add_argument("--max-leaf-nodes", type=_parse_optional_int, default=DEFAULT_MAX_LEAF_NODES)
parser.add_argument("--min-impurity-decrease", type=float, default=DEFAULT_MIN_IMPURITY_DECREASE)
parser.add_argument("--max-features", type=_parse_max_features, default=DEFAULT_MAX_FEATURES)
parser.add_argument("--bootstrap", type=_parse_bool, default=DEFAULT_BOOTSTRAP)
parser.add_argument("--max-samples", type=_parse_optional_max_samples, default=_parse_optional_max_samples(DEFAULT_MAX_SAMPLES_TOKEN))
parser.add_argument("--ccp-alpha", type=float, default=DEFAULT_CCP_ALPHA)
parser.add_argument("--n-jobs", type=_parse_optional_int, default=DEFAULT_N_JOBS)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
parser.add_argument("--enable-tuning", type=_parse_bool, default=DEFAULT_ENABLE_TUNING)
parser.add_argument("--tuning-method", choices=["grid", "random"], default=DEFAULT_TUNING_METHOD)
parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
parser.add_argument("--cv-scoring", choices=["rmse", "mae", "r2"], default=DEFAULT_CV_SCORING)
parser.add_argument("--cv-n-iter", type=int, default=DEFAULT_CV_N_ITER)
parser.add_argument("--cv-n-jobs", type=int, default=DEFAULT_CV_N_JOBS)
args = parser.parse_args()

if int(args.min_samples_split) < 2:
	raise ValueError("--min-samples-split must be >= 2")
if int(args.min_samples_leaf) < 1:
	raise ValueError("--min-samples-leaf must be >= 1")
if not (0.0 <= float(args.min_weight_fraction_leaf) <= 0.5):
	raise ValueError("--min-weight-fraction-leaf must be in range [0, 0.5]")
if args.max_leaf_nodes is not None and int(args.max_leaf_nodes) < 2:
	raise ValueError("--max-leaf-nodes must be >= 2 when provided")
if float(args.min_impurity_decrease) < 0.0:
	raise ValueError("--min-impurity-decrease must be >= 0")
if float(args.ccp_alpha) < 0.0:
	raise ValueError("--ccp-alpha must be >= 0")
if args.n_jobs == 0:
	raise ValueError("--n-jobs must be != 0 or none")
if int(args.cv_n_jobs) == 0:
	raise ValueError("--cv-n-jobs must be != 0")
if (not bool(args.bootstrap)) and args.max_samples is not None:
	raise ValueError("--max-samples requires --bootstrap=true")

SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
cv_verbose = 0 if training_verbose <= 1 else 2
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
# Load dataset
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
#   - TARGET_PREPROCESS
TARGET_COLUMN = "{{TARGET_COLUMN}}"
FEATURE_DROP_COLUMNS = {{FEATURE_DROP_COLUMNS}}

if TARGET_COLUMN not in df.columns:
	raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

# y is the supervised target; X is the feature space (minus optional drops).
y = df[TARGET_COLUMN]
y_original = y.copy()
{{TARGET_PREPROCESS}} # type: ignore
X = df.drop(columns=FEATURE_DROP_COLUMNS, errors="ignore")

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

# Preprocess: impute missing values, then scale numeric and one-hot encode categorical features.
preprocessor = _build_preprocessor(X_train)

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Bundle preprocessing + model into one inference-ready pipeline.
model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		(
			"regressor",
			RandomForestRegressor(
				random_state=args.random_state,
				n_estimators=int(args.n_estimators),
				max_depth=args.max_depth,
				min_samples_split=int(args.min_samples_split),
				min_samples_leaf=int(args.min_samples_leaf),
				min_weight_fraction_leaf=float(args.min_weight_fraction_leaf),
				max_leaf_nodes=args.max_leaf_nodes,
				min_impurity_decrease=float(args.min_impurity_decrease),
				max_features=args.max_features,
				bootstrap=bool(args.bootstrap),
				max_samples=args.max_samples,
				ccp_alpha=float(args.ccp_alpha),
				n_jobs=args.n_jobs,
				verbose=training_verbose,
			),
		),
	]
)
fit_time_seconds = 0.0

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
			f"Training started with tuning: method={args.tuning_method}, "
			f"cv={args.cv_folds}, scoring={args.cv_scoring}"
		)
	else:
		print("Training started: RandomForestRegressor")

if args.enable_tuning:
	search_space = build_random_forest_search_space(
		step_name="regressor",
		n_estimators=int(args.n_estimators),
		max_depth=args.max_depth,
		min_samples_split=int(args.min_samples_split),
		min_samples_leaf=int(args.min_samples_leaf),
		min_weight_fraction_leaf=float(args.min_weight_fraction_leaf),
		max_leaf_nodes=args.max_leaf_nodes,
		min_impurity_decrease=float(args.min_impurity_decrease),
		max_features=args.max_features,
		bootstrap=bool(args.bootstrap),
		max_samples=args.max_samples,
		ccp_alpha=float(args.ccp_alpha),
		random_state=int(args.random_state),
		config=RANDOM_FOREST_SEARCH_GRID_CONFIG,
	)
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
if training_verbose > 0:
	print(f"Training completed in {fit_time_seconds:.3f}s: RandomForestRegressor")

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

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model fit on data it learned from) ----
print("Train MSE:", _round_metric(train_mse))  # Mean Squared Error on training set (average squared residuals)
print("Train MAE:", _round_metric(train_mae))  # Mean Absolute Error on training set (average absolute prediction error)
print("Train RMSE:", _round_metric(train_rmse))  # Root Mean Squared Error on training set (error in original target units)
print("Train R2:", _round_metric(train_r2))  # R² on training set (variance explained by model)
print("Train Max Error:", _round_metric(train_max_error))  # Largest absolute prediction error on training data
print("Train Residual Mean:", _round_metric(train_residual_mean))  # Mean of residuals (should be near 0 if unbiased)
print("Train Residual Std:", _round_metric(train_residual_std))  # Standard deviation of residuals (spread of errors)

# ---- Test Metrics (model performance on unseen data) ----
print("Test MSE:", _round_metric(test_mse))  # Mean Squared Error on test set (average squared prediction errors)
print("Test MAE:", _round_metric(test_mae))  # Mean Absolute Error on test set (average absolute difference from true values)
print("Test RMSE:", _round_metric(test_rmse))  # Root Mean Squared Error on test set (interpretable error magnitude)
print("Test R2:", _round_metric(test_r2))  # R² on test set (generalization performance)
print("Test Max Error:", _round_metric(test_max_error))  # Worst-case absolute prediction error on test set

# ---- Dataset Context (distribution reference) ----
print("Target Mean:", _round_metric(y.mean()))  # Overall target mean (prefer y_train.mean() in production to avoid leakage)
print("Target Std:", _round_metric(y.std()))  # Overall target standard deviation (prefer y_train.std() for clean separation)

# ---- Sanity Checks ----
print("First 5 predictions:", predictions[:5])  # Quick sanity check of predicted values
print("First 5 true values:", y_test.iloc[:5].tolist())  # Corresponding true values for sanity check

# =============================================================
# ========= EXPORT ARTIFACTS & MODEL REGISTRY =================
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
			"mean": _round_metric(y_train.mean()),
			"std": _round_metric(y_train.std()),
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
	n_val = 0
	n_train_effective = int(len(X_train))
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

	artifacts_for_map = {
		"model": model_dir / "model.pkl",
		"preprocess": preprocess_dir / "preprocessor.pkl",
		"eval_metrics": eval_dir / "metrics.json",
		"eval_predictions_preview": eval_dir / "predictions_preview.csv",
		**schema_artifacts,
		"inference_example": inference_dir / "inference_example.py",
	}
	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": "scikit-learn",
		"task": args.task,
		"algorithm": "random_forest",
		"estimator_class": "RandomForestRegressor",
		"model_id": "sklearn.randomforestregressor",
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
				"n_train": n_train_effective,
				"n_val": n_val,
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
			"n_estimators": int(args.n_estimators),
			"max_depth": int(args.max_depth) if args.max_depth is not None else None,
			"min_samples_split": int(args.min_samples_split),
			"min_samples_leaf": int(args.min_samples_leaf),
			"min_weight_fraction_leaf": float(args.min_weight_fraction_leaf),
			"max_leaf_nodes": int(args.max_leaf_nodes) if args.max_leaf_nodes is not None else None,
			"min_impurity_decrease": float(args.min_impurity_decrease),
			"max_features": str(args.max_features) if args.max_features is not None else None,
			"bootstrap": bool(args.bootstrap),
			"max_samples": args.max_samples,
			"ccp_alpha": float(args.ccp_alpha),
			"n_jobs": int(args.n_jobs) if args.n_jobs is not None else None,
			"enable_tuning": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_folds": int(args.cv_folds) if tuning_summary["enabled"] else None,
			"cv_scoring": args.cv_scoring if tuning_summary["enabled"] else None,
			"cv_n_iter": int(args.cv_n_iter) if tuning_summary["enabled"] and args.tuning_method == "random" else None,
			"cv_n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"tuning": tuning_summary,
		"selection": training_control,
		"training_control": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": model.named_steps["regressor"].get_params().get("n_jobs"),
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
	_write_unified_registry_sqlite(
		project_root=_project_root(),
		run_dir=run_dir,
		run_metadata=run_metadata,
		metrics=metrics,
	)

	print(f"Artifacts exported to: {run_dir}")
