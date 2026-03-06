import argparse
import hashlib
import json
import math
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
import xgboost as xgb
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
	json_safe as _json_safe,
	parse_bool_flag as _parse_bool,
	post_transform_feature_count as _post_transform_feature_count,
	round_metric as _round_metric_base,
	select_estimator_params as _select_estimator_params,
	validate_etl_outputs as _validate_etl_outputs,
	write_unified_registry_sqlite as _write_unified_registry_sqlite,
	write_model_schemas as _write_model_schemas,
)
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.search_utils import cv_scoring_name as _cv_scoring_name, search_space_size as _search_space_size
from libraries.xgboost_search_space import XGBoostSearchGridConfig as _XGBoostSearchGridConfig, build_xgboost_search_space as _build_xgboost_search_space
from libraries.xgboost_template_utils import resolve_xgboost_device as _resolve_xgboost_device

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (XGBoost Regression)

# Core run options (auto-configured for all ML models generated)
#   --name <model_name>                             (model name used for registry and artifact folder; default: script filename)
#   --artifact-name-mode full|short                 (full = timestamp + UUID for unique runs; short = readable name but may overwrite previous runs)
#   --save-model true|false                         (save trained model and artifacts; false logs metrics only)
#   --verbose 0|1|2|auto                            (0=silent, 1=training progress, 2=training + tuning progress, auto=adaptive verbosity)
#   --metric-decimals <int>                         (decimal precision for logged metrics and artifacts)
#
# Reproducibility + data split
#   --task binary_classification|										(task type for metric calculation and logging)
# 				 multiclass_classification								^
#   --random-state <int>                            (random seed for reproducibility)
#   --test-size <float>                             (test set fraction; e.g., 0.2 = 80/20 split)
#   --booster gbtree|gblinear|dart             			(booster algorithm used for training)
#
# Hyperparameter tuning configuration
#   --enable-tuning true|false                  		(enable hyperparameter tuning with cross-validation)
# 
# Model configuration (direct-fit)              (used when --enable-tuning=false)
#   --device auto|cpu|gpu                           (training device selection)
#   --n-estimators <int>                            (number of boosting rounds)
#   --learning-rate <float>                         (step size shrinkage used during boosting)
#   --max-depth <int>                               (tree boosters only)
#   --subsample <float>                             (tree boosters only)
#   --colsample-bytree <float>                      (tree boosters only)
#   --min-child-weight <float>                      (tree boosters only)
#   --reg-lambda <float>                            (L2 regularization strength)
#   --reg-alpha <float>                             (L1 regularization strength)
#
# Training path
#   --early-stopping true|false                     (enable early stopping during training)
#   --validation-fraction <float>                   (fraction of training set used for validation)
#   --n-iter-no-change <int>                        (stop training if validation score does not improve for N rounds)
#   --enable-tuning true|false                      (enable hyperparameter tuning with cross-validation)
#
# Hyperparameter tuning                         (used when --enable-tuning=true)
#   --tuning-method grid|random                     (grid or randomized hyperparameter search)
#   --cv-folds <int>                                (number of cross-validation folds)
#   --cv-scoring rmse|mae|r2                        (metric used during CV tuning)
#   --cv-n-iter <int>                               (number of random search iterations)
#   --cv-n-jobs <int>                               (CV search parallelism; -1 uses all cores)
# ---------------------------------------------------------------------

# NOTE: Adjust these grids to customize search breadth for tuning.
XGBOOST_SEARCH_GRID_CONFIG = _XGBoostSearchGridConfig(
  n_estimators=[200, 400, 800],  # number of boosting rounds (trees added sequentially to correct residual error)
  learning_rate=[0.3, 0.1, 0.05, 0.03],  # step size shrinkage applied to each tree’s contribution (smaller = slower but often better learning)
  reg_lambda=[0.0, 1.0, 10.0],  # L2 regularization on leaf weights (larger values reduce overfitting by shrinking weights)
  reg_alpha=[0.0, 0.001, 0.01, 0.1],  # L1 regularization on leaf weights (encourages sparsity in tree leaf outputs)
  booster_when_auto=["gbtree", "gblinear"],  # boosters searched when --booster=auto (tree-based gradient boosting vs linear booster)
  max_depth=[3, 5, 7, 9],  # maximum depth of trees for tree boosters (controls model complexity and interaction depth)
  subsample=[0.6, 0.8, 1.0],  # fraction of rows sampled for each boosting round (stochastic gradient boosting; helps prevent overfitting)
  colsample_bytree=[0.6, 0.8, 1.0],  # fraction of features sampled when constructing each tree (similar concept to RF max_features)
  min_child_weight=[1.0, 3.0, 5.0, 10.0],  # minimum sum of Hessian (approx. sample weight) required in a child node to allow a split
)

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_BOOSTER = "{{BOOSTER}}"
DEFAULT_DEVICE = "{{DEVICE}}"
DEFAULT_EARLY_STOPPING = "{{EARLY_STOPPING_DEFAULT}}" == "True"
DEFAULT_VALIDATION_FRACTION = float("{{VALIDATION_FRACTION_DEFAULT}}")
DEFAULT_N_ITER_NO_CHANGE = int("{{N_ITER_NO_CHANGE_DEFAULT}}")
DEFAULT_N_ESTIMATORS = int("{{XGB_N_ESTIMATORS_DEFAULT}}")
DEFAULT_LEARNING_RATE = float("{{XGB_LEARNING_RATE_DEFAULT}}")
DEFAULT_MAX_DEPTH = int("{{XGB_MAX_DEPTH_DEFAULT}}")
DEFAULT_SUBSAMPLE = float("{{XGB_SUBSAMPLE_DEFAULT}}")
DEFAULT_COLSAMPLE_BYTREE = float("{{XGB_COLSAMPLE_BYTREE_DEFAULT}}")
DEFAULT_MIN_CHILD_WEIGHT = float("{{XGB_MIN_CHILD_WEIGHT_DEFAULT}}")
DEFAULT_REG_LAMBDA = float("{{XGB_REG_LAMBDA_DEFAULT}}")
DEFAULT_REG_ALPHA = float("{{XGB_REG_ALPHA_DEFAULT}}")
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4
DEFAULT_ENABLE_TUNING = "{{XGB_ENABLE_TUNING_DEFAULT}}" == "True"
DEFAULT_TUNING_METHOD = "{{XGB_TUNING_METHOD_DEFAULT}}"
DEFAULT_CV_FOLDS = int("{{XGB_CV_FOLDS_DEFAULT}}")
DEFAULT_CV_SCORING = "{{XGB_CV_SCORING_DEFAULT}}"
DEFAULT_CV_N_ITER = int("{{XGB_CV_N_ITER_DEFAULT}}")
DEFAULT_CV_N_JOBS = int("{{XGB_CV_N_JOBS_DEFAULT}}")

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="XGBoost Regressor baseline")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--booster", choices=["auto", "gbtree", "gblinear", "dart"], default=DEFAULT_BOOSTER)
parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default=DEFAULT_DEVICE)
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--early-stopping", type=_parse_bool, default=DEFAULT_EARLY_STOPPING)
parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
parser.add_argument("--n-iter-no-change", type=int, default=DEFAULT_N_ITER_NO_CHANGE)
parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
parser.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
parser.add_argument("--colsample-bytree", type=float, default=DEFAULT_COLSAMPLE_BYTREE)
parser.add_argument("--min-child-weight", type=float, default=DEFAULT_MIN_CHILD_WEIGHT)
parser.add_argument("--reg-lambda", type=float, default=DEFAULT_REG_LAMBDA)
parser.add_argument("--reg-alpha", type=float, default=DEFAULT_REG_ALPHA)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
parser.add_argument("--enable-tuning", type=_parse_bool, default=DEFAULT_ENABLE_TUNING)
parser.add_argument("--tuning-method", choices=["grid", "random"], default=DEFAULT_TUNING_METHOD)
parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
parser.add_argument("--cv-scoring", choices=["rmse", "mae", "r2"], default=DEFAULT_CV_SCORING)
parser.add_argument("--cv-n-iter", type=int, default=DEFAULT_CV_N_ITER)
parser.add_argument("--cv-n-jobs", type=int, default=DEFAULT_CV_N_JOBS)
args = parser.parse_args()
SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
xgb_model_verbosity = min(3, max(0, int(training_verbose)))
xgb_fit_verbose = bool(training_verbose > 0)
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
xgb_device_requested = args.device
xgb_device, xgb_device_warning = _resolve_xgboost_device(xgb_device_requested)
if xgb_device_warning is not None:
	print(f"Warning: {xgb_device_warning}")
if training_verbose > 0:
	print(f"Resolved XGBoost device: requested={xgb_device_requested}, effective={xgb_device}")

# Resolve auto booster for direct-fit pipeline construction.
_resolved_booster = "gbtree" if args.booster == "auto" else args.booster

xgb_eval_metric = "rmse"
model_kwargs = {
	"booster": _resolved_booster,
	"device": xgb_device,
	"random_state": args.random_state,
	"objective": "reg:squarederror",
	"eval_metric": xgb_eval_metric,
	"n_estimators": int(args.n_estimators),
	"learning_rate": float(args.learning_rate),
	"max_depth": int(args.max_depth),
	"subsample": float(args.subsample),
	"colsample_bytree": float(args.colsample_bytree),
	"min_child_weight": float(args.min_child_weight),
	"reg_lambda": float(args.reg_lambda),
	"reg_alpha": float(args.reg_alpha),
	"verbosity": xgb_model_verbosity,
}

model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		(
			"regressor",
			xgb.XGBRegressor(**model_kwargs),
		),
	]
)
n_train_effective = int(len(X_train))
n_val = 0

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================
# ---------------------------------------------------------------------
# EARLY STOPPING (optional)
# - Enabled with --early-stopping=true.
# - Uses --validation-fraction as holdout split from training data.
# - Stops when validation metric does not improve for --n-iter-no-change rounds.
# - When disabled, trains once on full training split.
# ---------------------------------------------------------------------
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

if args.enable_tuning:
	if training_verbose > 0:
		print(
			f"Training started with tuning: method={args.tuning_method}, "
			f"cv={args.cv_folds}, scoring={args.cv_scoring}"
		)
	selected_cv_scoring = _cv_scoring_name(
		args.cv_scoring,
		{"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
	)

	search_space = _build_xgboost_search_space(
		step_name="regressor",
		booster=args.booster,
		config=XGBOOST_SEARCH_GRID_CONFIG,
	)

	n_iter = int(args.cv_n_iter)
	n_candidates_upper = _search_space_size(search_space)
	if n_candidates_upper > 0:
		n_iter = min(n_iter, n_candidates_upper)
	if args.tuning_method == "random":
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
	else:
		search = GridSearchCV(
			estimator=model,
			param_grid=search_space,
			scoring=selected_cv_scoring,
			cv=int(args.cv_folds),
			n_jobs=int(args.cv_n_jobs),
			verbose=cv_verbose,
			refit=False,
		)
	fit_started_at = time.perf_counter()
	search.fit(X_train, y_train)
	best_params = dict(search.best_params_)
	model.set_params(**best_params)

	best_step = None
	best_validation_score = None
	stopped_early = False
	selected_n_estimators = None
	search_n_estimators = None
	n_train_effective = int(len(X_train))
	n_val = 0

	if args.early_stopping:
		X_inner_train, X_valid, y_inner_train, y_valid = train_test_split(
			X_train,
			y_train,
			test_size=args.validation_fraction,
			random_state=args.random_state,
		)

		inner_preprocessor = _build_preprocessor(X_inner_train)
		X_inner_train_processed = inner_preprocessor.fit_transform(X_inner_train)
		X_valid_processed = inner_preprocessor.transform(X_valid)
		n_train_effective = int(len(X_inner_train))
		n_val = int(len(X_valid))

		search_model_kwargs = dict(model_kwargs)
		for param_name, param_value in best_params.items():
			if param_name.startswith("regressor__"):
				search_model_kwargs[param_name.split("__", 1)[1]] = param_value
		search_model_kwargs["early_stopping_rounds"] = int(args.n_iter_no_change)
		search_regressor = xgb.XGBRegressor(**search_model_kwargs)
		search_regressor.fit(
			X_inner_train_processed,
			y_inner_train,
			eval_set=[(X_valid_processed, y_valid)],
			verbose=xgb_fit_verbose,
		)

		search_n_estimators_raw = search_model_kwargs.get("n_estimators")
		search_n_estimators = int(search_n_estimators_raw) if search_n_estimators_raw is not None else 100
		best_iteration = getattr(search_regressor, "best_iteration", None)
		best_score_raw = getattr(search_regressor, "best_score", None)
		try:
			best_validation_score = float(best_score_raw) if best_score_raw is not None else None
		except (TypeError, ValueError):
			best_validation_score = None

		if best_iteration is not None:
			selected_n_estimators = int(best_iteration) + 1
		else:
			selected_n_estimators = search_n_estimators

		best_step = int(best_iteration) + 1 if best_iteration is not None else None
		stopped_early = bool(best_iteration is not None and (int(best_iteration) + 1) < int(search_n_estimators))
		model.named_steps["regressor"].set_params(n_estimators=selected_n_estimators)

	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
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
		"n_iter": int(search.n_iter) if hasattr(search, "n_iter") else None,
		"n_candidates": n_candidates,
		"best_score": _round_metric(best_score),
		"best_score_std": _round_metric(best_std) if best_std is not None else None,
		"best_params": _compact_metadata(_json_safe(best_params)),
	}
	training_control = {
		"enabled": True,
		"type": "search_cv",
		"max_steps_configured": int(search.n_iter) if hasattr(search, "n_iter") else int(n_candidates) if n_candidates is not None else n_candidates_upper,
		"steps_completed": int(args.cv_folds) * int(n_candidates) if n_candidates is not None else None,
		"patience": int(args.n_iter_no_change) if args.early_stopping else None,
		"monitor_metric": xgb_eval_metric if args.early_stopping else f"cv_{args.cv_scoring}",
		"monitor_split": "val" if args.early_stopping else "cv",
		"monitor_direction": "min" if (args.early_stopping or args.cv_scoring in ("rmse", "mae")) else "max",
		"best_step": best_step,
		"best_score": _round_metric(best_validation_score) if args.early_stopping else tuning_summary["best_score"],
		"stopped_early": stopped_early,
	}
elif args.early_stopping:
	X_inner_train, X_valid, y_inner_train, y_valid = train_test_split(
		X_train,
		y_train,
		test_size=args.validation_fraction,
		random_state=args.random_state,
	)

	inner_preprocessor = _build_preprocessor(X_inner_train)

	X_inner_train_processed = inner_preprocessor.fit_transform(X_inner_train)
	X_valid_processed = inner_preprocessor.transform(X_valid)
	n_train_effective = int(len(X_inner_train))
	n_val = int(len(X_valid))

	search_model_kwargs = dict(model_kwargs)
	search_model_kwargs["early_stopping_rounds"] = int(args.n_iter_no_change)
	search_regressor = xgb.XGBRegressor(**search_model_kwargs)
	search_regressor.fit(
		X_inner_train_processed,
		y_inner_train,
		eval_set=[(X_valid_processed, y_valid)],
		verbose=xgb_fit_verbose,
	)

	search_n_estimators_raw = search_regressor.get_params().get("n_estimators")
	search_n_estimators = int(search_n_estimators_raw) if search_n_estimators_raw is not None else 100

	best_iteration = getattr(search_regressor, "best_iteration", None)
	best_score_raw = getattr(search_regressor, "best_score", None)
	try:
		best_validation_score = float(best_score_raw) if best_score_raw is not None else None
	except (TypeError, ValueError):
		best_validation_score = None

	if best_iteration is not None:
		selected_n_estimators = int(best_iteration) + 1
	else:
		selected_n_estimators = search_n_estimators

	best_step = int(best_iteration) + 1 if best_iteration is not None else None

	model.named_steps["regressor"].set_params(n_estimators=selected_n_estimators)

	# Refit final model on full training split with selected boosting rounds.
	fit_started_at = time.perf_counter()
	if training_verbose > 0:
		print("Training started: XGBRegressor")
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
	if training_verbose > 0:
		print(f"Training completed in {fit_time_seconds:.3f}s: XGBRegressor")

	training_control = {
		"enabled": True,
		"type": "boosting",
		"max_steps_configured": int(search_n_estimators),
		"steps_completed": int(selected_n_estimators),
		"patience": int(args.n_iter_no_change),
		"monitor_metric": xgb_eval_metric,
		"monitor_split": "val",
		"monitor_direction": "min",
		"best_step": best_step,
		"best_score": _round_metric(best_validation_score),
		"stopped_early": bool(best_iteration is not None and (int(best_iteration) + 1) < int(search_n_estimators)),
	}
else:
	# Fit on training data (pipeline fits preprocessors + model).
	fit_started_at = time.perf_counter()
	if training_verbose > 0:
		print("Training started: XGBRegressor")
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
	if training_verbose > 0:
		print(f"Training completed in {fit_time_seconds:.3f}s: XGBRegressor")

	configured_n_estimators_raw = model.named_steps["regressor"].get_params().get("n_estimators")
	configured_n_estimators = int(configured_n_estimators_raw) if configured_n_estimators_raw is not None else 100

	training_control = {
		"enabled": False,
		"type": "boosting",
		"max_steps_configured": configured_n_estimators,
		"steps_completed": configured_n_estimators,
		"patience": int(args.n_iter_no_change),
		"monitor_metric": None,
		"monitor_split": None,
		"monitor_direction": None,
		"best_step": None,
		"best_score": None,
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
print("Train Residual Mean:", _round_metric(train_residual_mean))  # Mean of residuals (should be ~0 for unbiased regression)
print("Train Residual Std:", _round_metric(train_residual_std))  # Standard deviation of residuals (spread of prediction errors)

# ---- Test Metrics (model performance on unseen data) ----
print("Test MSE:", _round_metric(test_mse))  # Mean Squared Error on test set (average squared prediction errors)
print("Test MAE:", _round_metric(test_mae))  # Mean Absolute Error on test set (average absolute difference from true values)
print("Test RMSE:", _round_metric(test_rmse))  # Root Mean Squared Error on test set (interpretable error in target units)
print("Test R2:", _round_metric(test_r2))  # R² score on test set (generalization performance)
print("Test Max Error:", _round_metric(test_max_error))  # Largest single absolute prediction error on test set (worst-case mistake)

# ---- Dataset Context ----
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
			"mean": _round_metric(target_mean_train),
			"std": _round_metric(target_std_train),
		},
		"data_sizes": {
			"n_train": n_train_effective,
			"n_val": n_val,
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
	regressor_params = _json_safe(model.named_steps["regressor"].get_params())
	estimator_params_compact = _select_estimator_params(
		regressor_params,
		[
			"objective",
			"booster",
			"eval_metric",
			"n_estimators",
			"max_depth",
			"learning_rate",
			"subsample",
			"colsample_bytree",
			"gamma",
			"min_child_weight",
			"reg_alpha",
			"reg_lambda",
			"random_state",
			"n_jobs",
			"tree_method",
			"device",
			"enable_categorical",
		],
	)

	run_metadata = {
		"run_id": run_id,
		"model_name": model_name,
		"timestamp": timestamp,
		"library": "xgboost",
		"task": args.task,
		"algorithm": "gradient_boosting",
		"estimator_class": "XGBRegressor",
		"model_id": "xgboost.xgbregressor",
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
				"enabled": bool(tuning_summary["enabled"] or training_control["enabled"]),
				"strategy": (
					f"{args.tuning_method}_search_cv"
					if tuning_summary["enabled"]
					else ("explicit_split" if training_control["enabled"] else None)
				),
				"validation_fraction": float(args.validation_fraction) if (training_control["enabled"] and not tuning_summary["enabled"]) else None,
				"random_state": int(args.random_state) if (tuning_summary["enabled"] or training_control["enabled"]) else None,
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
			"device_requested": xgb_device_requested,
			"device_effective": xgb_device,
			"device_resolution_warning": xgb_device_warning,
			"booster": args.booster,
			"n_estimators": int(args.n_estimators),
			"learning_rate": float(args.learning_rate),
			"max_depth": int(args.max_depth),
			"subsample": float(args.subsample),
			"colsample_bytree": float(args.colsample_bytree),
			"min_child_weight": float(args.min_child_weight),
			"reg_lambda": float(args.reg_lambda),
			"reg_alpha": float(args.reg_alpha),
			"objective": "reg:squarederror",
			"eval_metric": "rmse",
			"enable_tuning": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_folds": int(args.cv_folds) if tuning_summary["enabled"] else None,
			"cv_scoring": args.cv_scoring if tuning_summary["enabled"] else None,
			"cv_n_iter_requested": int(args.cv_n_iter) if (tuning_summary["enabled"] and args.tuning_method == "random") else None,
			"cv_n_iter": int(tuning_summary["n_iter"]) if tuning_summary["enabled"] and tuning_summary["n_iter"] is not None else None,
			"cv_n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"tuning": tuning_summary,
		"training_control": training_control,
		"selection": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"device_requested": xgb_device_requested,
			"device_effective": xgb_device,
			"n_jobs": model.named_steps["regressor"].get_params().get("n_jobs"),
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
			"xgboost": xgb.__version__,
		},
	}
	run_metadata = _compact_metadata(_json_safe(run_metadata))
	with (run_dir / "run.json").open("w", encoding="utf-8") as run_file:
		json.dump(run_metadata, run_file, indent=2, allow_nan=False)

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
				"device_requested": xgb_device_requested,
				"device_effective": xgb_device,
				"booster": args.booster,
				"random_state": int(args.random_state),
				"tuning_enabled": bool(tuning_summary["enabled"]),
				"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
				"cv_best_score": tuning_summary["best_score"],
				"mse": _round_metric(test_mse),
				"mae": _round_metric(test_mae),
				"rmse": _round_metric(test_rmse),
				"r2": _round_metric(test_r2),
				"max_error": _round_metric(test_max_error),
				"train_mse": _round_metric(train_mse),
				"train_mae": _round_metric(train_mae),
				"train_rmse": _round_metric(train_rmse),
				"train_r2": _round_metric(train_r2),
				"train_max_error": _round_metric(train_max_error),
				"n_train": int(n_train_effective),
				"n_val": int(n_val),
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
