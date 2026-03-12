import argparse
import hashlib
import json
import math
import pickle
from functools import partial
import numpy as np
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	precision_recall_fscore_support,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

# Ensure project root is importable so generated templates can load shared helpers.
_current_file = Path(__file__).resolve()
for _candidate in [_current_file.parent, *_current_file.parents]:
	if (_candidate / "libraries").is_dir():
		if str(_candidate) not in sys.path:
			sys.path.insert(0, str(_candidate))
		break

from libraries.model_template_helpers import (
	artifact_map as _artifact_map,
	build_training_control as _build_training_control,
	build_tuning_summary as _build_tuning_summary,
	compact_metadata as _compact_metadata,
	find_project_root as _project_root,
	initialize_artifact_run as _initialize_artifact_run,
	initialize_tuning_summary as _initialize_tuning_summary,
	json_safe as _json_safe,
	parse_bool_flag as _parse_bool,
	post_transform_feature_count as _post_transform_feature_count,
	round_metric as _round_metric_base,
	select_estimator_params as _select_estimator_params,
	set_deterministic_seeds as _set_deterministic_seeds,
	validate_artifact_contract as _validate_artifact_contract,
	validate_etl_outputs as _validate_etl_outputs,
	write_unified_registry_sqlite as _write_unified_registry_sqlite,
	write_model_schemas as _write_model_schemas,
)
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.search_utils import cv_scoring_name as _cv_scoring_name, enumerate_search_candidates as _enumerate_search_candidates, search_space_size as _search_space_size
from libraries.cli_helpers import lower_token as _lower_token
from libraries.xgboost_search_space import XGBoostSearchGridConfig as _XGBoostSearchGridConfig, build_xgboost_search_space as _build_xgboost_search_space
from libraries.xgboost_template_utils import resolve_xgboost_device as _resolve_xgboost_device

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (XGBoost Regression)
#
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
#   --max-depth <int>                               (tree boosters only; gbtree|dart; maximum tree depth for base learners)
#   --subsample <float>                             (tree boosters only; gbtree|dart; subsample ratio of the training instances)
#   --colsample-bytree <float>                      (tree boosters only; gbtree|dart; subsample ratio of columns when constructing each tree)
#   --min-child-weight <float>                      (tree boosters only; gbtree|dart; minimum sum of instance weight needed in a child)
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
#   --tuning-method grid|random|bayesian            (grid or randomized hyperparameter search; bayesian = Optuna-guided search)
#   --cv-folds <int>                                (number of cross-validation folds)
#   --cv-scoring rmse|mae|r2                        (metric used during CV tuning)
#   --cv-n-iter <int>                               (number of search iterations/trials for random or bayesian tuning)
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

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="XGBoost Classifier baseline")
parser.add_argument("--task", type=_lower_token, choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", type=_lower_token, choices=["full", "short"], default="full")
parser.add_argument("--booster", type=_lower_token, choices=["auto", "gbtree", "gblinear", "dart"], default="{{BOOSTER}}")
parser.add_argument("--device", type=_lower_token, choices=["auto", "cpu", "gpu"], default="{{DEVICE}}")
parser.add_argument("--save-model", type=_parse_bool, default=False)
parser.add_argument("--random-state", type=int, default=1)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--early-stopping", type=_parse_bool, default="{{EARLY_STOPPING_DEFAULT}}" == "True")
parser.add_argument("--validation-fraction", type=float, default=float("{{VALIDATION_FRACTION_DEFAULT}}"))
parser.add_argument("--n-iter-no-change", type=int, default=int("{{N_ITER_NO_CHANGE_DEFAULT}}"))
parser.add_argument("--n-estimators", type=int, default=int("{{XGB_N_ESTIMATORS_DEFAULT}}"))
parser.add_argument("--learning-rate", type=float, default=float("{{XGB_LEARNING_RATE_DEFAULT}}"))
parser.add_argument("--max-depth", type=int, default=int("{{XGB_MAX_DEPTH_DEFAULT}}"))
parser.add_argument("--subsample", type=float, default=float("{{XGB_SUBSAMPLE_DEFAULT}}"))
parser.add_argument("--colsample-bytree", type=float, default=float("{{XGB_COLSAMPLE_BYTREE_DEFAULT}}"))
parser.add_argument("--min-child-weight", type=float, default=float("{{XGB_MIN_CHILD_WEIGHT_DEFAULT}}"))
parser.add_argument("--reg-lambda", type=float, default=float("{{XGB_REG_LAMBDA_DEFAULT}}"))
parser.add_argument("--reg-alpha", type=float, default=float("{{XGB_REG_ALPHA_DEFAULT}}"))
parser.add_argument("--verbose", type=_lower_token, choices=["0", "1", "2", "auto"], default="2")
parser.add_argument("--metric-decimals", type=int, default=4)
parser.add_argument("--enable-tuning", type=_parse_bool, default="{{XGB_ENABLE_TUNING_DEFAULT}}" == "True")
parser.add_argument("--tuning-method", type=_lower_token, choices=["grid", "random", "bayesian"], default="{{XGB_TUNING_METHOD_DEFAULT}}")
parser.add_argument("--cv-folds", type=int, default=int("{{XGB_CV_FOLDS_DEFAULT}}"))
parser.add_argument("--cv-scoring", type=_lower_token, choices=["f1_macro", "accuracy", "roc_auc_ovr"], default="{{XGB_CV_SCORING_DEFAULT}}")
parser.add_argument("--cv-n-iter", type=int, default=int("{{XGB_CV_N_ITER_DEFAULT}}"))
parser.add_argument("--cv-n-jobs", type=int, default=int("{{XGB_CV_N_JOBS_DEFAULT}}"))
args = parser.parse_args()

SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
xgb_model_verbosity = min(3, max(0, int(training_verbose)))
xgb_fit_verbose = bool(training_verbose > 0)
cv_verbose = 0 if training_verbose <= 1 else 2
METRIC_DECIMALS = int(args.metric_decimals)
_round_metric = partial(_round_metric_base, decimals=METRIC_DECIMALS)
seed_control = _set_deterministic_seeds(int(args.random_state))

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
# For classification tasks, stratify to preserve class distribution.
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=args.test_size,
	random_state=args.random_state,
	stratify=y,
)

# Preprocess: impute missing values, then scale numeric and one-hot encode categorical features.
preprocessor = _build_preprocessor(X_train)

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Determine XGBoost objective and evaluation metric based on task type.
if args.task == "binary_classification":
	xgb_objective = "binary:logistic"
	xgb_eval_metric = "logloss"
	xgb_num_class = None
else:
	xgb_objective = "multi:softprob"
	xgb_eval_metric = "mlogloss"
	xgb_num_class = int(y.nunique())

# Bundle preprocessing + model into one inference-ready pipeline.
fit_time_seconds = 0.0
xgb_device_requested = args.device
xgb_device, xgb_device_warning = _resolve_xgboost_device(xgb_device_requested)
if xgb_device_warning is not None:
	print(f"Warning: {xgb_device_warning}")
if training_verbose > 0:
	print(f"Resolved XGBoost device: requested={xgb_device_requested}, effective={xgb_device}")

# Resolve auto booster for direct-fit pipeline construction.
_resolved_booster = "gbtree" if args.booster == "auto" else args.booster

model_kwargs = {
	"booster": _resolved_booster,
	"device": xgb_device,
	"random_state": args.random_state,
	"objective": xgb_objective,
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
if xgb_num_class is not None:
	model_kwargs["num_class"] = xgb_num_class

model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		(
			"classifier",
			xgb.XGBClassifier(**model_kwargs),
		),
	]
)

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
tuning_summary = _initialize_tuning_summary()
n_train_effective = int(len(X_train))
n_val = 0

if args.enable_tuning:
	if training_verbose > 0:
		print(
			f"Training started with tuning: method={args.tuning_method}, "
			f"cv={args.cv_folds}, scoring={args.cv_scoring}"
		)
	selected_cv_scoring = _cv_scoring_name(
		args.cv_scoring,
		{"f1_macro": "f1_macro", "accuracy": "accuracy", "roc_auc_ovr": "roc_auc_ovr"},
	)

	search_space = _build_xgboost_search_space(
		step_name="classifier",
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
	elif args.tuning_method == "grid":
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
		search = None
	fit_started_at = time.perf_counter()
	if args.tuning_method in {"random", "grid"}:
		search.fit(X_train, y_train)
		best_params = dict(search.best_params_)
		best_cv_score = float(search.best_score_)
		best_std = None
		if hasattr(search, "cv_results_") and "std_test_score" in search.cv_results_:
			best_std = float(search.cv_results_["std_test_score"][search.best_index_])
		n_candidates = int(len(search.cv_results_["params"])) if hasattr(search, "cv_results_") else None
	else:
		optuna.logging.set_verbosity(optuna.logging.WARNING)
		candidate_params = _enumerate_search_candidates(search_space)
		if len(candidate_params) == 0:
			raise ValueError("No tuning candidates available for bayesian optimization.")
		n_trials = int(n_iter)

		def _objective(trial: optuna.Trial) -> float:
			index = int(trial.suggest_int("candidate_index", 0, len(candidate_params) - 1))
			params = candidate_params[index]
			estimator = clone(model)
			estimator.set_params(**params)
			scores = cross_val_score(
				estimator,
				X_train,
				y_train,
				scoring=selected_cv_scoring,
				cv=int(args.cv_folds),
				n_jobs=int(args.cv_n_jobs),
			)
			trial.set_user_attr("score_std", float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
			trial.set_user_attr("params", _json_safe(params))
			return float(np.mean(scores))

		study = optuna.create_study(
			direction="maximize",
			sampler=optuna.samplers.TPESampler(seed=int(args.random_state)),
		)
		study.optimize(_objective, n_trials=int(n_trials), n_jobs=1, show_progress_bar=False)
		best_trial = study.best_trial
		best_index = int(best_trial.params["candidate_index"])
		best_params = dict(candidate_params[best_index])
		best_cv_score = float(best_trial.value)
		best_std = float(best_trial.user_attrs.get("score_std")) if best_trial.user_attrs.get("score_std") is not None else None
		n_candidates = int(len(candidate_params))
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
			stratify=y_train,
		)
		n_train_effective = int(len(X_inner_train))
		n_val = int(len(X_valid))

		inner_preprocessor = _build_preprocessor(X_inner_train)
		X_inner_train_processed = inner_preprocessor.fit_transform(X_inner_train)
		X_valid_processed = inner_preprocessor.transform(X_valid)

		search_model_kwargs = dict(model_kwargs)
		for param_name, param_value in best_params.items():
			if param_name.startswith("classifier__"):
				search_model_kwargs[param_name.split("__", 1)[1]] = param_value
		search_model_kwargs["early_stopping_rounds"] = int(args.n_iter_no_change)
		search_classifier = xgb.XGBClassifier(**search_model_kwargs)
		search_classifier.fit(
			X_inner_train_processed,
			y_inner_train,
			eval_set=[(X_valid_processed, y_valid)],
			verbose=xgb_fit_verbose,
		)

		search_n_estimators_raw = search_model_kwargs.get("n_estimators")
		search_n_estimators = int(search_n_estimators_raw) if search_n_estimators_raw is not None else 100
		best_iteration = getattr(search_classifier, "best_iteration", None)
		best_score_raw = getattr(search_classifier, "best_score", None)
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
		model.named_steps["classifier"].set_params(n_estimators=selected_n_estimators)

	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
	tuning_summary = _build_tuning_summary(
		enabled=True,
		method=args.tuning_method,
		cv_folds=int(args.cv_folds),
		scoring=args.cv_scoring,
		scoring_sklearn=selected_cv_scoring,
		n_iter=int(n_iter) if args.tuning_method in {"random", "bayesian"} else None,
		n_candidates=n_candidates,
		best_score=_round_metric(best_cv_score),
		best_score_std=_round_metric(best_std) if best_std is not None else None,
		best_params=_compact_metadata(_json_safe(best_params)),
	)
	if training_verbose >= 2:
		print(
			f"Tuning completed: candidates={tuning_summary['n_candidates']}, "
			f"best_{args.cv_scoring}={tuning_summary['best_score']}"
		)
		print(f"Tuning best params: {tuning_summary['best_params']}")
	training_control = _build_training_control(
		enabled=True,
		control_type="search_cv",
		max_steps_configured=int(n_iter) if args.tuning_method in {"random", "bayesian"} else int(n_candidates) if n_candidates is not None else n_candidates_upper,
		steps_completed=int(args.cv_folds) * int(n_candidates) if n_candidates is not None else None,
		patience=int(args.n_iter_no_change) if args.early_stopping else None,
		monitor_metric=xgb_eval_metric if args.early_stopping else f"cv_{args.cv_scoring}",
		monitor_split="val" if args.early_stopping else "cv",
		monitor_direction="max",
		best_step=best_step,
		best_score=_round_metric(best_validation_score) if args.early_stopping else tuning_summary["best_score"],
		stopped_early=stopped_early,
	)
elif args.early_stopping:
	X_inner_train, X_valid, y_inner_train, y_valid = train_test_split(
		X_train,
		y_train,
		test_size=args.validation_fraction,
		random_state=args.random_state,
		stratify=y_train,
	)
	n_train_effective = int(len(X_inner_train))
	n_val = int(len(X_valid))

	inner_preprocessor = _build_preprocessor(X_inner_train)

	X_inner_train_processed = inner_preprocessor.fit_transform(X_inner_train)
	X_valid_processed = inner_preprocessor.transform(X_valid)

	search_model_kwargs = dict(model_kwargs)
	search_model_kwargs["early_stopping_rounds"] = int(args.n_iter_no_change)
	search_classifier = xgb.XGBClassifier(**search_model_kwargs)
	search_classifier.fit(
		X_inner_train_processed,
		y_inner_train,
		eval_set=[(X_valid_processed, y_valid)],
		verbose=xgb_fit_verbose,
	)

	search_n_estimators_raw = search_classifier.get_params().get("n_estimators")
	search_n_estimators = int(search_n_estimators_raw) if search_n_estimators_raw is not None else 100

	best_iteration = getattr(search_classifier, "best_iteration", None)
	best_score_raw = getattr(search_classifier, "best_score", None)
	try:
		best_validation_score = float(best_score_raw) if best_score_raw is not None else None
	except (TypeError, ValueError):
		best_validation_score = None

	if best_iteration is not None:
		selected_n_estimators = int(best_iteration) + 1
	else:
		selected_n_estimators = search_n_estimators

	best_step = int(best_iteration) + 1 if best_iteration is not None else None

	model.named_steps["classifier"].set_params(n_estimators=selected_n_estimators)

	# Refit final model on full training split with selected boosting rounds.
	fit_started_at = time.perf_counter()
	if training_verbose > 0:
		print("Training started: XGBClassifier")
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
	if training_verbose > 0:
		print(f"Training completed in {fit_time_seconds:.3f}s: XGBClassifier")

	training_control = _build_training_control(
		enabled=True,
		control_type="boosting",
		max_steps_configured=int(search_n_estimators),
		steps_completed=int(selected_n_estimators),
		patience=int(args.n_iter_no_change),
		monitor_metric=xgb_eval_metric,
		monitor_split="val",
		monitor_direction="min",
		best_step=best_step,
		best_score=_round_metric(best_validation_score),
		stopped_early=bool(best_iteration is not None and (int(best_iteration) + 1) < int(search_n_estimators)),
	)
else:
	# Fit on training data (pipeline fits preprocessors + model).
	fit_started_at = time.perf_counter()
	if training_verbose > 0:
		print("Training started: XGBClassifier")
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)
	if training_verbose > 0:
		print(f"Training completed in {fit_time_seconds:.3f}s: XGBClassifier")

	configured_n_estimators_raw = model.named_steps["classifier"].get_params().get("n_estimators")
	configured_n_estimators = int(configured_n_estimators_raw) if configured_n_estimators_raw is not None else 100

	training_control = _build_training_control(
		enabled=False,
		control_type="boosting",
		max_steps_configured=configured_n_estimators,
		steps_completed=configured_n_estimators,
		patience=int(args.n_iter_no_change),
		monitor_metric=None,
		monitor_split=None,
		monitor_direction=None,
		best_step=None,
		best_score=None,
		stopped_early=False,
	)

# =============================================================
# ==================== EVALUATE MODEL =========================
# =============================================================

# Evaluate model on train/test splits.
predict_started_at = time.perf_counter()
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)

if isinstance(train_predictions, np.ndarray) and train_predictions.ndim > 1:
	if train_predictions.shape[1] == 1:
		train_predictions = train_predictions.ravel()
	else:
		train_predictions = np.argmax(train_predictions, axis=1)

if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
	if predictions.shape[1] == 1:
		predictions = predictions.ravel()
	else:
		predictions = np.argmax(predictions, axis=1)

train_predictions = np.asarray(train_predictions).ravel()
predictions = np.asarray(predictions).ravel()

if not np.issubdtype(train_predictions.dtype, np.integer):
	if len(model.named_steps["classifier"].classes_) == 2:
		train_predictions = (train_predictions >= 0.5).astype(int)
		predictions = (predictions >= 0.5).astype(int)
	else:
		train_predictions = np.rint(train_predictions).astype(int)
		predictions = np.rint(predictions).astype(int)

predict_time_seconds = float(time.perf_counter() - predict_started_at)
classifier_classes = model.named_steps["classifier"].classes_
train_accuracy = accuracy_score(y_train, train_predictions)
train_f1_macro = f1_score(y_train, train_predictions, average="macro", zero_division=0)
test_accuracy = accuracy_score(y_test, predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test, predictions)
test_precision_macro = precision_score(y_test, predictions, average="macro", zero_division=0)
test_recall_macro = recall_score(y_test, predictions, average="macro", zero_division=0)
test_f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)
test_confusion_matrix = confusion_matrix(y_test, predictions)
_, _, _, support_values = precision_recall_fscore_support(
	y_test,
	predictions,
	labels=classifier_classes,
	zero_division=0,
)
support_by_class = {str(label): int(count) for label, count in zip(classifier_classes, support_values)}
support_total = int(len(y_test))

# Calculate probability-based metrics when predict_proba is available.
probabilities = None
train_logloss_value = None
test_roc_auc_macro_ovr = None
test_pr_auc_macro_ovr = None
test_logloss_value = None
brier_score = None
roc_curve_points = None
y_test_binarized = None
if hasattr(model, "predict_proba"):
	train_probabilities = np.asarray(model.predict_proba(X_train))
	probabilities = np.asarray(model.predict_proba(X_test))
	if train_probabilities.ndim == 1:
		train_probabilities = np.column_stack([1.0 - train_probabilities, train_probabilities])
	if probabilities.ndim == 1:
		probabilities = np.column_stack([1.0 - probabilities, probabilities])
	if train_probabilities.ndim == 2 and train_probabilities.shape[1] == 1 and len(classifier_classes) == 2:
		positive_train = train_probabilities[:, 0]
		train_probabilities = np.column_stack([1.0 - positive_train, positive_train])
	if probabilities.ndim == 2 and probabilities.shape[1] == 1 and len(classifier_classes) == 2:
		positive_test = probabilities[:, 0]
		probabilities = np.column_stack([1.0 - positive_test, positive_test])
	train_logloss_value = float(log_loss(y_train, train_probabilities, labels=classifier_classes))
	test_logloss_value = float(log_loss(y_test, probabilities, labels=classifier_classes))
	is_binary_problem = len(classifier_classes) == 2
	if is_binary_problem:
		positive_class = classifier_classes[-1]
		positive_probabilities = probabilities[:, -1]
		y_true_binary = (y_test == positive_class).astype(int)
		test_roc_auc_macro_ovr = float(roc_auc_score(y_test, positive_probabilities))
		test_pr_auc_macro_ovr = float(average_precision_score(y_true_binary, positive_probabilities))
		brier_score = float(brier_score_loss(y_true_binary, positive_probabilities))
		fpr, tpr, thresholds = roc_curve(y_test, positive_probabilities, pos_label=positive_class)
		roc_curve_points = pd.DataFrame(
			{
				"fpr": fpr,
				"tpr": tpr,
				"threshold": thresholds,
			}
		)
	else:
		y_test_binarized = label_binarize(y_test, classes=classifier_classes)
		test_roc_auc_macro_ovr = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr", average="macro"))
		test_pr_auc_macro_ovr = float(average_precision_score(y_test_binarized, probabilities, average="macro"))
		brier_score = float(((probabilities - y_test_binarized) ** 2).sum(axis=1).mean())

is_binary_problem = len(classifier_classes) == 2

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model fit on data it learned from) ----
print("Train Accuracy:", _round_metric(train_accuracy))  # Proportion of correct predictions on training data
print("Train F1 Macro:", _round_metric(train_f1_macro))  # Macro-averaged F1 score on training set

# ---- Optional Probability-Based Train Metrics ----

if train_logloss_value is not None:
	print("Train Log Loss:", _round_metric(train_logloss_value))  # Cross-entropy loss on training set

# ---- Test Metrics (model performance on unseen data) ----
print("Test Accuracy:", _round_metric(test_accuracy))  # Overall proportion of correct predictions on test data
print("Test Balanced Accuracy:", _round_metric(test_balanced_accuracy))  # Average recall across classes (robust to imbalance)
print("Test Precision Macro:", _round_metric(test_precision_macro))  # Macro-averaged precision across classes
print("Test Recall Macro:", _round_metric(test_recall_macro))  # Macro-averaged recall across classes
print("Test F1 Macro:", _round_metric(test_f1_macro))  # Macro-averaged F1 score (balanced precision/recall)
print("Test Support Total:", support_total)  # Total number of true samples in test set
print("Test Support By Class:", support_by_class)  # True sample count per class (class distribution insight)

# ---- Optional Ranking Metrics (require probability outputs) ----
if test_roc_auc_macro_ovr is not None:
	print("Test ROC AUC Macro OVR:", _round_metric(test_roc_auc_macro_ovr))  # One-vs-rest macro ROC-AUC score

if test_pr_auc_macro_ovr is not None:
	print("Test PR AUC Macro OVR:", _round_metric(test_pr_auc_macro_ovr))  # One-vs-rest macro Precision-Recall AUC

# ---- Optional Probability / Calibration Metrics ----
if test_logloss_value is not None:
	print("Test Log Loss:", _round_metric(test_logloss_value))  # Cross-entropy loss on test probabilities

if brier_score is not None:
	print("Test Brier Score:", _round_metric(brier_score))  # Probability calibration metric (lower is better)

# ---- Training Control (early stopping / step tracking) ----
if training_control["enabled"]:
	print("Training Control Best Step:", training_control["best_step"])  # Iteration/epoch with best validation score
	print("Training Control Steps Completed:", training_control["steps_completed"])  # Total training iterations completed
	print("Training Control Best Score:", training_control["best_score"])  # Best validation score achieved

# ---- Sanity Checks ----
print("First 5 predictions:", predictions[:5])  # Sample predictions for quick sanity check
print("First 5 true labels:", y_test.iloc[:5].values)  # Corresponding true labels for sanity check

# =============================================================
# ========= EXPORT ARTIFACTS & MODEL REGISTRY =================
# =============================================================

# Artifact export and registry logging.
if SAVE_MODEL:
	model_name = args.name.strip() or Path(__file__).stem
	run_context = _initialize_artifact_run(
		project_root=_project_root(),
		model_name=model_name,
		artifact_name_mode=args.artifact_name_mode,
		data_path=data_path,
	)
	model_root_dir = run_context["model_root_dir"]
	timestamp = str(run_context["timestamp"])
	run_id = str(run_context["run_id"])
	data_hash = str(run_context["data_hash"])
	run_dir = run_context["run_dir"]
	model_dir = run_context["model_dir"]
	preprocess_dir = run_context["preprocess_dir"]
	eval_dir = run_context["eval_dir"]
	data_dir = run_context["data_dir"]
	inference_dir = run_context["inference_dir"]
	data_rows = int(len(df))
	data_columns = int(df.shape[1])

	with (model_dir / "model.pkl").open("wb") as model_file:
		pickle.dump(model, model_file)

	with (preprocess_dir / "preprocessor.pkl").open("wb") as preprocess_file:
		pickle.dump(model.named_steps["preprocess"], preprocess_file)

	metrics = {
		"train": {
			"accuracy": _round_metric(train_accuracy),
			"f1_macro": _round_metric(train_f1_macro),
			"log_loss": _round_metric(train_logloss_value),
		},
		"test": {
			"accuracy": _round_metric(test_accuracy),
			"balanced_accuracy": _round_metric(test_balanced_accuracy),
			"precision_macro": _round_metric(test_precision_macro),
			"recall_macro": _round_metric(test_recall_macro),
			"f1_macro": _round_metric(test_f1_macro),
			"roc_auc": {
				"average": "macro",
				"multi_class": "ovr",
				"value": _round_metric(test_roc_auc_macro_ovr),
			},
			"pr_auc": {
				"average": "macro",
				"multi_class": "ovr",
				"value": _round_metric(test_pr_auc_macro_ovr),
			},
			"log_loss": _round_metric(test_logloss_value),
			"brier_score": _round_metric(brier_score),
			"support_total": support_total,
			"support_by_class": support_by_class,
		},
		"data_sizes": {
			"n_train": n_train_effective,
			"n_val": n_val,
			"n_test": int(len(X_test)),
		},
		"primary_metric": {
			"name": "roc_auc" if is_binary_problem else "f1_macro",
			"split": "test",
			"direction": "max",
			"value": _round_metric(test_roc_auc_macro_ovr) if is_binary_problem else _round_metric(test_f1_macro),
		},
		"probabilities": {
			"source": "predict_proba" if hasattr(model, "predict_proba") else None,
			"calibrated": False if hasattr(model, "predict_proba") else None,
			"calibration_method": None,
		},
		"training_control": training_control,
		"tuning": tuning_summary,
	}
	metrics["calibration"] = metrics.get("probabilities")
	metrics["timing"] = {"fit_seconds": _round_metric(fit_time_seconds), "predict_seconds": _round_metric(predict_time_seconds)}
	with (eval_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
		json.dump(metrics, metrics_file, indent=2)

	confusion_matrix_df = pd.DataFrame(
		test_confusion_matrix,
		index=classifier_classes,
		columns=classifier_classes,
	)
	confusion_matrix_df.to_csv(eval_dir / "confusion_matrix.csv", index=True)

	cm_figure, cm_axis = plt.subplots(figsize=(6, 5))
	cm_display = ConfusionMatrixDisplay(
		confusion_matrix=test_confusion_matrix,
		display_labels=classifier_classes,
	)
	cm_display.plot(ax=cm_axis, cmap="Blues", colorbar=False)
	cm_axis.set_title("Confusion Matrix")
	cm_figure.tight_layout()
	cm_figure.savefig(eval_dir / "confusion_matrix.png", dpi=150)
	plt.close(cm_figure)

	if roc_curve_points is not None:
		roc_curve_points.to_csv(eval_dir / "roc_curve.csv", index=False)

	if test_roc_auc_macro_ovr is not None:
		roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
		if is_binary_problem and roc_curve_points is not None:
			roc_axis.plot(
				roc_curve_points["fpr"],
				roc_curve_points["tpr"],
				label=f"ROC AUC = {test_roc_auc_macro_ovr:.4f}",
			)
		else:
			for class_index, class_label in enumerate(classifier_classes):
				class_fpr, class_tpr, _ = roc_curve(y_test_binarized[:, class_index], probabilities[:, class_index])
				roc_axis.plot(class_fpr, class_tpr, label=f"Class {class_label}")
			roc_axis.text(0.6, 0.1, f"Macro OVR AUC = {test_roc_auc_macro_ovr:.4f}")

		roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
		roc_axis.set_xlim(0.0, 1.0)
		roc_axis.set_ylim(0.0, 1.05)
		roc_axis.set_xlabel("False Positive Rate")
		roc_axis.set_ylabel("True Positive Rate")
		roc_axis.set_title("ROC Curve")
		roc_axis.legend(loc="lower right")
		roc_figure.tight_layout()
		roc_figure.savefig(eval_dir / "roc_curve.png", dpi=150)
		plt.close(roc_figure)

	predictions_preview = pd.DataFrame(
		{
			"y_true": y_test.iloc[:50].tolist(),
			"y_pred": pd.Series(predictions[:50]).tolist(),
		}
	)
	predictions_preview.to_csv(eval_dir / "predictions_preview.csv", index=False)

	inference_rows = X_test.iloc[:5].to_dict(orient="records")
	expected_values = y_test.iloc[:5].tolist()
	class_labels = [str(label) for label in classifier_classes.tolist()]
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
class_labels = {json.dumps(class_labels, indent=2)}

with MODEL_PATH.open("rb") as model_file:
	model = pickle.load(model_file)

features = pd.DataFrame(sample_rows)
predictions = model.predict(features)
probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None

if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
	if predictions.shape[1] == 1:
		predictions = predictions.ravel()
	else:
		predictions = np.argmax(predictions, axis=1)
predictions = np.asarray(predictions).ravel()

print("Inference Example")
print("Input Rows:", len(features))
print("Predictions:", predictions.tolist())
print("Expected:", expected_y)
if probabilities is not None:
	print("Class Labels:", class_labels)
	print("Probabilities:", np.asarray(probabilities).tolist())

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
	classifier_params = _json_safe(model.named_steps["classifier"].get_params())
	estimator_params_compact = _select_estimator_params(
		classifier_params,
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
			"num_class",
			"enable_categorical",
		],
	)

	artifacts_for_map = {
		"model": model_dir / "model.pkl",
		"preprocess": preprocess_dir / "preprocessor.pkl",
		"eval_metrics": eval_dir / "metrics.json",
		"eval_confusion_matrix": eval_dir / "confusion_matrix.csv",
		"eval_confusion_matrix_plot": eval_dir / "confusion_matrix.png",
		"eval_predictions_preview": eval_dir / "predictions_preview.csv",
		"eval_roc_curve_plot": eval_dir / "roc_curve.png",
		"inference_example": inference_dir / "inference_example.py",
		**schema_artifacts,
	}
	roc_curve_csv = eval_dir / "roc_curve.csv"
	if roc_curve_csv.exists():
		artifacts_for_map["eval_roc_curve"] = roc_curve_csv

	run_metadata = {
		"run_id": run_id,
		"model_name": model_name,
		"timestamp": timestamp,
		"library": "xgboost",
		"task": args.task,
		"algorithm": "gradient_boosting",
		"estimator_class": "XGBClassifier",
		"model_id": "xgboost.xgbclassifier",
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
			"stratify": True,
			"validation": {
				"enabled": bool(tuning_summary["enabled"] or training_control["enabled"]),
				"strategy": (f"{args.tuning_method}_search_cv" if tuning_summary["enabled"] else ("explicit_split" if training_control["enabled"] else None)),
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
			"objective": xgb_objective,
			"eval_metric": xgb_eval_metric,
			"num_class": xgb_num_class,
			"enable_tuning": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_folds": int(args.cv_folds) if tuning_summary["enabled"] else None,
			"cv_scoring": args.cv_scoring if tuning_summary["enabled"] else None,
			"cv_n_iter_requested": int(args.cv_n_iter) if (tuning_summary["enabled"] and args.tuning_method in {"random", "bayesian"}) else None,
			"cv_n_iter": int(tuning_summary["n_iter"]) if tuning_summary["enabled"] and tuning_summary["n_iter"] is not None else None,
			"cv_n_jobs": int(args.cv_n_jobs) if tuning_summary["enabled"] else None,
		},
		"tuning": tuning_summary,
		"training_control": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"seed_control": seed_control,
			"device_requested": xgb_device_requested,
			"device_effective": xgb_device,
			"n_jobs": classifier_params.get("n_jobs"),
		},
		"artifacts": _artifact_map(run_dir, artifacts_for_map),
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
				"device_requested": xgb_device_requested,
				"device_effective": xgb_device,
				"booster": args.booster,
				"num_class": int(xgb_num_class) if xgb_num_class is not None else None,
				"random_state": int(args.random_state),
				"training_control_enabled": bool(training_control["enabled"]),
				"tuning_enabled": bool(tuning_summary["enabled"]),
				"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
				"cv_best_score": tuning_summary["best_score"],
				"training_control_best_score": float(training_control["best_score"]) if training_control["best_score"] is not None else None,
				"training_control_best_step": int(training_control["best_step"]) if training_control["best_step"] is not None else None,
				"training_control_steps_completed": int(training_control["steps_completed"]) if training_control["steps_completed"] is not None else None,
				"training_control_stopped_early": bool(training_control["stopped_early"]),
				"accuracy": _round_metric(test_accuracy),
				"balanced_accuracy": _round_metric(test_balanced_accuracy),
				"precision_macro": _round_metric(test_precision_macro),
				"recall_macro": _round_metric(test_recall_macro),
				"f1_macro": _round_metric(test_f1_macro),
				"support": support_total,
				"roc_auc_macro_ovr": _round_metric(test_roc_auc_macro_ovr) if test_roc_auc_macro_ovr is not None else None,
				"pr_auc_macro_ovr": _round_metric(test_pr_auc_macro_ovr) if test_pr_auc_macro_ovr is not None else None,
				"log_loss": _round_metric(test_logloss_value) if test_logloss_value is not None else None,
				"brier_score": _round_metric(brier_score) if brier_score is not None else None,
				"train_accuracy": _round_metric(train_accuracy),
				"train_f1_macro": _round_metric(train_f1_macro),
				"train_log_loss": _round_metric(train_logloss_value) if train_logloss_value is not None else None,
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
