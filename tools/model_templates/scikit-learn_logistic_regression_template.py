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

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
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
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.search_utils import cv_scoring_name as _cv_scoring_name, search_space_size as _search_space_size
from libraries.serialization_utils import json_safe_best_params as _json_safe_best_params
from libraries.sklearn_template_utils import resolved_n_iter as _resolved_n_iter
from libraries.logistic_regression_search_space import (
	LogisticRegressionSearchGridConfig,
	build_logistic_regression_search_space as _build_search_space,
)
from libraries.logistic_compat import (
	LOGISTIC_SOLVER_PENALTY_COMPAT as _SOLVER_PENALTY_COMPAT,
	NON_AUTO_LOGISTIC_SOLVERS as _NON_AUTO_LOGISTIC_SOLVERS,
)

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (Logistic Regression)
#
# Core run options (auto-configured for all ML models generated)
#   --name <model_name>                             (model name used for registry and artifact folder; default: script filename)
#   --artifact-name-mode full|short                 (full = timestamp + UUID for unique runs; short = readable name but may overwrite previous runs)
#   --save-model true|false                         (save trained model and artifacts; false logs metrics only)
#   --verbose 0|1|2|auto                            (0=silent, 1=training progress, 2=training + tuning progress, auto=adaptive verbosity)
#   --metric-decimals <int>                         (decimal precision for logged metrics and artifacts)
#
# Task + data split / reproducibility
#   --task binary_classification|										(task type for metric calculation and logging)
# 				 multiclass_classification								^
#   --random-state <int>                           	(random seed for reproducibility)
#   --test-size <float>                            	(test set fraction; e.g., 0.2 = 80/20 split)
#
# Hyperparameter tuning configuration
#   --enable-tuning true|false                  		(enable hyperparameter tuning with cross-validation)
# 
# Direct-fit model settings 										(used when --enable-tuning=false)
#   --penalty none|l1|l2|elasticnet                 (regularization type applied during training)
#   --solver lbfgs|liblinear|newton-cg|							^
# 					 newton-cholesky|sag|saga								(optimization algorithm used to fit the model)
#   --c <float>                                     (inverse regularization strength; smaller values = stronger regularization)
#   --class-weight none|balanced                    (adjust class weighting to handle class imbalance)
#
# Hyperparameter tuning 												(used when --enable-tuning=true)
#   --penalty auto|l1|l2|elasticnet               	(regularization type to search during tuning; auto lets template select valid combinations)
#   --solver auto|lbfgs|liblinear|newton-cg|				(solver to search during tuning; auto selects compatible solvers)
# 					 newton-cholesky|sag|saga  							^
#   --tuning-method grid|random                     (grid = exhaustive search over grid; random = randomized search over iterations)
#   --cv-n-iter <int>                               (random search iterations; only used when --tuning-method=random)
#   --cv-folds <int>                                (number of cross-validation folds)
#   --cv-scoring f1_macro|accuracy|roc_auc_ovr      (metric used during CV tuning)
#   --cv-n-jobs <int>                               (CV search parallelism; -1 uses all cores)
# 
# Common logistic regression hyperparameters (used in both direct fit and tuning)
#   --max-iter <int>                                (maximum number of optimization iterations)
# ---------------------------------------------------------------------

LOGISTIC_SOLVERS = ["auto", *list(_NON_AUTO_LOGISTIC_SOLVERS)]
LOGISTIC_PENALTIES = ["auto", "none", "l1", "l2", "elasticnet"]

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Logistic Regression baseline")
parser.add_argument("--task", choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=False)
parser.add_argument("--random-state", type=int, default=1)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--max-iter", type=int, default=int("{{MAX_ITER_DEFAULT}}"))
parser.add_argument("--c", type=float, default=float("{{LOGISTIC_C_DEFAULT}}"))
parser.add_argument("--solver", choices=LOGISTIC_SOLVERS, default="{{LOGISTIC_SOLVER_DEFAULT}}")
parser.add_argument("--penalty", choices=LOGISTIC_PENALTIES, default="{{LOGISTIC_PENALTY_DEFAULT}}")
parser.add_argument("--class-weight", choices=["none", "balanced"], default="{{LOGISTIC_CLASS_WEIGHT_DEFAULT}}")
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default="1")
parser.add_argument("--metric-decimals", type=int, default=4)
parser.add_argument("--enable-tuning", type=_parse_bool, default="{{LOGISTIC_ENABLE_TUNING_DEFAULT}}" == "True")
parser.add_argument("--tuning-method", choices=["grid", "random"], default="{{LOGISTIC_TUNING_METHOD_DEFAULT}}")
parser.add_argument("--cv-folds", type=int, default=int("{{LOGISTIC_CV_FOLDS_DEFAULT}}"))
parser.add_argument("--cv-scoring", choices=["f1_macro", "accuracy", "roc_auc_ovr"], default="{{LOGISTIC_CV_SCORING_DEFAULT}}")
parser.add_argument("--cv-n-iter", type=int, default=int("{{LOGISTIC_CV_N_ITER_DEFAULT}}"))
parser.add_argument("--cv-n-jobs", type=int, default=int("{{LOGISTIC_CV_N_JOBS_DEFAULT}}"))
args = parser.parse_args()
if not args.enable_tuning and args.solver == "auto":
	raise ValueError("--solver=auto requires --enable-tuning=true.")
if not args.enable_tuning and args.penalty == "auto":
	raise ValueError("--penalty=auto requires --enable-tuning=true.")

# Solver/penalty compatibility validation (direct-fit path only).
# When tuning is enabled the search space handles compatibility via
# separate sub-grids, so we skip this check.  --solver=auto defers
# solver selection to the tuning search space (or defaults to lbfgs
# for direct fit).
if not args.enable_tuning and args.solver != "auto":
	if args.penalty not in _SOLVER_PENALTY_COMPAT.get(args.solver, set()):
		valid_penalties = sorted(_SOLVER_PENALTY_COMPAT[args.solver])
		raise ValueError(
			f"Solver '{args.solver}' does not support penalty='{args.penalty}'. "
			f"Valid penalties for {args.solver}: {valid_penalties}"
		)

SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
cv_verbose = 0 if training_verbose <= 1 else 2
METRIC_DECIMALS = int(args.metric_decimals)
_round_metric = partial(_round_metric_base, decimals=METRIC_DECIMALS)

# NOTE: Adjust these grids to customize search breadth for tuning.
LOGISTIC_SEARCH_GRID_CONFIG = LogisticRegressionSearchGridConfig(
	c_grid=[0.01, 0.1, 1.0, 10.0],  # inverse regularization strength (smaller = stronger regularization)
	max_iter=[500, 1000, 2000],  # maximum optimization iterations allowed for solver convergence
	class_weight=[None, "balanced"],  # None = uniform weights; "balanced" adjusts weights for class imbalance
	elasticnet_l1_ratio=[0.2, 0.5, 0.8],  # elasticnet mixing ratio (0≈L2, 1≈L1); only used when penalty="elasticnet"
	solver_penalty_compat=_SOLVER_PENALTY_COMPAT,  # mapping of valid solver→penalty combinations to avoid illegal configs
	solver_order=_NON_AUTO_LOGISTIC_SOLVERS,  # deterministic order of concrete solvers (excluding "auto") for tuning/search
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

preprocessor = _build_preprocessor(X_train)

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Bundle preprocessing + model into one inference-ready pipeline.
# Input to inference should be raw feature columns.
# Resolve --solver=auto to a concrete solver for the initial pipeline.
# For direct fit this is the final solver; for tuning it's just a
# placeholder that the search will override.
_resolved_solver = args.solver if args.solver != "auto" else "lbfgs"
penalty_value = "none" if args.penalty == "none" else args.penalty
class_weight_value = None if args.class_weight == "none" else args.class_weight
classifier_kwargs = {
	"penalty": penalty_value,
	"C": float(args.c),
	"solver": _resolved_solver,
	"class_weight": class_weight_value,
	"max_iter": int(args.max_iter),
	"random_state": int(args.random_state),
	"verbose": int(training_verbose),
}
if penalty_value == "elasticnet":
	classifier_kwargs["l1_ratio"] = 0.5
classifier = LogisticRegression(**classifier_kwargs)
classifier_name = f"LogisticRegression(penalty={penalty_value}, C={float(args.c):.6g}, solver={_resolved_solver}, class_weight={class_weight_value})"

model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("classifier", classifier),
	]
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================

selected_cv_scoring = _cv_scoring_name(
	args.cv_scoring,
	{"f1_macro": "f1_macro", "accuracy": "accuracy", "roc_auc_ovr": "roc_auc_ovr"},
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
		print(f"Training started: {classifier_name}")

if args.enable_tuning:
	search_space = _build_search_space(
		args.solver,
		args.penalty,
		int(args.random_state),
		LOGISTIC_SEARCH_GRID_CONFIG,
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
	best_std = None
	if hasattr(search, "cv_results_") and "std_test_score" in search.cv_results_:
		best_std = float(search.cv_results_["std_test_score"][search.best_index_])
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
classifier_step = model.named_steps["classifier"]
resolved_n_iter = _resolved_n_iter(classifier_step)
if training_verbose > 0:
	print(f"Training completed in {fit_time_seconds:.3f}s: {type(classifier_step).__name__}")

training_control = {
	"enabled": bool(tuning_summary["enabled"]),
	"type": f"{args.tuning_method}_search_cv" if tuning_summary["enabled"] else None,
	"max_steps_configured": tuning_summary["n_candidates"] if args.tuning_method == "grid" else (int(tuning_summary["n_iter"]) if tuning_summary["enabled"] and tuning_summary["n_iter"] is not None else None),
	"steps_completed": int(args.cv_folds) * int(tuning_summary["n_candidates"]) if tuning_summary["enabled"] and tuning_summary["n_candidates"] is not None else (int(resolved_n_iter) if resolved_n_iter is not None else None),
	"patience": None,
	"monitor_metric": f"cv_{args.cv_scoring}" if tuning_summary["enabled"] else None,
	"monitor_split": "cv" if tuning_summary["enabled"] else None,
	"monitor_direction": "max" if tuning_summary["enabled"] else None,
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
train_logloss_value = None
test_roc_auc_macro_ovr = None
test_pr_auc_macro_ovr = None
test_logloss_value = None
brier_score = None
roc_curve_points = None
y_test_binarized = None

if hasattr(model, "predict_proba"):
	train_probabilities = model.predict_proba(X_train)
	probabilities = model.predict_proba(X_test)
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
		# ROC curve points are saved for binary classification in this template.
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
print("Train F1 Macro:", _round_metric(train_f1_macro))  # Macro-averaged F1 on training set (equal weight per class)

# ---- Optional Probability-Based Train Metrics ----

if train_logloss_value is not None:
	print("Train Log Loss:", _round_metric(train_logloss_value))  # Cross-entropy loss on training set (probability quality)

# ---- Test Metrics (model performance on unseen data) ----
print("Test Accuracy:", _round_metric(test_accuracy))  # Overall correctness on test set
print("Test Balanced Accuracy:", _round_metric(test_balanced_accuracy))  # Mean recall across classes (robust to imbalance)
print("Test Precision Macro:", _round_metric(test_precision_macro))  # Macro-averaged precision (mean of per-class precision)
print("Test Recall Macro:", _round_metric(test_recall_macro))  # Macro-averaged recall (mean of per-class recall)
print("Test F1 Macro:", _round_metric(test_f1_macro))  # Macro-averaged F1 (harmonic mean of precision & recall per class)
print("Test Support Total:", support_total)  # Total number of true samples in test set
print("Test Support By Class:", support_by_class)  # True sample count per class (class distribution)

# ---- Optional Ranking Metrics (require predict_proba / decision scores) ----
if test_roc_auc_macro_ovr is not None:
	print("Test ROC AUC Macro OVR:", _round_metric(test_roc_auc_macro_ovr))  # One-vs-Rest macro ROC-AUC (ranking quality across classes)

if test_pr_auc_macro_ovr is not None:
	print("Test PR AUC Macro OVR:", _round_metric(test_pr_auc_macro_ovr))  # One-vs-Rest macro PR-AUC (precision-recall tradeoff)

# ---- Optional Probability Calibration Metrics ----
if test_logloss_value is not None:
	print("Test Log Loss:", _round_metric(test_logloss_value))  # Cross-entropy loss on test set (penalizes confident wrong predictions)

if brier_score is not None:
	print("Test Brier Score:", _round_metric(brier_score))  # Mean squared error of predicted probabilities (calibration metric)

# ---- Sanity Checks ----
print("Classifier:", classifier_name)  # Model identifier for experiment tracking
print("First 5 predictions:", predictions[:5])  # Quick sanity check of output classes
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
		# Saves full inference-ready pipeline: preprocess + classifier.
		pickle.dump(model, model_file)

	with (preprocess_dir / "preprocessor.pkl").open("wb") as preprocess_file:
		pickle.dump(model.named_steps["preprocess"], preprocess_file)

	n_val_metrics = 0
	n_train_effective_metrics = int(len(X_train))
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
			"n_train": n_train_effective_metrics,
			"n_val": n_val_metrics,
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
		index=model.named_steps["classifier"].classes_,
		columns=model.named_steps["classifier"].classes_,
	)
	confusion_matrix_df.to_csv(eval_dir / "confusion_matrix.csv", index=True)

	cm_figure, cm_axis = plt.subplots(figsize=(6, 5))
	cm_display = ConfusionMatrixDisplay(
		confusion_matrix=test_confusion_matrix,
		display_labels=model.named_steps["classifier"].classes_,
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

	# Example inference rows are kept in raw feature format on purpose.
	# Do NOT pre-one-hot-encode these rows; model.pkl handles preprocessing.
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

print("Inference Example")
print("Input Rows:", len(features))
print("Predictions:", predictions.tolist())
print("Expected:", expected_y)
if probabilities is not None:
	print("Class Labels:", class_labels)
	print("Probabilities:", probabilities.tolist())

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
	if hasattr(classifier_step, "coef_"):
		coef_matrix = np.asarray(getattr(classifier_step, "coef_"), dtype=float)
		if coef_matrix.ndim == 1:
			coef_matrix = coef_matrix.reshape(1, -1)
		class_labels = [str(label) for label in classifier_step.classes_.tolist()]
		if coef_matrix.shape[0] == 1 and len(class_labels) == 2:
			coefficient_rows = [class_labels[1]]
		else:
			coefficient_rows = class_labels[: int(coef_matrix.shape[0])]
		usable_length = min(len(feature_names), int(coef_matrix.shape[1]))
		coef_records: list[dict[str, object]] = []
		for row_index, class_label in enumerate(coefficient_rows):
			for feature_index in range(usable_length):
				coef_records.append(
					{
						"feature": feature_names[feature_index],
						"coefficient": float(coef_matrix[row_index, feature_index]),
					}
				)
		coefficients_csv_path = eval_dir / "coefficients.csv"
		pd.DataFrame(coef_records).to_csv(coefficients_csv_path, index=False)
		intercepts = []
		if hasattr(classifier_step, "intercept_"):
			intercepts = np.asarray(getattr(classifier_step, "intercept_"), dtype=float).reshape(-1).tolist()
		coefficients_summary_path = eval_dir / "coefficients_summary.json"
		coefficients_summary_path.write_text(
			json.dumps(
				{
					"n_features": int(usable_length),
					"n_class_rows": int(len(coefficient_rows)),
					"intercepts": [_round_metric(value) for value in intercepts],
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
	n_val = 0
	n_train_effective = int(len(X_train))
	estimator_class = classifier_step.__class__.__name__
	estimator_params = classifier_step.get_params()
	estimator_params_compact = _select_estimator_params(
		estimator_params,
		[
			"penalty",
			"C",
			"fit_intercept",
			"solver",
			"max_iter",
			"multi_class",
			"class_weight",
			"random_state",
			"n_jobs",
			"l1_ratio",
			"tol",
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
		**coefficient_artifacts,
	}
	roc_curve_csv = eval_dir / "roc_curve.csv"
	if roc_curve_csv.exists():
		artifacts_for_map["eval_roc_curve"] = roc_curve_csv
	run_metadata = {
		"run_id": run_id,
		"model_name": model_name,
		"timestamp": timestamp,
		"library": "scikit-learn",
		"task": args.task,
		"algorithm": "logistic_regression",
		"estimator_class": estimator_class,
		"model_id": "sklearn.logisticregression",
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
			"max_iter": int(args.max_iter),
			"penalty": args.penalty,
			"c": float(args.c),
			"solver": args.solver,
			"class_weight": args.class_weight,
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
			"n_jobs": estimator_params.get("n_jobs"),
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
			"eval_confusion_matrix",
			"eval_confusion_matrix_plot",
			"eval_predictions_preview",
			"inference_example",
			"input_schema",
			"target_mapping_schema",
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
				"n_train": n_train_effective,
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
