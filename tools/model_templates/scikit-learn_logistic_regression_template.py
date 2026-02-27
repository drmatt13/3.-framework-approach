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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

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
#   --task binary_classification|multiclass_classification
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
#   --max-iter <int>
#   --penalty none|l1|l2|elasticnet
#   --c <float>
#   --solver lbfgs|liblinear|newton-cg|newton-cholesky|sag|saga
#   --class-weight none|balanced
#   --verbose 0|1|2|auto
#   --metric-decimals <int>
# ---------------------------------------------------------------------

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_MAX_ITER = int("{{MAX_ITER_DEFAULT}}")
DEFAULT_C = float("{{LOGISTIC_C_DEFAULT}}")
DEFAULT_SOLVER = "{{LOGISTIC_SOLVER_DEFAULT}}"
DEFAULT_PENALTY = "{{LOGISTIC_PENALTY_DEFAULT}}"
DEFAULT_CLASS_WEIGHT = "{{LOGISTIC_CLASS_WEIGHT_DEFAULT}}"
LOGISTIC_SOLVERS = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
LOGISTIC_PENALTIES = ["none", "l1", "l2", "elasticnet"]
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4

# Helper function: resolve n_iter_ for iterative classifiers, handling different formats.
def _resolved_n_iter(model_step) -> int | None:
	n_iter_value = getattr(model_step, "n_iter_", None)
	if n_iter_value is None:
		return None
	if hasattr(n_iter_value, "tolist"):
		n_iter_value = n_iter_value.tolist()
	if isinstance(n_iter_value, (list, tuple)):
		if not n_iter_value:
			return None
		return int(max(n_iter_value))
	return int(n_iter_value)

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Logistic Regression baseline")
parser.add_argument("--task", choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
parser.add_argument("--c", type=float, default=DEFAULT_C)
parser.add_argument("--solver", choices=LOGISTIC_SOLVERS, default=DEFAULT_SOLVER)
parser.add_argument("--penalty", choices=LOGISTIC_PENALTIES, default=DEFAULT_PENALTY)
parser.add_argument("--class-weight", choices=["none", "balanced"], default=DEFAULT_CLASS_WEIGHT)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
args = parser.parse_args()

# Solver/penalty compatibility validation.
_SOLVER_PENALTY_COMPAT = {
	"lbfgs": {"l2", "none"},
	"liblinear": {"l1", "l2"},
	"newton-cg": {"l2", "none"},
	"newton-cholesky": {"l2", "none"},
	"sag": {"l2", "none"},
	"saga": {"l1", "l2", "elasticnet", "none"},
}
if args.penalty not in _SOLVER_PENALTY_COMPAT.get(args.solver, set()):
	valid_penalties = sorted(_SOLVER_PENALTY_COMPAT[args.solver])
	raise ValueError(
		f"Solver '{args.solver}' does not support penalty='{args.penalty}'. "
		f"Valid penalties for {args.solver}: {valid_penalties}"
	)
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
# For classification tasks, stratify to preserve class distribution.
X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=args.test_size,
	random_state=args.random_state,
	stratify=y,
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
# Input to inference should be raw feature columns.
penalty_value = None if args.penalty == "none" else args.penalty
class_weight_value = None if args.class_weight == "none" else args.class_weight
l1_ratio_value = 0.5 if penalty_value == "elasticnet" else None
classifier = LogisticRegression(
	penalty=penalty_value,
	C=float(args.c),
	solver=args.solver,
	class_weight=class_weight_value,
	max_iter=args.max_iter,
	l1_ratio=l1_ratio_value,
	random_state=args.random_state,
	verbose=training_verbose,
)
classifier_name = f"LogisticRegression(penalty={penalty_value}, C={float(args.c):.6g}, solver={args.solver}, class_weight={class_weight_value})"

model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("classifier", classifier),
	]
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================
# ---------------------------------------------------------------------
# EARLY STOPPING (optional)
# - Enabled with --early-stopping=true.
# - Uses --validation-fraction as holdout split from training data.
# - Stops when validation score does not improve for --n-iter-no-change rounds.
# - When disabled, trains once on full training split.
# ---------------------------------------------------------------------

# Fit on training data (pipeline fits preprocessors + model).
fit_started_at = time.perf_counter()
if training_verbose > 0:
	print(f"Training started: {classifier_name}")
model.fit(X_train, y_train)
fit_time_seconds = float(time.perf_counter() - fit_started_at)
if training_verbose > 0:
	print(f"Training completed in {fit_time_seconds:.3f}s: {classifier_name}")

classifier_step = model.named_steps["classifier"]
resolved_n_iter = _resolved_n_iter(classifier_step)

training_control = {
	"enabled": False,
	"strategy": None,
	"monitor_name": None,
	"monitor_mode": None,
	"max_steps_configured": int(args.max_iter),
	"steps_completed": int(resolved_n_iter) if resolved_n_iter is not None else None,
	"best_step": None,
	"best_score": None,
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
	}
	metrics["selection"] = training_control
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

	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": "scikit-learn",
		"task": args.task,
		"algorithm": "logistic_regression",
		"estimator_class": estimator_class,
		"model_id": "sklearn.logisticregression",
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
			"stratify": True,
			"validation": {
				"enabled": False,
				"strategy": None,
				"validation_fraction": None,
				"random_state": None,
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
		},
		"selection": training_control,
		"optimization": {
			"optimizer": None,
			"learning_rate": None,
			"batch_size": None,
			"epochs_configured": None,
			"epochs_completed": None,
			"gradient_clip_norm": None,
			"lr_scheduler": None,
		},
		"training_control": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": estimator_params.get("n_jobs"),
		},
		"artifacts": _artifact_map(
				run_dir,
				{
				"model": model_dir / "model.pkl",
				"preprocess": preprocess_dir / "preprocessor.pkl",
				"eval_metrics": eval_dir / "metrics.json",
				"eval_confusion_matrix": eval_dir / "confusion_matrix.csv",
				"eval_confusion_matrix_plot": eval_dir / "confusion_matrix.png",
				"eval_predictions_preview": eval_dir / "predictions_preview.csv",
				"eval_roc_curve": eval_dir / "roc_curve.csv",
				"eval_roc_curve_plot": eval_dir / "roc_curve.png",
				"inference_example": inference_dir / "inference_example.py",
					**schema_artifacts,
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

	print(f"Artifacts exported to: {run_dir}")
