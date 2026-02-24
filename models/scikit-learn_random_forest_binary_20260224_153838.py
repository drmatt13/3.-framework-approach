import argparse
import hashlib
import json
import pickle
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

# ---------------------------------------------------------------------
# Supported CLI flags (common usage)
#   --library scikit-learn
#   --model random_forest
#   --task binary_classification|multiclass_classification
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
# ---------------------------------------------------------------------

# Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_EARLY_STOPPING = "True" == "True"
DEFAULT_VALIDATION_FRACTION = float("0.1")
DEFAULT_N_ITER_NO_CHANGE = int("5")
DEFAULT_MAX_ESTIMATORS = 500
DEFAULT_ESTIMATOR_STEP = 25
MIN_ESTIMATORS_FOR_STOP = 100
METRIC_DECIMALS = 4
RUN_SCHEMA_VERSION = "1.0"
RUN_METADATA_PROFILE = "compact"

# Helper function: round metrics for cleaner output.
def _round_metric(value):
	return None if value is None else round(float(value), METRIC_DECIMALS)

# Helper function: find project root using a marker file.
def _project_root() -> Path:
	current = Path(__file__).resolve().parent
	for candidate in [current, *current.parents]:
		if (candidate / "requirements.txt").exists():
			return candidate
	return Path(__file__).resolve().parents[1]

# Helper function: parse boolean CLI input.
def _parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "y"}:
		return True
	if normalized in {"0", "false", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError("Expected true/false")


def _post_transform_feature_count(preprocessor, sample_frame: pd.DataFrame) -> int | None:
	try:
		transformed = preprocessor.transform(sample_frame)
		return int(transformed.shape[1])
	except Exception:
		return None


def _artifact_map(base_dir: Path, artifacts: dict[str, Path]) -> dict[str, str]:
	resolved: dict[str, str] = {}
	for key, path in artifacts.items():
		if path.exists():
			resolved[key] = str(path.relative_to(base_dir))
	return resolved


def _compact_metadata(value):
	if isinstance(value, dict):
		compacted = {}
		for key, item in value.items():
			compacted_item = _compact_metadata(item)
			if compacted_item is None:
				continue
			if isinstance(compacted_item, (dict, list)) and len(compacted_item) == 0:
				continue
			compacted[key] = compacted_item
		return compacted
	if isinstance(value, list):
		compacted_list = []
		for item in value:
			compacted_item = _compact_metadata(item)
			if compacted_item is None:
				continue
			if isinstance(compacted_item, (dict, list)) and len(compacted_item) == 0:
				continue
			compacted_list.append(compacted_item)
		return compacted_list
	return value


def _select_estimator_params(params: dict, keys: list[str]) -> dict:
	return {key: params.get(key) for key in keys if key in params}

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="Random Forest Classifier baseline")
parser.add_argument("--library", choices=["scikit-learn"], default="scikit-learn")
parser.add_argument("--model", choices=["random_forest"], default="random_forest")
parser.add_argument("--task", choices=["binary_classification"], default="binary_classification")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--early-stopping", type=_parse_bool, default=DEFAULT_EARLY_STOPPING)
parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
parser.add_argument("--n-iter-no-change", type=int, default=DEFAULT_N_ITER_NO_CHANGE)
args = parser.parse_args()
SAVE_MODEL = args.save_model

project_root = _project_root()
data_path = project_root / "data" / "template_data" / "breast_cancer_wisconsin.csv"
df = pd.read_csv(data_path)
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# =============================================================
# ================== MODEL CODE STARTS HERE ===================
# =============================================================
# This section contains model definition, training, evaluation,
# and artifact generation logic.

# Load data.
y = df["diagnosis"]
y = y.map({"B": 0, "M": 1}).astype("int64") # type: ignore
X = df.drop(columns=["diagnosis", "id"]) # type: ignore

# =============================================================
# ============== ADDITIONAL FEATURE ENGINEERING ===============
# =============================================================
# Add optional feature transformations or derived features below.

#  -

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

# Preprocess: scale numeric features and one-hot encode categorical features.
preprocessor = ColumnTransformer(
	transformers=[
		("num", StandardScaler(), numerical_cols),
		("cat", one_hot_encoder, categorical_cols),
	],
	remainder="drop",
)

# Bundle preprocessing + model into one inference-ready pipeline.
fit_time_seconds = 0.0
if args.early_stopping:
	X_inner_train, X_valid, y_inner_train, y_valid = train_test_split(
		X_train,
		y_train,
		test_size=args.validation_fraction,
		random_state=args.random_state,
		stratify=y_train,
	)

	inner_categorical_cols = X_inner_train.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
	inner_numerical_cols = X_inner_train.select_dtypes(include=["number"]).columns.tolist()

	try:
		inner_one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		inner_one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

	inner_preprocessor = ColumnTransformer(
		transformers=[
			("num", StandardScaler(), inner_numerical_cols),
			("cat", inner_one_hot_encoder, inner_categorical_cols),
		],
		remainder="drop",
	)

	X_inner_train_processed = inner_preprocessor.fit_transform(X_inner_train)
	X_valid_processed = inner_preprocessor.transform(X_valid)

	search_classifier = RandomForestClassifier(
		random_state=args.random_state,
		warm_start=True,
		n_estimators=0,
	)

	best_validation_balanced_accuracy = float("-inf")
	best_n_estimators = DEFAULT_ESTIMATOR_STEP
	rounds_without_improvement = 0
	rounds_completed = 0

	for n_estimators in range(DEFAULT_ESTIMATOR_STEP, DEFAULT_MAX_ESTIMATORS + 1, DEFAULT_ESTIMATOR_STEP):
		search_classifier.set_params(n_estimators=n_estimators)
		search_classifier.fit(X_inner_train_processed, y_inner_train)

		validation_predictions = search_classifier.predict(X_valid_processed)
		validation_balanced_accuracy = balanced_accuracy_score(y_valid, validation_predictions)
		rounds_completed += 1

		if validation_balanced_accuracy > (best_validation_balanced_accuracy + 1e-6):
			best_validation_balanced_accuracy = float(validation_balanced_accuracy)
			best_n_estimators = n_estimators
			rounds_without_improvement = 0
		else:
			rounds_without_improvement += 1

		if n_estimators >= MIN_ESTIMATORS_FOR_STOP and rounds_without_improvement >= args.n_iter_no_change:
			break

	model = Pipeline(
		steps=[
			("preprocess", preprocessor),
			("classifier", RandomForestClassifier(random_state=args.random_state, n_estimators=best_n_estimators)),
		]
	)
	fit_started_at = time.perf_counter()
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)

	model_selection = {
		"enabled": True,
		"type": "incremental_search",
		"search_space": {
			"n_estimators": {
				"start": int(DEFAULT_ESTIMATOR_STEP),
				"stop": int(DEFAULT_MAX_ESTIMATORS),
				"step": int(DEFAULT_ESTIMATOR_STEP),
			},
		},
		"metric": "balanced_accuracy",
		"metric_split": "val",
		"direction": "max",
		"trials_completed": int(rounds_completed),
		"best_trial_index": int(best_n_estimators // DEFAULT_ESTIMATOR_STEP),
		"best_params": {"n_estimators": int(best_n_estimators)},
		"best_score": _round_metric(best_validation_balanced_accuracy),
		"stopped_early": bool(best_n_estimators < DEFAULT_MAX_ESTIMATORS),
		"patience": int(args.n_iter_no_change),
	}
else:
	model = Pipeline(
		steps=[
			("preprocess", preprocessor),
			("classifier", RandomForestClassifier(random_state=args.random_state)),
		]
	)

	# Fit on training data (pipeline fits preprocessors + model).
	fit_started_at = time.perf_counter()
	model.fit(X_train, y_train)
	fit_time_seconds = float(time.perf_counter() - fit_started_at)

	model_selection = {
		"enabled": False,
		"type": "incremental_search",
		"search_space": {
			"n_estimators": {
				"start": int(DEFAULT_ESTIMATOR_STEP),
				"stop": int(DEFAULT_MAX_ESTIMATORS),
				"step": int(DEFAULT_ESTIMATOR_STEP),
			},
		},
		"metric": "balanced_accuracy",
		"metric_split": "val",
		"direction": "max",
		"trials_completed": 0,
		"best_trial_index": None,
		"best_params": {"n_estimators": int(model.named_steps["classifier"].n_estimators)},
		"best_score": None,
		"stopped_early": False,
		"patience": int(args.n_iter_no_change),
	}

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
	if args.task == "binary_classification":
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

# ---- Train Metrics (model fit on data it learned from) ----
print("Train Accuracy:", _round_metric(train_accuracy))  # Proportion of correct predictions on training data
print("Train F1 Macro:", _round_metric(train_f1_macro))  # Macro-averaged F1 on training set (equal weight per class)

if train_logloss_value is not None:
	print("Train Log Loss:", _round_metric(train_logloss_value))  # Cross-entropy loss on training set (probability confidence quality)

# ---- Test Metrics (model generalization to unseen data) ----
print("Test Accuracy:", _round_metric(test_accuracy))  # Overall proportion of correct predictions on unseen test data
print("Test Balanced Accuracy:", _round_metric(test_balanced_accuracy))  # Average recall across classes (handles class imbalance)
print("Test Precision Macro:", _round_metric(test_precision_macro))  # Macro-averaged precision (mean per-class precision)
print("Test Recall Macro:", _round_metric(test_recall_macro))  # Macro-averaged recall (mean per-class recall)
print("Test F1 Macro:", _round_metric(test_f1_macro))  # Macro-averaged F1 score (harmonic mean of precision and recall per class)

print("Test Support Total:", support_total)  # Total number of true test samples used for evaluation
print("Test Support By Class:", support_by_class)  # True sample count per class (class distribution insight)

if test_roc_auc_macro_ovr is not None:
	print("Test ROC AUC Macro OVR:", _round_metric(test_roc_auc_macro_ovr))  # One-vs-Rest macro ROC-AUC (probability ranking quality)

if test_pr_auc_macro_ovr is not None:
	print("Test PR AUC Macro OVR:", _round_metric(test_pr_auc_macro_ovr))  # One-vs-Rest macro Precision-Recall AUC (imbalance-sensitive metric)

if test_logloss_value is not None:
	print("Test Log Loss:", _round_metric(test_logloss_value))  # Cross-entropy loss on test set (penalizes confident wrong predictions)

if brier_score is not None:
	print("Test Brier Score:", _round_metric(brier_score))  # Mean squared error of predicted probabilities (calibration metric)

if model_selection["enabled"]:
	print("Model Selection Best Trial:", model_selection["best_trial_index"])
	print("Model Selection Best Params:", model_selection["best_params"])
	print("Model Selection Best Score:", model_selection["best_score"])

print("First 5 predictions:", predictions[:5])  # Sample predictions for quick sanity check of output classes

# =============================================================
# ==================== MODEL CODE ENDS HERE ===================
# =============================================================
# End of model logic.

# Artifact export and registry logging.
if SAVE_MODEL:
	model_name = args.name.strip() or Path(__file__).stem
	model_root_dir = project_root / "artifacts" / "models" / model_name
	timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
	run_id = str(uuid.uuid4())
	data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
	data_rows = int(len(df))
	data_columns = int(df.shape[1])
	run_dir = model_root_dir / f"{timestamp}_{model_name}"

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
			"n_train": int(len(X_train)),
			"n_val": int(len(X_valid)) if model_selection["enabled"] else 0,
			"n_test": int(len(X_test)),
		},
		"primary_metric": {
			"name": "f1_macro" if args.task == "multiclass_classification" else "roc_auc",
			"split": "test",
			"direction": "max",
			"value": _round_metric(test_f1_macro) if args.task == "multiclass_classification" else _round_metric(test_roc_auc_macro_ovr),
		},
		"probabilities": {
			"source": "predict_proba" if hasattr(model, "predict_proba") else None,
			"calibrated": False if hasattr(model, "predict_proba") else None,
			"calibration_method": None,
		},
		"model_selection": model_selection,
	}
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
		if args.task == "binary_classification" and roc_curve_points is not None:
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
	inference_script = f'''import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

sample_rows = {json.dumps(inference_rows, indent=2)}
expected_y = {json.dumps(expected_values, indent=2)}

with MODEL_PATH.open("rb") as model_file:
	model = pickle.load(model_file)

features = pd.DataFrame(sample_rows)
predictions = model.predict(features)

print("Inference example (5 rows from X_test)")
print("Predictions:", predictions.tolist())
print("Expected y:", expected_y)

results = features.copy()
results["y_expected"] = expected_y
results["y_pred"] = predictions
print(results)
'''
	with (inference_dir / "inference_example.py").open("w", encoding="utf-8") as inference_file:
		inference_file.write(inference_script)

	feature_schema = {
		"target": "diagnosis",
		"feature_columns": X.columns.tolist(),
		"categorical_columns": categorical_cols,
		"numerical_columns": numerical_cols,
		"dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
	}
	with (data_dir / "feature_schema.json").open("w", encoding="utf-8") as schema_file:
		json.dump(feature_schema, schema_file, indent=2)

	post_transform_feature_count = _post_transform_feature_count(model.named_steps["preprocess"], X_train.iloc[:1])
	n_val = int(len(X_valid)) if model_selection["enabled"] else 0
	n_train_effective = int(len(X_inner_train)) if model_selection["enabled"] else int(len(X_train))
	classifier_params = model.named_steps["classifier"].get_params()
	estimator_params_compact = _select_estimator_params(
		classifier_params,
		[
			"n_estimators",
			"criterion",
			"max_depth",
			"max_features",
			"min_samples_split",
			"min_samples_leaf",
			"class_weight",
			"bootstrap",
			"max_samples",
			"random_state",
			"n_jobs",
		],
	)

	run_metadata = {
		"schema_version": RUN_SCHEMA_VERSION,
		"metadata_profile": RUN_METADATA_PROFILE,
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": args.library,
		"task": args.task,
		"algorithm": "random_forest",
		"estimator_class": "RandomForestClassifier",
		"model_id": "sklearn.randomforestclassifier",
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
				"enabled": bool(model_selection["enabled"]),
				"strategy": "explicit_split" if model_selection["enabled"] else None,
				"validation_fraction": float(args.validation_fraction) if model_selection["enabled"] else None,
				"random_state": int(args.random_state) if model_selection["enabled"] else None,
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
			"imputer": None,
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
		"model_selection": model_selection,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": classifier_params.get("n_jobs"),
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
				"model_selection_enabled": bool(model_selection["enabled"]),
				"model_selection_best_score": float(model_selection["best_score"]) if model_selection["best_score"] is not None else None,
				"model_selection_best_trial_index": int(model_selection["best_trial_index"]) if model_selection["best_trial_index"] is not None else None,
				"model_selection_best_n_estimators": int(model_selection["best_params"]["n_estimators"]) if model_selection["best_params"] is not None else None,
				"model_selection_stopped_early": bool(model_selection["stopped_early"]),
				"accuracy": float(test_accuracy),
				"balanced_accuracy": float(test_balanced_accuracy),
				"precision_macro": float(test_precision_macro),
				"recall_macro": float(test_recall_macro),
				"f1_macro": float(test_f1_macro),
				"support": support_total,
				"roc_auc_macro_ovr": float(test_roc_auc_macro_ovr) if test_roc_auc_macro_ovr is not None else None,
				"pr_auc_macro_ovr": float(test_pr_auc_macro_ovr) if test_pr_auc_macro_ovr is not None else None,
				"log_loss": float(test_logloss_value) if test_logloss_value is not None else None,
				"brier_score": float(brier_score) if brier_score is not None else None,
				"train_accuracy": float(train_accuracy),
				"train_f1_macro": float(train_f1_macro),
				"train_log_loss": float(train_logloss_value) if train_logloss_value is not None else None,
				"n_train": int(len(X_train)),
				"n_test": int(len(X_test)),
			}
		]
	)
	registry_df = pd.concat([registry_df, registry_row], ignore_index=True)
	registry_df.to_csv(registry_path, index=False)

	print(f"Artifacts exported to: {run_dir}")
