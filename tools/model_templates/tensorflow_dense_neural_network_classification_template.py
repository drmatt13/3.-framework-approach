import argparse
import hashlib
import json
from functools import partial
import pickle
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_recall_fscore_support,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

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
	json_safe as _json_safe,
	parse_bool_flag as _parse_bool,
	post_transform_feature_count as _post_transform_feature_count,
	round_metric as _round_metric_base,
	to_dense_float32 as _to_dense_float32,
	validate_etl_outputs as _validate_etl_outputs,
	write_model_schemas as _write_model_schemas,
)

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (common usage)
#   --task {{TASK_VALUE}}
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
#   --test-size <float>
#   --optimizer adam|sgd|rmsprop|adagrad|adamw
#   --learning-rate <float>
#   --epochs <int>
#   --batch-size <int>
#   --early-stopping true|false
#   --validation-fraction <float>
#   --n-iter-no-change <int>
#   --verbose 0|1|2|auto
#   --metric-decimals <int>
# ---------------------------------------------------------------------

# # Default values for optional parameters. These can be overridden via CLI.
SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1
DEFAULT_OPTIMIZER_NAME = "{{OPTIMIZER_NAME}}"
DEFAULT_LEARNING_RATE = float("{{LEARNING_RATE}}")
DEFAULT_EPOCHS = int("{{EPOCHS}}")
DEFAULT_BATCH_SIZE = int("{{BATCH_SIZE}}")
DEFAULT_EARLY_STOPPING = "{{EARLY_STOPPING_DEFAULT}}" == "True"
DEFAULT_VALIDATION_FRACTION = float("{{VALIDATION_FRACTION_DEFAULT}}")
DEFAULT_N_ITER_NO_CHANGE = int("{{N_ITER_NO_CHANGE_DEFAULT}}")
DEFAULT_VERBOSE = "1"
DEFAULT_METRIC_DECIMALS = 4

parser = argparse.ArgumentParser(description="TensorFlow Dense Neural Network Classifier baseline")
parser.add_argument("--task", choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"], default=DEFAULT_OPTIMIZER_NAME)
parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
parser.add_argument("--early-stopping", type=_parse_bool, default=DEFAULT_EARLY_STOPPING)
parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
parser.add_argument("--n-iter-no-change", type=int, default=DEFAULT_N_ITER_NO_CHANGE)
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default=DEFAULT_VERBOSE)
parser.add_argument("--metric-decimals", type=int, default=DEFAULT_METRIC_DECIMALS)
args = parser.parse_args()
SAVE_MODEL = args.save_model
training_verbose = "auto" if args.verbose == "auto" else int(args.verbose)
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

# ---------------------------------------------------------
# Target normalization (classification-safe integer labels)
# ---------------------------------------------------------

# - If target is non-numeric, map categories to stable integer class IDs.
# - If target is numeric, coerce to numeric and drop non-convertible rows.
if not pd.api.types.is_numeric_dtype(y):
	y = y.astype("category").cat.codes.astype("int64")
else:
	y = pd.to_numeric(y, errors="coerce")
	valid_target_mask = y.notna()
	X = X.loc[valid_target_mask].copy()
	y = y.loc[valid_target_mask].astype("int64")
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
preprocessor = ColumnTransformer(
	transformers=[
		("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_cols),
		("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder)]), categorical_cols),
	],
	remainder="drop",
)

# Transform features to dense float32 arrays for TensorFlow compatibility.
y_train_array = np.asarray(y_train, dtype=np.int64)
y_test_array = np.asarray(y_test, dtype=np.int64)

X_inner_train_processed = None
X_valid_processed = None
y_inner_train_array = y_train_array
y_valid_array = None
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
	X_inner_train_processed = _to_dense_float32(preprocessor.fit_transform(X_inner_train))
	X_valid_processed = _to_dense_float32(preprocessor.transform(X_valid))
	X_train_processed = _to_dense_float32(preprocessor.transform(X_train))
	y_inner_train_array = np.asarray(y_inner_train, dtype=np.int64)
	y_valid_array = np.asarray(y_valid, dtype=np.int64)
	n_train_effective = int(len(X_inner_train))
	n_val = int(len(X_valid))
else:
	X_train_processed = _to_dense_float32(preprocessor.fit_transform(X_train))

X_test_processed = _to_dense_float32(preprocessor.transform(X_test))

# Derive class metadata from training labels.
classes = np.sort(np.unique(y_train_array))
num_classes = int(len(classes))
if num_classes < 2:
	raise ValueError("Need at least 2 classes in the training split.")
is_binary = num_classes == 2

output_units = 1 if is_binary else num_classes
output_activation = "sigmoid" if is_binary else "softmax"
loss_fn = "binary_crossentropy" if is_binary else "sparse_categorical_crossentropy"
generated_defaults = {"output_units": int("{{OUTPUT_UNITS}}"), "output_activation": "{{OUTPUT_ACTIVATION}}", "loss_fn": "{{LOSS_FN}}"}

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(int(args.random_state))

# Create optimizer with specified learning rate
optimizer = {{OPTIMIZER_CTOR}}(learning_rate=float(args.learning_rate))

# ---------------------------------------------------------------------
# TENSORFLOW DENSE MODEL TUNING BLOCK
# Edit this tf.keras.Sequential block to tune layer widths/depth,
# activations, and regularization for your dataset.
#
# Regularization examples (uncomment to enable):
#   tf.keras.layers.Dense(
#       128,
#       activation="relu",
#       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
#   )
#   tf.keras.layers.Dense(
#       64,
#       activation="relu",
#       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#   )
#   tf.keras.layers.Dropout(0.2)
# ---------------------------------------------------------------------

# Adjust the architecture and hyperparameters of the model as needed for your dataset/task.
keras_model = tf.keras.Sequential(
	[
		tf.keras.layers.Input(shape=(int(X_train_processed.shape[1]),)),
		tf.keras.layers.Dense(128, activation="relu"),
		tf.keras.layers.Dense(64, activation="relu"),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(32, activation="relu"),
		tf.keras.layers.Dense(output_units, activation=output_activation),
	]
)

# Compile the model with the specified optimizer, loss function, and metrics.
keras_model.compile(
	optimizer=optimizer,
	loss=loss_fn,
	metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")] if is_binary else [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

# =============================================================
# ===================== TRAIN MODEL ===========================
# =============================================================

# Fit on training data and capture training history/timing.
callbacks = []
fit_features = X_train_processed
fit_targets = y_train_array
validation_data = None
if args.early_stopping:
	early_stopping_callback = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss",
		patience=int(args.n_iter_no_change),
		mode="min",
		restore_best_weights=True,
	)
	callbacks.append(early_stopping_callback)
	fit_features = X_inner_train_processed
	fit_targets = y_inner_train_array
	validation_data = (X_valid_processed, y_valid_array)

fit_started_at = time.perf_counter()
history = keras_model.fit(
	fit_features,
	fit_targets,
	epochs=int(args.epochs),
	batch_size=int(args.batch_size),
	callbacks=callbacks,
	validation_data=validation_data,
	verbose=training_verbose,
)
fit_time_seconds = float(time.perf_counter() - fit_started_at)

history_epochs = int(len(history.history.get("loss", [])))
if history_epochs <= 0:
	history_epochs = int(args.epochs)

best_step = None
best_score_raw = None
if args.early_stopping:
	validation_loss_history = history.history.get("val_loss", [])
	if len(validation_loss_history) > 0:
		best_index = int(np.argmin(validation_loss_history))
		best_step = best_index + 1
		best_score_raw = float(validation_loss_history[best_index])

training_control = {
	"enabled": bool(args.early_stopping),
	"type": "epochs",
	"max_steps_configured": int(args.epochs),
	"steps_completed": history_epochs,
	"patience": int(args.n_iter_no_change),
	"monitor_metric": "val_loss" if args.early_stopping else None,
	"monitor_split": "val" if args.early_stopping else None,
	"monitor_direction": "min" if args.early_stopping else None,
	"best_step": int(best_step) if best_step is not None else None,
	"best_score": _round_metric(best_score_raw),
	"stopped_early": bool(args.early_stopping and history_epochs < int(args.epochs)),
}

# =============================================================
# ==================== EVALUATE MODEL =========================
# =============================================================

# Evaluate model on train/test splits.
# Predict on train/test splits and measure inference time.
predict_started_at = time.perf_counter()
train_raw = np.asarray(keras_model.predict(X_train_processed, verbose=training_verbose))
test_raw = np.asarray(keras_model.predict(X_test_processed, verbose=training_verbose))
predict_time_seconds = float(time.perf_counter() - predict_started_at)

# Convert model outputs to probabilities and class predictions.
if is_binary:
	train_pos = train_raw.reshape(-1)
	test_pos = test_raw.reshape(-1)
	train_probabilities = np.column_stack([1.0 - train_pos, train_pos])
	probabilities = np.column_stack([1.0 - test_pos, test_pos])
	train_predictions = (train_pos >= 0.5).astype(int)
	predictions = (test_pos >= 0.5).astype(int)
else:
	train_probabilities = train_raw
	probabilities = test_raw
	train_predictions = np.argmax(train_probabilities, axis=1).astype(int)
	predictions = np.argmax(probabilities, axis=1).astype(int)

# Compute core classification metrics.
train_accuracy = accuracy_score(y_train_array, train_predictions)
train_f1_macro = f1_score(y_train_array, train_predictions, average="macro", zero_division=0)
test_accuracy = accuracy_score(y_test_array, predictions)
test_balanced_accuracy = balanced_accuracy_score(y_test_array, predictions)
test_precision_macro = precision_score(y_test_array, predictions, average="macro", zero_division=0)
test_recall_macro = recall_score(y_test_array, predictions, average="macro", zero_division=0)
test_f1_macro = f1_score(y_test_array, predictions, average="macro", zero_division=0)
train_log_loss = float(log_loss(y_train_array, train_probabilities, labels=classes))
test_log_loss = float(log_loss(y_test_array, probabilities, labels=classes))
test_cm = confusion_matrix(y_test_array, predictions, labels=classes)
_, _, _, support_values = precision_recall_fscore_support(y_test_array, predictions, labels=classes, zero_division=0)
support_by_class = {str(k): int(v) for k, v in zip(classes.tolist(), support_values.tolist())}
support_total = int(len(y_test_array))

# Compute probability-based diagnostics (binary vs multiclass handling).
roc_auc_macro_ovr = None
pr_auc_macro_ovr = None
brier_score_value = None
roc_curve_points = None
y_test_binarized = None
if is_binary:
	pos_probs = probabilities[:, -1]
	roc_auc_macro_ovr = float(roc_auc_score(y_test_array, pos_probs))
	pr_auc_macro_ovr = float(average_precision_score(y_test_array, pos_probs))
	brier_score_value = float(brier_score_loss(y_test_array, pos_probs))
	fpr, tpr, thresholds = roc_curve(y_test_array, pos_probs)
	roc_curve_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
else:
	y_test_binarized = label_binarize(y_test_array, classes=classes)
	roc_auc_macro_ovr = float(roc_auc_score(y_test_binarized, probabilities, multi_class="ovr", average="macro"))
	pr_auc_macro_ovr = float(average_precision_score(y_test_binarized, probabilities, average="macro"))
	brier_score_value = float(((probabilities - y_test_binarized) ** 2).sum(axis=1).mean())

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model fit on data it learned from) ----
print("Train Accuracy:", _round_metric(train_accuracy))  # Proportion of correctly classified training samples
print("Train F1 Macro:", _round_metric(train_f1_macro))  # Macro-averaged harmonic mean of precision and recall on training set
print("Train Log Loss:", _round_metric(train_log_loss))  # Cross-entropy loss on training probabilities

# ---- Test Metrics (model performance on unseen data) ----
print("Test Accuracy:", _round_metric(test_accuracy))  # Overall classification accuracy on test data
print("Test Balanced Accuracy:", _round_metric(test_balanced_accuracy))  # Mean recall across classes (robust to class imbalance)
print("Test Precision Macro:", _round_metric(test_precision_macro))  # Macro-averaged precision across all classes
print("Test Recall Macro:", _round_metric(test_recall_macro))  # Macro-averaged recall across all classes
print("Test F1 Macro:", _round_metric(test_f1_macro))  # Macro-averaged F1 score (balanced precision/recall measure)
print("Test ROC AUC Macro OVR:", _round_metric(roc_auc_macro_ovr))  # One-vs-rest ROC-AUC macro average (ranking quality across classes)
print("Test PR AUC Macro OVR:", _round_metric(pr_auc_macro_ovr))  # One-vs-rest precision-recall AUC macro average (better for imbalance)
print("Test Log Loss:", _round_metric(test_log_loss))  # Cross-entropy loss on test probabilities (penalizes overconfident errors)
print("Test Brier Score:", _round_metric(brier_score_value))  # Probability calibration error (lower is better)

# ---- Training Control (early stopping / step tracking) ----
if training_control["enabled"]:
	print("Training Control Best Step:", training_control["best_step"])  # Iteration/epoch with best validation score
	print("Training Control Steps Completed:", training_control["steps_completed"])  # Total training iterations completed
	print("Training Control Best Score:", training_control["best_score"])  # Best validation score achieved during training

# ---- Sanity Checks ----
print("First 5 predictions:", predictions[:5].tolist())  # Sample predicted classes
print("First 5 true values:", y_test.iloc[:5].tolist())  # Corresponding ground-truth classes

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

	# Save TensorFlow model and fitted preprocessor.
	keras_model.save(model_dir / "model.keras")
	with (preprocess_dir / "preprocessor.pkl").open("wb") as f:
		pickle.dump(preprocessor, f)

	metrics = {
		"train": {"accuracy": _round_metric(train_accuracy), "f1_macro": _round_metric(train_f1_macro), "log_loss": _round_metric(train_log_loss)},
		"test": {
			"accuracy": _round_metric(test_accuracy),
			"balanced_accuracy": _round_metric(test_balanced_accuracy),
			"precision_macro": _round_metric(test_precision_macro),
			"recall_macro": _round_metric(test_recall_macro),
			"f1_macro": _round_metric(test_f1_macro),
			"roc_auc_macro_ovr": _round_metric(roc_auc_macro_ovr),
			"pr_auc_macro_ovr": _round_metric(pr_auc_macro_ovr),
			"log_loss": _round_metric(test_log_loss),
			"brier_score": _round_metric(brier_score_value),
			"support_total": support_total,
			"support_by_class": support_by_class,
		},
		"data_sizes": {
			"n_train": int(n_train_effective),
			"n_val": int(n_val),
			"n_test": int(len(X_test)),
		},
		"primary_metric": {
			"name": "roc_auc" if is_binary else "f1_macro",
			"split": "test",
			"direction": "max",
			"value": _round_metric(roc_auc_macro_ovr) if is_binary else _round_metric(test_f1_macro),
		},
		"probabilities": {
			"source": "predict",
			"calibrated": False,
			"calibration_method": None,
		},
		"training_control": training_control,
		"model_selection": training_control,
		"timing": {"fit_seconds": _round_metric(fit_time_seconds), "predict_seconds": _round_metric(predict_time_seconds)},
	}
	metrics["selection"] = training_control
	metrics["calibration"] = metrics.get("probabilities")
	rounded_history = {
		k: [_round_metric(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for v in values]
		for k, values in history.history.items()
	}
	(eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
	(eval_dir / "training_history.json").write_text(json.dumps(_json_safe(rounded_history), indent=2), encoding="utf-8")
	pd.DataFrame(test_cm, index=classes, columns=classes).to_csv(eval_dir / "confusion_matrix.csv")
	pd.DataFrame({"y_true": y_test.iloc[:50].tolist(), "y_pred": pd.Series(predictions[:50]).tolist()}).to_csv(
		eval_dir / "predictions_preview.csv", index=False
	)

	cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
	ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=classes).plot(ax=cm_ax, cmap="Blues", colorbar=False)
	cm_ax.set_title("Confusion Matrix")
	cm_fig.tight_layout()
	cm_fig.savefig(eval_dir / "confusion_matrix.png", dpi=150)
	plt.close(cm_fig)

	if roc_curve_points is not None:
		roc_curve_points.to_csv(eval_dir / "roc_curve.csv", index=False)
	if roc_auc_macro_ovr is not None:
		roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
		if is_binary and roc_curve_points is not None:
			roc_ax.plot(roc_curve_points["fpr"], roc_curve_points["tpr"], label=f"ROC AUC = {roc_auc_macro_ovr:.4f}")
		else:
			for i, class_label in enumerate(classes):
				fpr_i, tpr_i, _ = roc_curve(y_test_binarized[:, i], probabilities[:, i])
				roc_ax.plot(fpr_i, tpr_i, label=f"Class {class_label}")
			roc_ax.text(0.58, 0.1, f"Macro OVR AUC = {roc_auc_macro_ovr:.4f}")
		roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
		roc_ax.set_xlabel("False Positive Rate")
		roc_ax.set_ylabel("True Positive Rate")
		roc_ax.set_title("ROC Curve")
		roc_ax.legend(loc="lower right")
		roc_fig.tight_layout()
		roc_fig.savefig(eval_dir / "roc_curve.png", dpi=150)
		plt.close(roc_fig)

	inference_rows = X_test.iloc[:5].to_dict(orient="records")
	expected_values = y_test.iloc[:5].tolist()
	class_labels = [str(label) for label in classes.tolist()]
	sample_rows_literal = json.dumps(inference_rows, indent=2)
	sample_rows_literal = sample_rows_literal.replace(": NaN", ": np.nan").replace(": Infinity", ": np.inf").replace(": -Infinity", ": -np.inf")
	inference_verbose_literal = '"auto"' if args.verbose == "auto" else str(int(args.verbose))
	inference_script = f'''import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.keras"
PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / "preprocess" / "preprocessor.pkl"
sample_rows = {sample_rows_literal}
expected_y = {json.dumps(expected_values, indent=2)}
class_labels = {json.dumps(class_labels, indent=2)}

with PREPROCESSOR_PATH.open("rb") as f:
	preprocessor = pickle.load(f)
model = tf.keras.models.load_model(MODEL_PATH)
features = pd.DataFrame(sample_rows)
X = preprocessor.transform(features)
if hasattr(X, "toarray"):
	X = X.toarray()
X = np.asarray(X, dtype=np.float32)
raw = np.asarray(model.predict(X, verbose={inference_verbose_literal}))
if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
	pos = raw.reshape(-1)
	probabilities = np.column_stack([1.0 - pos, pos])
	predictions = (pos >= 0.5).astype(int)
else:
	probabilities = raw
	predictions = np.argmax(probabilities, axis=1).astype(int)
print("Inference Example")
print("Predictions:", predictions.tolist())
print("Expected:", expected_y)
print("Class Labels:", class_labels)
print("Probabilities:", probabilities.tolist())
'''
	(inference_dir / "inference_example.py").write_text(inference_script, encoding="utf-8")

	schema_artifacts = _write_model_schemas(
		schema_dir=data_dir,
		X_raw=X,
		y_model=y,
		target_column_name=target_column_name,
		transformed_features=X_train_processed[:1],
		preprocessor=preprocessor,
		y_original=y_original,
	)

	run_metadata = {
		"run_id": run_id,
		"name": model_name,
		"timestamp": timestamp,
		"library": "tensorflow",
		"task": args.task,
		"algorithm": "dense_neural_network",
		"estimator_class": "tf.keras.Sequential",
		"model_id": "tensorflow.keras.sequential.dense_nn.classification",
		"dataset": {"path": str(data_path.relative_to(_project_root())), "sha256": data_hash, "rows": int(len(df)), "columns": int(df.shape[1])},
		"data_split": {
			"strategy": "train_test_split",
			"test_size": float(args.test_size),
			"random_state": int(args.random_state),
			"stratify": True,
			"validation": {
				"enabled": bool(training_control["enabled"]),
				"strategy": "explicit_split" if training_control["enabled"] else None,
				"validation_fraction": float(args.validation_fraction) if training_control["enabled"] else None,
				"random_state": int(args.random_state) if training_control["enabled"] else None,
			},
			"sizes": {
				"n_rows": int(len(df)),
				"n_train": int(n_train_effective),
				"n_val": int(n_val),
				"n_test": int(len(X_test)),
			},
		},
		"params": {
			"optimizer": args.optimizer,
			"learning_rate": float(args.learning_rate),
			"epochs": int(args.epochs),
			"batch_size": int(args.batch_size),
			"early_stopping": bool(args.early_stopping),
			"validation_fraction": float(args.validation_fraction),
			"n_iter_no_change": int(args.n_iter_no_change),
			"random_state": int(args.random_state),
			"hidden_layers": [128, 64, 32],
			"output_units": output_units,
			"output_activation": output_activation,
			"loss_fn": loss_fn,
			"generator_defaults": generated_defaults,
		},
		"preprocessing": {"feature_count": {"raw": int(X.shape[1]), "post_transform": _post_transform_feature_count(preprocessor, X_train.iloc[:1])}},
		"selection": training_control,
		"training_control": training_control,
		"model_selection": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
		},
		"artifacts": _artifact_map(
			run_dir,
			{
				"model": model_dir / "model.keras",
				"preprocess": preprocess_dir / "preprocessor.pkl",
				"metrics": eval_dir / "metrics.json",
				"history": eval_dir / "training_history.json",
				"confusion_matrix": eval_dir / "confusion_matrix.csv",
				"confusion_matrix_plot": eval_dir / "confusion_matrix.png",
				"roc_curve": eval_dir / "roc_curve.csv",
				"roc_curve_plot": eval_dir / "roc_curve.png",
				"predictions_preview": eval_dir / "predictions_preview.csv",
				**schema_artifacts,
				"inference_example": inference_dir / "inference_example.py",
			},
		),
		"versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "tensorflow": tf.__version__},
	}
	(run_dir / "run.json").write_text(json.dumps(_compact_metadata(_json_safe(run_metadata)), indent=2), encoding="utf-8")

	registry_path = model_root_dir / "model_registry.csv"
	registry_df = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
	next_id = int(registry_df["model_id"].max()) + 1 if ("model_id" in registry_df.columns and not registry_df.empty) else 1
	registry_row = pd.DataFrame(
		[{
			"model_id": next_id,
			"run_id": run_id,
			"name": model_name,
			"timestamp": timestamp,
			"optimizer": args.optimizer,
			"learning_rate": float(args.learning_rate),
			"epochs": int(args.epochs),
			"batch_size": int(args.batch_size),
			"random_state": int(args.random_state),
			"training_control_enabled": bool(training_control["enabled"]),
			"training_control_best_score": float(training_control["best_score"]) if training_control["best_score"] is not None else None,
			"training_control_best_step": int(training_control["best_step"]) if training_control["best_step"] is not None else None,
			"training_control_steps_completed": int(training_control["steps_completed"]) if training_control["steps_completed"] is not None else None,
			"training_control_stopped_early": bool(training_control["stopped_early"]),
			"model_selection_enabled": bool(training_control["enabled"]),
			"model_selection_best_score": float(training_control["best_score"]) if training_control["best_score"] is not None else None,
			"model_selection_best_step": int(training_control["best_step"]) if training_control["best_step"] is not None else None,
			"model_selection_steps_completed": int(training_control["steps_completed"]) if training_control["steps_completed"] is not None else None,
			"model_selection_stopped_early": bool(training_control["stopped_early"]),
			"num_class": int(num_classes),
			"accuracy": _round_metric(test_accuracy),
			"balanced_accuracy": _round_metric(test_balanced_accuracy),
			"precision_macro": _round_metric(test_precision_macro),
			"recall_macro": _round_metric(test_recall_macro),
			"f1_macro": _round_metric(test_f1_macro),
			"roc_auc_macro_ovr": _round_metric(roc_auc_macro_ovr) if roc_auc_macro_ovr is not None else None,
			"pr_auc_macro_ovr": _round_metric(pr_auc_macro_ovr) if pr_auc_macro_ovr is not None else None,
			"log_loss": _round_metric(test_log_loss),
			"brier_score": _round_metric(brier_score_value) if brier_score_value is not None else None,
			"support": support_total,
			"n_train": int(len(X_train)),
			"n_test": int(len(X_test)),
		}]
	)
	pd.concat([registry_df, registry_row], ignore_index=True).to_csv(registry_path, index=False)
	print(f"Artifacts exported to: {run_dir}")

