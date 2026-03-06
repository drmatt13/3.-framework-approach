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

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

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
	to_dense_float32 as _to_dense_float32,
	validate_etl_outputs as _validate_etl_outputs,
	write_unified_registry_sqlite as _write_unified_registry_sqlite,
	write_model_schemas as _write_model_schemas,
)
from libraries.preprocessing_utils import build_tabular_preprocessor as _build_preprocessor, normalize_string_columns as _normalize_string_columns
from libraries.tensorflow_template_utils import (
	build_dense_regressor as _build_dense_regressor,
	build_direct_fit_dense_config as _build_direct_fit_dense_config,
	build_optimizer as _build_optimizer,
	summarize_dense_model_metadata as _summarize_dense_model_metadata,
)
from libraries.tensorflow_search_space import DenseNNSearchGridConfig, build_dense_nn_search_candidates as _build_dense_nn_search_candidates

# =============================================================
# =============== CONFIGURATION / CLI FLAGS ===================
# =============================================================

# ---------------------------------------------------------------------
# Supported CLI flags (TensorFlow Regression)
#
# Core run options (auto-configured for all ML models generated)
#   --name <model_name>                             (model name used for registry and artifact folder; default: script filename)
#   --artifact-name-mode full|short                 (full = timestamp + UUID for unique runs; short = readable name but may overwrite previous runs)
#   --save-model true|false                         (save trained model and artifacts; false logs metrics only)
#   --verbose 0|1|2|auto                            (0=silent, 1=training progress, 2=training + tuning progress, auto=adaptive verbosity)
#   --metric-decimals <int>                         (decimal precision for logged metrics and artifacts)
#
# Reproducibility + data split
#   --random-state <int>                            (random seed for reproducibility)
#   --test-size <float>                             (test set fraction; e.g., 0.2 = 80/20 split)
#
# Hyperparameter tuning configuration
#   --enable-tuning true|false                  		(enable hyperparameter tuning with cross-validation)
# 
# Model configuration (direct-fit)              (used when --enable-tuning=false)
#   --optimizer adam|sgd|rmsprop|										(optimizer used during training)
# 							adagrad|adamw									 			^
#   --learning-rate <float>                         (optimizer learning rate)
#   --epochs <int>                                  (maximum number of training epochs)
#   --batch-size <int>                              (number of samples processed per gradient update)
#
# Training path
#   --early-stopping true|false                     (enable validation-based early stopping)
#   --validation-fraction <float>                   (fraction of training data reserved for validation)
#   --n-iter-no-change <int>                        (stop training if validation metric does not improve for N epochs)
#
# Hyperparameter tuning                         (used when --enable-tuning=true)
#   --tuning-method grid|random                     (grid = exhaustive search over candidates; random = random subset)
#   --cv-scoring rmse                               (metric used to select best model during tuning)
#   --cv-n-iter <int>                               (number of random search iterations)
#   --tuning-optimizer auto|adam|sgd|rmsprop|adagrad|adamw
#   --tuning-activation auto|relu|gelu|tanh
#   --tuning-regularization auto|none|l1|l2|l1_l2
# ---------------------------------------------------------------------

# Command-line argument parsing.
parser = argparse.ArgumentParser(description="TensorFlow Dense Neural Network Regressor baseline")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--artifact-name-mode", choices=["full", "short"], default="full")
parser.add_argument("--save-model", type=_parse_bool, default=False)
parser.add_argument("--random-state", type=int, default=1)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--optimizer", choices=["auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"], default="{{OPTIMIZER_NAME}}")
parser.add_argument("--learning-rate", type=float, default=float("{{LEARNING_RATE}}"))
parser.add_argument("--epochs", type=int, default=int("{{EPOCHS}}"))
parser.add_argument("--batch-size", type=int, default=int("{{BATCH_SIZE}}"))
parser.add_argument("--early-stopping", type=_parse_bool, default="{{EARLY_STOPPING_DEFAULT}}" == "True")
parser.add_argument("--validation-fraction", type=float, default=float("{{VALIDATION_FRACTION_DEFAULT}}"))
parser.add_argument("--n-iter-no-change", type=int, default=int("{{N_ITER_NO_CHANGE_DEFAULT}}"))
parser.add_argument("--verbose", choices=["0", "1", "2", "auto"], default="1")
parser.add_argument("--metric-decimals", type=int, default=4)
parser.add_argument("--enable-tuning", type=_parse_bool, default="{{TF_ENABLE_TUNING_DEFAULT}}" == "True")
parser.add_argument("--tuning-method", choices=["grid", "random"], default="{{TF_TUNING_METHOD_DEFAULT}}")
parser.add_argument("--cv-scoring", choices=["rmse"], default="{{TF_CV_SCORING_DEFAULT}}")
parser.add_argument("--cv-n-iter", type=int, default=int("{{TF_CV_N_ITER_DEFAULT}}"))
parser.add_argument("--tuning-optimizer", choices=["auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"], default="{{TF_TUNING_OPTIMIZER_DEFAULT}}")
parser.add_argument("--tuning-activation", choices=["auto", "relu", "gelu", "tanh"], default="{{TF_TUNING_ACTIVATION_DEFAULT}}")
parser.add_argument("--tuning-regularization", choices=["auto", "none", "l1", "l2", "l1_l2"], default="{{TF_TUNING_REGULARIZATION_DEFAULT}}")
args = parser.parse_args()

SAVE_MODEL = args.save_model
training_verbose = 1 if args.verbose == "auto" else int(args.verbose)
phase_logs_enabled = training_verbose > 0
keras_fit_verbose = 1 if training_verbose >= 2 else 0
trial_fit_verbose = 1 if training_verbose >= 2 else 0
METRIC_DECIMALS = int(args.metric_decimals)
_round_metric = partial(_round_metric_base, decimals=METRIC_DECIMALS)

# NOTE: Adjust these grids to customize search breadth for tuning.
DENSE_NN_SEARCH_GRID_CONFIG = DenseNNSearchGridConfig(
	hidden_units=[
		[64],
		[128],
		[128, 64],
		[256, 128],
		[256, 128, 64],
	],
	activation=["relu", "gelu", "tanh"],
	optimizer=["adam", "rmsprop", "sgd", "adagrad", "adamw"],
	learning_rate=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
	batch_size=[16, 32, 64, 128],
	dropout=[0.0, 0.1, 0.2, 0.3],
	l1=[0.0, 1e-5, 1e-4, 1e-3],
	l2=[0.0, 1e-5, 1e-4, 1e-3],
)

#  NOTE: This config is used for direct-fit runs when tuning is disabled.
DIRECT_FIT_CONFIG = _build_direct_fit_dense_config(
	optimizer_name=("adam" if args.optimizer == "auto" else args.optimizer),
	learning_rate=float(args.learning_rate),
	hidden_layers=[128, 64],
	activations=["relu", "relu"],
	dropouts=[0.0, 0.0],
	l1s=[0.0, 0.0],
	l2s=[0.0, 0.0],
	output_units=1,
	output_activation="linear",
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
y = pd.to_numeric(y.loc[valid_target_mask], errors="coerce")
y_original = y_original.loc[valid_target_mask].copy()

# ---------------------------------------------------------
# Target normalization (regression numeric target)
# ---------------------------------------------------------

# Coerce to numeric and drop rows that cannot be converted.
valid_target_mask = y.notna()
X = X.loc[valid_target_mask].copy()
y = y.loc[valid_target_mask].astype("float64")
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

# Transform features to dense float32 arrays for TensorFlow compatibility.
y_train_array = np.asarray(y_train, dtype=np.float32)
y_test_array = np.asarray(y_test, dtype=np.float32)

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
	)
	X_inner_train_processed = _to_dense_float32(preprocessor.fit_transform(X_inner_train))
	X_valid_processed = _to_dense_float32(preprocessor.transform(X_valid))
	X_train_processed = _to_dense_float32(preprocessor.transform(X_train))
	y_inner_train_array = np.asarray(y_inner_train, dtype=np.float32)
	y_valid_array = np.asarray(y_valid, dtype=np.float32)
	n_train_effective = int(len(X_inner_train))
	n_val = int(len(X_valid))
else:
	X_train_processed = _to_dense_float32(preprocessor.fit_transform(X_train))

X_test_processed = _to_dense_float32(preprocessor.transform(X_test))

# =============================================================
# ================= BUILD MODEL PIPELINE ======================
# =============================================================

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(int(args.random_state))

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
	if phase_logs_enabled:
		print(
			f"Training started with tuning: method={args.tuning_method}, "
			f"scoring={args.cv_scoring}, n_iter={args.cv_n_iter}"
		)
	rng = np.random.default_rng(int(args.random_state))
	X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
		X_train_processed,
		y_train_array,
		test_size=float(args.validation_fraction),
		random_state=int(args.random_state),
	)
	candidates = _build_dense_nn_search_candidates(
		config=DENSE_NN_SEARCH_GRID_CONFIG,
		optimizer=args.tuning_optimizer,
		activation=args.tuning_activation,
		regularization=args.tuning_regularization,
	)
	if len(candidates) == 0:
		raise ValueError("No TensorFlow tuning candidates remain after applying optimizer/activation/regularization filters.")
	if args.tuning_method == "random":
		trial_indices = rng.choice(len(candidates), size=min(int(args.cv_n_iter), len(candidates)), replace=False).tolist()
	else:
		trial_indices = list(range(len(candidates)))
	best_candidate = None
	best_candidate_score = np.inf
	for trial_index, index in enumerate(trial_indices, start=1):
		candidate = candidates[int(index)]
		if training_verbose >= 2:
			print(f"Tuning trial {trial_index}/{len(trial_indices)}: {candidate}")
		tf.keras.backend.clear_session()
		trial_model = _build_dense_regressor(
			input_dim=int(X_train_processed.shape[1]),
			optimizer_name=candidate["optimizer"],
			learning_rate=float(candidate["learning_rate"]),
			hidden_layers=list(candidate["hidden_layers"]),
			dropout=float(candidate["dropout"]),
			hidden_activation=str(candidate["activation"]),
			l1=float(candidate["l1"]),
			l2=float(candidate["l2"]),
		)
		trial_callbacks = [
			tf.keras.callbacks.EarlyStopping(
				monitor="val_loss",
				patience=max(2, int(args.n_iter_no_change)),
				mode="min",
				restore_best_weights=True,
			)
		]
		trial_model.fit(
			X_tune_train,
			y_tune_train,
			epochs=max(10, min(100, int(args.epochs))),
			batch_size=int(candidate["batch_size"]),
			validation_data=(X_tune_val, y_tune_val),
			callbacks=trial_callbacks,
			verbose=trial_fit_verbose,
		)
		val_predictions = np.asarray(trial_model(X_tune_val, training=False).numpy()).reshape(-1)
		score = float(root_mean_squared_error(y_tune_val, val_predictions))
		if score < best_candidate_score:
			best_candidate_score = score
			best_candidate = candidate
	if best_candidate is None:
		raise ValueError("Tuning did not find a valid TensorFlow candidate.")

	selected_optimizer = str(best_candidate["optimizer"])
	selected_learning_rate = float(best_candidate["learning_rate"])
	selected_hidden_layers = [int(v) for v in best_candidate["hidden_layers"]]
	selected_dropout = float(best_candidate["dropout"])
	selected_activation = str(best_candidate["activation"])
	selected_l1 = float(best_candidate["l1"])
	selected_l2 = float(best_candidate["l2"])
	args.batch_size = int(best_candidate["batch_size"])
	best_candidate_for_artifacts = {
		**best_candidate,
		"optimizer_when_auto": best_candidate.get("optimizer"),
		"activation_when_auto": best_candidate.get("activation"),
	}
	tuning_summary = {
		"enabled": True,
		"method": args.tuning_method,
		"cv_folds": None,
		"scoring": args.cv_scoring,
		"scoring_sklearn": args.cv_scoring,
		"n_iter": int(len(trial_indices)) if args.tuning_method == "random" else None,
		"n_candidates": int(len(trial_indices)),
		"best_score": _round_metric(best_candidate_score),
		"best_score_std": None,
		"best_params": _compact_metadata(_json_safe(best_candidate_for_artifacts)),
	}

if args.enable_tuning and tuning_summary["enabled"]:
	tf.keras.backend.clear_session()
	keras_model = _build_dense_regressor(
		input_dim=int(X_train_processed.shape[1]),
		optimizer_name=selected_optimizer,
		learning_rate=float(selected_learning_rate),
		hidden_layers=selected_hidden_layers,
		dropout=selected_dropout,
		hidden_activation=selected_activation,
		l1=float(selected_l1),
		l2=float(selected_l2),
	)
else:
	tf.keras.backend.clear_session()
	selected_optimizer = str(DIRECT_FIT_CONFIG["optimizer"])
	selected_learning_rate = float(DIRECT_FIT_CONFIG["learning_rate"])
	selected_hidden_layers = [int(v) for v in DIRECT_FIT_CONFIG["hidden_layers"]]
	selected_activations = [str(v) for v in DIRECT_FIT_CONFIG["activations"]]
	selected_dropouts = [float(v) for v in DIRECT_FIT_CONFIG["dropouts"]]
	selected_l1s = [float(v) for v in DIRECT_FIT_CONFIG["l1s"]]
	selected_l2s = [float(v) for v in DIRECT_FIT_CONFIG["l2s"]]
	selected_output_units = int(DIRECT_FIT_CONFIG["output_units"] if DIRECT_FIT_CONFIG["output_units"] is not None else 1)
	selected_output_activation = str(
		DIRECT_FIT_CONFIG["output_activation"] if DIRECT_FIT_CONFIG["output_activation"] is not None else "linear"
	)

	layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(int(X_train_processed.shape[1]),))]
	for units, activation, dropout_rate, l1_value, l2_value in zip(
		selected_hidden_layers,
		selected_activations,
		selected_dropouts,
		selected_l1s,
		selected_l2s,
	):
		layer_regularizer = None
		if float(l1_value) > 0.0 or float(l2_value) > 0.0:
			layer_regularizer = tf.keras.regularizers.l1_l2(l1=float(l1_value), l2=float(l2_value))
		layers.append(
			tf.keras.layers.Dense(
				int(units),
				activation=str(activation),
				kernel_regularizer=layer_regularizer,
			)
		)
		if float(dropout_rate) > 0.0:
			layers.append(tf.keras.layers.Dropout(float(dropout_rate)))

	layers.append(tf.keras.layers.Dense(selected_output_units, activation=selected_output_activation))
	keras_model = tf.keras.Sequential(layers)
	keras_model.compile(
		optimizer=_build_optimizer(selected_optimizer, float(selected_learning_rate)),
		loss="mse",
		metrics=[
			tf.keras.metrics.RootMeanSquaredError(name="rmse"),
			tf.keras.metrics.MeanAbsoluteError(name="mae"),
		],
	)

model_metadata = _summarize_dense_model_metadata(
	keras_model,
	fallback_optimizer_name=selected_optimizer,
	fallback_learning_rate=float(selected_learning_rate),
)
selected_optimizer = str(model_metadata["optimizer"])
selected_learning_rate = float(model_metadata["learning_rate"] if model_metadata["learning_rate"] is not None else selected_learning_rate)
selected_hidden_layers = [int(v) for v in model_metadata["hidden_layers"]]
selected_activation = str(model_metadata["activation"])
selected_dropout = float(model_metadata["dropout"])
selected_l1 = float(model_metadata["l1"])
selected_l2 = float(model_metadata["l2"])
selected_architecture = model_metadata["architecture"]

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
if phase_logs_enabled and not args.enable_tuning:
	print("Training started: TensorFlow Dense Regressor")
history = keras_model.fit(
	fit_features,
	fit_targets,
	epochs=int(args.epochs),
	batch_size=int(args.batch_size),
	callbacks=callbacks,
	validation_data=validation_data,
	verbose=keras_fit_verbose,
)
fit_time_seconds = float(time.perf_counter() - fit_started_at)
if phase_logs_enabled:
	print(f"Training completed in {fit_time_seconds:.3f}s: TensorFlow Dense Regressor")

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

if tuning_summary["enabled"]:
	training_control = {
		"enabled": True,
		"type": "search_cv",
		"max_steps_configured": int(tuning_summary["n_candidates"]),
		"steps_completed": int(tuning_summary["n_candidates"]),
		"patience": None,
		"monitor_metric": f"cv_{args.cv_scoring}",
		"monitor_split": "cv",
		"monitor_direction": "min",
		"best_step": None,
		"best_score": tuning_summary["best_score"],
		"stopped_early": False,
	}
else:
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
predict_started_at = time.perf_counter()
train_predictions = np.asarray(keras_model(X_train_processed, training=False).numpy()).reshape(-1)
predictions = np.asarray(keras_model(X_test_processed, training=False).numpy()).reshape(-1)
predict_time_seconds = float(time.perf_counter() - predict_started_at)

train_mse = mean_squared_error(y_train, train_predictions)
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = root_mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)
train_max_err = max_error(y_train, train_predictions)
train_residuals = y_train.to_numpy(dtype=np.float64) - train_predictions
train_residual_mean = float(np.mean(train_residuals))
train_residual_std = float(np.std(train_residuals, ddof=1)) if len(train_residuals) > 1 else 0.0

test_mse = mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_rmse = root_mean_squared_error(y_test, predictions)
test_r2 = r2_score(y_test, predictions)
test_max_err = max_error(y_test, predictions)

# =============================================================
# ============== MODEL METRICS / LOGGING ======================
# =============================================================

# ---- Train Metrics (model fit on data it learned from) ----
print("Train MSE:", _round_metric(train_mse))  # Mean Squared Error on training set (average squared residuals; penalizes large errors heavily)
print("Train MAE:", _round_metric(train_mae))  # Mean Absolute Error on training set (average absolute prediction error)
print("Train RMSE:", _round_metric(train_rmse))  # Root Mean Squared Error on training set (error in original target units)
print("Train R2:", _round_metric(train_r2))  # R² on training set (proportion of variance explained by model)
print("Train Max Error:", _round_metric(train_max_err))  # Largest single absolute prediction error on training set
print("Train Residual Mean:", _round_metric(train_residual_mean))  # Mean of residuals (should be ~0 for unbiased regression)
print("Train Residual Std:", _round_metric(train_residual_std))  # Standard deviation of residuals (spread of prediction errors)

# ---- Test Metrics (model performance on unseen data) ----
print("Test MSE:", _round_metric(test_mse))  # Mean Squared Error on test set (average squared prediction errors)
print("Test MAE:", _round_metric(test_mae))  # Mean Absolute Error on test set (average absolute difference from true values)
print("Test RMSE:", _round_metric(test_rmse))  # Root Mean Squared Error on test set (interpretable error in target units)
print("Test R2:", _round_metric(test_r2))  # R² score on test set (generalization performance)
print("Test Max Error:", _round_metric(test_max_err))  # Largest single absolute prediction error on test set (worst-case mistake)

# ---- Training Control (early stopping / step tracking) ----
if training_control["enabled"]:
	print("Training Control Best Step:", training_control["best_step"])  # Iteration/epoch with best validation score
	print("Training Control Steps Completed:", training_control["steps_completed"])  # Total training iterations completed
	print("Training Control Best Score:", training_control["best_score"])  # Best validation score achieved during training

# ---- Sanity Checks ----
print("First 5 predictions:", predictions[:5].tolist())  # Sample of predicted values for quick sanity check
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
		"train": {
			"mse": _round_metric(train_mse),
			"mae": _round_metric(train_mae),
			"rmse": _round_metric(train_rmse),
			"r2": _round_metric(train_r2),
			"max_error": _round_metric(train_max_err),
			"residual_mean": _round_metric(train_residual_mean),
			"residual_std": _round_metric(train_residual_std),
		},
		"test": {
			"mse": _round_metric(test_mse),
			"mae": _round_metric(test_mae),
			"rmse": _round_metric(test_rmse),
			"r2": _round_metric(test_r2),
			"max_error": _round_metric(test_max_err),
		},
		"target_summary": {
			"split": "train",
			"mean": _round_metric(float(y_train.mean())),
			"std": _round_metric(float(y_train.std())),
		},
		"data_sizes": {
			"n_train": int(n_train_effective),
			"n_val": int(n_val),
			"n_test": int(len(X_test)),
		},
		"primary_metric": {
			"name": "rmse",
			"split": "test",
			"direction": "min",
			"value": _round_metric(test_rmse),
		},
		"tuning": tuning_summary,
		"training_control": training_control,
		"timing": {"fit_seconds": _round_metric(fit_time_seconds), "predict_seconds": _round_metric(predict_time_seconds)},
	}
	metrics["calibration"] = {"source": None, "calibrated": None, "calibration_method": None}
	rounded_history = {
		k: [_round_metric(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for v in values]
		for k, values in history.history.items()
	}
	with (eval_dir / "metrics.json").open("w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	with (eval_dir / "training_history.json").open("w", encoding="utf-8") as f:
		json.dump(_json_safe(rounded_history), f, indent=2)
	pd.DataFrame({"y_true": y_test.iloc[:50].tolist(), "y_pred": predictions[:50].tolist()}).to_csv(
		eval_dir / "predictions_preview.csv", index=False
	)

	inference_rows = X_test.iloc[:5].to_dict(orient="records")
	expected_values = y_test.iloc[:5].tolist()
	sample_rows_literal = json.dumps(inference_rows, indent=2)
	sample_rows_literal = sample_rows_literal.replace(": NaN", ": np.nan").replace(": Infinity", ": np.inf").replace(": -Infinity", ": -np.inf")
	inference_verbose_literal = str(1 if training_verbose >= 2 else 0)
	inference_script = f'''import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.keras"
PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / "preprocess" / "preprocessor.pkl"
sample_rows = {sample_rows_literal}
expected_y = {json.dumps(expected_values, indent=2)}

with PREPROCESSOR_PATH.open("rb") as f:
	preprocessor = pickle.load(f)
model = tf.keras.models.load_model(MODEL_PATH)
features = pd.DataFrame(sample_rows)
X = preprocessor.transform(features)
if hasattr(X, "toarray"):
	X = X.toarray()
X = np.asarray(X, dtype=np.float32)
predictions = np.asarray(model(X, training=False).numpy()).reshape(-1)
print("Inference Example")
print("Predictions:", predictions.tolist())
print("Expected:", expected_y)
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
		"model_name": model_name,
		"timestamp": timestamp,
		"library": "tensorflow",
		"task": args.task,
		"algorithm": "dense_neural_network",
		"estimator_class": "tf.keras.Sequential",
		"model_id": "tensorflow.keras.sequential.dense_nn.regression",
		"dataset": {"path": str(data_path.relative_to(_project_root())), "sha256": data_hash, "rows": int(len(df)), "columns": int(df.shape[1])},
		"data_split": {
			"strategy": "train_test_split",
			"test_size": float(args.test_size),
			"random_state": int(args.random_state),
			"stratify": None,
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
			"optimizer": selected_optimizer,
			"learning_rate": float(selected_learning_rate),
			"activation": selected_activation,
			"l1": float(selected_l1),
			"l2": float(selected_l2),
			"epochs": int(args.epochs),
			"batch_size": int(args.batch_size),
			"early_stopping": bool(args.early_stopping),
			"validation_fraction": float(args.validation_fraction),
			"n_iter_no_change": int(args.n_iter_no_change),
			"enable_tuning": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_scoring": args.cv_scoring if tuning_summary["enabled"] else None,
			"cv_n_iter": int(args.cv_n_iter) if tuning_summary["enabled"] else None,
			"tuning_optimizer": args.tuning_optimizer if tuning_summary["enabled"] else None,
			"tuning_activation": args.tuning_activation if tuning_summary["enabled"] else None,
			"tuning_regularization": args.tuning_regularization if tuning_summary["enabled"] else None,
			"random_state": int(args.random_state),
			"hidden_layers": selected_hidden_layers,
			"dropout": float(selected_dropout),
			"architecture": selected_architecture,
		},
		"tuning": tuning_summary,
		"preprocessing": {"feature_count": {"raw": int(X.shape[1]), "post_transform": _post_transform_feature_count(preprocessor, X_train.iloc[:1])}},
		"training_control": training_control,
		"fit_summary": {
			"fit_time_seconds": _round_metric(fit_time_seconds),
			"predict_time_seconds": _round_metric(predict_time_seconds),
			"random_state_effective": int(args.random_state),
			"n_jobs": None,
		},
		"artifacts": _artifact_map(
			run_dir,
			{
				"model": model_dir / "model.keras",
				"preprocess": preprocess_dir / "preprocessor.pkl",
				"eval_metrics": eval_dir / "metrics.json",
				"eval_training_history": eval_dir / "training_history.json",
				"eval_predictions_preview": eval_dir / "predictions_preview.csv",
				**schema_artifacts,
				"inference_example": inference_dir / "inference_example.py",
			},
		),
		"versions": {"python": platform.python_version(), "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "tensorflow": tf.__version__},
	}
	(run_dir / "run.json").write_text(json.dumps(_compact_metadata(_json_safe(run_metadata)), indent=2), encoding="utf-8")

	registry_path = model_root_dir / "registry.csv"
	registry_df = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
	next_id = int(registry_df["model_id"].max()) + 1 if ("model_id" in registry_df.columns and not registry_df.empty) else 1
	registry_row = pd.DataFrame(
		[{
			"model_id": next_id,
			"run_id": run_id,
			"model_name": model_name,
			"timestamp": timestamp,
			"dataset_sha256": data_hash,
			"dataset_rows": int(len(df)),
			"dataset_columns": int(df.shape[1]),
			"optimizer": selected_optimizer,
			"learning_rate": float(selected_learning_rate),
			"activation": selected_activation,
			"l1": float(selected_l1),
			"l2": float(selected_l2),
			"epochs": int(args.epochs),
			"batch_size": int(args.batch_size),
			"random_state": int(args.random_state),
			"tuning_enabled": bool(tuning_summary["enabled"]),
			"tuning_method": args.tuning_method if tuning_summary["enabled"] else None,
			"cv_best_score": tuning_summary["best_score"],
			"training_control_enabled": bool(training_control["enabled"]),
			"training_control_best_score": float(training_control["best_score"]) if training_control["best_score"] is not None else None,
			"training_control_best_step": int(training_control["best_step"]) if training_control["best_step"] is not None else None,
			"training_control_steps_completed": int(training_control["steps_completed"]) if training_control["steps_completed"] is not None else None,
			"training_control_stopped_early": bool(training_control["stopped_early"]),
			"mse": _round_metric(test_mse),
			"mae": _round_metric(test_mae),
			"rmse": _round_metric(test_rmse),
			"r2": _round_metric(test_r2),
			"n_train": int(len(X_train)),
			"n_test": int(len(X_test)),
		}]
	)
	pd.concat([registry_df, registry_row], ignore_index=True).to_csv(registry_path, index=False)
	_write_unified_registry_sqlite(
		project_root=_project_root(),
		run_dir=run_dir,
		run_metadata=_compact_metadata(_json_safe(run_metadata)),
		metrics=metrics,
	)
	print(f"Artifacts exported to: {run_dir}")

