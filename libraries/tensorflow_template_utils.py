import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


def _activation_name(activation) -> str | None:
	if activation is None:
		return None
	if isinstance(activation, str):
		return activation
	name = getattr(activation, "__name__", None)
	if isinstance(name, str) and name:
		return name
	return str(activation)


def _serialize_keras_object(value):
	if value is None:
		return None
	try:
		return tf.keras.utils.serialize_keras_object(value)
	except Exception:
		if hasattr(value, "get_config"):
			try:
				return {
					"class_name": value.__class__.__name__,
					"config": value.get_config(),
				}
			except Exception:
				return {"class_name": value.__class__.__name__}
		return str(value)


def _resolve_learning_rate(optimizer, fallback: float | None = None) -> float | None:
	if optimizer is None:
		return fallback
	learning_rate = getattr(optimizer, "learning_rate", None)
	if learning_rate is None:
		return fallback
	try:
		return float(tf.keras.backend.get_value(learning_rate))
	except Exception:
		try:
			return float(learning_rate)
		except Exception:
			return fallback


def _layer_l1_l2(layer: tf.keras.layers.Layer) -> tuple[float, float]:
	regularizer = getattr(layer, "kernel_regularizer", None)
	if regularizer is None:
		return 0.0, 0.0
	return (
		float(getattr(regularizer, "l1", 0.0) or 0.0),
		float(getattr(regularizer, "l2", 0.0) or 0.0),
	)


def summarize_dense_model_metadata(
	model: tf.keras.Sequential,
	*,
	fallback_optimizer_name: str | None = None,
	fallback_learning_rate: float | None = None,
) -> dict:
	optimizer_obj = getattr(model, "optimizer", None)
	optimizer_name = str(getattr(optimizer_obj, "name", fallback_optimizer_name or "unknown"))
	learning_rate = _resolve_learning_rate(optimizer_obj, fallback=fallback_learning_rate)

	layers_payload: list[dict] = []
	dense_layers: list[tf.keras.layers.Dense] = []
	hidden_dropout_rates: list[float] = []
	for index, layer in enumerate(model.layers):
		entry = {
			"index": int(index),
			"name": str(getattr(layer, "name", layer.__class__.__name__)),
			"class_name": layer.__class__.__name__,
			"trainable": bool(getattr(layer, "trainable", True)),
		}
		if isinstance(layer, tf.keras.layers.Dense):
			dense_layers.append(layer)
			entry.update(
				{
					"units": int(layer.units),
					"activation": _activation_name(getattr(layer, "activation", None)),
					"kernel_regularizer": _serialize_keras_object(getattr(layer, "kernel_regularizer", None)),
				}
			)
		elif isinstance(layer, tf.keras.layers.Dropout):
			rate = float(getattr(layer, "rate", 0.0))
			hidden_dropout_rates.append(rate)
			entry.update({"rate": rate})
		layers_payload.append(entry)

	hidden_dense_layers = dense_layers[:-1] if len(dense_layers) > 1 else []
	hidden_layers = [int(layer.units) for layer in hidden_dense_layers]
	hidden_activations = [_activation_name(getattr(layer, "activation", None)) for layer in hidden_dense_layers]
	hidden_l1_l2 = [_layer_l1_l2(layer) for layer in hidden_dense_layers]

	representative_activation = "relu"
	if len(hidden_activations) > 0:
		unique_activations = {str(v) for v in hidden_activations if v is not None}
		if len(unique_activations) == 1:
			representative_activation = str(hidden_activations[0])
		else:
			representative_activation = "mixed"

	representative_dropout = float(hidden_dropout_rates[0]) if len(hidden_dropout_rates) > 0 else 0.0
	representative_l1 = float(hidden_l1_l2[0][0]) if len(hidden_l1_l2) > 0 else 0.0
	representative_l2 = float(hidden_l1_l2[0][1]) if len(hidden_l1_l2) > 0 else 0.0

	output_layer = dense_layers[-1] if len(dense_layers) > 0 else None
	architecture = {
		"layers": layers_payload,
		"hidden": {
			"units": hidden_layers,
			"activations": hidden_activations,
			"dropout_rates": hidden_dropout_rates,
			"regularization_l1_l2": [
				{"l1": float(l1_value), "l2": float(l2_value)}
				for l1_value, l2_value in hidden_l1_l2
			],
		},
		"output": {
			"units": int(output_layer.units) if output_layer is not None else None,
			"activation": _activation_name(getattr(output_layer, "activation", None)) if output_layer is not None else None,
		},
	}

	return {
		"optimizer": optimizer_name,
		"learning_rate": learning_rate,
		"hidden_layers": hidden_layers,
		"activation": representative_activation,
		"dropout": representative_dropout,
		"l1": representative_l1,
		"l2": representative_l2,
		"architecture": architecture,
	}


def _kernel_regularizer(l1: float, l2: float):
	l1_value = float(l1)
	l2_value = float(l2)
	if l1_value <= 0.0 and l2_value <= 0.0:
		return None
	if l1_value > 0.0 and l2_value > 0.0:
		return tf.keras.regularizers.l1_l2(l1=l1_value, l2=l2_value)
	if l1_value > 0.0:
		return tf.keras.regularizers.l1(l1_value)
	return tf.keras.regularizers.l2(l2_value)


def build_direct_fit_dense_config(
	*,
	optimizer_name: str,
	learning_rate: float,
	hidden_layers: list[int],
	activations: list[str],
	dropouts: list[float],
	l1s: list[float],
	l2s: list[float],
	output_units: int | None = None,
	output_activation: str | None = None,
) -> dict:
	hidden_layers_values = [int(v) for v in hidden_layers]
	activation_values = [str(v) for v in activations]
	dropout_values = [float(v) for v in dropouts]
	l1_values = [float(v) for v in l1s]
	l2_values = [float(v) for v in l2s]

	if len(hidden_layers_values) == 0:
		raise ValueError("Direct-fit dense config must define at least one hidden layer.")

	if not (
		len(hidden_layers_values)
		== len(activation_values)
		== len(dropout_values)
		== len(l1_values)
		== len(l2_values)
	):
		raise ValueError(
			"Direct-fit dense config lengths must match for hidden_layers, activations, dropouts, l1s, and l2s."
		)

	for units in hidden_layers_values:
		if int(units) <= 0:
			raise ValueError("Hidden layer units must be > 0.")
	for dropout_rate in dropout_values:
		if float(dropout_rate) < 0.0 or float(dropout_rate) >= 1.0:
			raise ValueError("Dropout rates must be in [0.0, 1.0).")
	for regularization in [*l1_values, *l2_values]:
		if float(regularization) < 0.0:
			raise ValueError("L1/L2 regularization values must be >= 0.0.")

	if output_units is not None and int(output_units) <= 0:
		raise ValueError("Output units must be > 0 when provided.")
	if output_activation is not None and str(output_activation).strip() == "":
		raise ValueError("Output activation must be a non-empty string when provided.")

	representative_activation = activation_values[0]
	if len({str(v) for v in activation_values}) > 1:
		representative_activation = "mixed"

	return {
		"optimizer": str(optimizer_name),
		"learning_rate": float(learning_rate),
		"hidden_layers": hidden_layers_values,
		"activations": activation_values,
		"dropouts": dropout_values,
		"l1s": l1_values,
		"l2s": l2_values,
		"activation": representative_activation,
		"dropout": float(dropout_values[0]),
		"l1": float(l1_values[0]),
		"l2": float(l2_values[0]),
		"output_units": int(output_units) if output_units is not None else None,
		"output_activation": str(output_activation) if output_activation is not None else None,
	}


def resolve_classification_output_config(num_classes: int) -> dict:
	class_count = int(num_classes)
	if class_count < 2:
		raise ValueError("Need at least 2 classes in the training split.")
	is_binary = class_count == 2
	return {
		"is_binary": bool(is_binary),
		"output_units": 1 if is_binary else class_count,
		"output_activation": "sigmoid" if is_binary else "softmax",
		"loss_fn": "binary_crossentropy" if is_binary else "sparse_categorical_crossentropy",
	}


def build_optimizer(optimizer_name: str, learning_rate: float):
	name = str(optimizer_name).strip().lower()
	if name == "adam":
		return tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
	if name == "sgd":
		return tf.keras.optimizers.SGD(learning_rate=float(learning_rate))
	if name == "rmsprop":
		return tf.keras.optimizers.RMSprop(learning_rate=float(learning_rate))
	if name == "adagrad":
		return tf.keras.optimizers.Adagrad(learning_rate=float(learning_rate))
	if name == "adamw":
		return tf.keras.optimizers.AdamW(learning_rate=float(learning_rate))
	raise ValueError(f"Unsupported TensorFlow optimizer: {optimizer_name}")


def build_dense_classifier(input_dim: int, output_units: int, output_activation: str, optimizer_name: str, learning_rate: float, hidden_layers: list[int], dropout: float, is_binary: bool, hidden_activation: str = "relu", l1: float = 0.0, l2: float = 0.0) -> tf.keras.Sequential:
	optimizer = build_optimizer(optimizer_name=optimizer_name, learning_rate=float(learning_rate))
	hidden_regularizer = _kernel_regularizer(l1=l1, l2=l2)
	layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(int(input_dim),))]
	for width in hidden_layers:
		layers.append(tf.keras.layers.Dense(int(width), activation=hidden_activation, kernel_regularizer=hidden_regularizer))
	if float(dropout) > 0:
		layers.append(tf.keras.layers.Dropout(float(dropout)))
	layers.append(tf.keras.layers.Dense(output_units, activation=output_activation))
	model = tf.keras.Sequential(layers)
	model.compile(
		optimizer=optimizer,
		loss="binary_crossentropy" if is_binary else "sparse_categorical_crossentropy",
		metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")] if is_binary else [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
	)
	return model


def build_dense_regressor(input_dim: int, optimizer_name: str, learning_rate: float, hidden_layers: list[int], dropout: float, hidden_activation: str = "relu", l1: float = 0.0, l2: float = 0.0) -> tf.keras.Sequential:
	optimizer = build_optimizer(optimizer_name=optimizer_name, learning_rate=float(learning_rate))
	hidden_regularizer = _kernel_regularizer(l1=l1, l2=l2)
	layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(int(input_dim),))]
	for width in hidden_layers:
		layers.append(tf.keras.layers.Dense(int(width), activation=hidden_activation, kernel_regularizer=hidden_regularizer))
	if float(dropout) > 0:
		layers.append(tf.keras.layers.Dropout(float(dropout)))
	layers.append(tf.keras.layers.Dense(1))
	model = tf.keras.Sequential(layers)
	model.compile(
		optimizer=optimizer,
		loss="mse",
		metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
	)
	return model


def predict_class_probabilities(model: tf.keras.Sequential, features: np.ndarray, is_binary: bool) -> np.ndarray:
	raw = np.asarray(model(features, training=False).numpy())
	if is_binary:
		pos = raw.reshape(-1)
		return np.column_stack([1.0 - pos, pos])
	return raw


def classification_score(y_true: np.ndarray, probabilities: np.ndarray, is_binary: bool, scoring: str) -> float:
	if is_binary:
		preds = (probabilities[:, -1] >= 0.5).astype(int)
	else:
		preds = np.argmax(probabilities, axis=1).astype(int)
	if scoring == "f1_macro":
		return float(f1_score(y_true, preds, average="macro", zero_division=0))
	raise ValueError(f"Unsupported --cv-scoring '{scoring}'")
