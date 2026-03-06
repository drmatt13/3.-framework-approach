import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


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


def build_dense_classifier(input_dim: int, output_units: int, output_activation: str, optimizer_name: str, learning_rate: float, hidden_layers: list[int], dropout: float, is_binary: bool, hidden_activation: str = "relu", l1: float = 0.0, l2: float = 0.0) -> tf.keras.Sequential:
	optimizer_ctor = {
		"adam": tf.keras.optimizers.Adam,
		"sgd": tf.keras.optimizers.SGD,
		"rmsprop": tf.keras.optimizers.RMSprop,
		"adagrad": tf.keras.optimizers.Adagrad,
		"adamw": tf.keras.optimizers.AdamW,
	}[optimizer_name]
	optimizer = optimizer_ctor(learning_rate=float(learning_rate))
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
	optimizer_ctor = {
		"adam": tf.keras.optimizers.Adam,
		"sgd": tf.keras.optimizers.SGD,
		"rmsprop": tf.keras.optimizers.RMSprop,
		"adagrad": tf.keras.optimizers.Adagrad,
		"adamw": tf.keras.optimizers.AdamW,
	}[optimizer_name]
	optimizer = optimizer_ctor(learning_rate=float(learning_rate))
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
