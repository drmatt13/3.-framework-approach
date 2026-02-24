import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAVE_MODEL = False


def _parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "y"}:
		return True
	if normalized in {"0", "false", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError("Expected true/false")


parser = argparse.ArgumentParser(description="TensorFlow Dense NN regression baseline")
parser.add_argument("--library", choices=["tensorflow"], default="tensorflow")
parser.add_argument("--model", choices=["dense_nn"], default="dense_nn")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"], default="{{OPTIMIZER_NAME}}")
parser.add_argument("--learning_rate", type=float, default={{LEARNING_RATE}})
parser.add_argument("--epochs", type=int, default={{EPOCHS}})
parser.add_argument("--batch_size", type=int, default={{BATCH_SIZE}})
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
args = parser.parse_args()
SAVE_MODEL = args.save_model

optimizer_map = {
	"adam": tf.keras.optimizers.Adam,
	"sgd": tf.keras.optimizers.SGD,
	"rmsprop": tf.keras.optimizers.RMSprop,
	"adagrad": tf.keras.optimizers.Adagrad,
	"adamw": tf.keras.optimizers.AdamW,
}

data_path = Path(__file__).resolve().parents[1] / "data" / "template_data" / "california_housing.csv"
df = pd.read_csv(data_path).dropna()

X = df.drop(columns=["median_house_value"])
X = pd.get_dummies(X, drop_first=True)
X = X.to_numpy(dtype="float32")
y = df["median_house_value"].to_numpy(dtype="float32")

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
)

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(3,)),
	tf.keras.layers.Dense(16, activation="relu"),
	tf.keras.layers.Dense(8, activation="relu"),
	tf.keras.layers.Dense(1, activation="linear"),
])

model.compile(
	optimizer=optimizer_map[args.optimizer](learning_rate=args.learning_rate),
	loss="mse",
	metrics=["mae"],
)

model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("MSE:", loss)
print("MAE:", mae)

if SAVE_MODEL:
	model_path = Path(__file__).resolve().with_name(f"{Path(__file__).stem}_model.keras")
	model.save(model_path)
	print(f"Saved model to: {model_path}")
