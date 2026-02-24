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


parser = argparse.ArgumentParser(description="TensorFlow CNN classification baseline")
parser.add_argument("--library", choices=["tensorflow"], default="tensorflow")
parser.add_argument("--model", choices=["cnn"], default="cnn")
parser.add_argument("--task", choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
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

data_path = Path(__file__).resolve().parents[1] / "data" / "template_data" / "{{DATA_FILE}}"
df = pd.read_csv(data_path)
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

y = df["{{TARGET_COLUMN}}"]
{{TARGET_PREPROCESS}}
X = df.drop(columns={{FEATURE_DROP_COLUMNS}})
X = pd.get_dummies(X, drop_first=True)

X_values = X.to_numpy(dtype="float32")
n_samples, n_features = X_values.shape
side = int(np.ceil(np.sqrt(n_features)))

padded = np.zeros((n_samples, side * side), dtype="float32")
padded[:, :n_features] = X_values
X_images = padded.reshape(n_samples, side, side, 1)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
	X_images,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y,
)

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(side, side, 1)),
	tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
	tf.keras.layers.MaxPool2D((2, 2)),
	tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
	tf.keras.layers.MaxPool2D((2, 2)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense({{OUTPUT_UNITS}}, activation="{{OUTPUT_ACTIVATION}}"),
])

model.compile(
	optimizer=optimizer_map[args.optimizer](learning_rate=args.learning_rate),
	loss="{{LOSS_FN}}",
	metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
loss, metric = model.evaluate(X_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", metric)

if SAVE_MODEL:
	model_path = Path(__file__).resolve().with_name(f"{Path(__file__).stem}_model.keras")
	model.save(model_path)
	print(f"Saved model to: {model_path}")
