from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
	optimizer={{OPTIMIZER_CTOR}}(learning_rate={{LEARNING_RATE}}),
	loss="{{LOSS_FN}}",
	metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs={{EPOCHS}}, batch_size={{BATCH_SIZE}}, verbose=1)
loss, metric = model.evaluate(X_test, y_test, verbose=0)
print("Loss:", loss)
print("Accuracy:", metric)
