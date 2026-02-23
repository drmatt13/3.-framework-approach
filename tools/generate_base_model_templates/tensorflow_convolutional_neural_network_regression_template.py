from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data_path = Path(__file__).resolve().parents[1] / "data" / "template_data" / "california_housing.csv"
df = pd.read_csv(data_path).dropna()

X = df.drop(columns=["median_house_value"])
X = pd.get_dummies(X, drop_first=True)
y = df["median_house_value"].to_numpy(dtype="float32")

X_values = X.to_numpy(dtype="float32")
n_samples, n_features = X_values.shape
side = int(np.ceil(np.sqrt(n_features)))

padded = np.zeros((n_samples, side * side), dtype="float32")
padded[:, :n_features] = X_values
X_images = padded.reshape(n_samples, side, side, 1)

X_train, X_test, y_train, y_test = train_test_split(
	X_images,
	y,
	test_size=0.2,
	random_state=42,
)

model = tf.keras.Sequential([
	tf.keras.layers.Input(shape=(side, side, 1)),
	tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
	tf.keras.layers.MaxPool2D((2, 2)),
	tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
	tf.keras.layers.MaxPool2D((2, 2)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense(1, activation="linear"),
])

model.compile(
	optimizer={{OPTIMIZER_CTOR}}(learning_rate={{LEARNING_RATE}}),
	loss="mse",
	metrics=["mae"],
)

model.fit(X_train, y_train, epochs={{EPOCHS}}, batch_size={{BATCH_SIZE}}, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("MSE:", loss)
print("MAE:", mae)
