from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
	optimizer={{OPTIMIZER_CTOR}}(learning_rate={{LEARNING_RATE}}),
	loss="mse",
	metrics=["mae"],
)

model.fit(X_train, y_train, epochs={{EPOCHS}}, batch_size={{BATCH_SIZE}}, verbose=1)
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("MSE:", loss)
print("MAE:", mae)
