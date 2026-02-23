from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data_path = Path(__file__).resolve().parents[1] / "data" / "template_data" / "california_housing.csv"
df = pd.read_csv(data_path).dropna()

X = df.drop(columns=["median_house_value"])
X = pd.get_dummies(X, drop_first=True)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
