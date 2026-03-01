import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
	for column in frame.select_dtypes(include=["object", "string"]).columns:
		series = frame[column].astype("string").str.strip()
		series = series.replace("", np.nan)
		frame[column] = series.astype("object")
	return frame.replace({pd.NA: np.nan})


def build_tabular_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
	categorical_cols = frame.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
	numerical_cols = frame.select_dtypes(include=["number"]).columns.tolist()
	try:
		one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", one_hot_encoder),
		]
	)
	return ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numerical_cols),
			("cat", categorical_transformer, categorical_cols),
		],
		remainder="drop",
	)
