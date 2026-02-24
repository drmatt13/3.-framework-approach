import argparse
import hashlib
import json
import pickle
import platform
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------
# Supported CLI flags (common usage)
#   --library scikit-learn
#   --model random_forest
#   --task regression
#   --name <model_name>
#   --save-model true|false
#   --random-state <int>
# ---------------------------------------------------------------------

SAVE_MODEL = False
DEFAULT_RANDOM_STATE = 1


def _project_root() -> Path:
	current = Path(__file__).resolve().parent
	for candidate in [current, *current.parents]:
		if (candidate / "requirements.txt").exists():
			return candidate
	return Path(__file__).resolve().parents[1]


def _parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "y"}:
		return True
	if normalized in {"0", "false", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError("Expected true/false")


parser = argparse.ArgumentParser(description="Random Forest Regressor baseline")
parser.add_argument("--library", choices=["scikit-learn"], default="scikit-learn")
parser.add_argument("--model", choices=["random_forest"], default="random_forest")
parser.add_argument("--task", choices=["regression"], default="regression")
parser.add_argument("--name", default=Path(__file__).stem)
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
args = parser.parse_args()
SAVE_MODEL = args.save_model

# Load data
project_root = _project_root()
data_path = project_root / "data" / "template_data" / "california_housing.csv"
df = pd.read_csv(data_path).dropna()

y = df["median_house_value"]
X = df.drop(columns=["median_house_value"])

# Additional Feature engineer here if needed
# ******************************************
# ******************************************
# ******************************************

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=args.random_state,
)

categorical_cols = X_train.select_dtypes(include=["object", "category", "bool", "str"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

try:
	one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
	one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
	transformers=[
		("num", StandardScaler(), numerical_cols),
		("cat", one_hot_encoder, categorical_cols),
	],
	remainder="drop",
)

model = Pipeline(
	steps=[
		("preprocess", preprocessor),
		("regressor", RandomForestRegressor(random_state=args.random_state)),
	]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, predictions)
test_mae = mean_absolute_error(y_test, predictions)
test_rmse = root_mean_squared_error(y_test, predictions)
test_r2 = r2_score(y_test, predictions)

print("MSE:", test_mse)
print("MAE:", test_mae)
print("RMSE:", test_rmse)
print("R2:", test_r2)
print("First 5 predictions:", predictions[:5])

if SAVE_MODEL:
	model_name = args.name.strip() or Path(__file__).stem
	model_root_dir = project_root / "artifacts" / "models" / model_name
	timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
	run_id = str(uuid.uuid4())
	data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
	data_rows = int(len(df))
	data_columns = int(df.shape[1])
	run_dir = model_root_dir / f"{timestamp}_{model_name}"

	model_dir = run_dir / "model"
	preprocess_dir = run_dir / "preprocess"
	eval_dir = run_dir / "eval"
	data_dir = run_dir / "data"
	inference_dir = run_dir / "inference"

	for directory in (model_dir, preprocess_dir, eval_dir, data_dir, inference_dir):
		directory.mkdir(parents=True, exist_ok=True)

	with (model_dir / "model.pkl").open("wb") as model_file:
		pickle.dump(model, model_file)

	with (preprocess_dir / "preprocessor.pkl").open("wb") as preprocess_file:
		pickle.dump(model.named_steps["preprocess"], preprocess_file)

	metrics = {
		"mse": float(test_mse),
		"mae": float(test_mae),
		"rmse": float(test_rmse),
		"r2": float(test_r2),
		"n_train": int(len(X_train)),
		"n_test": int(len(X_test)),
	}
	with (eval_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
		json.dump(metrics, metrics_file, indent=2)

	predictions_preview = pd.DataFrame(
		{
			"y_true": y_test.iloc[:50].tolist(),
			"y_pred": predictions[:50].tolist(),
		}
	)
	predictions_preview.to_csv(eval_dir / "predictions_preview.csv", index=False)

	inference_rows = X_test.iloc[:5].to_dict(orient="records")
	expected_values = y_test.iloc[:5].tolist()
	inference_script = f'''import pickle
from pathlib import Path

import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

sample_rows = {json.dumps(inference_rows, indent=2)}
expected_y = {json.dumps(expected_values, indent=2)}

with MODEL_PATH.open("rb") as model_file:
	model = pickle.load(model_file)

features = pd.DataFrame(sample_rows)
predictions = model.predict(features)

print("Inference example (5 rows from X_test)")
print("Predictions:", predictions.tolist())
print("Expected y:", expected_y)

results = features.copy()
results["y_expected"] = expected_y
results["y_pred"] = predictions
print(results)
'''
	with (inference_dir / "inference_example.py").open("w", encoding="utf-8") as inference_file:
		inference_file.write(inference_script)

	feature_schema = {
		"target": "median_house_value",
		"feature_columns": X.columns.tolist(),
		"categorical_columns": categorical_cols,
		"numerical_columns": numerical_cols,
		"dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
	}
	with (data_dir / "feature_schema.json").open("w", encoding="utf-8") as schema_file:
		json.dump(feature_schema, schema_file, indent=2)

	run_metadata = {
		"run_id": run_id,
		"library": args.library,
		"model": args.model,
		"name": model_name,
		"task": args.task,
		"timestamp": timestamp,
		"dataset": {
			"path": str(data_path.relative_to(project_root)),
			"sha256": data_hash,
			"rows": data_rows,
			"columns": data_columns,
		},
		"artifacts": {
			"model": str((model_dir / "model.pkl").relative_to(project_root)),
			"preprocess": str((preprocess_dir / "preprocessor.pkl").relative_to(project_root)),
			"eval_metrics": str((eval_dir / "metrics.json").relative_to(project_root)),
			"eval_predictions_preview": str((eval_dir / "predictions_preview.csv").relative_to(project_root)),
			"feature_schema": str((data_dir / "feature_schema.json").relative_to(project_root)),
			"inference_example": str((inference_dir / "inference_example.py").relative_to(project_root)),
		},
		"params": {
			"test_size": 0.2,
			"random_state": args.random_state,
			"one_hot_handle_unknown": "ignore",
			"scaler": "StandardScaler",
			"regressor": "RandomForestRegressor",
		},
		"versions": {
			"python": platform.python_version(),
			"pandas": pd.__version__,
			"scikit-learn": sklearn.__version__,
		},
	}
	with (run_dir / "run.json").open("w", encoding="utf-8") as run_file:
		json.dump(run_metadata, run_file, indent=2)

	registry_path = model_root_dir / "model_registry.csv"
	if registry_path.exists():
		registry_df = pd.read_csv(registry_path)
		if "model_id" in registry_df.columns and not registry_df.empty:
			model_id = int(registry_df["model_id"].max()) + 1
		else:
			model_id = 1
	else:
		registry_df = pd.DataFrame()
		model_id = 1

	registry_row = pd.DataFrame(
		[
			{
				"model_id": model_id,
				"run_id": run_id,
				"name": model_name,
				"timestamp": timestamp,
				"dataset_sha256": data_hash,
				"dataset_rows": data_rows,
				"dataset_columns": data_columns,
				"random_state": int(args.random_state),
				"mse": float(test_mse),
				"mae": float(test_mae),
				"rmse": float(test_rmse),
				"r2": float(test_r2),
				"n_train": int(len(X_train)),
				"n_test": int(len(X_test)),
			}
		]
	)
	registry_df = pd.concat([registry_df, registry_row], ignore_index=True)
	registry_df.to_csv(registry_path, index=False)

	print(f"Artifacts exported to: {run_dir}")
