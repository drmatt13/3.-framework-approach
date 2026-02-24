import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
{{SKLEARN_METRIC_IMPORT}}

SAVE_MODEL = False


def _parse_bool(value: str) -> bool:
	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "y"}:
		return True
	if normalized in {"0", "false", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError("Expected true/false")


parser = argparse.ArgumentParser(description="XGBoost baseline")
parser.add_argument("--library", choices=["xgboost"], default="xgboost")
parser.add_argument("--task", choices=["{{TASK_VALUE}}"], default="{{TASK_VALUE}}")
parser.add_argument("--booster", choices=["gbtree", "gblinear", "dart"], default="{{BOOSTER}}")
parser.add_argument("--save-model", type=_parse_bool, default=SAVE_MODEL)
args = parser.parse_args()
SAVE_MODEL = args.save_model

data_path = Path(__file__).resolve().parents[1] / "data" / "template_data" / "{{DATA_FILE}}"
df = pd.read_csv(data_path)
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

y = df["{{TARGET_COLUMN}}"]
{{TARGET_PREPROCESS}}
X = df.drop(columns={{FEATURE_DROP_COLUMNS}})
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	{{STRATIFY_ARG}}
)

model = xgb.{{XGB_ESTIMATOR}}(
	booster=args.booster,
	random_state=42,
	{{XGB_TASK_KWARGS}}
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
{{PREDICTION_POST_PROCESS}}
print("{{METRIC_LABEL}}:", {{METRIC_CALL}})

if SAVE_MODEL:
	model_path = Path(__file__).resolve().with_name(f"{Path(__file__).stem}_model.pkl")
	with model_path.open("wb") as model_file:
		pickle.dump(model, model_file)
	print(f"Saved model to: {model_path}")
