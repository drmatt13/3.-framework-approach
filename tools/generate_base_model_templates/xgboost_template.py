from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
{{SKLEARN_METRIC_IMPORT}}

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
	booster="{{BOOSTER}}",
	random_state=42,
	{{XGB_TASK_KWARGS}}
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
{{PREDICTION_POST_PROCESS}}
print("{{METRIC_LABEL}}:", {{METRIC_CALL}})
