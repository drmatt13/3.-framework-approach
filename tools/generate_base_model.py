import argparse
from pathlib import Path
import sys

# ---------------------------------------------------------
# Flag combinations:
# ---------------------------------------------------------

# scikit-learn
# ---------------------------------------------------------

# Linear Regression (regression only)
#   --library=scikit-learn 
#   --model=linear_regression 
#   --task=regression 
#   --name=...

# Logistic Regression (classification only)
#   --library=scikit-learn 
#   --model=logistic_regression 
#   --task=binary_classification|multiclass_classification 
#   --name=...

# Random Forest
#   --library=scikit-learn 
#   --model=random_forest 
#   --task=regression|binary_classification|multiclass_classification 
#   --name=...


# xgboost
# ---------------------------------------------------------

#   --library=xgboost 
#   --task=regression|binary_classification|multiclass_classification 
#   --name=...
#
#   (optional)
#   --booster=gbtree|gblinear|dart


# tensorflow
# ---------------------------------------------------------

# Dense Neural Network (MLP-style)
#   --library=tensorflow 
#   --model=dense_nn 
#   --task=regression|binary_classification|multiclass_classification 
#   --optimizer=adam|sgd|rmsprop|adagrad|adamw
#   --learning_rate=<float>
#   --epochs=<int>
#   --batch_size=<int>
#   --name=...

# Convolutional Neural Network
#   --library=tensorflow 
#   --model=cnn 
#   --task=regression|binary_classification|multiclass_classification 
#   --optimizer=adam|sgd|rmsprop|adagrad|adamw
#   --learning_rate=<float>
#   --epochs=<int>
#   --batch_size=<int>
#   --name=...

# ---------------------------------------------------------

SKLEARN_MODELS = {"linear_regression", "logistic_regression", "random_forest"}
TENSORFLOW_MODELS = {"dense_nn", "cnn"}
TASKS = {"regression", "binary_classification", "multiclass_classification"}
XGBOOST_BOOSTERS = {"gbtree", "gblinear", "dart"}
TENSORFLOW_OPTIMIZERS = {"adam", "sgd", "rmsprop", "adagrad", "adamw"}

SKLEARN_MODEL_TASKS = {
    "linear_regression": {"regression"},
    "logistic_regression": {"binary_classification", "multiclass_classification"},
    "random_forest": TASKS,
}

TENSORFLOW_MODEL_TASKS = {
    "dense_nn": TASKS,
    "cnn": TASKS,
}

OPTIMIZER_CLASS_MAP = {
    "adam": "tf.keras.optimizers.Adam",
    "sgd": "tf.keras.optimizers.SGD",
    "rmsprop": "tf.keras.optimizers.RMSprop",
    "adagrad": "tf.keras.optimizers.Adagrad",
    "adamw": "tf.keras.optimizers.AdamW",
}


# ---------------------------------------------------------
# Templates
# ---------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent / "generate_base_model_templates"


def task_family(task: str) -> str:
    return "classification" if task in {"binary_classification", "multiclass_classification"} else "regression"


def read_template(filename: str) -> str:
    template_path = TEMPLATES_DIR / filename
    if not template_path.exists():
        raise ValueError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def template_filename(args: argparse.Namespace) -> str:
    family = task_family(args.task)

    if args.library == "scikit-learn":
        if args.model == "linear_regression":
            return "scikit-learn_linear_regression_template.py"
        if args.model == "logistic_regression":
            return "scikit-learn_logistic_regression_template.py"
        if args.model == "random_forest":
            return "scikit-learn_random_forest_template.py"

    if args.library == "xgboost":
        return "xgboost_template.py"

    if args.library == "tensorflow":
        if args.model == "dense_nn":
            return (
                "tensorflow_dense_neural_network_classification_template.py"
                if family == "classification"
                else "tensorflow_dense_neural_network_regression_template.py"
            )
        if args.model == "cnn":
            return (
                "tensorflow_convolutional_neural_network_classification_template.py"
                if family == "classification"
                else "tensorflow_convolutional_neural_network_regression_template.py"
            )

    raise ValueError("No template mapping found for provided flags")


def template_replacements(args: argparse.Namespace) -> dict[str, str]:
    family = task_family(args.task)
    replacements: dict[str, str] = {}

    if args.library == "scikit-learn" and args.model == "logistic_regression":
        if args.task == "binary_classification":
            replacements.update(
                {
                    "DATA_FILE": "breast_cancer_wisconsin.csv",
                    "TARGET_COLUMN": "diagnosis",
                    "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                    "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                }
            )
        else:
            replacements.update(
                {
                    "DATA_FILE": "iris.csv",
                    "TARGET_COLUMN": "species",
                    "FEATURE_DROP_COLUMNS": '["species"]',
                    "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                }
            )

    if args.library == "scikit-learn" and args.model == "random_forest":
        if family == "regression":
            replacements.update(
                {
                    "RF_ESTIMATOR": "RandomForestRegressor",
                    "SKLEARN_METRIC_IMPORT": "from sklearn.metrics import mean_squared_error",
                    "DATA_FILE": "california_housing.csv",
                    "TARGET_COLUMN": "median_house_value",
                    "FEATURE_DROP_COLUMNS": '["median_house_value"]',
                    "TARGET_PREPROCESS": "",
                    "STRATIFY_ARG": "",
                    "METRIC_LABEL": "MSE",
                    "METRIC_CALL": "mean_squared_error(y_test, predictions)",
                }
            )
        else:
            if args.task == "binary_classification":
                data_file = "breast_cancer_wisconsin.csv"
                target_column = "diagnosis"
                feature_drop_columns = '["diagnosis", "id"]'
                target_preprocess = 'y = y.map({"B": 0, "M": 1}).astype("int64")'
            else:
                data_file = "iris.csv"
                target_column = "species"
                feature_drop_columns = '["species"]'
                target_preprocess = 'y = y.astype("category").cat.codes.astype("int64")'

            replacements.update(
                {
                    "RF_ESTIMATOR": "RandomForestClassifier",
                    "SKLEARN_METRIC_IMPORT": "from sklearn.metrics import accuracy_score",
                    "DATA_FILE": data_file,
                    "TARGET_COLUMN": target_column,
                    "FEATURE_DROP_COLUMNS": feature_drop_columns,
                    "TARGET_PREPROCESS": target_preprocess,
                    "STRATIFY_ARG": "stratify=y",
                    "METRIC_LABEL": "Accuracy",
                    "METRIC_CALL": "accuracy_score(y_test, predictions)",
                }
            )

    if args.library == "xgboost":
        booster = args.booster or "gbtree"
        if args.task == "regression":
            replacements.update(
                {
                    "XGB_ESTIMATOR": "XGBRegressor",
                    "SKLEARN_METRIC_IMPORT": "from sklearn.metrics import mean_squared_error",
                    "DATA_FILE": "california_housing.csv",
                    "TARGET_COLUMN": "median_house_value",
                    "FEATURE_DROP_COLUMNS": '["median_house_value"]',
                    "TARGET_PREPROCESS": "",
                    "STRATIFY_ARG": "",
                    "XGB_TASK_KWARGS": 'objective="reg:squarederror", eval_metric="rmse"',
                    "PREDICTION_POST_PROCESS": "",
                    "METRIC_LABEL": "MSE",
                    "METRIC_CALL": "mean_squared_error(y_test, predictions)",
                    "BOOSTER": booster,
                }
            )
        elif args.task == "binary_classification":
            replacements.update(
                {
                    "XGB_ESTIMATOR": "XGBClassifier",
                    "SKLEARN_METRIC_IMPORT": "from sklearn.metrics import accuracy_score",
                    "DATA_FILE": "breast_cancer_wisconsin.csv",
                    "TARGET_COLUMN": "diagnosis",
                    "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                    "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                    "STRATIFY_ARG": "stratify=y",
                    "XGB_TASK_KWARGS": 'objective="binary:logistic", eval_metric="logloss"',
                    "PREDICTION_POST_PROCESS": "",
                    "METRIC_LABEL": "Accuracy",
                    "METRIC_CALL": "accuracy_score(y_test, predictions)",
                    "BOOSTER": booster,
                }
            )
        else:
            replacements.update(
                {
                    "XGB_ESTIMATOR": "XGBClassifier",
                    "SKLEARN_METRIC_IMPORT": "from sklearn.metrics import accuracy_score",
                    "DATA_FILE": "iris.csv",
                    "TARGET_COLUMN": "species",
                    "FEATURE_DROP_COLUMNS": '["species"]',
                    "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                    "STRATIFY_ARG": "stratify=y",
                    "XGB_TASK_KWARGS": 'objective="multi:softprob", num_class=3, eval_metric="mlogloss"',
                    "PREDICTION_POST_PROCESS": "if predictions.ndim > 1:\n    predictions = np.argmax(predictions, axis=1)",
                    "METRIC_LABEL": "Accuracy",
                    "METRIC_CALL": "accuracy_score(y_test, predictions)",
                    "BOOSTER": booster,
                }
            )

    if args.library == "tensorflow":
        replacements.update(
            {
                "OPTIMIZER_CTOR": OPTIMIZER_CLASS_MAP[args.optimizer],
                "LEARNING_RATE": str(args.learning_rate),
                "EPOCHS": str(args.epochs),
                "BATCH_SIZE": str(args.batch_size),
            }
        )

        if family == "classification":
            if args.task == "binary_classification":
                replacements.update(
                    {
                        "DATA_FILE": "breast_cancer_wisconsin.csv",
                        "TARGET_COLUMN": "diagnosis",
                        "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                        "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                        "OUTPUT_UNITS": "1",
                        "OUTPUT_ACTIVATION": "sigmoid",
                        "LOSS_FN": "binary_crossentropy",
                    }
                )
            else:
                replacements.update(
                    {
                        "DATA_FILE": "iris.csv",
                        "TARGET_COLUMN": "species",
                        "FEATURE_DROP_COLUMNS": '["species"]',
                        "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                        "OUTPUT_UNITS": "3",
                        "OUTPUT_ACTIVATION": "softmax",
                        "LOSS_FN": "sparse_categorical_crossentropy",
                    }
                )

    return replacements


# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    # --name is required by argparse; here we just sanity check blank strings
    if not args.name or not args.name.strip():
        raise ValueError("--name cannot be empty")

    if args.task not in TASKS:
        allowed = ", ".join(sorted(TASKS))
        raise ValueError(f"Invalid --task '{args.task}'. Allowed: {allowed}")

    if args.library == "xgboost":
        if args.model is not None:
            raise ValueError("Invalid flags: xgboost does not use --model. Omit --model entirely.")
        if args.booster is not None and args.booster not in XGBOOST_BOOSTERS:
            allowed = ", ".join(sorted(XGBOOST_BOOSTERS))
            raise ValueError(f"Invalid xgboost --booster '{args.booster}'. Allowed: {allowed}")
        if args.optimizer is not None or args.learning_rate is not None or args.epochs is not None or args.batch_size is not None:
            raise ValueError("Invalid flags: --optimizer/--learning_rate/--epochs/--batch_size are tensorflow-only")
        return

    if args.library == "scikit-learn":
        if not args.model:
            raise ValueError("--model is required for scikit-learn")
        if args.model not in SKLEARN_MODELS:
            allowed = ", ".join(sorted(SKLEARN_MODELS))
            raise ValueError(f"Invalid scikit-learn --model '{args.model}'. Allowed: {allowed}")
        allowed_tasks = SKLEARN_MODEL_TASKS[args.model]
        if args.task not in allowed_tasks:
            allowed = ", ".join(sorted(allowed_tasks))
            raise ValueError(f"Invalid --task '{args.task}' for scikit-learn model '{args.model}'. Allowed: {allowed}")
        if args.booster is not None:
            raise ValueError("Invalid flags: --booster is xgboost-only")
        if args.optimizer is not None or args.learning_rate is not None or args.epochs is not None or args.batch_size is not None:
            raise ValueError("Invalid flags: --optimizer/--learning_rate/--epochs/--batch_size are tensorflow-only")
        return

    if args.library == "tensorflow":
        if not args.model:
            raise ValueError("--model is required for tensorflow")
        if args.model not in TENSORFLOW_MODELS:
            allowed = ", ".join(sorted(TENSORFLOW_MODELS))
            raise ValueError(f"Invalid tensorflow --model '{args.model}'. Allowed: {allowed}")
        allowed_tasks = TENSORFLOW_MODEL_TASKS[args.model]
        if args.task not in allowed_tasks:
            allowed = ", ".join(sorted(allowed_tasks))
            raise ValueError(f"Invalid --task '{args.task}' for tensorflow model '{args.model}'. Allowed: {allowed}")
        if args.booster is not None:
            raise ValueError("Invalid flags: --booster is xgboost-only")
        if args.optimizer is None:
            raise ValueError("--optimizer is required for tensorflow")
        if args.optimizer not in TENSORFLOW_OPTIMIZERS:
            allowed = ", ".join(sorted(TENSORFLOW_OPTIMIZERS))
            raise ValueError(f"Invalid tensorflow --optimizer '{args.optimizer}'. Allowed: {allowed}")
        if args.learning_rate is None:
            raise ValueError("--learning_rate is required for tensorflow")
        if args.epochs is None:
            raise ValueError("--epochs is required for tensorflow")
        if args.batch_size is None:
            raise ValueError("--batch_size is required for tensorflow")
        return

    raise ValueError(f"Unsupported library: {args.library}")


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate base ML model file")
    parser.add_argument(
        "--library",
        required=True,
        choices=["scikit-learn", "xgboost", "tensorflow"],
        help="Which ML library/framework to generate for",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Model template to generate (required for scikit-learn and tensorflow)",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["regression", "binary_classification", "multiclass_classification"],
        help="ML task type",
    )
    parser.add_argument(
        "--booster",
        required=False,
        choices=["gbtree", "gblinear", "dart"],
        help="xgboost booster (optional; xgboost only)",
    )
    parser.add_argument(
        "--optimizer",
        required=False,
        choices=["adam", "sgd", "rmsprop", "adagrad", "adamw"],
        help="tensorflow optimizer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="tensorflow learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="tensorflow epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="tensorflow batch size",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of generated model file (without .py)",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional explicit output path. If omitted, writes to <repo>/models/<name>.py",
    )

    args = parser.parse_args()

    try:
        validate_args(args)

        template_name = template_filename(args)
        template = read_template(template_name)
        content = render_template(template, template_replacements(args))

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            script_dir = Path(__file__).resolve().parent
            models_dir = script_dir.parent / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            output_path = models_dir / f"{args.name}.py"

        output_path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"Generated file: {output_path.resolve()}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()