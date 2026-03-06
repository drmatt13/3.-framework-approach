import argparse
from pathlib import Path
import sys

_current_file = Path(__file__).resolve()
for _candidate in [_current_file.parent, *_current_file.parents]:
    if (_candidate / "libraries").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from libraries.cli_helpers import parse_bool_flag as _parse_bool

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
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto
#   --penalty=auto|none|l1|l2|elasticnet
#   --alpha=<float>
#   --enable-tuning=true|false
#   --tuning-method=grid|random
#   --cv-folds=<int>
#   --cv-scoring=rmse|mae|r2
#   --cv-n-iter=<int>
#   --cv-n-jobs=<int>

# Logistic Regression (classification only)
#   --library=scikit-learn 
#   --model=logistic_regression 
#   --task=binary_classification|multiclass_classification 
#   --name=...
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto

# Random Forest
#   --library=scikit-learn 
#   --model=random_forest 
#   --task=regression|binary_classification|multiclass_classification 
#   --name=...
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto
#   --early-stopping=true|false
#   --validation-fraction=<float>
#   --n-iter-no-change=<int>

# xgboost
# ---------------------------------------------------------

#   --library=xgboost 
#   --task=regression|binary_classification|multiclass_classification 
#   --name=...
#
#   (optional)
#   --booster=auto|gbtree|gblinear|dart
#   --device=auto|cpu|gpu
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto
#   --early-stopping=true|false
#   --validation-fraction=<float>
#   --n-iter-no-change=<int>

# tensorflow
# ---------------------------------------------------------

# Dense Neural Network (MLP-style)
#   --library=tensorflow 
#   --model=dense_nn 
#   --task=regression|binary_classification|multiclass_classification 
#   --optimizer=auto|adam|sgd|rmsprop|adagrad|adamw
#   --learning_rate=<float>
#   --epochs=<int>
#   --batch_size=<int>
#   --name=...
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto

# Convolutional Neural Network
#   --library=tensorflow 
#   --model=cnn 
#   --task=regression|binary_classification|multiclass_classification 
#   --optimizer=auto|adam|sgd|rmsprop|adagrad|adamw
#   --learning_rate=<float>
#   --epochs=<int>
#   --batch_size=<int>
#   --name=...
#
#   Generated template runtime options (configurable defaults):
#   --verbose=0|1|2|auto

# ---------------------------------------------------------

SKLEARN_MODELS = {"linear_regression", "logistic_regression", "random_forest"}
TENSORFLOW_MODELS = {"dense_nn"}
TASKS = {"regression", "binary_classification", "multiclass_classification"}
XGBOOST_BOOSTERS = {"auto", "gbtree", "gblinear", "dart"}
XGBOOST_DEVICES = {"auto", "cpu", "gpu"}
TENSORFLOW_OPTIMIZERS = {"auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"}
SKLEARN_LOGISTIC_SOLVERS = {"auto", "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
CLASSIFICATION_CV_SCORINGS = {"f1_macro", "accuracy", "roc_auc_ovr"}
REGRESSION_CV_SCORINGS = {"rmse", "mae", "r2"}
RF_MAX_FEATURE_PRESETS = {"auto", "sqrt", "log2", "none"}

SKLEARN_MODEL_TASKS = {
    "linear_regression": {"regression"},
    "logistic_regression": {"binary_classification", "multiclass_classification"},
    "random_forest": TASKS,
}

TENSORFLOW_MODEL_TASKS = {
    "dense_nn": TASKS,
}

OPTIMIZER_CLASS_MAP = {
    "auto": "tf.keras.optimizers.Adam",
    "adam": "tf.keras.optimizers.Adam",
    "sgd": "tf.keras.optimizers.SGD",
    "rmsprop": "tf.keras.optimizers.RMSprop",
    "adagrad": "tf.keras.optimizers.Adagrad",
    "adamw": "tf.keras.optimizers.AdamW",
}

DEFAULT_EARLY_STOPPING_BY_TEMPLATE = {
    ("xgboost", None, "classification"): True,
    ("xgboost", None, "regression"): True,
    ("tensorflow", "dense_nn", "classification"): True,
    ("tensorflow", "dense_nn", "regression"): True,
}

DEFAULT_VALIDATION_FRACTION_BY_TEMPLATE = {
    ("xgboost", None, "classification"): 0.1,
    ("xgboost", None, "regression"): 0.1,
    ("tensorflow", "dense_nn", "classification"): 0.1,
    ("tensorflow", "dense_nn", "regression"): 0.1,
}

DEFAULT_N_ITER_NO_CHANGE_BY_TEMPLATE = {
    ("xgboost", None, "classification"): 20,
    ("xgboost", None, "regression"): 20,
    ("tensorflow", "dense_nn", "classification"): 5,
    ("tensorflow", "dense_nn", "regression"): 5,
}

DEFAULT_MAX_ITER_BY_TEMPLATE = {
    ("scikit-learn", "logistic_regression", "classification"): 1000,
}

DEFAULT_XGBOOST_PARAMS_BY_TEMPLATE = {
    ("xgboost", "classification"): {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "f1_macro",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
    ("xgboost", "regression"): {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "rmse",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
}

DEFAULT_LOGISTIC_PARAMS_BY_TEMPLATE = {
    ("scikit-learn", "logistic_regression", "classification"): {
        "c": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
        "class_weight": "none",
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "f1_macro",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
}

DEFAULT_RANDOM_FOREST_PARAMS_BY_TEMPLATE = {
    ("scikit-learn", "random_forest", "classification"): {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "max_features": "sqrt",
        "bootstrap": True,
        "max_samples": 1.0,
        "ccp_alpha": 0.0,
        "n_jobs": -1,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "f1_macro",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
    ("scikit-learn", "random_forest", "regression"): {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "max_features": "1.0",
        "bootstrap": True,
        "max_samples": 1.0,
        "ccp_alpha": 0.0,
        "n_jobs": -1,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "rmse",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
}

DEFAULT_TENSORFLOW_DENSE_PARAMS_BY_TEMPLATE = {
    ("tensorflow", "dense_nn", "classification"): {
        "hidden_layers": [128, 64, 32],
        "dropout": 0.1,
        "hidden_activation": "relu",
        "l1": 0.0,
        "l2": 0.0,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_scoring": "f1_macro",
        "cv_n_iter": 10,
        "tuning_optimizer": "auto",
        "tuning_activation": "auto",
        "tuning_regularization": "auto",
    },
    ("tensorflow", "dense_nn", "regression"): {
        "hidden_layers": [128, 64, 32],
        "dropout": 0.0,
        "hidden_activation": "relu",
        "l1": 0.0,
        "l2": 0.0,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_scoring": "rmse",
        "cv_n_iter": 10,
        "tuning_optimizer": "auto",
        "tuning_activation": "auto",
        "tuning_regularization": "auto",
    },
}

DEFAULT_LINEAR_REGRESSION_PARAMS_BY_TEMPLATE = {
    ("scikit-learn", "linear_regression", "regression"): {
        "penalty": "none",
        "alpha": 1.0,
        "fit_intercept": True,
        "l1_ratio": 0.5,
        "enable_tuning": False,
        "tuning_method": "grid",
        "cv_folds": 5,
        "cv_scoring": "rmse",
        "cv_n_iter": 20,
        "cv_n_jobs": -1,
    },
}

STARTER_DATASETS_BY_FAMILY = {
    "regression": ["ames_housing.csv", "california_housing.csv", "insurance.csv"],
    "binary_classification": ["adult_income.csv", "breast_cancer_wisconsin.csv", "mushrooms.csv", "titanic.csv"],
    "multiclass_classification": [
        "car_evaluation.csv",
        "dry_bean.csv",
        "forest_cover_type.csv",
        "iris.csv",
        "wine_quality.csv",
    ],
}


STARTER_DATASET_CONFIG = {
    "ames_housing.csv": {
        "data_file": "ames_housing.csv",
        "target_column": "SalePrice",
        "feature_drop_columns": '["SalePrice", "Order", "PID"]',
        "target_preprocess": 'y = y.astype("float64")',
    },
    "california_housing.csv": {
        "data_file": "california_housing.csv",
        "target_column": "median_house_value",
        "feature_drop_columns": '["median_house_value"]',
        "target_preprocess": 'y = y.astype("float64")',
    },
    "insurance.csv": {
        "data_file": "insurance.csv",
        "target_column": "charges",
        "feature_drop_columns": '["charges"]',
        "target_preprocess": 'y = y.astype("float64")',
    },
    "adult_income.csv": {
        "data_file": "adult_income.csv",
        "target_column": "income",
        "feature_drop_columns": '["income"]',
        "target_preprocess": 'y = y.astype("str").str.strip().str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1}).astype("int64")',
    },
    "breast_cancer_wisconsin.csv": {
        "data_file": "breast_cancer_wisconsin.csv",
        "target_column": "diagnosis",
        "feature_drop_columns": '["diagnosis", "id"]',
        "target_preprocess": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
    },
    "titanic.csv": {
        "data_file": "titanic.csv",
        "target_column": "Survived",
        "feature_drop_columns": '["Survived", "PassengerId"]',
        "target_preprocess": 'y = y.astype("int64")',
    },
    "car_evaluation.csv": {
        "data_file": "car_evaluation.csv",
        "target_column": "class",
        "feature_drop_columns": '["class"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
        "read_csv_extra_args": ", header=None",
        "header_names": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
    },
    "dry_bean.csv": {
        "data_file": "dry_bean.csv",
        "target_column": "Class",
        "feature_drop_columns": '["Class"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
    },
    "forest_cover_type.csv": {
        "data_file": "forest_cover_type.csv",
        "target_column": "Cover_Type",
        "feature_drop_columns": '["Cover_Type"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
    },
    "iris.csv": {
        "data_file": "iris.csv",
        "target_column": "species",
        "feature_drop_columns": '["species"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
    },
    "mushrooms.csv": {
        "data_file": "mushrooms.csv",
        "target_column": "class",
        "feature_drop_columns": '["class"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
    },
    "wine_quality.csv": {
        "data_file": "wine_quality.csv",
        "target_column": "quality",
        "feature_drop_columns": '["quality"]',
        "target_preprocess": 'y = y.astype("category").cat.codes.astype("int64")',
    },
}


# ---------------------------------------------------------
# Templates
# ---------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent / "model_templates"


def validate_shared_helper_modules() -> None:
    required_modules = [
        "preprocessing_utils.py",
        "search_utils.py",
        "serialization_utils.py",
        "sklearn_template_utils.py",
        "xgboost_search_space.py",
        "xgboost_template_utils.py",
        "tensorflow_template_utils.py",
    ]
    libraries_dir = Path(__file__).resolve().parent.parent / "libraries"
    missing = [name for name in required_modules if not (libraries_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            "Required shared helper module(s) are missing in libraries/: "
            f"{missing_list}. Restore these files before generating models."
        )


def task_family(task: str) -> str:
    return "classification" if task in {"binary_classification", "multiclass_classification"} else "regression"


def _parse_optional_int_token(value: str | int | None, *, arg_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        return int(normalized)
    except Exception as exc:
        raise ValueError(f"Invalid {arg_name}. Expected integer or 'none'.") from exc


def _stringify_optional_int(value: int | None) -> str:
    return "none" if value is None else str(int(value))


def _parse_optional_max_samples_token(value: str | int | float | None, *, arg_name: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = value
    else:
        normalized = str(value).strip().lower()
        if normalized in {"none", "null"}:
            return None
        try:
            parsed = float(normalized) if "." in normalized else int(normalized)
        except Exception as exc:
            raise ValueError(f"Invalid {arg_name}. Expected int, float in (0,1], or 'none'.") from exc
    if isinstance(parsed, int):
        if parsed <= 0:
            raise ValueError(f"Invalid {arg_name}. Integer value must be > 0.")
        return parsed
    parsed_float = float(parsed)
    if not (0.0 < parsed_float <= 1.0):
        raise ValueError(f"Invalid {arg_name}. Float value must be in range (0, 1].")
    return parsed_float


def _stringify_optional_max_samples(value: int | float | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(int(value))
    return str(float(value))


def _validate_rf_max_features_default(value: str, *, arg_name: str) -> None:
    normalized = str(value).strip().lower()
    if normalized in RF_MAX_FEATURE_PRESETS:
        return
    try:
        parsed = float(normalized)
    except Exception as exc:
        raise ValueError(
            f"Invalid {arg_name}. Allowed: auto, sqrt, log2, none, or float in (0, 1]."
        ) from exc
    if not (0.0 < parsed <= 1.0):
        raise ValueError(
            f"Invalid {arg_name}. Float value must be in range (0, 1]."
        )


def _starter_dataset_for_args(args: argparse.Namespace) -> dict | None:
    if not args.starter_dataset:
        return None
    return STARTER_DATASET_CONFIG[args.starter_dataset]


def _parse_hidden_layers_csv(value: str, *, arg_name: str) -> list[int]:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"Invalid {arg_name}. Provide comma-separated positive integers, e.g. 128,64,32")
    hidden_layers: list[int] = []
    for token in tokens:
        try:
            units = int(token)
        except Exception as exc:
            raise ValueError(f"Invalid {arg_name}. '{token}' is not an integer") from exc
        if units <= 0:
            raise ValueError(f"Invalid {arg_name}. Hidden layer units must be positive")
        hidden_layers.append(units)
    return hidden_layers


def _task_dataset_dir(task: str) -> str:
    return task


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


def _supports_early_stopping_defaults(args: argparse.Namespace) -> bool:
    if args.library == "tensorflow" and args.model == "dense_nn":
        return True
    if args.library == "xgboost":
        return True
    return False


def _supports_validation_n_iter_defaults(args: argparse.Namespace) -> bool:
    if args.library == "tensorflow" and args.model == "dense_nn":
        return True
    if args.library == "xgboost":
        return True
    return False


def _supports_max_iter_default(args: argparse.Namespace) -> bool:
    family = task_family(args.task)
    return args.library == "scikit-learn" and args.model == "logistic_regression" and family == "classification"


def template_filename(args: argparse.Namespace) -> str:
    family = task_family(args.task)

    if args.library == "scikit-learn":
        if args.model == "linear_regression":
            return "scikit-learn_linear_regression_template.py"
        if args.model == "logistic_regression":
            return "scikit-learn_logistic_regression_template.py"
        if args.model == "random_forest":
            return (
                "scikit-learn_random_forest_regression_template.py"
                if family == "regression"
                else "scikit-learn_random_forest_classification_template.py"
            )

    if args.library == "xgboost":
        return (
            "xgboost_regression_template.py"
            if family == "regression"
            else "xgboost_classification_template.py"
        )

    if args.library == "tensorflow":
        if args.model == "dense_nn":
            return (
                "tensorflow_dense_neural_network_classification_template.py"
                if family == "classification"
                else "tensorflow_dense_neural_network_regression_template.py"
            )

    raise ValueError("No template mapping found for provided flags")


def template_replacements(args: argparse.Namespace) -> dict[str, str]:
    family = task_family(args.task)
    replacements: dict[str, str] = {}
    starter_dataset = _starter_dataset_for_args(args)

    if args.library == "scikit-learn" and args.model == "linear_regression":
        linear_dataset = starter_dataset if starter_dataset is not None else STARTER_DATASET_CONFIG["california_housing.csv"]
        lr_defaults = DEFAULT_LINEAR_REGRESSION_PARAMS_BY_TEMPLATE[("scikit-learn", "linear_regression", "regression")]
        default_lr_penalty = args.default_lr_penalty if args.default_lr_penalty is not None else lr_defaults["penalty"]
        default_lr_alpha = args.default_lr_alpha if args.default_lr_alpha is not None else lr_defaults["alpha"]
        default_lr_fit_intercept = args.default_lr_fit_intercept if args.default_lr_fit_intercept is not None else lr_defaults["fit_intercept"]
        default_lr_l1_ratio = args.default_lr_l1_ratio if args.default_lr_l1_ratio is not None else lr_defaults["l1_ratio"]
        default_lr_enable_tuning = (
            args.default_lr_enable_tuning if args.default_lr_enable_tuning is not None else lr_defaults["enable_tuning"]
        )
        default_lr_tuning_method = (
            args.default_lr_tuning_method if args.default_lr_tuning_method is not None else lr_defaults["tuning_method"]
        )
        default_lr_cv_folds = args.default_lr_cv_folds if args.default_lr_cv_folds is not None else lr_defaults["cv_folds"]
        default_lr_cv_scoring = (
            args.default_lr_cv_scoring if args.default_lr_cv_scoring is not None else lr_defaults["cv_scoring"]
        )
        default_lr_cv_n_iter = args.default_lr_cv_n_iter if args.default_lr_cv_n_iter is not None else lr_defaults["cv_n_iter"]
        default_lr_cv_n_jobs = args.default_lr_cv_n_jobs if args.default_lr_cv_n_jobs is not None else lr_defaults["cv_n_jobs"]
        replacements.update(
            {
                "DATA_TASK_DIR": _task_dataset_dir(args.task),
                "DATA_FILE": linear_dataset["data_file"],
                "TARGET_COLUMN": linear_dataset["target_column"],
                "FEATURE_DROP_COLUMNS": linear_dataset["feature_drop_columns"],
                "TARGET_PREPROCESS": linear_dataset["target_preprocess"],
                "LR_PENALTY_DEFAULT": str(default_lr_penalty),
                "LR_ALPHA_DEFAULT": str(float(default_lr_alpha)),
                "LR_FIT_INTERCEPT_DEFAULT": "True" if default_lr_fit_intercept else "False",
                "LR_L1_RATIO_DEFAULT": str(float(default_lr_l1_ratio)),
                "LR_ENABLE_TUNING_DEFAULT": "True" if default_lr_enable_tuning else "False",
                "LR_TUNING_METHOD_DEFAULT": str(default_lr_tuning_method),
                "LR_CV_FOLDS_DEFAULT": str(int(default_lr_cv_folds)),
                "LR_CV_SCORING_DEFAULT": str(default_lr_cv_scoring),
                "LR_CV_N_ITER_DEFAULT": str(int(default_lr_cv_n_iter)),
                "LR_CV_N_JOBS_DEFAULT": str(int(default_lr_cv_n_jobs)),
            }
        )

    if _supports_early_stopping_defaults(args):
        key = (args.library, args.model if args.library in {"scikit-learn", "tensorflow"} else None, family)
        early_stopping_default = args.default_early_stopping
        if early_stopping_default is None:
            early_stopping_default = DEFAULT_EARLY_STOPPING_BY_TEMPLATE[key]

        replacements.update(
            {
                "EARLY_STOPPING_DEFAULT": "True" if early_stopping_default else "False",
            }
        )

    if _supports_validation_n_iter_defaults(args):
        key = (args.library, args.model if args.library in {"scikit-learn", "tensorflow"} else None, family)

        validation_fraction_default = args.default_validation_fraction
        if validation_fraction_default is None:
            validation_fraction_default = DEFAULT_VALIDATION_FRACTION_BY_TEMPLATE[key]

        n_iter_no_change_default = args.default_n_iter_no_change
        if n_iter_no_change_default is None:
            n_iter_no_change_default = DEFAULT_N_ITER_NO_CHANGE_BY_TEMPLATE[key]

        replacements.update(
            {
                "VALIDATION_FRACTION_DEFAULT": str(validation_fraction_default),
                "N_ITER_NO_CHANGE_DEFAULT": str(n_iter_no_change_default),
            }
        )

    if _supports_max_iter_default(args):
        key = ("scikit-learn", "logistic_regression", family)
        max_iter_default = args.default_max_iter
        if max_iter_default is None:
            max_iter_default = DEFAULT_MAX_ITER_BY_TEMPLATE[key]
        replacements.update({"MAX_ITER_DEFAULT": str(int(max_iter_default))})

    if args.library == "scikit-learn" and args.model == "logistic_regression":
        logistic_defaults = DEFAULT_LOGISTIC_PARAMS_BY_TEMPLATE[("scikit-learn", "logistic_regression", family)]
        default_c = args.default_c if args.default_c is not None else logistic_defaults["c"]
        default_solver = args.default_solver if args.default_solver is not None else logistic_defaults["solver"]
        default_penalty = args.default_logistic_penalty if args.default_logistic_penalty is not None else logistic_defaults["penalty"]
        default_class_weight = args.default_logistic_class_weight if args.default_logistic_class_weight is not None else logistic_defaults["class_weight"]
        default_enable_tuning = args.default_logistic_enable_tuning if args.default_logistic_enable_tuning is not None else logistic_defaults["enable_tuning"]
        default_tuning_method = args.default_logistic_tuning_method if args.default_logistic_tuning_method is not None else logistic_defaults["tuning_method"]
        default_cv_folds = args.default_logistic_cv_folds if args.default_logistic_cv_folds is not None else logistic_defaults["cv_folds"]
        default_cv_scoring = args.default_logistic_cv_scoring if args.default_logistic_cv_scoring is not None else logistic_defaults["cv_scoring"]
        default_cv_n_iter = args.default_logistic_cv_n_iter if args.default_logistic_cv_n_iter is not None else logistic_defaults["cv_n_iter"]
        default_cv_n_jobs = args.default_logistic_cv_n_jobs if args.default_logistic_cv_n_jobs is not None else logistic_defaults["cv_n_jobs"]
        replacements.update(
            {
                "LOGISTIC_C_DEFAULT": str(float(default_c)),
                "LOGISTIC_SOLVER_DEFAULT": str(default_solver),
                "LOGISTIC_PENALTY_DEFAULT": str(default_penalty),
                "LOGISTIC_CLASS_WEIGHT_DEFAULT": str(default_class_weight),
                "LOGISTIC_ENABLE_TUNING_DEFAULT": "True" if default_enable_tuning else "False",
                "LOGISTIC_TUNING_METHOD_DEFAULT": str(default_tuning_method),
                "LOGISTIC_CV_FOLDS_DEFAULT": str(int(default_cv_folds)),
                "LOGISTIC_CV_SCORING_DEFAULT": str(default_cv_scoring),
                "LOGISTIC_CV_N_ITER_DEFAULT": str(int(default_cv_n_iter)),
                "LOGISTIC_CV_N_JOBS_DEFAULT": str(int(default_cv_n_jobs)),
            }
        )

        if starter_dataset is not None:
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": starter_dataset["data_file"],
                    "TARGET_COLUMN": starter_dataset["target_column"],
                    "FEATURE_DROP_COLUMNS": starter_dataset["feature_drop_columns"],
                    "TARGET_PREPROCESS": starter_dataset["target_preprocess"],
                }
            )
        elif args.task == "binary_classification":
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": "breast_cancer_wisconsin.csv",
                    "TARGET_COLUMN": "diagnosis",
                    "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                    "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                }
            )
        else:
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": "iris.csv",
                    "TARGET_COLUMN": "species",
                    "FEATURE_DROP_COLUMNS": '["species"]',
                    "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                }
            )

    if args.library == "scikit-learn" and args.model == "random_forest":
        rf_defaults = DEFAULT_RANDOM_FOREST_PARAMS_BY_TEMPLATE[("scikit-learn", "random_forest", family)]
        default_rf_n_estimators = (
            args.default_rf_n_estimators if args.default_rf_n_estimators is not None else rf_defaults["n_estimators"]
        )
        default_rf_max_depth = _parse_optional_int_token(
            args.default_rf_max_depth,
            arg_name="--default-rf-max-depth",
        ) if args.default_rf_max_depth is not None else rf_defaults["max_depth"]
        default_rf_min_samples_split = (
            args.default_rf_min_samples_split
            if args.default_rf_min_samples_split is not None
            else rf_defaults["min_samples_split"]
        )
        default_rf_min_samples_leaf = (
            args.default_rf_min_samples_leaf
            if args.default_rf_min_samples_leaf is not None
            else rf_defaults["min_samples_leaf"]
        )
        default_rf_min_weight_fraction_leaf = (
            args.default_rf_min_weight_fraction_leaf
            if args.default_rf_min_weight_fraction_leaf is not None
            else rf_defaults["min_weight_fraction_leaf"]
        )
        default_rf_max_leaf_nodes = _parse_optional_int_token(
            args.default_rf_max_leaf_nodes,
            arg_name="--default-rf-max-leaf-nodes",
        ) if args.default_rf_max_leaf_nodes is not None else rf_defaults["max_leaf_nodes"]
        default_rf_min_impurity_decrease = (
            args.default_rf_min_impurity_decrease
            if args.default_rf_min_impurity_decrease is not None
            else rf_defaults["min_impurity_decrease"]
        )
        default_rf_max_features = (
            args.default_rf_max_features
            if args.default_rf_max_features is not None
            else rf_defaults["max_features"]
        )
        default_rf_bootstrap = (
            args.default_rf_bootstrap
            if args.default_rf_bootstrap is not None
            else rf_defaults["bootstrap"]
        )
        default_rf_max_samples = _parse_optional_max_samples_token(
            args.default_rf_max_samples,
            arg_name="--default-rf-max-samples",
        ) if args.default_rf_max_samples is not None else rf_defaults["max_samples"]
        default_rf_ccp_alpha = (
            args.default_rf_ccp_alpha
            if args.default_rf_ccp_alpha is not None
            else rf_defaults["ccp_alpha"]
        )
        default_rf_n_jobs = _parse_optional_int_token(
            args.default_rf_n_jobs,
            arg_name="--default-rf-n-jobs",
        ) if args.default_rf_n_jobs is not None else rf_defaults["n_jobs"]
        default_enable_tuning = args.default_rf_enable_tuning if args.default_rf_enable_tuning is not None else rf_defaults["enable_tuning"]
        default_tuning_method = args.default_rf_tuning_method if args.default_rf_tuning_method is not None else rf_defaults["tuning_method"]
        default_cv_folds = args.default_rf_cv_folds if args.default_rf_cv_folds is not None else rf_defaults["cv_folds"]
        default_cv_scoring = args.default_rf_cv_scoring if args.default_rf_cv_scoring is not None else rf_defaults["cv_scoring"]
        default_cv_n_iter = args.default_rf_cv_n_iter if args.default_rf_cv_n_iter is not None else rf_defaults["cv_n_iter"]
        default_cv_n_jobs = args.default_rf_cv_n_jobs if args.default_rf_cv_n_jobs is not None else rf_defaults["cv_n_jobs"]
        replacements.update(
            {
                "RF_N_ESTIMATORS_DEFAULT": str(int(default_rf_n_estimators)),
                "RF_MAX_DEPTH_DEFAULT": _stringify_optional_int(default_rf_max_depth),
                "RF_MIN_SAMPLES_SPLIT_DEFAULT": str(int(default_rf_min_samples_split)),
                "RF_MIN_SAMPLES_LEAF_DEFAULT": str(int(default_rf_min_samples_leaf)),
                "RF_MIN_WEIGHT_FRACTION_LEAF_DEFAULT": str(float(default_rf_min_weight_fraction_leaf)),
                "RF_MAX_LEAF_NODES_DEFAULT": _stringify_optional_int(default_rf_max_leaf_nodes),
                "RF_MIN_IMPURITY_DECREASE_DEFAULT": str(float(default_rf_min_impurity_decrease)),
                "RF_MAX_FEATURES_DEFAULT": str(default_rf_max_features),
                "RF_BOOTSTRAP_DEFAULT": "True" if default_rf_bootstrap else "False",
                "RF_MAX_SAMPLES_DEFAULT": _stringify_optional_max_samples(default_rf_max_samples),
                "RF_CCP_ALPHA_DEFAULT": str(float(default_rf_ccp_alpha)),
                "RF_N_JOBS_DEFAULT": _stringify_optional_int(default_rf_n_jobs),
                "RF_ENABLE_TUNING_DEFAULT": "True" if default_enable_tuning else "False",
                "RF_TUNING_METHOD_DEFAULT": str(default_tuning_method),
                "RF_CV_FOLDS_DEFAULT": str(int(default_cv_folds)),
                "RF_CV_SCORING_DEFAULT": str(default_cv_scoring),
                "RF_CV_N_ITER_DEFAULT": str(int(default_cv_n_iter)),
                "RF_CV_N_JOBS_DEFAULT": str(int(default_cv_n_jobs)),
            }
        )

        if family == "classification":
            if starter_dataset is not None:
                replacements.update(
                    {
                        "TASK_VALUE": args.task,
                        "DATA_FILE": starter_dataset["data_file"],
                        "TARGET_COLUMN": starter_dataset["target_column"],
                        "FEATURE_DROP_COLUMNS": starter_dataset["feature_drop_columns"],
                        "TARGET_PREPROCESS": starter_dataset["target_preprocess"],
                    }
                )
            elif args.task == "binary_classification":
                replacements.update(
                    {
                        "TASK_VALUE": args.task,
                        "DATA_FILE": "breast_cancer_wisconsin.csv",
                        "TARGET_COLUMN": "diagnosis",
                        "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                        "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                    }
                )
            else:
                replacements.update(
                    {
                        "TASK_VALUE": args.task,
                        "DATA_FILE": "iris.csv",
                        "TARGET_COLUMN": "species",
                        "FEATURE_DROP_COLUMNS": '["species"]',
                        "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                    }
                )
        else:
            regression_dataset = starter_dataset if starter_dataset is not None else STARTER_DATASET_CONFIG["california_housing.csv"]
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": regression_dataset["data_file"],
                    "TARGET_COLUMN": regression_dataset["target_column"],
                    "FEATURE_DROP_COLUMNS": regression_dataset["feature_drop_columns"],
                    "TARGET_PREPROCESS": regression_dataset["target_preprocess"],
                }
            )

    if args.library == "xgboost":
        booster = args.booster or "gbtree"
        device = args.device or "cpu"
        xgboost_defaults = DEFAULT_XGBOOST_PARAMS_BY_TEMPLATE[("xgboost", family)]
        default_n_estimators = args.default_n_estimators if args.default_n_estimators is not None else xgboost_defaults["n_estimators"]
        default_learning_rate = (
            args.default_learning_rate if args.default_learning_rate is not None else xgboost_defaults["learning_rate"]
        )
        default_max_depth = args.default_max_depth if args.default_max_depth is not None else xgboost_defaults["max_depth"]
        default_subsample = args.default_subsample if args.default_subsample is not None else xgboost_defaults["subsample"]
        default_colsample_bytree = (
            args.default_colsample_bytree
            if args.default_colsample_bytree is not None
            else xgboost_defaults["colsample_bytree"]
        )
        default_min_child_weight = (
            args.default_xgb_min_child_weight
            if args.default_xgb_min_child_weight is not None
            else xgboost_defaults["min_child_weight"]
        )
        default_reg_lambda = (
            args.default_xgb_reg_lambda
            if args.default_xgb_reg_lambda is not None
            else xgboost_defaults["reg_lambda"]
        )
        default_reg_alpha = (
            args.default_xgb_reg_alpha
            if args.default_xgb_reg_alpha is not None
            else xgboost_defaults["reg_alpha"]
        )
        default_enable_tuning = args.default_xgb_enable_tuning if args.default_xgb_enable_tuning is not None else xgboost_defaults["enable_tuning"]
        default_tuning_method = args.default_xgb_tuning_method if args.default_xgb_tuning_method is not None else xgboost_defaults["tuning_method"]
        default_cv_folds = args.default_xgb_cv_folds if args.default_xgb_cv_folds is not None else xgboost_defaults["cv_folds"]
        default_cv_scoring = args.default_xgb_cv_scoring if args.default_xgb_cv_scoring is not None else xgboost_defaults["cv_scoring"]
        default_cv_n_iter = args.default_xgb_cv_n_iter if args.default_xgb_cv_n_iter is not None else xgboost_defaults["cv_n_iter"]
        default_cv_n_jobs = args.default_xgb_cv_n_jobs if args.default_xgb_cv_n_jobs is not None else xgboost_defaults["cv_n_jobs"]

        replacements.update(
            {
                "XGB_N_ESTIMATORS_DEFAULT": str(int(default_n_estimators)),
                "XGB_LEARNING_RATE_DEFAULT": str(float(default_learning_rate)),
                "XGB_MAX_DEPTH_DEFAULT": str(int(default_max_depth)),
                "XGB_SUBSAMPLE_DEFAULT": str(float(default_subsample)),
                "XGB_COLSAMPLE_BYTREE_DEFAULT": str(float(default_colsample_bytree)),
                "XGB_MIN_CHILD_WEIGHT_DEFAULT": str(float(default_min_child_weight)),
                "XGB_REG_LAMBDA_DEFAULT": str(float(default_reg_lambda)),
                "XGB_REG_ALPHA_DEFAULT": str(float(default_reg_alpha)),
                "XGB_ENABLE_TUNING_DEFAULT": "True" if default_enable_tuning else "False",
                "XGB_TUNING_METHOD_DEFAULT": str(default_tuning_method),
                "XGB_CV_FOLDS_DEFAULT": str(int(default_cv_folds)),
                "XGB_CV_SCORING_DEFAULT": str(default_cv_scoring),
                "XGB_CV_N_ITER_DEFAULT": str(int(default_cv_n_iter)),
                "XGB_CV_N_JOBS_DEFAULT": str(int(default_cv_n_jobs)),
            }
        )

        if starter_dataset is not None:
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": starter_dataset["data_file"],
                    "TARGET_COLUMN": starter_dataset["target_column"],
                    "FEATURE_DROP_COLUMNS": starter_dataset["feature_drop_columns"],
                    "TARGET_PREPROCESS": starter_dataset["target_preprocess"],
                    "BOOSTER": booster,
                    "DEVICE": device,
                }
            )
        elif args.task == "regression":
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": "california_housing.csv",
                    "TARGET_COLUMN": "median_house_value",
                    "FEATURE_DROP_COLUMNS": '["median_house_value"]',
                    "TARGET_PREPROCESS": 'y = y.astype("float64")',
                    "BOOSTER": booster,
                    "DEVICE": device,
                }
            )
        elif args.task == "binary_classification":
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": "breast_cancer_wisconsin.csv",
                    "TARGET_COLUMN": "diagnosis",
                    "FEATURE_DROP_COLUMNS": '["diagnosis", "id"]',
                    "TARGET_PREPROCESS": 'y = y.map({"B": 0, "M": 1}).astype("int64")',
                    "BOOSTER": booster,
                    "DEVICE": device,
                }
            )
        else:
            replacements.update(
                {
                    "TASK_VALUE": args.task,
                    "DATA_FILE": "iris.csv",
                    "TARGET_COLUMN": "species",
                    "FEATURE_DROP_COLUMNS": '["species"]',
                    "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                    "BOOSTER": booster,
                    "DEVICE": device,
                }
            )

    if args.library == "tensorflow":
        tf_defaults = DEFAULT_TENSORFLOW_DENSE_PARAMS_BY_TEMPLATE[("tensorflow", "dense_nn", family)]
        default_hidden_layers = (
            _parse_hidden_layers_csv(args.default_tf_hidden_layers, arg_name="--default-tf-hidden-layers")
            if args.default_tf_hidden_layers is not None
            else tf_defaults["hidden_layers"]
        )
        default_dropout = args.default_tf_dropout if args.default_tf_dropout is not None else tf_defaults["dropout"]
        default_hidden_activation = (
            args.default_tf_hidden_activation
            if args.default_tf_hidden_activation is not None
            else tf_defaults["hidden_activation"]
        )
        default_l1 = args.default_tf_l1 if args.default_tf_l1 is not None else tf_defaults["l1"]
        default_l2 = args.default_tf_l2 if args.default_tf_l2 is not None else tf_defaults["l2"]
        default_enable_tuning = args.default_tf_enable_tuning if args.default_tf_enable_tuning is not None else tf_defaults["enable_tuning"]
        default_tuning_method = args.default_tf_tuning_method if args.default_tf_tuning_method is not None else tf_defaults["tuning_method"]
        default_cv_scoring = args.default_tf_cv_scoring if args.default_tf_cv_scoring is not None else tf_defaults["cv_scoring"]
        default_cv_n_iter = args.default_tf_cv_n_iter if args.default_tf_cv_n_iter is not None else tf_defaults["cv_n_iter"]
        default_tuning_optimizer = args.default_tf_tuning_optimizer if args.default_tf_tuning_optimizer is not None else tf_defaults["tuning_optimizer"]
        default_tuning_activation = args.default_tf_tuning_activation if args.default_tf_tuning_activation is not None else tf_defaults["tuning_activation"]
        default_tuning_regularization = args.default_tf_tuning_regularization if args.default_tf_tuning_regularization is not None else tf_defaults["tuning_regularization"]
        replacements.update(
            {
                "OPTIMIZER_CTOR": OPTIMIZER_CLASS_MAP[args.optimizer],
                "OPTIMIZER_NAME": args.optimizer,
                "LEARNING_RATE": str(args.learning_rate),
                "EPOCHS": str(args.epochs),
                "BATCH_SIZE": str(args.batch_size),
                "TF_DEFAULT_HIDDEN_LAYERS": str(default_hidden_layers),
                "TF_DEFAULT_DROPOUT": str(float(default_dropout)),
                "TF_DEFAULT_HIDDEN_ACTIVATION": str(default_hidden_activation),
                "TF_DEFAULT_L1": str(float(default_l1)),
                "TF_DEFAULT_L2": str(float(default_l2)),
                "TF_ENABLE_TUNING_DEFAULT": "True" if default_enable_tuning else "False",
                "TF_TUNING_METHOD_DEFAULT": str(default_tuning_method),
                "TF_CV_SCORING_DEFAULT": str(default_cv_scoring),
                "TF_CV_N_ITER_DEFAULT": str(int(default_cv_n_iter)),
                "TF_TUNING_OPTIMIZER_DEFAULT": str(default_tuning_optimizer),
                "TF_TUNING_ACTIVATION_DEFAULT": str(default_tuning_activation),
                "TF_TUNING_REGULARIZATION_DEFAULT": str(default_tuning_regularization),
            }
        )

        if family == "regression":
            if starter_dataset is not None:
                replacements.update(
                    {
                        "DATA_FILE": starter_dataset["data_file"],
                        "TARGET_COLUMN": starter_dataset["target_column"],
                        "FEATURE_DROP_COLUMNS": starter_dataset["feature_drop_columns"],
                        "TARGET_PREPROCESS": starter_dataset["target_preprocess"],
                    }
                )
            else:
                replacements.update(
                    {
                        "DATA_FILE": "california_housing.csv",
                        "TARGET_COLUMN": "median_house_value",
                        "FEATURE_DROP_COLUMNS": '["median_house_value"]',
                        "TARGET_PREPROCESS": 'y = y.astype("float64")',
                    }
                )

        if family == "classification":
            if starter_dataset is not None:
                output_units = "1" if args.task == "binary_classification" else "3"
                output_activation = "sigmoid" if args.task == "binary_classification" else "softmax"
                loss_fn = "binary_crossentropy" if args.task == "binary_classification" else "sparse_categorical_crossentropy"
                replacements.update(
                    {
                        "TASK_VALUE": args.task,
                        "DATA_FILE": starter_dataset["data_file"],
                        "TARGET_COLUMN": starter_dataset["target_column"],
                        "FEATURE_DROP_COLUMNS": starter_dataset["feature_drop_columns"],
                        "TARGET_PREPROCESS": starter_dataset["target_preprocess"],
                        "OUTPUT_UNITS": output_units,
                        "OUTPUT_ACTIVATION": output_activation,
                        "LOSS_FN": loss_fn,
                    }
                )
            elif args.task == "binary_classification":
                replacements.update(
                    {
                        "TASK_VALUE": args.task,
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
                        "TASK_VALUE": args.task,
                        "DATA_FILE": "iris.csv",
                        "TARGET_COLUMN": "species",
                        "FEATURE_DROP_COLUMNS": '["species"]',
                        "TARGET_PREPROCESS": 'y = y.astype("category").cat.codes.astype("int64")',
                        "OUTPUT_UNITS": "3",
                        "OUTPUT_ACTIVATION": "softmax",
                        "LOSS_FN": "sparse_categorical_crossentropy",
                    }
                )

    if "DATA_FILE" in replacements:
        replacements["DATA_TASK_DIR"] = _task_dataset_dir(args.task)

    if "DATA_FILE" in replacements:
        read_csv_statement = "df = pd.read_csv(data_path)"
        post_read_setup = ""
        if starter_dataset is not None:
            read_csv_extra_args = starter_dataset.get("read_csv_extra_args", "")
            if read_csv_extra_args:
                read_csv_statement = f"df = pd.read_csv(data_path{read_csv_extra_args})"
            header_names = starter_dataset.get("header_names")
            if header_names is not None:
                post_read_setup = f"df.columns = {repr(header_names)}"
        replacements["READ_CSV_STATEMENT"] = read_csv_statement
        replacements["POST_READ_DATASET_SETUP"] = post_read_setup

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

    if args.default_validation_fraction is not None and not (0.0 < args.default_validation_fraction < 1.0):
        raise ValueError("Invalid --default-validation-fraction. Allowed range: 0 < value < 1")
    if args.default_n_iter_no_change is not None and args.default_n_iter_no_change <= 0:
        raise ValueError("Invalid --default-n-iter-no-change. Must be a positive integer")
    if args.default_max_iter is not None and args.default_max_iter <= 0:
        raise ValueError("Invalid --default-max-iter. Must be a positive integer")
    if args.default_n_estimators is not None and args.default_n_estimators <= 0:
        raise ValueError("Invalid --default-n-estimators. Must be a positive integer")
    if args.default_learning_rate is not None and args.default_learning_rate <= 0:
        raise ValueError("Invalid --default-learning-rate. Must be a positive number")
    if args.default_max_depth is not None and args.default_max_depth <= 0:
        raise ValueError("Invalid --default-max-depth. Must be a positive integer")
    if args.default_subsample is not None and not (0.0 < args.default_subsample <= 1.0):
        raise ValueError("Invalid --default-subsample. Allowed range: 0 < value <= 1")
    if args.default_colsample_bytree is not None and not (0.0 < args.default_colsample_bytree <= 1.0):
        raise ValueError("Invalid --default-colsample-bytree. Allowed range: 0 < value <= 1")
    if args.default_c is not None and args.default_c <= 0:
        raise ValueError("Invalid --default-c. Must be a positive number")
    if args.default_solver is not None and args.default_solver not in SKLEARN_LOGISTIC_SOLVERS:
        allowed = ", ".join(sorted(SKLEARN_LOGISTIC_SOLVERS))
        raise ValueError(f"Invalid --default-solver '{args.default_solver}'. Allowed: {allowed}")
    if args.default_rf_n_estimators is not None and args.default_rf_n_estimators <= 0:
        raise ValueError("Invalid --default-rf-n-estimators. Must be a positive integer")
    if args.default_rf_max_depth is not None:
        parsed_default_rf_max_depth = _parse_optional_int_token(
            args.default_rf_max_depth,
            arg_name="--default-rf-max-depth",
        )
        if parsed_default_rf_max_depth is not None and parsed_default_rf_max_depth <= 0:
            raise ValueError("Invalid --default-rf-max-depth. Must be a positive integer or 'none'")
    if args.default_rf_min_samples_split is not None and args.default_rf_min_samples_split < 2:
        raise ValueError("Invalid --default-rf-min-samples-split. Must be >= 2")
    if args.default_rf_min_samples_leaf is not None and args.default_rf_min_samples_leaf < 1:
        raise ValueError("Invalid --default-rf-min-samples-leaf. Must be >= 1")
    if args.default_rf_min_weight_fraction_leaf is not None and not (0.0 <= args.default_rf_min_weight_fraction_leaf <= 0.5):
        raise ValueError("Invalid --default-rf-min-weight-fraction-leaf. Allowed range: 0 <= value <= 0.5")
    if args.default_rf_max_leaf_nodes is not None:
        parsed_default_rf_max_leaf_nodes = _parse_optional_int_token(
            args.default_rf_max_leaf_nodes,
            arg_name="--default-rf-max-leaf-nodes",
        )
        if parsed_default_rf_max_leaf_nodes is not None and parsed_default_rf_max_leaf_nodes < 2:
            raise ValueError("Invalid --default-rf-max-leaf-nodes. Must be >= 2 or 'none'")
    if args.default_rf_min_impurity_decrease is not None and args.default_rf_min_impurity_decrease < 0.0:
        raise ValueError("Invalid --default-rf-min-impurity-decrease. Must be >= 0")
    if args.default_rf_max_features is not None:
        _validate_rf_max_features_default(args.default_rf_max_features, arg_name="--default-rf-max-features")
    if args.default_rf_max_samples is not None:
        _parse_optional_max_samples_token(args.default_rf_max_samples, arg_name="--default-rf-max-samples")
    if args.default_rf_ccp_alpha is not None and args.default_rf_ccp_alpha < 0.0:
        raise ValueError("Invalid --default-rf-ccp-alpha. Must be >= 0")
    if args.default_rf_n_jobs is not None:
        parsed_default_rf_n_jobs = _parse_optional_int_token(
            args.default_rf_n_jobs,
            arg_name="--default-rf-n-jobs",
        )
        if parsed_default_rf_n_jobs == 0:
            raise ValueError("Invalid --default-rf-n-jobs. Must be != 0 or 'none'")
    if args.default_lr_l1_ratio is not None and not (0.0 <= args.default_lr_l1_ratio <= 1.0):
        raise ValueError("Invalid --default-lr-l1-ratio. Allowed range: 0 <= value <= 1")
    if args.default_lr_alpha is not None and args.default_lr_alpha <= 0:
        raise ValueError("Invalid --default-lr-alpha. Must be a positive number")
    if args.default_xgb_min_child_weight is not None and args.default_xgb_min_child_weight <= 0:
        raise ValueError("Invalid --default-xgb-min-child-weight. Must be a positive number")
    if args.default_xgb_reg_lambda is not None and args.default_xgb_reg_lambda < 0:
        raise ValueError("Invalid --default-xgb-reg-lambda. Must be >= 0")
    if args.default_xgb_reg_alpha is not None and args.default_xgb_reg_alpha < 0:
        raise ValueError("Invalid --default-xgb-reg-alpha. Must be >= 0")
    if args.default_lr_cv_folds is not None and args.default_lr_cv_folds < 2:
        raise ValueError("Invalid --default-lr-cv-folds. Must be >= 2")
    if args.default_lr_cv_n_iter is not None and args.default_lr_cv_n_iter <= 0:
        raise ValueError("Invalid --default-lr-cv-n-iter. Must be a positive integer")
    if args.default_lr_cv_n_jobs is not None and args.default_lr_cv_n_jobs == 0:
        raise ValueError("Invalid --default-lr-cv-n-jobs. Must be != 0 (use -1 for all cores)")
    if args.default_logistic_cv_folds is not None and args.default_logistic_cv_folds < 2:
        raise ValueError("Invalid --default-logistic-cv-folds. Must be >= 2")
    if args.default_logistic_cv_n_iter is not None and args.default_logistic_cv_n_iter <= 0:
        raise ValueError("Invalid --default-logistic-cv-n-iter. Must be a positive integer")
    if args.default_logistic_cv_n_jobs is not None and args.default_logistic_cv_n_jobs == 0:
        raise ValueError("Invalid --default-logistic-cv-n-jobs. Must be != 0 (use -1 for all cores)")
    if args.default_rf_cv_folds is not None and args.default_rf_cv_folds < 2:
        raise ValueError("Invalid --default-rf-cv-folds. Must be >= 2")
    if args.default_rf_cv_n_iter is not None and args.default_rf_cv_n_iter <= 0:
        raise ValueError("Invalid --default-rf-cv-n-iter. Must be a positive integer")
    if args.default_rf_cv_n_jobs is not None and args.default_rf_cv_n_jobs == 0:
        raise ValueError("Invalid --default-rf-cv-n-jobs. Must be != 0 (use -1 for all cores)")
    if args.default_xgb_cv_folds is not None and args.default_xgb_cv_folds < 2:
        raise ValueError("Invalid --default-xgb-cv-folds. Must be >= 2")
    if args.default_xgb_cv_n_iter is not None and args.default_xgb_cv_n_iter <= 0:
        raise ValueError("Invalid --default-xgb-cv-n-iter. Must be a positive integer")
    if args.default_xgb_cv_n_jobs is not None and args.default_xgb_cv_n_jobs == 0:
        raise ValueError("Invalid --default-xgb-cv-n-jobs. Must be != 0 (use -1 for all cores)")
    if args.default_tf_cv_n_iter is not None and args.default_tf_cv_n_iter <= 0:
        raise ValueError("Invalid --default-tf-cv-n-iter. Must be a positive integer")
    if args.default_tf_tuning_optimizer is not None and args.default_tf_tuning_optimizer not in TENSORFLOW_OPTIMIZERS:
        allowed = ", ".join(sorted(TENSORFLOW_OPTIMIZERS))
        raise ValueError(f"Invalid --default-tf-tuning-optimizer '{args.default_tf_tuning_optimizer}'. Allowed: {allowed}")
    if args.default_tf_tuning_activation is not None and args.default_tf_tuning_activation not in {"auto", "relu", "gelu", "tanh"}:
        raise ValueError("Invalid --default-tf-tuning-activation. Allowed: auto, relu, gelu, tanh")
    if args.default_tf_tuning_regularization is not None and args.default_tf_tuning_regularization not in {"auto", "none", "l1", "l2", "l1_l2"}:
        raise ValueError("Invalid --default-tf-tuning-regularization. Allowed: auto, none, l1, l2, l1_l2")
    if args.default_tf_hidden_layers is not None:
        _parse_hidden_layers_csv(args.default_tf_hidden_layers, arg_name="--default-tf-hidden-layers")
    if args.default_tf_dropout is not None and not (0.0 <= args.default_tf_dropout < 1.0):
        raise ValueError("Invalid --default-tf-dropout. Allowed range: 0 <= value < 1")
    if args.default_tf_l1 is not None and args.default_tf_l1 < 0.0:
        raise ValueError("Invalid --default-tf-l1. Must be >= 0")
    if args.default_tf_l2 is not None and args.default_tf_l2 < 0.0:
        raise ValueError("Invalid --default-tf-l2. Must be >= 0")

    if args.starter_dataset is not None:
        allowed = STARTER_DATASETS_BY_FAMILY[args.task]
        if args.starter_dataset not in allowed:
            allowed_str = ", ".join(allowed)
            raise ValueError(
                f"Invalid --starter-dataset '{args.starter_dataset}' for task '{args.task}'. Allowed: {allowed_str}"
            )

    early_stopping_defaults_provided = any(
        value is not None
        for value in (
            args.default_early_stopping,
            args.default_validation_fraction,
            args.default_n_iter_no_change,
        )
    )

    max_iter_default_provided = args.default_max_iter is not None

    if args.library == "xgboost":
        family = task_family(args.task)
        if args.model is not None:
            raise ValueError("Invalid flags: xgboost does not use --model. Omit --model entirely.")
        if args.booster is not None and args.booster not in XGBOOST_BOOSTERS:
            allowed = ", ".join(sorted(XGBOOST_BOOSTERS))
            raise ValueError(f"Invalid xgboost --booster '{args.booster}'. Allowed: {allowed}")
        if args.device is not None and args.device not in XGBOOST_DEVICES:
            allowed = ", ".join(sorted(XGBOOST_DEVICES))
            raise ValueError(f"Invalid xgboost --device '{args.device}'. Allowed: {allowed}")
        if args.optimizer is not None or args.learning_rate is not None or args.epochs is not None or args.batch_size is not None:
            raise ValueError("Invalid flags: --optimizer/--learning_rate/--epochs/--batch_size are tensorflow-only")
        if early_stopping_defaults_provided and not _supports_early_stopping_defaults(args):
            raise ValueError(
                "Invalid flags: --default-early-stopping/--default-validation-fraction/"
                "--default-n-iter-no-change are not supported for this template"
            )
        if max_iter_default_provided:
            raise ValueError("Invalid flag: --default-max-iter is not supported for xgboost")
        if args.default_c is not None or args.default_solver is not None:
            raise ValueError("Invalid flags: --default-c/--default-solver are scikit-learn logistic-only")
        if (
            args.default_rf_n_estimators is not None
            or args.default_rf_max_depth is not None
            or args.default_rf_min_samples_split is not None
            or args.default_rf_min_samples_leaf is not None
            or args.default_rf_min_weight_fraction_leaf is not None
            or args.default_rf_max_leaf_nodes is not None
            or args.default_rf_min_impurity_decrease is not None
            or args.default_rf_max_features is not None
            or args.default_rf_bootstrap is not None
            or args.default_rf_max_samples is not None
            or args.default_rf_ccp_alpha is not None
            or args.default_rf_n_jobs is not None
        ):
            raise ValueError(
                "Invalid flags: --default-rf-n-estimators/--default-rf-max-depth/"
                "--default-rf-min-samples-split/--default-rf-min-samples-leaf/--default-rf-min-weight-fraction-leaf/"
                "--default-rf-max-leaf-nodes/--default-rf-min-impurity-decrease/--default-rf-max-features/"
                "--default-rf-bootstrap/--default-rf-max-samples/--default-rf-ccp-alpha/--default-rf-n-jobs are scikit-learn random_forest-only"
            )
        if (
            args.default_lr_penalty is not None
            or args.default_lr_alpha is not None
            or args.default_lr_fit_intercept is not None
            or args.default_lr_l1_ratio is not None
            or args.default_lr_enable_tuning is not None
            or args.default_lr_tuning_method is not None
            or args.default_lr_cv_folds is not None
            or args.default_lr_cv_scoring is not None
            or args.default_lr_cv_n_iter is not None
            or args.default_lr_cv_n_jobs is not None
        ):
            raise ValueError(
                "Invalid flags: --default-lr-penalty/--default-lr-alpha/--default-lr-fit-intercept/--default-lr-l1-ratio/"
                "--default-lr-enable-tuning/--default-lr-tuning-method/--default-lr-cv-folds/--default-lr-cv-scoring/"
                "--default-lr-cv-n-iter/--default-lr-cv-n-jobs are scikit-learn linear_regression-only"
            )
        if (
            args.default_logistic_enable_tuning is not None
            or args.default_logistic_tuning_method is not None
            or args.default_logistic_cv_folds is not None
            or args.default_logistic_cv_scoring is not None
            or args.default_logistic_cv_n_iter is not None
            or args.default_logistic_cv_n_jobs is not None
            or args.default_rf_enable_tuning is not None
            or args.default_rf_tuning_method is not None
            or args.default_rf_cv_folds is not None
            or args.default_rf_cv_scoring is not None
            or args.default_rf_cv_n_iter is not None
            or args.default_rf_cv_n_jobs is not None
        ):
            raise ValueError(
                "Invalid flags: scikit-learn tuning defaults are not supported for xgboost"
            )
        if args.default_xgb_tuning_method is not None and args.default_xgb_tuning_method not in {"grid", "random"}:
            raise ValueError("Invalid --default-xgb-tuning-method. Allowed: grid, random")
        if args.default_xgb_cv_scoring is not None:
            allowed_scoring = REGRESSION_CV_SCORINGS if family == "regression" else CLASSIFICATION_CV_SCORINGS
            if args.default_xgb_cv_scoring not in allowed_scoring:
                allowed = ", ".join(sorted(allowed_scoring))
                raise ValueError(
                    f"Invalid --default-xgb-cv-scoring '{args.default_xgb_cv_scoring}' for task '{args.task}'. "
                    f"Allowed: {allowed}"
                )
        if (
            args.default_tf_enable_tuning is not None
            or args.default_tf_tuning_method is not None
            or args.default_tf_cv_scoring is not None
            or args.default_tf_cv_n_iter is not None
            or args.default_tf_tuning_optimizer is not None
            or args.default_tf_tuning_activation is not None
            or args.default_tf_tuning_regularization is not None
            or args.default_tf_hidden_layers is not None
            or args.default_tf_hidden_activation is not None
            or args.default_tf_dropout is not None
            or args.default_tf_l1 is not None
            or args.default_tf_l2 is not None
        ):
            raise ValueError(
                "Invalid flags: tensorflow defaults are tensorflow-only"
            )
        return

    if args.library == "scikit-learn":
        family = task_family(args.task)
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
        if args.device is not None:
            raise ValueError("Invalid flags: --device is xgboost-only")
        if args.optimizer is not None or args.learning_rate is not None or args.epochs is not None or args.batch_size is not None:
            raise ValueError("Invalid flags: --optimizer/--learning_rate/--epochs/--batch_size are tensorflow-only")
        if early_stopping_defaults_provided and not _supports_early_stopping_defaults(args):
            raise ValueError(
                "Invalid flags: --default-early-stopping/--default-validation-fraction/"
                "--default-n-iter-no-change are not supported for this template"
            )
        if max_iter_default_provided and not _supports_max_iter_default(args):
            raise ValueError(
                "Invalid flag: --default-max-iter is only supported for scikit-learn logistic_regression classification templates"
            )
        if (
            args.default_n_estimators is not None
            or args.default_learning_rate is not None
            or args.default_max_depth is not None
            or args.default_subsample is not None
            or args.default_colsample_bytree is not None
            or args.default_xgb_min_child_weight is not None
            or args.default_xgb_reg_lambda is not None
            or args.default_xgb_reg_alpha is not None
        ):
            raise ValueError(
                "Invalid flags: --default-n-estimators/--default-learning-rate/--default-max-depth/"
                "--default-subsample/--default-colsample-bytree/--default-xgb-min-child-weight/"
                "--default-xgb-reg-lambda/--default-xgb-reg-alpha are xgboost-only"
            )

        if args.model == "logistic_regression":
            logistic_defaults = DEFAULT_LOGISTIC_PARAMS_BY_TEMPLATE[("scikit-learn", "logistic_regression", family)]
            effective_logistic_enable_tuning = (
                args.default_logistic_enable_tuning
                if args.default_logistic_enable_tuning is not None
                else logistic_defaults["enable_tuning"]
            )
            effective_logistic_penalty = (
                args.default_logistic_penalty
                if args.default_logistic_penalty is not None
                else logistic_defaults["penalty"]
            )
            if effective_logistic_enable_tuning and str(effective_logistic_penalty).strip().lower() == "none":
                raise ValueError(
                    "Invalid default combination: --default-logistic-penalty=none is not allowed when "
                    "--default-logistic-enable-tuning=true"
                )
            if (
                args.default_rf_n_estimators is not None
                or args.default_rf_max_depth is not None
                or args.default_rf_min_samples_split is not None
                or args.default_rf_min_samples_leaf is not None
                or args.default_rf_min_weight_fraction_leaf is not None
                or args.default_rf_max_leaf_nodes is not None
                or args.default_rf_min_impurity_decrease is not None
                or args.default_rf_max_features is not None
                or args.default_rf_bootstrap is not None
                or args.default_rf_max_samples is not None
                or args.default_rf_ccp_alpha is not None
                or args.default_rf_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-rf-n-estimators/--default-rf-max-depth/"
                    "--default-rf-min-samples-split/--default-rf-min-samples-leaf/--default-rf-min-weight-fraction-leaf/"
                    "--default-rf-max-leaf-nodes/--default-rf-min-impurity-decrease/--default-rf-max-features/"
                    "--default-rf-bootstrap/--default-rf-max-samples/--default-rf-ccp-alpha/--default-rf-n-jobs are scikit-learn random_forest-only"
                )
            if (
                args.default_lr_penalty is not None
                or args.default_lr_alpha is not None
                or args.default_lr_fit_intercept is not None
                or args.default_lr_l1_ratio is not None
                or args.default_lr_enable_tuning is not None
                or args.default_lr_tuning_method is not None
                or args.default_lr_cv_folds is not None
                or args.default_lr_cv_scoring is not None
                or args.default_lr_cv_n_iter is not None
                or args.default_lr_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-lr-penalty/--default-lr-alpha/--default-lr-fit-intercept/--default-lr-l1-ratio/"
                    "--default-lr-enable-tuning/--default-lr-tuning-method/--default-lr-cv-folds/--default-lr-cv-scoring/"
                    "--default-lr-cv-n-iter/--default-lr-cv-n-jobs are scikit-learn linear_regression-only"
                )
            if (
                args.default_rf_enable_tuning is not None
                or args.default_rf_tuning_method is not None
                or args.default_rf_cv_folds is not None
                or args.default_rf_cv_scoring is not None
                or args.default_rf_cv_n_iter is not None
                or args.default_rf_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-rf-enable-tuning/--default-rf-tuning-method/--default-rf-cv-folds/"
                    "--default-rf-cv-scoring/--default-rf-cv-n-iter/--default-rf-cv-n-jobs are scikit-learn random_forest-only"
                )
            if args.default_logistic_tuning_method is not None and args.default_logistic_tuning_method not in {"grid", "random"}:
                raise ValueError("Invalid --default-logistic-tuning-method. Allowed: grid, random")
        elif args.model == "linear_regression":
            linear_defaults = DEFAULT_LINEAR_REGRESSION_PARAMS_BY_TEMPLATE[("scikit-learn", "linear_regression", family)]
            effective_lr_enable_tuning = (
                args.default_lr_enable_tuning
                if args.default_lr_enable_tuning is not None
                else linear_defaults["enable_tuning"]
            )
            effective_lr_penalty = (
                args.default_lr_penalty
                if args.default_lr_penalty is not None
                else linear_defaults["penalty"]
            )
            if effective_lr_enable_tuning and str(effective_lr_penalty).strip().lower() == "none":
                raise ValueError(
                    "Invalid default combination: --default-lr-penalty=none is not allowed when "
                    "--default-lr-enable-tuning=true"
                )
            if args.default_c is not None or args.default_solver is not None or args.default_logistic_penalty is not None or args.default_logistic_class_weight is not None:
                raise ValueError("Invalid flags: --default-c/--default-solver/--default-logistic-penalty/--default-logistic-class-weight are scikit-learn logistic-only")
            if (
                args.default_rf_n_estimators is not None
                or args.default_rf_max_depth is not None
                or args.default_rf_min_samples_leaf is not None
                or args.default_rf_max_features is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-rf-n-estimators/--default-rf-max-depth/"
                    "--default-rf-min-samples-leaf/--default-rf-max-features are scikit-learn random_forest-only"
                )
            if (
                args.default_logistic_enable_tuning is not None
                or args.default_logistic_tuning_method is not None
                or args.default_logistic_cv_folds is not None
                or args.default_logistic_cv_scoring is not None
                or args.default_logistic_cv_n_iter is not None
                or args.default_logistic_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-logistic-enable-tuning/--default-logistic-tuning-method/--default-logistic-cv-folds/"
                    "--default-logistic-cv-scoring/--default-logistic-cv-n-iter/--default-logistic-cv-n-jobs are scikit-learn logistic-only"
                )
            if (
                args.default_rf_enable_tuning is not None
                or args.default_rf_tuning_method is not None
                or args.default_rf_cv_folds is not None
                or args.default_rf_cv_scoring is not None
                or args.default_rf_cv_n_iter is not None
                or args.default_rf_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-rf-enable-tuning/--default-rf-tuning-method/--default-rf-cv-folds/"
                    "--default-rf-cv-scoring/--default-rf-cv-n-iter/--default-rf-cv-n-jobs are scikit-learn random_forest-only"
                )
            if args.default_lr_tuning_method is not None and args.default_lr_tuning_method not in {"grid", "random"}:
                raise ValueError("Invalid --default-lr-tuning-method. Allowed: grid, random")
            if args.default_lr_cv_scoring is not None and args.default_lr_cv_scoring not in {"rmse", "mae", "r2"}:
                raise ValueError("Invalid --default-lr-cv-scoring. Allowed: rmse, mae, r2")
        else:
            if args.default_c is not None or args.default_solver is not None or args.default_logistic_penalty is not None or args.default_logistic_class_weight is not None:
                raise ValueError("Invalid flags: --default-c/--default-solver/--default-logistic-penalty/--default-logistic-class-weight are scikit-learn logistic-only")
            if (
                args.default_lr_penalty is not None
                or args.default_lr_alpha is not None
                or args.default_lr_fit_intercept is not None
                or args.default_lr_l1_ratio is not None
                or args.default_lr_enable_tuning is not None
                or args.default_lr_tuning_method is not None
                or args.default_lr_cv_folds is not None
                or args.default_lr_cv_scoring is not None
                or args.default_lr_cv_n_iter is not None
                or args.default_lr_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-lr-penalty/--default-lr-alpha/--default-lr-fit-intercept/--default-lr-l1-ratio/"
                    "--default-lr-enable-tuning/--default-lr-tuning-method/--default-lr-cv-folds/--default-lr-cv-scoring/"
                    "--default-lr-cv-n-iter/--default-lr-cv-n-jobs are scikit-learn linear_regression-only"
                )
            if (
                args.default_logistic_enable_tuning is not None
                or args.default_logistic_tuning_method is not None
                or args.default_logistic_cv_folds is not None
                or args.default_logistic_cv_scoring is not None
                or args.default_logistic_cv_n_iter is not None
                or args.default_logistic_cv_n_jobs is not None
            ):
                raise ValueError(
                    "Invalid flags: --default-logistic-enable-tuning/--default-logistic-tuning-method/--default-logistic-cv-folds/"
                    "--default-logistic-cv-scoring/--default-logistic-cv-n-iter/--default-logistic-cv-n-jobs are scikit-learn logistic-only"
                )
            if args.default_rf_tuning_method is not None and args.default_rf_tuning_method not in {"grid", "random"}:
                raise ValueError("Invalid --default-rf-tuning-method. Allowed: grid, random")
            if args.default_rf_cv_scoring is not None:
                allowed_scoring = REGRESSION_CV_SCORINGS if family == "regression" else CLASSIFICATION_CV_SCORINGS
                if args.default_rf_cv_scoring not in allowed_scoring:
                    allowed = ", ".join(sorted(allowed_scoring))
                    raise ValueError(
                        f"Invalid --default-rf-cv-scoring '{args.default_rf_cv_scoring}' for task '{args.task}'. "
                        f"Allowed: {allowed}"
                    )
            if args.default_rf_bootstrap is False and args.default_rf_max_samples is not None:
                raise ValueError("Invalid default combination: --default-rf-max-samples requires --default-rf-bootstrap=true")

        if (
            args.default_xgb_enable_tuning is not None
            or args.default_xgb_tuning_method is not None
            or args.default_xgb_cv_folds is not None
            or args.default_xgb_cv_scoring is not None
            or args.default_xgb_cv_n_iter is not None
            or args.default_xgb_cv_n_jobs is not None
            or args.default_tf_enable_tuning is not None
            or args.default_tf_tuning_method is not None
            or args.default_tf_cv_scoring is not None
            or args.default_tf_cv_n_iter is not None
            or args.default_tf_tuning_optimizer is not None
            or args.default_tf_tuning_activation is not None
            or args.default_tf_tuning_regularization is not None
            or args.default_tf_hidden_layers is not None
            or args.default_tf_hidden_activation is not None
            or args.default_tf_dropout is not None
            or args.default_tf_l1 is not None
            or args.default_tf_l2 is not None
        ):
            raise ValueError("Invalid flags: xgboost/tensorflow defaults are not supported for scikit-learn")

        return

    if args.library == "tensorflow":
        family = task_family(args.task)
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
        if args.device is not None:
            raise ValueError("Invalid flags: --device is xgboost-only")
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
        if early_stopping_defaults_provided and not _supports_early_stopping_defaults(args):
            raise ValueError(
                "Invalid flags: --default-early-stopping/--default-validation-fraction/"
                "--default-n-iter-no-change are not supported for this template"
            )
        if max_iter_default_provided:
            raise ValueError("Invalid flag: --default-max-iter is not supported for tensorflow")
        if (
            args.default_n_estimators is not None
            or args.default_learning_rate is not None
            or args.default_max_depth is not None
            or args.default_subsample is not None
            or args.default_colsample_bytree is not None
            or args.default_xgb_min_child_weight is not None
            or args.default_xgb_reg_lambda is not None
            or args.default_xgb_reg_alpha is not None
            or args.default_c is not None
            or args.default_solver is not None
            or args.default_logistic_penalty is not None
            or args.default_logistic_class_weight is not None
            or args.default_rf_n_estimators is not None
            or args.default_rf_max_depth is not None
            or args.default_rf_min_samples_split is not None
            or args.default_rf_min_samples_leaf is not None
            or args.default_rf_min_weight_fraction_leaf is not None
            or args.default_rf_max_leaf_nodes is not None
            or args.default_rf_min_impurity_decrease is not None
            or args.default_rf_max_features is not None
            or args.default_rf_bootstrap is not None
            or args.default_rf_max_samples is not None
            or args.default_rf_ccp_alpha is not None
            or args.default_rf_n_jobs is not None
            or args.default_lr_penalty is not None
            or args.default_lr_alpha is not None
            or args.default_lr_fit_intercept is not None
            or args.default_lr_l1_ratio is not None
            or args.default_logistic_enable_tuning is not None
            or args.default_logistic_tuning_method is not None
            or args.default_logistic_cv_folds is not None
            or args.default_logistic_cv_scoring is not None
            or args.default_logistic_cv_n_iter is not None
            or args.default_logistic_cv_n_jobs is not None
            or args.default_rf_enable_tuning is not None
            or args.default_rf_tuning_method is not None
            or args.default_rf_cv_folds is not None
            or args.default_rf_cv_scoring is not None
            or args.default_rf_cv_n_iter is not None
            or args.default_rf_cv_n_jobs is not None
            or args.default_xgb_enable_tuning is not None
            or args.default_xgb_tuning_method is not None
            or args.default_xgb_cv_folds is not None
            or args.default_xgb_cv_scoring is not None
            or args.default_xgb_cv_n_iter is not None
            or args.default_xgb_cv_n_jobs is not None
        ):
            raise ValueError(
                "Invalid flags: xgboost/scikit-learn default hyperparameter flags are not supported for tensorflow"
            )
        if args.default_tf_cv_scoring is not None:
            allowed_scoring = {"rmse"} if family == "regression" else {"f1_macro"}
            if args.default_tf_cv_scoring not in allowed_scoring:
                allowed = ", ".join(sorted(allowed_scoring))
                raise ValueError(
                    f"Invalid --default-tf-cv-scoring '{args.default_tf_cv_scoring}' for task '{args.task}'. "
                    f"Allowed: {allowed}"
                )
        if args.default_tf_tuning_method is not None and args.default_tf_tuning_method not in {"grid", "random"}:
            raise ValueError("Invalid --default-tf-tuning-method. Allowed: grid, random")
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
        choices=["auto", "gbtree", "gblinear", "dart"],
        help="xgboost booster (optional; xgboost only)",
    )
    parser.add_argument(
        "--device",
        required=False,
        choices=["auto", "cpu", "gpu"],
        help="xgboost device (optional; xgboost only)",
    )
    parser.add_argument(
        "--default-max-iter",
        type=int,
        required=False,
        help="Default value injected into generated logistic classification template --max-iter flag",
    )
    parser.add_argument(
        "--default-c",
        type=float,
        required=False,
        help="Default value injected into generated logistic classification template --c flag",
    )
    parser.add_argument(
        "--default-solver",
        required=False,
        choices=["auto", "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        help="Default value injected into generated logistic classification template --solver flag",
    )
    parser.add_argument(
        "--default-rf-n-estimators",
        type=int,
        required=False,
        help="Default value injected into generated random forest template --n-estimators flag",
    )
    parser.add_argument(
        "--default-rf-max-depth",
        required=False,
        help="Default value injected into generated random forest template --max-depth flag (integer or 'none')",
    )
    parser.add_argument(
        "--default-rf-min-samples-split",
        type=int,
        required=False,
        help="Default value injected into generated random forest template --min-samples-split flag",
    )
    parser.add_argument(
        "--default-rf-min-weight-fraction-leaf",
        type=float,
        required=False,
        help="Default value injected into generated random forest template --min-weight-fraction-leaf flag",
    )
    parser.add_argument(
        "--default-rf-max-leaf-nodes",
        required=False,
        help="Default value injected into generated random forest template --max-leaf-nodes flag (integer or 'none')",
    )
    parser.add_argument(
        "--default-rf-min-impurity-decrease",
        type=float,
        required=False,
        help="Default value injected into generated random forest template --min-impurity-decrease flag",
    )
    parser.add_argument(
        "--default-rf-min-samples-leaf",
        type=int,
        required=False,
        help="Default value injected into generated random forest template --min-samples-leaf flag",
    )
    parser.add_argument(
        "--default-rf-max-features",
        required=False,
        help="Default value injected into generated random forest template --max-features flag (e.g. sqrt, log2, 1.0)",
    )
    parser.add_argument(
        "--default-rf-bootstrap",
        type=_parse_bool,
        required=False,
        help="Default value injected into generated random forest template --bootstrap flag",
    )
    parser.add_argument(
        "--default-rf-max-samples",
        required=False,
        help="Default value injected into generated random forest template --max-samples flag (int, float, or 'none')",
    )
    parser.add_argument(
        "--default-rf-ccp-alpha",
        type=float,
        required=False,
        help="Default value injected into generated random forest template --ccp-alpha flag",
    )
    parser.add_argument(
        "--default-rf-n-jobs",
        required=False,
        help="Default value injected into generated random forest template --n-jobs flag (integer or 'none')",
    )
    parser.add_argument(
        "--default-n-estimators",
        type=int,
        required=False,
        help="Default value injected into generated xgboost template --n-estimators flag",
    )
    parser.add_argument(
        "--default-learning-rate",
        type=float,
        required=False,
        help="Default value injected into generated xgboost template --learning-rate flag",
    )
    parser.add_argument(
        "--default-max-depth",
        type=int,
        required=False,
        help="Default value injected into generated xgboost template --max-depth flag",
    )
    parser.add_argument(
        "--default-subsample",
        type=float,
        required=False,
        help="Default value injected into generated xgboost template --subsample flag",
    )
    parser.add_argument(
        "--default-colsample-bytree",
        type=float,
        required=False,
        help="Default value injected into generated xgboost template --colsample-bytree flag",
    )
    parser.add_argument(
        "--default-early-stopping",
        type=_parse_bool,
        required=False,
        help="Default value injected into generated template --early-stopping flag",
    )
    parser.add_argument(
        "--default-validation-fraction",
        type=float,
        required=False,
        help="Default value injected into generated template --validation-fraction flag",
    )
    parser.add_argument(
        "--default-n-iter-no-change",
        type=int,
        required=False,
        help="Default value injected into generated template --n-iter-no-change flag",
    )
    parser.add_argument(
        "--default-lr-penalty",
        required=False,
        choices=["auto", "none", "l1", "l2", "elasticnet"],
        help="Default regularization penalty for linear regression template",
    )
    parser.add_argument(
        "--default-lr-alpha",
        type=float,
        required=False,
        help="Default regularization strength (alpha) for linear regression template",
    )
    parser.add_argument(
        "--default-lr-fit-intercept",
        type=_parse_bool,
        required=False,
        help="Default fit_intercept for linear regression template",
    )
    parser.add_argument(
        "--default-lr-l1-ratio",
        type=float,
        required=False,
        help="Default l1_ratio for linear regression ElasticNet (0-1)",
    )
    parser.add_argument(
        "--default-lr-enable-tuning",
        type=_parse_bool,
        required=False,
        help="Default enable_tuning for linear regression template",
    )
    parser.add_argument(
        "--default-lr-tuning-method",
        required=False,
        choices=["grid", "random"],
        help="Default tuning method for linear regression template",
    )
    parser.add_argument(
        "--default-lr-cv-folds",
        type=int,
        required=False,
        help="Default cross-validation folds for linear regression template",
    )
    parser.add_argument(
        "--default-lr-cv-scoring",
        required=False,
        choices=["rmse", "mae", "r2"],
        help="Default CV scoring for linear regression template",
    )
    parser.add_argument(
        "--default-lr-cv-n-iter",
        type=int,
        required=False,
        help="Default random-search iterations for linear regression template",
    )
    parser.add_argument(
        "--default-lr-cv-n-jobs",
        type=int,
        required=False,
        help="Default n_jobs for linear regression tuning CV",
    )
    parser.add_argument(
        "--default-logistic-penalty",
        required=False,
        choices=["auto", "none", "l1", "l2", "elasticnet"],
        help="Default penalty for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-class-weight",
        required=False,
        choices=["none", "balanced"],
        help="Default class_weight for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-enable-tuning",
        type=_parse_bool,
        required=False,
        help="Default enable_tuning for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-tuning-method",
        required=False,
        choices=["grid", "random"],
        help="Default tuning method for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-cv-folds",
        type=int,
        required=False,
        help="Default CV folds for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-cv-scoring",
        required=False,
        choices=["f1_macro", "accuracy", "roc_auc_ovr"],
        help="Default CV scoring for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-cv-n-iter",
        type=int,
        required=False,
        help="Default random-search iterations for logistic regression template",
    )
    parser.add_argument(
        "--default-logistic-cv-n-jobs",
        type=int,
        required=False,
        help="Default n_jobs for logistic regression tuning CV",
    )
    parser.add_argument(
        "--default-rf-enable-tuning",
        type=_parse_bool,
        required=False,
        help="Default enable_tuning for random forest templates",
    )
    parser.add_argument(
        "--default-rf-tuning-method",
        required=False,
        choices=["grid", "random"],
        help="Default tuning method for random forest templates",
    )
    parser.add_argument(
        "--default-rf-cv-folds",
        type=int,
        required=False,
        help="Default CV folds for random forest templates",
    )
    parser.add_argument(
        "--default-rf-cv-scoring",
        required=False,
        choices=["f1_macro", "accuracy", "roc_auc_ovr", "rmse", "mae", "r2"],
        help="Default CV scoring for random forest templates",
    )
    parser.add_argument(
        "--default-rf-cv-n-iter",
        type=int,
        required=False,
        help="Default random-search iterations for random forest templates",
    )
    parser.add_argument(
        "--default-rf-cv-n-jobs",
        type=int,
        required=False,
        help="Default n_jobs for random forest tuning CV",
    )
    parser.add_argument(
        "--default-xgb-min-child-weight",
        type=float,
        required=False,
        help="Default min_child_weight for xgboost template",
    )
    parser.add_argument(
        "--default-xgb-reg-lambda",
        type=float,
        required=False,
        help="Default reg_lambda (L2) for xgboost template",
    )
    parser.add_argument(
        "--default-xgb-reg-alpha",
        type=float,
        required=False,
        help="Default reg_alpha (L1) for xgboost template",
    )
    parser.add_argument(
        "--default-xgb-enable-tuning",
        type=_parse_bool,
        required=False,
        help="Default enable_tuning for xgboost templates",
    )
    parser.add_argument(
        "--default-xgb-tuning-method",
        required=False,
        choices=["grid", "random"],
        help="Default tuning method for xgboost templates",
    )
    parser.add_argument(
        "--default-xgb-cv-folds",
        type=int,
        required=False,
        help="Default CV folds for xgboost templates",
    )
    parser.add_argument(
        "--default-xgb-cv-scoring",
        required=False,
        choices=["f1_macro", "accuracy", "roc_auc_ovr", "rmse", "mae", "r2"],
        help="Default CV scoring for xgboost templates",
    )
    parser.add_argument(
        "--default-xgb-cv-n-iter",
        type=int,
        required=False,
        help="Default random-search iterations for xgboost templates",
    )
    parser.add_argument(
        "--default-xgb-cv-n-jobs",
        type=int,
        required=False,
        help="Default n_jobs for xgboost tuning CV",
    )
    parser.add_argument(
        "--default-tf-enable-tuning",
        type=_parse_bool,
        required=False,
        help="Default enable_tuning for tensorflow dense templates",
    )
    parser.add_argument(
        "--default-tf-tuning-method",
        required=False,
        choices=["grid", "random"],
        help="Default tuning method for tensorflow dense templates",
    )
    parser.add_argument(
        "--default-tf-cv-scoring",
        required=False,
        choices=["f1_macro", "rmse"],
        help="Default tuning scoring for tensorflow dense templates",
    )
    parser.add_argument(
        "--default-tf-cv-n-iter",
        type=int,
        required=False,
        help="Default random-search iterations for tensorflow dense templates",
    )
    parser.add_argument(
        "--default-tf-tuning-optimizer",
        required=False,
        choices=["auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"],
        help="Default optimizer filter for tensorflow tuning (auto searches all)",
    )
    parser.add_argument(
        "--default-tf-tuning-activation",
        required=False,
        choices=["auto", "relu", "gelu", "tanh"],
        help="Default activation filter for tensorflow tuning (auto searches all)",
    )
    parser.add_argument(
        "--default-tf-tuning-regularization",
        required=False,
        choices=["auto", "none", "l1", "l2", "l1_l2"],
        help="Default regularization filter for tensorflow tuning",
    )
    parser.add_argument(
        "--default-tf-hidden-layers",
        required=False,
        help="Default hidden-layer units for non-tuning tensorflow runs (comma-separated, e.g. 128,64,32)",
    )
    parser.add_argument(
        "--default-tf-hidden-activation",
        required=False,
        choices=["relu", "gelu", "tanh"],
        help="Default hidden-layer activation for non-tuning tensorflow runs",
    )
    parser.add_argument(
        "--default-tf-dropout",
        type=float,
        required=False,
        help="Default dropout for non-tuning tensorflow runs (0 <= value < 1)",
    )
    parser.add_argument(
        "--default-tf-l1",
        type=float,
        required=False,
        help="Default L1 regularization for non-tuning tensorflow runs (>= 0)",
    )
    parser.add_argument(
        "--default-tf-l2",
        type=float,
        required=False,
        help="Default L2 regularization for non-tuning tensorflow runs (>= 0)",
    )
    parser.add_argument(
        "--optimizer",
        required=False,
        choices=["auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"],
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
        "--starter-dataset",
        required=False,
        help=(
            "Optional starter dataset file name for the chosen task family. "
            "Examples: ames_housing.csv, breast_cancer_wisconsin.csv, iris.csv"
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional explicit output path. If omitted, writes to <repo>/models/<name>.py",
    )

    args = parser.parse_args()

    try:
        validate_shared_helper_modules()
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

        if output_path.exists():
            raise ValueError(
                f"A model file already exists for --name '{args.name}': {output_path.resolve()}\n"
                "Pick a new --name (for example, --name your_model_v2) or provide --output to a different path."
            )

        output_path.write_text(content.strip() + "\n", encoding="utf-8")
        print(f"Generated file: {output_path.resolve()}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
