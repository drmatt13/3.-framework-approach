import subprocess
import sys
from pathlib import Path
from prompt_toolkit.styles import Style

try:
    import questionary
except ImportError:
    print("Missing dependency: questionary. Install with: pip install questionary", file=sys.stderr)
    sys.exit(1)

SKLEARN_MODELS = ["linear_regression", "logistic_regression", "random_forest"]
TENSORFLOW_MODELS = ["dense_nn"]

# Valid tasks per model/library (based on your flag combinations)
TASKS_BY_LIBRARY_MODEL = {
    ("scikit-learn", "linear_regression"): ["regression"],
    ("scikit-learn", "logistic_regression"): ["binary_classification", "multiclass_classification"],
    ("scikit-learn", "random_forest"): ["regression", "binary_classification", "multiclass_classification"],
    ("tensorflow", "dense_nn"): ["regression", "binary_classification", "multiclass_classification"],
    # xgboost has no --model in your interface; task applies directly to library
    ("xgboost", None): ["regression", "binary_classification", "multiclass_classification"],
}

XGBOOST_BOOSTERS = ["gbtree", "gblinear", "dart"]
XGBOOST_DEVICE_DEFAULTS = ["cpu", "gpu"]
SKLEARN_LOGISTIC_SOLVERS = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]

# Only meaningful for TensorFlow models (gradient-based training)
TENSORFLOW_OPTIMIZERS = ["adam", "sgd", "rmsprop", "adagrad", "adamw"]

STARTER_DATASETS_BY_TASK = {
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

CUSTOM_STYLE = Style.from_dict(
    {
        "qmark": "fg:#f8b808 bold",  # Question mark
        "question": "bold",  # Question text
        "answer": "fg:#3fb0f0 bold",  # Selected answer after choice
        "pointer": "fg:#f8b808 bold",  # Arrow pointer (>)
        "highlighted": "fg:#ffffff bg:#222222 bold",  # Highlighted option in menu
        "selected": "fg:#8ab4f8 bold",  # Selected checkbox item (if used)
    }
)

# Keep defaulted select prompts visually consistent with regular selects.
# questionary.select with `default=...` uses the `selected` token for the
# preselected item; using a non-blue style here avoids sticky blue highlights.
DEFAULT_SELECT_STYLE = Style.from_dict(
    {
        "qmark": "fg:#f8b808 bold",  # Question mark
        "question": "bold",  # Question text
        "answer": "fg:#3fb0f0 bold",  # Selected answer after choice
        "pointer": "fg:#f8b808 bold",  # Arrow pointer (>)
        "highlighted": "fg:#ffffff bg:#222222 bold",  # Highlighted option in menu
        "selected": "fg:#8ab4f8 bold",  # Selected checkbox item (if used)
    }
)


def _ask_text(prompt: str, *, validate_fn, default: str | None = None) -> str | None:
    """
    Wrapper to apply consistent style + validation to text prompts.

    IMPORTANT: questionary/prompt_toolkit can crash if default=None is passed.
    So we only pass `default` when it's a real string.
    """
    kwargs = {
        "validate": validate_fn,
        "style": CUSTOM_STYLE,
    }
    if default is not None:
        kwargs["default"] = default
    return questionary.text(prompt, **kwargs).ask()


def _ask_select(prompt: str, *, choices, default: str | None = None):
    """Wrapper to apply consistent style + behavior to select prompts."""
    kwargs = {
        "choices": choices,
        "use_shortcuts": True,
        "style": DEFAULT_SELECT_STYLE if default is not None else CUSTOM_STYLE,
    }
    if default is not None:
        kwargs["default"] = default
    return questionary.select(prompt, **kwargs).ask()


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _stringify_setting(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _should_omit_resolved_key(key: str, values_by_key: dict[str, object]) -> bool:
    tuning_dependencies: dict[str, tuple[str, ...]] = {
        "default_xgb_enable_tuning": (
            "default_xgb_tuning_method",
            "default_xgb_cv_folds",
            "default_xgb_cv_scoring",
            "default_xgb_cv_n_iter",
            "default_xgb_cv_n_jobs",
        ),
        "default_logistic_enable_tuning": (
            "default_logistic_tuning_method",
            "default_logistic_cv_folds",
            "default_logistic_cv_scoring",
            "default_logistic_cv_n_iter",
            "default_logistic_cv_n_jobs",
        ),
        "default_rf_enable_tuning": (
            "default_rf_tuning_method",
            "default_rf_cv_folds",
            "default_rf_cv_scoring",
            "default_rf_cv_n_iter",
            "default_rf_cv_n_jobs",
        ),
        "default_lr_enable_tuning": (
            "default_lr_tuning_method",
            "default_lr_cv_folds",
            "default_lr_cv_scoring",
            "default_lr_cv_n_iter",
            "default_lr_cv_n_jobs",
        ),
        "default_tf_enable_tuning": (
            "default_tf_tuning_method",
            "default_tf_cv_scoring",
            "default_tf_cv_n_iter",
        ),
    }

    for enable_key, dependent_keys in tuning_dependencies.items():
        if key in dependent_keys and enable_key in values_by_key:
            return not _is_truthy(values_by_key[enable_key])

    return False


def _supports_early_stopping_defaults(library: str, model: str | None, task: str) -> bool:
    if library == "tensorflow" and model == "dense_nn":
        return True
    if library == "xgboost":
        return True
    return False


def _supports_default_max_iter(library: str, model: str | None, task: str) -> bool:
    return library == "scikit-learn" and model == "logistic_regression" and task in {
        "binary_classification",
        "multiclass_classification",
    }


def _recommended_es_defaults(library: str, model: str | None) -> tuple[bool, float, int]:
    if library == "xgboost":
        return True, 0.1, 20
    if library == "tensorflow" and model == "dense_nn":
        return True, 0.1, 5
    return True, 0.1, 5


# ---------------------------------------------------------
# Dataset metadata (for size-aware profile defaults)
# ---------------------------------------------------------

DATASET_META = {
    "ames_housing.csv": {"rows": 2930, "features": 82},
    "california_housing.csv": {"rows": 20640, "features": 9},
    "insurance.csv": {"rows": 1338, "features": 7},
    "adult_income.csv": {"rows": 32561, "features": 15},
    "breast_cancer_wisconsin.csv": {"rows": 569, "features": 32},
    "mushrooms.csv": {"rows": 8124, "features": 23},
    "titanic.csv": {"rows": 891, "features": 12},
    "car_evaluation.csv": {"rows": 1728, "features": 7},
    "dry_bean.csv": {"rows": 13611, "features": 17},
    "forest_cover_type.csv": {"rows": 581012, "features": 55},
    "iris.csv": {"rows": 150, "features": 5},
    "wine_quality.csv": {"rows": 6497, "features": 13},
}


def _dataset_size_bucket(dataset_name: str | None) -> str:
    """Return 'small', 'medium', or 'large' based on known row counts."""
    if dataset_name is None or dataset_name not in DATASET_META:
        return "medium"
    rows = DATASET_META[dataset_name]["rows"]
    if rows < 2000:
        return "small"
    if rows <= 20000:
        return "medium"
    return "large"


# ---------------------------------------------------------
# Training profiles
# ---------------------------------------------------------

def _get_profile_defaults(
    library: str,
    model: str | None,
    task: str,
    profile: str,
    size_bucket: str,
) -> dict:
    """Return hyperparameter defaults for the given profile.

    Keys match the variable names used in per-param prompts.
    A value of None means the param is not applicable.
    """
    defaults: dict = {}

    if library == "xgboost":
        presets = {
            "Quick": {
                "n_est": 100,
                "lr": 0.3,
                "depth": 3,
                "sub": 0.8,
                "col": 0.8,
                "mcw": 1.0,
                "rl": 1.0,
                "ra": 0.0,
                "es": True,
                "vf": 0.1,
                "nic": 10,
                "enable_tuning": False,
                "tuning_method": "random",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 10,
                "cv_n_jobs": -1,
            },
            "Balanced": {
                "n_est": 300,
                "lr": 0.1,
                "depth": 6,
                "sub": 1.0,
                "col": 1.0,
                "mcw": 1.0,
                "rl": 1.0,
                "ra": 0.0,
                "es": True,
                "vf": 0.1,
                "nic": 20,
                "enable_tuning": False,
                "tuning_method": "random",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 20,
                "cv_n_jobs": -1,
            },
            "Thorough": {
                "n_est": 1000,
                "lr": 0.05,
                "depth": 8,
                "sub": 0.8,
                "col": 0.8,
                "mcw": 1.0,
                "rl": 1.0,
                "ra": 0.1,
                "es": True,
                "vf": 0.1,
                "nic": 30,
                "enable_tuning": True,
                "tuning_method": "random",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 30,
                "cv_n_jobs": -1,
            },
        }
        if size_bucket == "large":
            presets["Quick"]["n_est"] = 200
            presets["Thorough"]["n_est"] = 1500
        elif size_bucket == "small":
            presets["Quick"]["depth"] = 2
            presets["Balanced"]["depth"] = 4
            presets["Thorough"]["n_est"] = 500

        p = presets.get(profile, presets["Balanced"])
        defaults = {
            "default_n_estimators": str(p["n_est"]),
            "default_learning_rate": str(p["lr"]),
            "default_max_depth": str(p["depth"]),
            "default_subsample": str(p["sub"]),
            "default_colsample_bytree": str(p["col"]),
            "default_xgb_min_child_weight": str(p["mcw"]),
            "default_xgb_reg_lambda": str(p["rl"]),
            "default_xgb_reg_alpha": str(p["ra"]),
            "default_early_stopping": p["es"],
            "default_validation_fraction": p["vf"],
            "default_n_iter_no_change": p["nic"],
            "default_xgb_enable_tuning": p["enable_tuning"],
            "default_xgb_tuning_method": p["tuning_method"],
            "default_xgb_cv_folds": str(p["cv_folds"]),
            "default_xgb_cv_scoring": p["cv_scoring"],
            "default_xgb_cv_n_iter": str(p["cv_n_iter"]),
            "default_xgb_cv_n_jobs": str(p["cv_n_jobs"]),
        }

    elif library == "scikit-learn" and model == "random_forest":
        presets = {
            "Quick": {
                "n_est": 100,
                "depth": 8,
                "msl": 5,
                "mf": "sqrt",
                "enable_tuning": False,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 20,
                "cv_n_jobs": -1,
            },
            "Balanced": {
                "n_est": 300,
                "depth": 16,
                "msl": 1,
                "mf": "sqrt",
                "enable_tuning": False,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 20,
                "cv_n_jobs": -1,
            },
            "Thorough": {
                "n_est": 500,
                "depth": 32,
                "msl": 1,
                "mf": "sqrt",
                "enable_tuning": True,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro" if task != "regression" else "rmse",
                "cv_n_iter": 30,
                "cv_n_jobs": -1,
            },
        }
        if size_bucket == "large":
            presets["Thorough"]["n_est"] = 800
        elif size_bucket == "small":
            presets["Balanced"]["n_est"] = 200
            presets["Quick"]["depth"] = 6

        p = presets.get(profile, presets["Balanced"])
        defaults = {
            "default_rf_n_estimators": str(p["n_est"]),
            "default_rf_max_depth": str(p["depth"]),
            "default_rf_min_samples_leaf": str(p["msl"]),
            "default_rf_max_features": str(p["mf"]),
            "default_rf_enable_tuning": p["enable_tuning"],
            "default_rf_tuning_method": p["tuning_method"],
            "default_rf_cv_folds": str(p["cv_folds"]),
            "default_rf_cv_scoring": p["cv_scoring"],
            "default_rf_cv_n_iter": str(p["cv_n_iter"]),
            "default_rf_cv_n_jobs": str(p["cv_n_jobs"]),
        }

    elif library == "scikit-learn" and model == "logistic_regression":
        presets = {
            "Quick": {
                "c": 1.0,
                "solver": "lbfgs",
                "max_iter": 500,
                "penalty": "l2",
                "class_weight": "none",
                "enable_tuning": False,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro",
                "cv_n_iter": 20,
                "cv_n_jobs": -1,
            },
            "Balanced": {
                "c": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "penalty": "l2",
                "class_weight": "none",
                "enable_tuning": False,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro",
                "cv_n_iter": 20,
                "cv_n_jobs": -1,
            },
            "Thorough": {
                "c": 0.5,
                "solver": "saga",
                "max_iter": 2000,
                "penalty": "l2",
                "class_weight": "none",
                "enable_tuning": True,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "f1_macro",
                "cv_n_iter": 30,
                "cv_n_jobs": -1,
            },
        }
        p = presets.get(profile, presets["Balanced"])
        defaults = {
            "default_c": str(p["c"]),
            "default_solver": p["solver"],
            "default_max_iter": p["max_iter"],
            "default_logistic_penalty": p["penalty"],
            "default_logistic_class_weight": p["class_weight"],
            "default_logistic_enable_tuning": p["enable_tuning"],
            "default_logistic_tuning_method": p["tuning_method"],
            "default_logistic_cv_folds": str(p["cv_folds"]),
            "default_logistic_cv_scoring": p["cv_scoring"],
            "default_logistic_cv_n_iter": str(p["cv_n_iter"]),
            "default_logistic_cv_n_jobs": str(p["cv_n_jobs"]),
        }

    elif library == "scikit-learn" and model == "linear_regression":
        presets = {
            "Quick": {
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
            "Balanced": {
                "penalty": "l2",
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
            "Thorough": {
                "penalty": "l2",
                "alpha": 0.1,
                "fit_intercept": True,
                "l1_ratio": 0.5,
                "enable_tuning": True,
                "tuning_method": "grid",
                "cv_folds": 5,
                "cv_scoring": "rmse",
                "cv_n_iter": 30,
                "cv_n_jobs": -1,
            },
        }
        p = presets.get(profile, presets["Balanced"])
        defaults = {
            "default_lr_penalty": p["penalty"],
            "default_lr_alpha": str(p["alpha"]),
            "default_lr_fit_intercept": p["fit_intercept"],
            "default_lr_l1_ratio": str(p["l1_ratio"]),
            "default_lr_enable_tuning": p["enable_tuning"],
            "default_lr_tuning_method": p["tuning_method"],
            "default_lr_cv_folds": str(p["cv_folds"]),
            "default_lr_cv_scoring": p["cv_scoring"],
            "default_lr_cv_n_iter": str(p["cv_n_iter"]),
            "default_lr_cv_n_jobs": str(p["cv_n_jobs"]),
        }

    return defaults


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    generator_path = script_dir / "tools" / "generate_model.py"

    if not generator_path.exists():
        print(f"Could not find generator script at: {generator_path}", file=sys.stderr)
        return 1

    library = _ask_select(
        "Select library:",
        choices=["scikit-learn", "xgboost", "tensorflow"],
    )

    if library is None:
        print("Cancelled.")
        return 0

    model = None
    if library == "scikit-learn":
        model = _ask_select(
            "Select scikit-learn model:",
            choices=SKLEARN_MODELS,
        )
        if model is None:
            print("Cancelled.")
            return 0

    if library == "tensorflow":
        model = _ask_select(
            "Select tensorflow model:",
            choices=TENSORFLOW_MODELS,
        )
        if model is None:
            print("Cancelled.")
            return 0

    # Task selection (context-aware)
    task_choices = TASKS_BY_LIBRARY_MODEL.get((library, model))
    if not task_choices:
        # Fallback (shouldn't happen if mappings are correct)
        task_choices = ["regression", "binary_classification", "multiclass_classification"]

    if len(task_choices) == 1:
        task = task_choices[0]
        print(f"Task auto-selected: {task}")
    else:
        task = _ask_select(
            "Select task:",
            choices=task_choices,
        )

        if task is None:
            print("Cancelled.")
            return 0

    # Optional xgboost booster selection
    booster = None
    device = None
    if library == "xgboost":
        booster = _ask_select(
            "Default xgboost booster:",
            choices=XGBOOST_BOOSTERS,
        )

        if booster is None:
            print("Cancelled.")
            return 0

        device = _ask_select(
            "Default xgboost device for generated template:",
            choices=XGBOOST_DEVICE_DEFAULTS,
        )

        if device is None:
            print("Cancelled.")
            return 0

    # Optional starter dataset selection (task-aware)
    starter_dataset = None
    starter_choices = STARTER_DATASETS_BY_TASK.get(task, [])
    if starter_choices:
        starter_dataset = _ask_select(
            "Select starter template dataset:",
            choices=starter_choices,
        )

        if starter_dataset is None:
            print("Cancelled.")
            return 0

    # ---------------------------------------------------------
    # Training profile selection (non-TensorFlow only)
    # ---------------------------------------------------------
    profile = None
    profile_defaults = {}
    size_bucket = _dataset_size_bucket(starter_dataset)

    if library != "tensorflow":
        profile = _ask_select(
            "Select training profile:",
            choices=[
                questionary.Choice("Quick", description="Fast iteration, smaller models"),
                questionary.Choice("Balanced", description="Solid defaults for most tasks"),
                questionary.Choice("Thorough", description="Larger search, more compute"),
                questionary.Choice("Custom", description="Set every hyperparameter manually"),
            ],
        )

        if profile is None:
            print("Cancelled.")
            return 0

        if profile != "Custom":
            profile_defaults = _get_profile_defaults(library, model, task, profile, size_bucket)

    # TensorFlow training knobs (ONLY where necessary)
    optimizer = None
    learning_rate = None
    epochs = None
    batch_size = None

    if library == "tensorflow":
        optimizer = _ask_select(
            "Select optimizer:",
            choices=TENSORFLOW_OPTIMIZERS,
        )
        if optimizer is None:
            print("Cancelled.")
            return 0

        learning_rate = _ask_text(
            "Enter learning rate (e.g., 0.001):",
            default="0.001",
            validate_fn=lambda s: True
            if (_is_float(s) and float(s) > 0)
            else "Must be a positive number",
        )
        if learning_rate is None:
            print("Cancelled.")
            return 0

        epochs = _ask_text(
            "Enter epochs (e.g., 100):",
            default="100",
            validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
        )
        if epochs is None:
            print("Cancelled.")
            return 0

        batch_size = _ask_text(
            "Enter batch size (e.g., 32):",
            default="32",
            validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
        )
        if batch_size is None:
            print("Cancelled.")
            return 0

    default_early_stopping = None
    default_validation_fraction = None
    default_n_iter_no_change = None
    default_max_iter = None
    default_n_estimators = None
    default_learning_rate = None
    default_max_depth = None
    default_subsample = None
    default_colsample_bytree = None
    default_c = None
    default_solver = None
    default_rf_n_estimators = None
    default_rf_max_depth = None
    default_rf_min_samples_leaf = None
    default_rf_max_features = None
    default_logistic_penalty = None
    default_logistic_class_weight = None
    default_logistic_enable_tuning = None
    default_logistic_tuning_method = None
    default_logistic_cv_folds = None
    default_logistic_cv_scoring = None
    default_logistic_cv_n_iter = None
    default_logistic_cv_n_jobs = None
    default_rf_enable_tuning = None
    default_rf_tuning_method = None
    default_rf_cv_folds = None
    default_rf_cv_scoring = None
    default_rf_cv_n_iter = None
    default_rf_cv_n_jobs = None
    default_lr_penalty = None
    default_lr_alpha = None
    default_lr_fit_intercept = None
    default_lr_l1_ratio = None
    default_lr_enable_tuning = None
    default_lr_tuning_method = None
    default_lr_cv_folds = None
    default_lr_cv_scoring = None
    default_lr_cv_n_iter = None
    default_lr_cv_n_jobs = None
    default_xgb_min_child_weight = None
    default_xgb_reg_lambda = None
    default_xgb_reg_alpha = None
    default_xgb_enable_tuning = None
    default_xgb_tuning_method = None
    default_xgb_cv_folds = None
    default_xgb_cv_scoring = None
    default_xgb_cv_n_iter = None
    default_xgb_cv_n_jobs = None
    default_tf_enable_tuning = None
    default_tf_tuning_method = None
    default_tf_cv_scoring = None
    default_tf_cv_n_iter = None

    # When a profile provides defaults, use them and skip prompts.
    use_custom = profile == "Custom" or library == "tensorflow"

    if library == "xgboost":
        if use_custom:
            default_n_estimators = _ask_text(
                "Default n_estimators for template --n-estimators:",
                default="300",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_n_estimators is None:
                print("Cancelled.")
                return 0

            default_learning_rate = _ask_text(
                "Default learning rate for template --learning-rate:",
                default="0.1",
                validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
            )
            if default_learning_rate is None:
                print("Cancelled.")
                return 0

            default_max_depth = _ask_text(
                "Default max depth for template --max-depth:",
                default="6",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_max_depth is None:
                print("Cancelled.")
                return 0

            default_subsample = _ask_text(
                "Default subsample for template --subsample (0 < value <= 1):",
                default="1.0",
                validate_fn=lambda s: True if (_is_float(s) and 0 < float(s) <= 1.0) else "Must be in range (0, 1]",
            )
            if default_subsample is None:
                print("Cancelled.")
                return 0

            default_colsample_bytree = _ask_text(
                "Default colsample_bytree for template --colsample-bytree (0 < value <= 1):",
                default="1.0",
                validate_fn=lambda s: True if (_is_float(s) and 0 < float(s) <= 1.0) else "Must be in range (0, 1]",
            )
            if default_colsample_bytree is None:
                print("Cancelled.")
                return 0

            default_xgb_min_child_weight = _ask_text(
                "Default min_child_weight for template --min-child-weight:",
                default="1.0",
                validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0) else "Must be a non-negative number",
            )
            if default_xgb_min_child_weight is None:
                print("Cancelled.")
                return 0

            default_xgb_reg_lambda = _ask_text(
                "Default reg_lambda (L2) for template --reg-lambda:",
                default="1.0",
                validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0) else "Must be a non-negative number",
            )
            if default_xgb_reg_lambda is None:
                print("Cancelled.")
                return 0

            default_xgb_reg_alpha = _ask_text(
                "Default reg_alpha (L1) for template --reg-alpha:",
                default="0.0",
                validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0) else "Must be a non-negative number",
            )
            if default_xgb_reg_alpha is None:
                print("Cancelled.")
                return 0

            default_xgb_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if default_xgb_enable_tuning is None:
                print("Cancelled.")
                return 0
            if default_xgb_enable_tuning:
                default_xgb_tuning_method = "random"
                default_xgb_cv_folds = _ask_text(
                    "Default CV folds (--cv-folds, >=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if default_xgb_cv_folds is None:
                    print("Cancelled.")
                    return 0
                default_xgb_cv_scoring = _ask_select(
                    "Default CV scoring (--cv-scoring):",
                    choices=["rmse", "mae", "r2"] if task == "regression" else ["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if default_xgb_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                default_xgb_cv_n_iter = _ask_text(
                    "Default random-search iterations (--cv-n-iter, >0):",
                    default="20",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                )
                if default_xgb_cv_n_iter is None:
                    print("Cancelled.")
                    return 0
                default_xgb_cv_n_jobs = _ask_text(
                    "Default CV parallel jobs (--cv-n-jobs, e.g., -1):",
                    default="-1",
                    validate_fn=lambda s: True if _is_int(s) else "Must be an integer",
                )
                if default_xgb_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            default_n_estimators = profile_defaults.get("default_n_estimators", "300")
            default_learning_rate = profile_defaults.get("default_learning_rate", "0.1")
            default_max_depth = profile_defaults.get("default_max_depth", "6")
            default_subsample = profile_defaults.get("default_subsample", "1.0")
            default_colsample_bytree = profile_defaults.get("default_colsample_bytree", "1.0")
            default_xgb_min_child_weight = profile_defaults.get("default_xgb_min_child_weight", "1.0")
            default_xgb_reg_lambda = profile_defaults.get("default_xgb_reg_lambda", "1.0")
            default_xgb_reg_alpha = profile_defaults.get("default_xgb_reg_alpha", "0.0")
            default_xgb_enable_tuning = profile_defaults.get("default_xgb_enable_tuning", False)
            default_xgb_tuning_method = profile_defaults.get("default_xgb_tuning_method", "random")
            default_xgb_cv_folds = profile_defaults.get("default_xgb_cv_folds", "5")
            default_xgb_cv_scoring = profile_defaults.get(
                "default_xgb_cv_scoring",
                "rmse" if task == "regression" else "f1_macro",
            )
            default_xgb_cv_n_iter = profile_defaults.get("default_xgb_cv_n_iter", "20")
            default_xgb_cv_n_jobs = profile_defaults.get("default_xgb_cv_n_jobs", "-1")

    if library == "scikit-learn" and model == "logistic_regression":
        if use_custom:
            default_c = _ask_text(
                "Default C for template --c:",
                default="1.0",
                validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
            )
            if default_c is None:
                print("Cancelled.")
                return 0

            default_solver = _ask_select(
                "Default solver for template --solver:",
                choices=SKLEARN_LOGISTIC_SOLVERS,
            )
            if default_solver is None:
                print("Cancelled.")
                return 0

            default_logistic_penalty = _ask_select(
                "Default penalty for template --penalty:",
                choices=["none", "l1", "l2", "elasticnet"],
            )
            if default_logistic_penalty is None:
                print("Cancelled.")
                return 0

            default_logistic_class_weight = _ask_select(
                "Default class_weight for template --class-weight:",
                choices=["none", "balanced"],
            )
            if default_logistic_class_weight is None:
                print("Cancelled.")
                return 0

            default_logistic_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if default_logistic_enable_tuning is None:
                print("Cancelled.")
                return 0
            if default_logistic_enable_tuning:
                default_logistic_tuning_method = _ask_select(
                    "Default tuning method (--tuning-method):",
                    choices=["grid", "random"],
                )
                if default_logistic_tuning_method is None:
                    print("Cancelled.")
                    return 0
                default_logistic_cv_folds = _ask_text(
                    "Default CV folds (--cv-folds, >=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if default_logistic_cv_folds is None:
                    print("Cancelled.")
                    return 0
                default_logistic_cv_scoring = _ask_select(
                    "Default CV scoring (--cv-scoring):",
                    choices=["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if default_logistic_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                if default_logistic_tuning_method == "random":
                    default_logistic_cv_n_iter = _ask_text(
                        "Default random-search iterations (--cv-n-iter, >0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if default_logistic_cv_n_iter is None:
                        print("Cancelled.")
                        return 0
                default_logistic_cv_n_jobs = _ask_text(
                    "Default CV parallel jobs (--cv-n-jobs, e.g., -1):",
                    default="-1",
                    validate_fn=lambda s: True if _is_int(s) else "Must be an integer",
                )
                if default_logistic_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            default_c = profile_defaults.get("default_c", "1.0")
            default_solver = profile_defaults.get("default_solver", "lbfgs")
            default_logistic_penalty = profile_defaults.get("default_logistic_penalty", "l2")
            default_logistic_class_weight = profile_defaults.get("default_logistic_class_weight", "none")
            default_logistic_enable_tuning = profile_defaults.get("default_logistic_enable_tuning", False)
            default_logistic_tuning_method = profile_defaults.get("default_logistic_tuning_method", "grid")
            default_logistic_cv_folds = profile_defaults.get("default_logistic_cv_folds", "5")
            default_logistic_cv_scoring = profile_defaults.get("default_logistic_cv_scoring", "f1_macro")
            default_logistic_cv_n_iter = profile_defaults.get("default_logistic_cv_n_iter", "20")
            default_logistic_cv_n_jobs = profile_defaults.get("default_logistic_cv_n_jobs", "-1")

    if library == "scikit-learn" and model == "random_forest":
        if use_custom:
            default_rf_n_estimators = _ask_text(
                "Default n_estimators for template --n-estimators:",
                default="300",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_rf_n_estimators is None:
                print("Cancelled.")
                return 0

            default_rf_max_depth = _ask_text(
                "Default max depth for template --max-depth:",
                default="16",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_rf_max_depth is None:
                print("Cancelled.")
                return 0

            default_rf_min_samples_leaf = _ask_text(
                "Default min_samples_leaf for template --min-samples-leaf:",
                default="1",
                validate_fn=lambda s: True if (_is_int(s) and int(s) >= 1) else "Must be an integer >= 1",
            )
            if default_rf_min_samples_leaf is None:
                print("Cancelled.")
                return 0

            default_rf_max_features = _ask_select(
                "Default max_features for template --max-features:",
                choices=["sqrt", "log2", "1.0"],
            )
            if default_rf_max_features is None:
                print("Cancelled.")
                return 0

            default_rf_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if default_rf_enable_tuning is None:
                print("Cancelled.")
                return 0
            if default_rf_enable_tuning:
                default_rf_tuning_method = _ask_select(
                    "Default tuning method (--tuning-method):",
                    choices=["grid", "random"],
                )
                if default_rf_tuning_method is None:
                    print("Cancelled.")
                    return 0
                default_rf_cv_folds = _ask_text(
                    "Default CV folds (--cv-folds, >=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if default_rf_cv_folds is None:
                    print("Cancelled.")
                    return 0
                default_rf_cv_scoring = _ask_select(
                    "Default CV scoring (--cv-scoring):",
                    choices=["rmse", "mae", "r2"] if task == "regression" else ["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if default_rf_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                if default_rf_tuning_method == "random":
                    default_rf_cv_n_iter = _ask_text(
                        "Default random-search iterations (--cv-n-iter, >0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if default_rf_cv_n_iter is None:
                        print("Cancelled.")
                        return 0
                default_rf_cv_n_jobs = _ask_text(
                    "Default CV parallel jobs (--cv-n-jobs, e.g., -1):",
                    default="-1",
                    validate_fn=lambda s: True if _is_int(s) else "Must be an integer",
                )
                if default_rf_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            default_rf_n_estimators = profile_defaults.get("default_rf_n_estimators", "300")
            default_rf_max_depth = profile_defaults.get("default_rf_max_depth", "16")
            default_rf_min_samples_leaf = profile_defaults.get("default_rf_min_samples_leaf", "1")
            default_rf_max_features = profile_defaults.get("default_rf_max_features", "sqrt")
            default_rf_enable_tuning = profile_defaults.get("default_rf_enable_tuning", False)
            default_rf_tuning_method = profile_defaults.get("default_rf_tuning_method", "grid")
            default_rf_cv_folds = profile_defaults.get("default_rf_cv_folds", "5")
            default_rf_cv_scoring = profile_defaults.get(
                "default_rf_cv_scoring",
                "rmse" if task == "regression" else "f1_macro",
            )
            default_rf_cv_n_iter = profile_defaults.get("default_rf_cv_n_iter", "20")
            default_rf_cv_n_jobs = profile_defaults.get("default_rf_cv_n_jobs", "-1")

    if library == "tensorflow" and model == "dense_nn":
        default_tf_enable_tuning = questionary.confirm(
            "Enable hyperparameter tuning by default? (--enable-tuning)",
            default=False,
            style=CUSTOM_STYLE,
        ).ask()
        if default_tf_enable_tuning is None:
            print("Cancelled.")
            return 0
        if default_tf_enable_tuning:
            default_tf_tuning_method = "random"
            default_tf_cv_scoring = _ask_select(
                "Default tuning scoring (--cv-scoring):",
                choices=["rmse"] if task == "regression" else ["f1_macro"],
            )
            if default_tf_cv_scoring is None:
                print("Cancelled.")
                return 0
            default_tf_cv_n_iter = _ask_text(
                "Default random-search iterations (--cv-n-iter, >0):",
                default="10",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_tf_cv_n_iter is None:
                print("Cancelled.")
                return 0

    if library == "scikit-learn" and model == "linear_regression":
        if use_custom:
            default_lr_penalty = _ask_select(
                "Default penalty for template --penalty:",
                choices=["none", "l1", "l2", "elasticnet"],
            )
            if default_lr_penalty is None:
                print("Cancelled.")
                return 0

            if default_lr_penalty != "none":
                default_lr_alpha = _ask_text(
                    "Default alpha (regularization strength) for template --alpha:",
                    default="1.0",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
                )
                if default_lr_alpha is None:
                    print("Cancelled.")
                    return 0
            else:
                default_lr_alpha = "1.0"

            default_lr_fit_intercept = questionary.confirm(
                "Fit intercept in default template? (--fit-intercept)",
                default=True,
                style=CUSTOM_STYLE,
            ).ask()
            if default_lr_fit_intercept is None:
                print("Cancelled.")
                return 0

            if default_lr_penalty == "elasticnet":
                default_lr_l1_ratio = _ask_text(
                    "Default l1_ratio for ElasticNet (0 = pure L2, 1 = pure L1) --l1-ratio:",
                    default="0.5",
                    validate_fn=lambda s: True if (_is_float(s) and 0.0 <= float(s) <= 1.0) else "Must be in range [0, 1]",
                )
                if default_lr_l1_ratio is None:
                    print("Cancelled.")
                    return 0
            else:
                default_lr_l1_ratio = "0.5"

            default_lr_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default in generated template? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if default_lr_enable_tuning is None:
                print("Cancelled.")
                return 0

            if default_lr_enable_tuning:
                if default_lr_penalty == "none":
                    print(
                        "Note: penalty=none with tuning enabled uses a small search space "
                        "(primarily fit_intercept)."
                    )

                default_lr_tuning_method = _ask_select(
                    "Default tuning method (--tuning-method):",
                    choices=["grid", "random"],
                )
                if default_lr_tuning_method is None:
                    print("Cancelled.")
                    return 0

                default_lr_cv_folds = _ask_text(
                    "Default CV folds (--cv-folds, >=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if default_lr_cv_folds is None:
                    print("Cancelled.")
                    return 0

                default_lr_cv_scoring = _ask_select(
                    "Default CV scoring (--cv-scoring):",
                    choices=["rmse", "mae", "r2"],
                )
                if default_lr_cv_scoring is None:
                    print("Cancelled.")
                    return 0

                if default_lr_tuning_method == "random":
                    default_lr_cv_n_iter = _ask_text(
                        "Default random-search iterations (--cv-n-iter, >0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if default_lr_cv_n_iter is None:
                        print("Cancelled.")
                        return 0

                default_lr_cv_n_jobs = _ask_text(
                    "Default CV parallel jobs (--cv-n-jobs, e.g., -1):",
                    default="-1",
                    validate_fn=lambda s: True if _is_int(s) else "Must be an integer",
                )
                if default_lr_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            default_lr_penalty = profile_defaults.get("default_lr_penalty", "none")
            default_lr_alpha = profile_defaults.get("default_lr_alpha", "1.0")
            default_lr_fit_intercept = profile_defaults.get("default_lr_fit_intercept", True)
            default_lr_l1_ratio = profile_defaults.get("default_lr_l1_ratio", "0.5")
            default_lr_enable_tuning = profile_defaults.get("default_lr_enable_tuning", False)
            default_lr_tuning_method = profile_defaults.get("default_lr_tuning_method", "grid")
            default_lr_cv_folds = profile_defaults.get("default_lr_cv_folds", "5")
            default_lr_cv_scoring = profile_defaults.get("default_lr_cv_scoring", "rmse")
            default_lr_cv_n_iter = profile_defaults.get("default_lr_cv_n_iter", "20")
            default_lr_cv_n_jobs = profile_defaults.get("default_lr_cv_n_jobs", "-1")

    if _supports_default_max_iter(library, model, task):
        if use_custom:
            default_max_iter = _ask_text(
                "Default max iterations for template --max-iter:",
                default="1000",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if default_max_iter is None:
                print("Cancelled.")
                return 0
            default_max_iter = int(default_max_iter)
        else:
            default_max_iter = int(profile_defaults.get("default_max_iter", 1000))

    if _supports_early_stopping_defaults(library, model, task):
        if use_custom:
            recommended_early_stopping, recommended_validation_fraction, recommended_n_iter_no_change = _recommended_es_defaults(
                library,
                model,
            )

            default_early_stopping = questionary.confirm(
                "Enable early stopping by default in the generated template (--early-stopping)?",
                default=bool(recommended_early_stopping),
                style=CUSTOM_STYLE,
            ).ask()

            if default_early_stopping is None:
                print("Cancelled.")
                return 0

            use_recommended_defaults = questionary.confirm(
                "Use recommended preset values for --validation-fraction and --n-iter-no-change?",
                default=True,
                style=CUSTOM_STYLE,
            ).ask()

            if use_recommended_defaults is None:
                print("Cancelled.")
                return 0

            if use_recommended_defaults:
                default_validation_fraction = recommended_validation_fraction
                default_n_iter_no_change = recommended_n_iter_no_change
            else:
                default_validation_fraction = _ask_text(
                    "Default validation fraction for template --validation-fraction (0 < value < 1):",
                    default=str(recommended_validation_fraction),
                    validate_fn=lambda s: True
                    if (_is_float(s) and 0.0 < float(s) < 1.0)
                    else "Must be a number where 0 < value < 1",
                )
                if default_validation_fraction is None:
                    print("Cancelled.")
                    return 0

                default_n_iter_no_change = _ask_text(
                    "Default n_iter_no_change for template --n-iter-no-change:",
                    default=str(recommended_n_iter_no_change),
                    validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                )
                if default_n_iter_no_change is None:
                    print("Cancelled.")
                    return 0

                default_validation_fraction = float(default_validation_fraction)
                default_n_iter_no_change = int(default_n_iter_no_change)
        else:
            # Profile provides ES defaults
            default_early_stopping = profile_defaults.get("default_early_stopping", True)
            default_validation_fraction = profile_defaults.get("default_validation_fraction", 0.1)
            default_n_iter_no_change = profile_defaults.get("default_n_iter_no_change", 5)

    resolved_defaults: list[tuple[str, object]] = [
        ("library", library),
        ("model", model if model is not None else "n/a"),
        ("task", task),
        ("starter_dataset", starter_dataset if starter_dataset is not None else "n/a"),
        ("training_profile", profile if profile is not None else "n/a"),
    ]

    optional_defaults: list[tuple[str, object | None]] = [
        ("booster", booster),
        ("device", device),
        ("optimizer", optimizer),
        ("learning_rate", learning_rate),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("default_early_stopping", default_early_stopping),
        ("default_validation_fraction", default_validation_fraction),
        ("default_n_iter_no_change", default_n_iter_no_change),
        ("default_max_iter", default_max_iter),
        ("default_n_estimators", default_n_estimators),
        ("default_learning_rate", default_learning_rate),
        ("default_max_depth", default_max_depth),
        ("default_subsample", default_subsample),
        ("default_colsample_bytree", default_colsample_bytree),
        ("default_xgb_min_child_weight", default_xgb_min_child_weight),
        ("default_xgb_reg_lambda", default_xgb_reg_lambda),
        ("default_xgb_reg_alpha", default_xgb_reg_alpha),
        ("default_xgb_enable_tuning", default_xgb_enable_tuning),
        ("default_xgb_tuning_method", default_xgb_tuning_method),
        ("default_xgb_cv_folds", default_xgb_cv_folds),
        ("default_xgb_cv_scoring", default_xgb_cv_scoring),
        ("default_xgb_cv_n_iter", default_xgb_cv_n_iter),
        ("default_xgb_cv_n_jobs", default_xgb_cv_n_jobs),
        ("default_c", default_c),
        ("default_solver", default_solver),
        ("default_logistic_penalty", default_logistic_penalty),
        ("default_logistic_class_weight", default_logistic_class_weight),
        ("default_logistic_enable_tuning", default_logistic_enable_tuning),
        ("default_logistic_tuning_method", default_logistic_tuning_method),
        ("default_logistic_cv_folds", default_logistic_cv_folds),
        ("default_logistic_cv_scoring", default_logistic_cv_scoring),
        ("default_logistic_cv_n_iter", default_logistic_cv_n_iter),
        ("default_logistic_cv_n_jobs", default_logistic_cv_n_jobs),
        ("default_rf_n_estimators", default_rf_n_estimators),
        ("default_rf_max_depth", default_rf_max_depth),
        ("default_rf_min_samples_leaf", default_rf_min_samples_leaf),
        ("default_rf_max_features", default_rf_max_features),
        ("default_rf_enable_tuning", default_rf_enable_tuning),
        ("default_rf_tuning_method", default_rf_tuning_method),
        ("default_rf_cv_folds", default_rf_cv_folds),
        ("default_rf_cv_scoring", default_rf_cv_scoring),
        ("default_rf_cv_n_iter", default_rf_cv_n_iter),
        ("default_rf_cv_n_jobs", default_rf_cv_n_jobs),
        ("default_lr_penalty", default_lr_penalty),
        ("default_lr_alpha", default_lr_alpha),
        ("default_lr_fit_intercept", default_lr_fit_intercept),
        ("default_lr_l1_ratio", default_lr_l1_ratio),
        ("default_lr_enable_tuning", default_lr_enable_tuning),
        ("default_lr_tuning_method", default_lr_tuning_method),
        ("default_lr_cv_folds", default_lr_cv_folds),
        ("default_lr_cv_scoring", default_lr_cv_scoring),
        ("default_lr_cv_n_iter", default_lr_cv_n_iter),
        ("default_lr_cv_n_jobs", default_lr_cv_n_jobs),
        ("default_tf_enable_tuning", default_tf_enable_tuning),
        ("default_tf_tuning_method", default_tf_tuning_method),
        ("default_tf_cv_scoring", default_tf_cv_scoring),
        ("default_tf_cv_n_iter", default_tf_cv_n_iter),
    ]

    optional_values_by_key = {key: value for key, value in optional_defaults if value is not None}

    for key, value in optional_defaults:
        if value is None:
            continue
        if _should_omit_resolved_key(key, optional_values_by_key):
            continue
        resolved_defaults.append((key, value))

    print("\nResolved defaults:")
    for key, value in resolved_defaults:
        print(f"  {key}={_stringify_setting(value)}")

    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    name = None
    while True:
        name = _ask_text(
            "Enter model name (no .py):",
            validate_fn=lambda s: True if s.strip() else "Name cannot be empty",
        )

        if name is None:
            print("Cancelled.")
            return 0

        output_candidate = models_dir / f"{name.strip()}.py"
        if output_candidate.exists():
            print(f"Name already exists: {output_candidate}")
            print("Try another model name.\n")
            continue
        break

    cmd = [
        sys.executable,
        str(generator_path),
        "--library",
        library,
        "--task",
        task,
        "--name",
        name.strip(),
    ]

    if model is not None:
        cmd.extend(["--model", model])

    if library == "xgboost":
        cmd.extend(["--booster", booster])
        cmd.extend(["--device", device])

    if starter_dataset is not None:
        cmd.extend(["--starter-dataset", starter_dataset])

    if _supports_early_stopping_defaults(library, model, task):
        cmd.extend(["--default-early-stopping", "true" if default_early_stopping else "false"])
        cmd.extend(["--default-validation-fraction", str(float(default_validation_fraction))])
        cmd.extend(["--default-n-iter-no-change", str(int(default_n_iter_no_change))])

    if default_max_iter is not None:
        cmd.extend(["--default-max-iter", str(int(default_max_iter))])

    if default_n_estimators is not None:
        cmd.extend(["--default-n-estimators", str(int(default_n_estimators))])
    if default_learning_rate is not None:
        cmd.extend(["--default-learning-rate", str(float(default_learning_rate))])
    if default_max_depth is not None:
        cmd.extend(["--default-max-depth", str(int(default_max_depth))])
    if default_subsample is not None:
        cmd.extend(["--default-subsample", str(float(default_subsample))])
    if default_colsample_bytree is not None:
        cmd.extend(["--default-colsample-bytree", str(float(default_colsample_bytree))])
    if default_xgb_min_child_weight is not None:
        cmd.extend(["--default-xgb-min-child-weight", str(float(default_xgb_min_child_weight))])
    if default_xgb_reg_lambda is not None:
        cmd.extend(["--default-xgb-reg-lambda", str(float(default_xgb_reg_lambda))])
    if default_xgb_reg_alpha is not None:
        cmd.extend(["--default-xgb-reg-alpha", str(float(default_xgb_reg_alpha))])

    if default_c is not None:
        cmd.extend(["--default-c", str(float(default_c))])
    if default_solver is not None:
        cmd.extend(["--default-solver", str(default_solver)])
    if default_logistic_penalty is not None:
        cmd.extend(["--default-logistic-penalty", str(default_logistic_penalty)])
    if default_logistic_class_weight is not None:
        cmd.extend(["--default-logistic-class-weight", str(default_logistic_class_weight)])
    if default_logistic_enable_tuning is not None:
        cmd.extend(["--default-logistic-enable-tuning", "true" if default_logistic_enable_tuning else "false"])
    if default_logistic_tuning_method is not None:
        cmd.extend(["--default-logistic-tuning-method", str(default_logistic_tuning_method)])
    if default_logistic_cv_folds is not None:
        cmd.extend(["--default-logistic-cv-folds", str(int(default_logistic_cv_folds))])
    if default_logistic_cv_scoring is not None:
        cmd.extend(["--default-logistic-cv-scoring", str(default_logistic_cv_scoring)])
    if default_logistic_tuning_method == "random" and default_logistic_cv_n_iter is not None:
        cmd.extend(["--default-logistic-cv-n-iter", str(int(default_logistic_cv_n_iter))])
    if default_logistic_cv_n_jobs is not None:
        cmd.extend(["--default-logistic-cv-n-jobs", str(int(default_logistic_cv_n_jobs))])

    if default_rf_n_estimators is not None:
        cmd.extend(["--default-rf-n-estimators", str(int(default_rf_n_estimators))])
    if default_rf_max_depth is not None:
        cmd.extend(["--default-rf-max-depth", str(int(default_rf_max_depth))])
    if default_rf_min_samples_leaf is not None:
        cmd.extend(["--default-rf-min-samples-leaf", str(int(default_rf_min_samples_leaf))])
    if default_rf_max_features is not None:
        cmd.extend(["--default-rf-max-features", str(default_rf_max_features)])
    if default_rf_enable_tuning is not None:
        cmd.extend(["--default-rf-enable-tuning", "true" if default_rf_enable_tuning else "false"])
    if default_rf_tuning_method is not None:
        cmd.extend(["--default-rf-tuning-method", str(default_rf_tuning_method)])
    if default_rf_cv_folds is not None:
        cmd.extend(["--default-rf-cv-folds", str(int(default_rf_cv_folds))])
    if default_rf_cv_scoring is not None:
        cmd.extend(["--default-rf-cv-scoring", str(default_rf_cv_scoring)])
    if default_rf_tuning_method == "random" and default_rf_cv_n_iter is not None:
        cmd.extend(["--default-rf-cv-n-iter", str(int(default_rf_cv_n_iter))])
    if default_rf_cv_n_jobs is not None:
        cmd.extend(["--default-rf-cv-n-jobs", str(int(default_rf_cv_n_jobs))])

    if default_xgb_enable_tuning is not None:
        cmd.extend(["--default-xgb-enable-tuning", "true" if default_xgb_enable_tuning else "false"])
    if default_xgb_tuning_method is not None:
        cmd.extend(["--default-xgb-tuning-method", str(default_xgb_tuning_method)])
    if default_xgb_cv_folds is not None:
        cmd.extend(["--default-xgb-cv-folds", str(int(default_xgb_cv_folds))])
    if default_xgb_cv_scoring is not None:
        cmd.extend(["--default-xgb-cv-scoring", str(default_xgb_cv_scoring)])
    if default_xgb_cv_n_iter is not None:
        cmd.extend(["--default-xgb-cv-n-iter", str(int(default_xgb_cv_n_iter))])
    if default_xgb_cv_n_jobs is not None:
        cmd.extend(["--default-xgb-cv-n-jobs", str(int(default_xgb_cv_n_jobs))])

    if default_lr_penalty is not None:
        cmd.extend(["--default-lr-penalty", str(default_lr_penalty)])
    if default_lr_alpha is not None:
        cmd.extend(["--default-lr-alpha", str(float(default_lr_alpha))])
    if default_lr_fit_intercept is not None:
        cmd.extend(["--default-lr-fit-intercept", "true" if default_lr_fit_intercept else "false"])
    if default_lr_l1_ratio is not None:
        cmd.extend(["--default-lr-l1-ratio", str(float(default_lr_l1_ratio))])
    if default_lr_enable_tuning is not None:
        cmd.extend(["--default-lr-enable-tuning", "true" if default_lr_enable_tuning else "false"])
    if default_lr_tuning_method is not None:
        cmd.extend(["--default-lr-tuning-method", str(default_lr_tuning_method)])
    if default_lr_cv_folds is not None:
        cmd.extend(["--default-lr-cv-folds", str(int(default_lr_cv_folds))])
    if default_lr_cv_scoring is not None:
        cmd.extend(["--default-lr-cv-scoring", str(default_lr_cv_scoring)])
    if default_lr_tuning_method == "random" and default_lr_cv_n_iter is not None:
        cmd.extend(["--default-lr-cv-n-iter", str(int(default_lr_cv_n_iter))])
    if default_lr_cv_n_jobs is not None:
        cmd.extend(["--default-lr-cv-n-jobs", str(int(default_lr_cv_n_jobs))])

    # Add TensorFlow-only flags where necessary
    if library == "tensorflow":
        # These variables are guaranteed non-None here due to the prompts above.
        cmd.extend(["--optimizer", optimizer])
        cmd.extend(["--learning_rate", str(float(learning_rate))])
        cmd.extend(["--epochs", str(int(epochs))])
        cmd.extend(["--batch_size", str(int(batch_size))])
        if default_tf_enable_tuning is not None:
            cmd.extend(["--default-tf-enable-tuning", "true" if default_tf_enable_tuning else "false"])
        if default_tf_tuning_method is not None:
            cmd.extend(["--default-tf-tuning-method", str(default_tf_tuning_method)])
        if default_tf_cv_scoring is not None:
            cmd.extend(["--default-tf-cv-scoring", str(default_tf_cv_scoring)])
        if default_tf_cv_n_iter is not None:
            cmd.extend(["--default-tf-cv-n-iter", str(int(default_tf_cv_n_iter))])

    print("\nRunning:")
    print("  " + " ".join(cmd) + "\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0 and "already exists for --name" in (result.stdout + result.stderr):
        print("That name is taken. Re-run this interactive command and choose a different output name.")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())