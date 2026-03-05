import subprocess
import sys
from pathlib import Path
from prompt_toolkit.styles import Style
from libraries.logistic_compat import (
    NON_AUTO_LOGISTIC_SOLVERS,
    compatible_solvers_for_penalty,
)

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

XGBOOST_BOOSTERS = ["auto", "gbtree", "gblinear", "dart"]
XGBOOST_DEVICE_DEFAULTS = ["cpu", "gpu"]
SKLEARN_LOGISTIC_SOLVERS = ["auto", *list(NON_AUTO_LOGISTIC_SOLVERS)]

# Only meaningful for TensorFlow models (gradient-based training)
TENSORFLOW_OPTIMIZERS = ["auto", "adam", "sgd", "rmsprop", "adagrad", "adamw"]

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
        "style": CUSTOM_STYLE,
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


def _is_optional_positive_int_text(s: str) -> bool:
    v = s.strip().lower()
    return v in {"none", "null"} or (_is_int(v) and int(v) > 0)


def _is_optional_nonzero_int_text(s: str) -> bool:
    v = s.strip().lower()
    return v in {"none", "null"} or (_is_int(v) and int(v) != 0)


def _is_rf_max_features_text(s: str) -> bool:
    v = s.strip().lower()
    if v in {"auto", "sqrt", "log2", "none"}:
        return True
    if not _is_float(v):
        return False
    return 0.0 < float(v) <= 1.0


def _is_rf_max_samples_text(s: str) -> bool:
    v = s.strip().lower()
    if v in {"none", "null"}:
        return True
    if _is_int(v):
        return int(v) > 0
    if _is_float(v):
        return 0.0 < float(v) <= 1.0
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
    # Hide random-search iteration settings when tuning is disabled or method is grid.
    n_iter_by_method: tuple[tuple[str, str], ...] = (
        ("xgb_cv_n_iter", "xgb_tuning_method"),
        ("logistic_cv_n_iter", "logistic_tuning_method"),
        ("rf_cv_n_iter", "rf_tuning_method"),
        ("lr_cv_n_iter", "lr_tuning_method"),
        ("tf_cv_n_iter", "tf_tuning_method"),
    )
    for n_iter_key, method_key in n_iter_by_method:
        if key == n_iter_key:
            enable_key = f"{n_iter_key.removesuffix('_cv_n_iter')}_enable_tuning"
            if enable_key in values_by_key and not _is_truthy(values_by_key[enable_key]):
                return True
            if method_key in values_by_key:
                return str(values_by_key[method_key]).strip().lower() == "grid"
            return False

    tuning_dependencies: dict[str, tuple[str, ...]] = {
        "xgb_enable_tuning": (
            "xgb_tuning_method",
            "xgb_cv_folds",
            "xgb_cv_scoring",
            "xgb_cv_n_iter",
            "xgb_cv_n_jobs",
        ),
        "logistic_enable_tuning": (
            "logistic_tuning_method",
            "logistic_cv_folds",
            "logistic_cv_scoring",
            "logistic_cv_n_iter",
            "logistic_cv_n_jobs",
        ),
        "rf_enable_tuning": (
            "rf_tuning_method",
            "rf_cv_folds",
            "rf_cv_scoring",
            "rf_cv_n_iter",
            "rf_cv_n_jobs",
        ),
        "lr_enable_tuning": (
            "lr_tuning_method",
            "lr_cv_folds",
            "lr_cv_scoring",
            "lr_cv_n_iter",
            "lr_cv_n_jobs",
        ),
        "tf_enable_tuning": (
            "tf_tuning_method",
            "tf_cv_scoring",
            "tf_cv_n_iter",
        ),
    }

    for enable_key, dependent_keys in tuning_dependencies.items():
        if key in dependent_keys and enable_key in values_by_key:
            return not _is_truthy(values_by_key[enable_key])

    if key == "rf_max_samples":
        if "rf_enable_tuning" in values_by_key and _is_truthy(values_by_key["rf_enable_tuning"]):
            return True
        if "rf_bootstrap" in values_by_key:
            return not _is_truthy(values_by_key["rf_bootstrap"])

    # When tuning is enabled, omit direct estimator defaults to simplify summary.
    direct_defaults_by_enable: dict[str, tuple[str, ...]] = {
        "xgb_enable_tuning": (
            "n_estimators",
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "xgb_min_child_weight",
            "xgb_reg_lambda",
            "xgb_reg_alpha",
        ),
        "logistic_enable_tuning": (
            "c",
            "logistic_class_weight",
        ),
        "rf_enable_tuning": (
            "rf_n_estimators",
            "rf_max_depth",
            "rf_min_samples_split",
            "rf_min_samples_leaf",
            "rf_min_weight_fraction_leaf",
            "rf_max_leaf_nodes",
            "rf_min_impurity_decrease",
            "rf_max_features",
            "rf_bootstrap",
            "rf_max_samples",
            "rf_ccp_alpha",
            "rf_n_jobs",
        ),
        "lr_enable_tuning": (
            "lr_alpha",
            "lr_fit_intercept",
            "lr_l1_ratio",
        ),
    }
    for enable_key, direct_keys in direct_defaults_by_enable.items():
        if key in direct_keys and enable_key in values_by_key:
            return _is_truthy(values_by_key[enable_key])

    return False


def _resolved_display_key(key: str) -> str:
    explicit_map = {
        "xgb_min_child_weight": "min_child_weight",
        "xgb_reg_lambda": "reg_lambda",
        "xgb_reg_alpha": "reg_alpha",
        "logistic_penalty": "penalty",
        "logistic_class_weight": "class_weight",
        "rf_n_estimators": "n_estimators",
        "rf_max_depth": "max_depth",
        "rf_min_samples_split": "min_samples_split",
        "rf_min_samples_leaf": "min_samples_leaf",
        "rf_min_weight_fraction_leaf": "min_weight_fraction_leaf",
        "rf_max_leaf_nodes": "max_leaf_nodes",
        "rf_min_impurity_decrease": "min_impurity_decrease",
        "rf_max_features": "max_features",
        "rf_bootstrap": "bootstrap",
        "rf_max_samples": "max_samples",
        "rf_ccp_alpha": "ccp_alpha",
        "rf_n_jobs": "n_jobs",
        "lr_penalty": "penalty",
        "lr_alpha": "alpha",
        "lr_fit_intercept": "fit_intercept",
        "lr_l1_ratio": "l1_ratio",
        "tf_learning_rate": "learning_rate",
    }
    if key in explicit_map:
        return explicit_map[key]

    suffix_groups = (
        "enable_tuning",
        "tuning_method",
        "cv_folds",
        "cv_scoring",
        "cv_n_iter",
        "cv_n_jobs",
    )
    for suffix in suffix_groups:
        if key.endswith(f"_{suffix}"):
            return suffix

    return key


def _supports_early_stopping_defaults(library: str, model: str | None, task: str) -> bool:
    if library == "tensorflow" and model == "dense_nn":
        return True
    if library == "xgboost":
        return True
    return False


def _supports_validation_n_iter_defaults(library: str, model: str | None, task: str) -> bool:
    if library == "tensorflow" and model == "dense_nn":
        return True
    if library == "xgboost":
        return True
    return False


def _supports_max_iter(library: str, model: str | None, task: str) -> bool:
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


def _xgb_booster_uses_tree_params(booster: str | None) -> bool:
    return booster in {"gbtree", "dart", "auto"}


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
            "n_estimators": str(p["n_est"]),
            "learning_rate": str(p["lr"]),
            "max_depth": str(p["depth"]),
            "subsample": str(p["sub"]),
            "colsample_bytree": str(p["col"]),
            "xgb_min_child_weight": str(p["mcw"]),
            "xgb_reg_lambda": str(p["rl"]),
            "xgb_reg_alpha": str(p["ra"]),
            "early_stopping": p["es"],
            "validation_fraction": p["vf"],
            "n_iter_no_change": p["nic"],
            "xgb_enable_tuning": p["enable_tuning"],
            "xgb_tuning_method": p["tuning_method"],
            "xgb_cv_folds": str(p["cv_folds"]),
            "xgb_cv_scoring": p["cv_scoring"],
            "xgb_cv_n_iter": str(p["cv_n_iter"]),
            "xgb_cv_n_jobs": str(p["cv_n_jobs"]),
        }

    elif library == "scikit-learn" and model == "random_forest":
        presets = {
            "Quick": {
                "n_est": 100,
                "depth": 8,
                "mss": 4,
                "msl": 2,
                "mwfl": 0.0,
                "mln": None,
                "mid": 0.0,
                "mf": "sqrt",
                "bootstrap": True,
                "max_samples": 1.0,
                "ccp_alpha": 0.0,
                "n_jobs": -1,
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
                "mss": 4,
                "msl": 2,
                "mwfl": 0.0,
                "mln": None,
                "mid": 0.0,
                "mf": "sqrt",
                "bootstrap": True,
                "max_samples": 1.0,
                "ccp_alpha": 0.0,
                "n_jobs": -1,
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
                "mss": 4,
                "msl": 2,
                "mwfl": 0.0,
                "mln": None,
                "mid": 0.0,
                "mf": "sqrt",
                "bootstrap": True,
                "max_samples": 1.0,
                "ccp_alpha": 0.0,
                "n_jobs": -1,
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
            "rf_n_estimators": str(p["n_est"]),
            "rf_max_depth": str(p["depth"]),
            "rf_min_samples_split": str(p["mss"]),
            "rf_min_samples_leaf": str(p["msl"]),
            "rf_min_weight_fraction_leaf": str(p["mwfl"]),
            "rf_max_leaf_nodes": "none" if p["mln"] is None else str(p["mln"]),
            "rf_min_impurity_decrease": str(p["mid"]),
            "rf_max_features": str(p["mf"]),
            "rf_bootstrap": p["bootstrap"],
            "rf_max_samples": str(p["max_samples"]),
            "rf_ccp_alpha": str(p["ccp_alpha"]),
            "rf_n_jobs": str(p["n_jobs"]),
            "rf_enable_tuning": p["enable_tuning"],
            "rf_tuning_method": p["tuning_method"],
            "rf_cv_folds": str(p["cv_folds"]),
            "rf_cv_scoring": p["cv_scoring"],
            "rf_cv_n_iter": str(p["cv_n_iter"]),
            "rf_cv_n_jobs": str(p["cv_n_jobs"]),
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
            "c": str(p["c"]),
            "solver": p["solver"],
            "max_iter": p["max_iter"],
            "logistic_penalty": p["penalty"],
            "logistic_class_weight": p["class_weight"],
            "logistic_enable_tuning": p["enable_tuning"],
            "logistic_tuning_method": p["tuning_method"],
            "logistic_cv_folds": str(p["cv_folds"]),
            "logistic_cv_scoring": p["cv_scoring"],
            "logistic_cv_n_iter": str(p["cv_n_iter"]),
            "logistic_cv_n_jobs": str(p["cv_n_jobs"]),
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
            "lr_penalty": p["penalty"],
            "lr_alpha": str(p["alpha"]),
            "lr_fit_intercept": p["fit_intercept"],
            "lr_l1_ratio": str(p["l1_ratio"]),
            "lr_enable_tuning": p["enable_tuning"],
            "lr_tuning_method": p["tuning_method"],
            "lr_cv_folds": str(p["cv_folds"]),
            "lr_cv_scoring": p["cv_scoring"],
            "lr_cv_n_iter": str(p["cv_n_iter"]),
            "lr_cv_n_jobs": str(p["cv_n_jobs"]),
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

    # Optional xgboost defaults (asked after tuning mode is known)
    booster = None
    device = None

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
    tf_learning_rate = None
    epochs = None
    batch_size = None

    early_stopping = None
    validation_fraction = None
    n_iter_no_change = None
    max_iter = None
    n_estimators = None
    learning_rate = None
    max_depth = None
    subsample = None
    colsample_bytree = None
    c = None
    solver = None
    rf_n_estimators = None
    rf_max_depth = None
    rf_min_samples_split = None
    rf_min_samples_leaf = None
    rf_min_weight_fraction_leaf = None
    rf_max_leaf_nodes = None
    rf_min_impurity_decrease = None
    rf_max_features = None
    rf_bootstrap = None
    rf_max_samples = None
    rf_ccp_alpha = None
    rf_n_jobs = None
    logistic_penalty = None
    logistic_class_weight = None
    logistic_enable_tuning = None
    logistic_tuning_method = None
    logistic_cv_folds = None
    logistic_cv_scoring = None
    logistic_cv_n_iter = None
    logistic_cv_n_jobs = None
    rf_enable_tuning = None
    rf_tuning_method = None
    rf_cv_folds = None
    rf_cv_scoring = None
    rf_cv_n_iter = None
    rf_cv_n_jobs = None
    lr_penalty = None
    lr_alpha = None
    lr_fit_intercept = None
    lr_l1_ratio = None
    lr_enable_tuning = None
    lr_tuning_method = None
    lr_cv_folds = None
    lr_cv_scoring = None
    lr_cv_n_iter = None
    lr_cv_n_jobs = None
    xgb_min_child_weight = None
    xgb_reg_lambda = None
    xgb_reg_alpha = None
    xgb_enable_tuning = None
    xgb_tuning_method = None
    xgb_cv_folds = None
    xgb_cv_scoring = None
    xgb_cv_n_iter = None
    xgb_cv_n_jobs = None
    tf_enable_tuning = None
    tf_tuning_method = None
    tf_cv_scoring = None
    tf_cv_n_iter = None

    # When a profile provides defaults, use them and skip prompts.
    use_custom = profile == "Custom" or library == "tensorflow"

    if library == "xgboost":
        if use_custom:
            booster = _ask_select(
                "Select xgboost booster family:",
                choices=["gbtree", "gblinear", "dart"],
            )
            if booster is None:
                print("Cancelled.")
                return 0

            xgb_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if xgb_enable_tuning is None:
                print("Cancelled.")
                return 0

            if xgb_enable_tuning:
                enable_auto_booster = questionary.confirm(
                    "Allow tuning to search across booster families (--booster auto)?",
                    default=False,
                    style=CUSTOM_STYLE,
                ).ask()
                if enable_auto_booster is None:
                    print("Cancelled.")
                    return 0
                if enable_auto_booster:
                    booster = "auto"
                    print("Note: booster=auto enables cross-family search in tuning.")
                else:
                    print(
                        f"Note: tuning search will stay within the selected booster family ({booster})."
                    )

            device = _ask_select(
                "Select xgboost device:",
                choices=XGBOOST_DEVICE_DEFAULTS,
            )
            if device is None:
                print("Cancelled.")
                return 0

            if xgb_enable_tuning:
                xgb_tuning_method = "random"
                xgb_cv_n_iter = _ask_text(
                    "Enter random-search iterations (>0):",
                    default="20",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                )
                if xgb_cv_n_iter is None:
                    print("Cancelled.")
                    return 0
                xgb_cv_folds = _ask_text(
                    "Enter CV folds (>=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if xgb_cv_folds is None:
                    print("Cancelled.")
                    return 0
                xgb_cv_scoring = _ask_select(
                    "Select CV scoring metric:",
                    choices=["rmse", "mae", "r2"] if task == "regression" else ["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if xgb_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                xgb_cv_n_jobs = _ask_text(
                    "Enter cv_n_jobs (integer != 0; -1 for all cores):",
                    default="-1",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) != 0) else "Must be an integer != 0",
                )
                if xgb_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0

            if not xgb_enable_tuning:
                n_estimators = _ask_text(
                    "Enter n_estimators:",
                    default="300",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                )
                if n_estimators is None:
                    print("Cancelled.")
                    return 0

                learning_rate = _ask_text(
                    "Enter learning rate:",
                    default="0.1",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
                )
                if learning_rate is None:
                    print("Cancelled.")
                    return 0

                if _xgb_booster_uses_tree_params(booster):
                    max_depth = _ask_text(
                        "Enter max depth:",
                        default="6",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if max_depth is None:
                        print("Cancelled.")
                        return 0

                    subsample = _ask_text(
                        "Enter subsample (0 < value <= 1):",
                        default="1.0",
                        validate_fn=lambda s: True if (_is_float(s) and 0 < float(s) <= 1.0) else "Must be in range (0, 1]",
                    )
                    if subsample is None:
                        print("Cancelled.")
                        return 0

                    colsample_bytree = _ask_text(
                        "Enter colsample_bytree (0 < value <= 1):",
                        default="1.0",
                        validate_fn=lambda s: True if (_is_float(s) and 0 < float(s) <= 1.0) else "Must be in range (0, 1]",
                    )
                    if colsample_bytree is None:
                        print("Cancelled.")
                        return 0

                    xgb_min_child_weight = _ask_text(
                        "Enter min_child_weight:",
                        default="1.0",
                        validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
                    )
                    if xgb_min_child_weight is None:
                        print("Cancelled.")
                        return 0
                else:
                    max_depth = None
                    subsample = None
                    colsample_bytree = None
                    xgb_min_child_weight = None
                    print("Note: booster=gblinear ignores tree-only params (max_depth/subsample/colsample_bytree/min_child_weight).")

                xgb_reg_lambda = _ask_text(
                    "Enter reg_lambda (L2 regularization):",
                    default="1.0",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0) else "Must be a non-negative number",
                )
                if xgb_reg_lambda is None:
                    print("Cancelled.")
                    return 0

                xgb_reg_alpha = _ask_text(
                    "Enter reg_alpha (L1 regularization):",
                    default="0.0",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0) else "Must be a non-negative number",
                )
                if xgb_reg_alpha is None:
                    print("Cancelled.")
                    return 0
        else:
            n_estimators = profile_defaults.get("n_estimators", "300")
            learning_rate = profile_defaults.get("learning_rate", "0.1")
            max_depth = profile_defaults.get("max_depth", "6")
            subsample = profile_defaults.get("subsample", "1.0")
            colsample_bytree = profile_defaults.get("colsample_bytree", "1.0")
            xgb_min_child_weight = profile_defaults.get("xgb_min_child_weight", "1.0")
            xgb_reg_lambda = profile_defaults.get("xgb_reg_lambda", "1.0")
            xgb_reg_alpha = profile_defaults.get("xgb_reg_alpha", "0.0")
            xgb_enable_tuning = profile_defaults.get("xgb_enable_tuning", False)
            xgb_tuning_method = profile_defaults.get("xgb_tuning_method", "random")
            xgb_cv_folds = profile_defaults.get("xgb_cv_folds", "5")
            xgb_cv_scoring = profile_defaults.get(
                "xgb_cv_scoring",
                "rmse" if task == "regression" else "f1_macro",
            )
            xgb_cv_n_iter = profile_defaults.get("xgb_cv_n_iter", "20")
            xgb_cv_n_jobs = profile_defaults.get("xgb_cv_n_jobs", "-1")

            booster_choices = ["gbtree", "auto", "gblinear", "dart"] if _is_truthy(xgb_enable_tuning) else ["gbtree", "gblinear", "dart"]
            booster = _ask_select(
                "Select xgboost booster:",
                choices=booster_choices,
            )
            if booster is None:
                print("Cancelled.")
                return 0

            if _is_truthy(xgb_enable_tuning):
                if booster == "auto":
                    print("Note: booster=auto enables cross-family search in tuning.")
                else:
                    print(
                        f"Note: tuning search will stay within the selected booster family ({booster})."
                    )
            elif not _xgb_booster_uses_tree_params(booster):
                max_depth = None
                subsample = None
                colsample_bytree = None
                xgb_min_child_weight = None

            device = _ask_select(
                "Select xgboost device:",
                choices=XGBOOST_DEVICE_DEFAULTS,
            )
            if device is None:
                print("Cancelled.")
                return 0

    if library == "scikit-learn" and model == "logistic_regression":
        if use_custom:
            logistic_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if logistic_enable_tuning is None:
                print("Cancelled.")
                return 0

            penalty_choices = ["auto", "l1", "l2", "elasticnet"] if logistic_enable_tuning else ["none", "l1", "l2", "elasticnet"]
            logistic_penalty = _ask_select(
                "Select penalty:",
                choices=penalty_choices,
            )
            if logistic_penalty is None:
                print("Cancelled.")
                return 0

            if logistic_enable_tuning and logistic_penalty == "auto":
                solver_choices = SKLEARN_LOGISTIC_SOLVERS
            else:
                non_auto_solver_choices = compatible_solvers_for_penalty(logistic_penalty)
                solver_choices = ["auto", *non_auto_solver_choices] if logistic_enable_tuning else non_auto_solver_choices

            if len(solver_choices) == 1:
                solver = solver_choices[0]
                print(
                    f"Note: penalty={logistic_penalty} constrains "
                    f"solver to '{solver}'."
                )
            else:
                solver = _ask_select(
                    "Select solver:",
                    choices=solver_choices,
                )
                if solver is None:
                    print("Cancelled.")
                    return 0

            if logistic_enable_tuning:
                logistic_tuning_method = _ask_select(
                    "Select tuning method:",
                    choices=["grid", "random"],
                )
                if logistic_tuning_method is None:
                    print("Cancelled.")
                    return 0
                if logistic_tuning_method == "random":
                    logistic_cv_n_iter = _ask_text(
                        "Enter random-search iterations (>0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if logistic_cv_n_iter is None:
                        print("Cancelled.")
                        return 0
                logistic_cv_folds = _ask_text(
                    "Enter CV folds (>=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if logistic_cv_folds is None:
                    print("Cancelled.")
                    return 0
                logistic_cv_scoring = _ask_select(
                    "Select CV scoring metric:",
                    choices=["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if logistic_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                logistic_cv_n_jobs = _ask_text(
                    "Enter cv_n_jobs (integer != 0; -1 for all cores):",
                    default="-1",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) != 0) else "Must be an integer != 0",
                )
                if logistic_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0

            if not logistic_enable_tuning:
                if logistic_penalty != "none":
                    c = _ask_text(
                        "Enter C (inverse regularization strength):",
                        default="1.0",
                        validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
                    )
                    if c is None:
                        print("Cancelled.")
                        return 0
                else:
                    c = None

                logistic_class_weight = _ask_select(
                    "Select class_weight:",
                    choices=["none", "balanced"],
                )
                if logistic_class_weight is None:
                    print("Cancelled.")
                    return 0
        else:
            c = profile_defaults.get("c", "1.0")
            solver = profile_defaults.get("solver", "lbfgs")
            logistic_penalty = profile_defaults.get("logistic_penalty", "l2")
            logistic_class_weight = profile_defaults.get("logistic_class_weight", "none")
            logistic_enable_tuning = profile_defaults.get("logistic_enable_tuning", False)
            logistic_tuning_method = profile_defaults.get("logistic_tuning_method", "grid")
            logistic_cv_folds = profile_defaults.get("logistic_cv_folds", "5")
            logistic_cv_scoring = profile_defaults.get("logistic_cv_scoring", "f1_macro")
            logistic_cv_n_iter = profile_defaults.get("logistic_cv_n_iter", "20")
            logistic_cv_n_jobs = profile_defaults.get("logistic_cv_n_jobs", "-1")

    if library == "scikit-learn" and model == "random_forest":
        if use_custom:
            rf_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if rf_enable_tuning is None:
                print("Cancelled.")
                return 0
            if rf_enable_tuning:
                rf_tuning_method = _ask_select(
                    "Select tuning method:",
                    choices=["grid", "random"],
                )
                if rf_tuning_method is None:
                    print("Cancelled.")
                    return 0
                if rf_tuning_method == "random":
                    rf_cv_n_iter = _ask_text(
                        "Enter random-search iterations (>0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if rf_cv_n_iter is None:
                        print("Cancelled.")
                        return 0
                rf_cv_folds = _ask_text(
                    "Enter CV folds (>=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if rf_cv_folds is None:
                    print("Cancelled.")
                    return 0
                rf_cv_scoring = _ask_select(
                    "Select CV scoring metric:",
                    choices=["rmse", "mae", "r2"] if task == "regression" else ["f1_macro", "accuracy", "roc_auc_ovr"],
                )
                if rf_cv_scoring is None:
                    print("Cancelled.")
                    return 0
                rf_cv_n_jobs = _ask_text(
                    "Enter cv_n_jobs (integer != 0; -1 for all cores):",
                    default="-1",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) != 0) else "Must be an integer != 0",
                )
                if rf_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0

            if not rf_enable_tuning:
                rf_n_estimators = _ask_text(
                    "Enter n_estimators:",
                    default="300",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                )
                if rf_n_estimators is None:
                    print("Cancelled.")
                    return 0

                rf_max_depth_preset = _ask_select(
                    "Select max_depth:",
                    choices=["16", "32", "64", "unlimited", "custom"],
                )
                if rf_max_depth_preset is None:
                    print("Cancelled.")
                    return 0

                if rf_max_depth_preset == "custom":
                    rf_max_depth = _ask_text(
                        "Enter max_depth:",
                        default="16",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if rf_max_depth is None:
                        print("Cancelled.")
                        return 0
                elif rf_max_depth_preset == "unlimited":
                    rf_max_depth = "none"
                else:
                    rf_max_depth = rf_max_depth_preset

                rf_max_leaf_nodes_preset = _ask_select(
                    "Select max_leaf_nodes:",
                    choices=["unlimited", "64", "128", "256", "512", "custom"],
                )
                if rf_max_leaf_nodes_preset is None:
                    print("Cancelled.")
                    return 0

                if rf_max_leaf_nodes_preset == "custom":
                    rf_max_leaf_nodes = _ask_text(
                        "Enter max_leaf_nodes (>=2):",
                        default="128",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                    )
                    if rf_max_leaf_nodes is None:
                        print("Cancelled.")
                        return 0
                elif rf_max_leaf_nodes_preset == "unlimited":
                    rf_max_leaf_nodes = "none"
                else:
                    rf_max_leaf_nodes = rf_max_leaf_nodes_preset

                rf_min_samples_split = _ask_text(
                    "Enter min_samples_split:",
                    default="4",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if rf_min_samples_split is None:
                    print("Cancelled.")
                    return 0

                rf_min_samples_leaf = _ask_text(
                    "Enter min_samples_leaf:",
                    default="2",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 1) else "Must be an integer >= 1",
                )
                if rf_min_samples_leaf is None:
                    print("Cancelled.")
                    return 0

                rf_min_impurity_decrease = _ask_text(
                    "Enter min_impurity_decrease:",
                    default="0.0",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0.0) else "Must be a non-negative number",
                )
                if rf_min_impurity_decrease is None:
                    print("Cancelled.")
                    return 0

                rf_min_weight_fraction_leaf = _ask_text(
                    "Enter min_weight_fraction_leaf (0 to 0.5):",
                    default="0.0",
                    validate_fn=lambda s: True if (_is_float(s) and 0.0 <= float(s) <= 0.5) else "Must be a number in range [0, 0.5]",
                )
                if rf_min_weight_fraction_leaf is None:
                    print("Cancelled.")
                    return 0

                rf_max_features_preset = _ask_select(
                    "Select max_features:",
                    choices=["sqrt", "log2", "auto", "none", "custom"],
                )
                if rf_max_features_preset is None:
                    print("Cancelled.")
                    return 0

                if rf_max_features_preset == "custom":
                    rf_max_features = _ask_text(
                        "Enter max_features as fraction (0 < value <= 1):",
                        default="0.5",
                        validate_fn=lambda s: True if (_is_float(s) and 0.0 < float(s) <= 1.0) else "Must be a float in (0, 1]",
                    )
                    if rf_max_features is None:
                        print("Cancelled.")
                        return 0
                else:
                    rf_max_features = rf_max_features_preset

                rf_bootstrap = questionary.confirm(
                    "Enable bootstrap sampling?",
                    default=True,
                    style=CUSTOM_STYLE,
                ).ask()
                if rf_bootstrap is None:
                    print("Cancelled.")
                    return 0

                if rf_bootstrap:
                    rf_max_samples_preset = _ask_select(
                        "Select max_samples:",
                        choices=["1.0 (all rows)", "0.8", "0.7", "0.5", "custom"],
                    )
                    if rf_max_samples_preset is None:
                        print("Cancelled.")
                        return 0

                    if rf_max_samples_preset == "custom":
                        rf_max_samples = _ask_text(
                            "Enter max_samples (int > 0 or float in (0,1]):",
                            default="0.8",
                            validate_fn=lambda s: True if _is_rf_max_samples_text(s) and s.strip().lower() not in {"none", "null"} else "Must be int > 0 or float in (0,1]",
                        )
                        if rf_max_samples is None:
                            print("Cancelled.")
                            return 0
                    elif rf_max_samples_preset == "1.0 (all rows)":
                        rf_max_samples = "1.0"
                    else:
                        rf_max_samples = rf_max_samples_preset
                else:
                    rf_max_samples = "none"

                rf_ccp_alpha = _ask_text(
                    "Enter ccp_alpha (cost-complexity pruning):",
                    default="0.0",
                    validate_fn=lambda s: True if (_is_float(s) and float(s) >= 0.0) else "Must be a non-negative number",
                )
                if rf_ccp_alpha is None:
                    print("Cancelled.")
                    return 0

                rf_n_jobs = _ask_text(
                    "Enter n_jobs (integer != 0; -1 for all cores):",
                    default="-1",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) != 0) else "Must be an integer != 0",
                )
                if rf_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            rf_n_estimators = profile_defaults.get("rf_n_estimators", "300")
            rf_max_depth = profile_defaults.get("rf_max_depth", "16")
            rf_min_samples_split = profile_defaults.get("rf_min_samples_split", "4")
            rf_min_samples_leaf = profile_defaults.get("rf_min_samples_leaf", "2")
            rf_min_weight_fraction_leaf = profile_defaults.get("rf_min_weight_fraction_leaf", "0.0")
            rf_max_leaf_nodes = profile_defaults.get("rf_max_leaf_nodes", "none")
            rf_min_impurity_decrease = profile_defaults.get("rf_min_impurity_decrease", "0.0")
            rf_max_features = profile_defaults.get("rf_max_features", "sqrt")
            rf_bootstrap = profile_defaults.get("rf_bootstrap", True)
            rf_max_samples = profile_defaults.get("rf_max_samples", "1.0")
            rf_ccp_alpha = profile_defaults.get("rf_ccp_alpha", "0.0")
            rf_n_jobs = profile_defaults.get("rf_n_jobs", "-1")
            rf_enable_tuning = profile_defaults.get("rf_enable_tuning", False)
            rf_tuning_method = profile_defaults.get("rf_tuning_method", "grid")
            rf_cv_folds = profile_defaults.get("rf_cv_folds", "5")
            rf_cv_scoring = profile_defaults.get(
                "rf_cv_scoring",
                "rmse" if task == "regression" else "f1_macro",
            )
            rf_cv_n_iter = profile_defaults.get("rf_cv_n_iter", "20")
            rf_cv_n_jobs = profile_defaults.get("rf_cv_n_jobs", "-1")

    if library == "tensorflow" and model == "dense_nn":
        tf_enable_tuning = questionary.confirm(
            "Enable hyperparameter tuning by default? (--enable-tuning)",
            default=False,
            style=CUSTOM_STYLE,
        ).ask()
        if tf_enable_tuning is None:
            print("Cancelled.")
            return 0

        optimizer_choices = ["adam", "auto", "sgd", "rmsprop", "adagrad", "adamw"] if tf_enable_tuning else ["adam", "sgd", "rmsprop", "adagrad", "adamw"]
        optimizer = _ask_select(
            "Select optimizer:",
            choices=optimizer_choices,
        )
        if optimizer is None:
            print("Cancelled.")
            return 0

        if tf_enable_tuning:
            if optimizer == "auto":
                print("Note: optimizer=auto lets tuning search across all optimizer families.")
            else:
                print(f"Note: tuning will keep optimizer fixed to '{optimizer}'.")

        tf_learning_rate = _ask_text(
            "Enter learning rate (e.g., 0.001):",
            default="0.001",
            validate_fn=lambda s: True
            if (_is_float(s) and float(s) > 0)
            else "Must be a positive number",
        )
        if tf_learning_rate is None:
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

        if tf_enable_tuning:
            tf_tuning_method = "random"
            tf_cv_n_iter = _ask_text(
                "Enter random-search iterations (>0):",
                default="10",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if tf_cv_n_iter is None:
                print("Cancelled.")
                return 0
            tf_cv_scoring = _ask_select(
                "Select tuning scoring metric:",
                choices=["rmse"] if task == "regression" else ["f1_macro"],
            )
            if tf_cv_scoring is None:
                print("Cancelled.")
                return 0

    if library == "scikit-learn" and model == "linear_regression":
        if use_custom:
            lr_enable_tuning = questionary.confirm(
                "Enable hyperparameter tuning by default in generated template? (--enable-tuning)",
                default=False,
                style=CUSTOM_STYLE,
            ).ask()
            if lr_enable_tuning is None:
                print("Cancelled.")
                return 0

            lr_penalty_choices = ["auto", "l1", "l2", "elasticnet"] if lr_enable_tuning else ["none", "l1", "l2", "elasticnet"]
            lr_penalty = _ask_select(
                "Select penalty:",
                choices=lr_penalty_choices,
            )
            if lr_penalty is None:
                print("Cancelled.")
                return 0

            if not lr_enable_tuning:
                if lr_penalty == "elasticnet":
                    lr_l1_ratio = _ask_text(
                        "Enter l1_ratio for ElasticNet (0=pure L2, 1=pure L1):",
                        default="0.5",
                        validate_fn=lambda s: True if (_is_float(s) and 0.0 <= float(s) <= 1.0) else "Must be in range [0, 1]",
                    )
                    if lr_l1_ratio is None:
                        print("Cancelled.")
                        return 0
                else:
                    lr_l1_ratio = "0.5"

                if lr_penalty != "none":
                    lr_alpha = _ask_text(
                        "Enter alpha (regularization strength):",
                        default="1.0",
                        validate_fn=lambda s: True if (_is_float(s) and float(s) > 0) else "Must be a positive number",
                    )
                    if lr_alpha is None:
                        print("Cancelled.")
                        return 0
                else:
                    lr_alpha = "1.0"

                lr_fit_intercept = questionary.confirm(
                    "Fit intercept?",
                    default=True,
                    style=CUSTOM_STYLE,
                ).ask()
                if lr_fit_intercept is None:
                    print("Cancelled.")
                    return 0
            else:
                lr_alpha = None
                lr_fit_intercept = None
                lr_l1_ratio = None

            if lr_enable_tuning:
                lr_tuning_method = _ask_select(
                    "Select tuning method:",
                    choices=["grid", "random"],
                )
                if lr_tuning_method is None:
                    print("Cancelled.")
                    return 0
                if lr_tuning_method == "random":
                    lr_cv_n_iter = _ask_text(
                        "Enter random-search iterations (>0):",
                        default="20",
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if lr_cv_n_iter is None:
                        print("Cancelled.")
                        return 0

                lr_cv_folds = _ask_text(
                    "Enter CV folds (>=2):",
                    default="5",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) >= 2) else "Must be an integer >= 2",
                )
                if lr_cv_folds is None:
                    print("Cancelled.")
                    return 0

                lr_cv_scoring = _ask_select(
                    "Select CV scoring metric:",
                    choices=["rmse", "mae", "r2"],
                )
                if lr_cv_scoring is None:
                    print("Cancelled.")
                    return 0

                lr_cv_n_jobs = _ask_text(
                    "Enter cv_n_jobs (integer != 0; -1 for all cores):",
                    default="-1",
                    validate_fn=lambda s: True if (_is_int(s) and int(s) != 0) else "Must be an integer != 0",
                )
                if lr_cv_n_jobs is None:
                    print("Cancelled.")
                    return 0
        else:
            lr_penalty = profile_defaults.get("lr_penalty", "none")
            lr_alpha = profile_defaults.get("lr_alpha", "1.0")
            lr_fit_intercept = profile_defaults.get("lr_fit_intercept", True)
            lr_l1_ratio = profile_defaults.get("lr_l1_ratio", "0.5")
            lr_enable_tuning = profile_defaults.get("lr_enable_tuning", False)
            lr_tuning_method = profile_defaults.get("lr_tuning_method", "grid")
            lr_cv_folds = profile_defaults.get("lr_cv_folds", "5")
            lr_cv_scoring = profile_defaults.get("lr_cv_scoring", "rmse")
            lr_cv_n_iter = profile_defaults.get("lr_cv_n_iter", "20")
            lr_cv_n_jobs = profile_defaults.get("lr_cv_n_jobs", "-1")

    if _supports_max_iter(library, model, task):
        if use_custom:
            max_iter = _ask_text(
                "Enter max iterations:",
                default="1000",
                validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
            )
            if max_iter is None:
                print("Cancelled.")
                return 0
            max_iter = int(max_iter)
        else:
            max_iter = int(profile_defaults.get("max_iter", 1000))

    if _supports_early_stopping_defaults(library, model, task):
        supports_validation_defaults = _supports_validation_n_iter_defaults(library, model, task)
        if use_custom:
            recommended_early_stopping, recommended_validation_fraction, recommended_n_iter_no_change = _recommended_es_defaults(
                library,
                model,
            )

            early_stopping = questionary.confirm(
                "Enable early stopping?",
                default=bool(recommended_early_stopping),
                style=CUSTOM_STYLE,
            ).ask()

            if early_stopping is None:
                print("Cancelled.")
                return 0

            if supports_validation_defaults:
                use_recommended_defaults = questionary.confirm(
                    "Use recommended preset values for validation fraction and n_iter_no_change?",
                    default=True,
                    style=CUSTOM_STYLE,
                ).ask()

                if use_recommended_defaults is None:
                    print("Cancelled.")
                    return 0

                if use_recommended_defaults:
                    validation_fraction = recommended_validation_fraction
                    n_iter_no_change = recommended_n_iter_no_change
                else:
                    validation_fraction = _ask_text(
                        "Enter validation fraction (0 < value < 1):",
                        default=str(recommended_validation_fraction),
                        validate_fn=lambda s: True
                        if (_is_float(s) and 0.0 < float(s) < 1.0)
                        else "Must be a number where 0 < value < 1",
                    )
                    if validation_fraction is None:
                        print("Cancelled.")
                        return 0

                    n_iter_no_change = _ask_text(
                        "Enter n_iter_no_change:",
                        default=str(recommended_n_iter_no_change),
                        validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
                    )
                    if n_iter_no_change is None:
                        print("Cancelled.")
                        return 0

                    validation_fraction = float(validation_fraction)
                    n_iter_no_change = int(n_iter_no_change)
        else:
            # Profile provides ES defaults
            recommended_early_stopping, _, _ = _recommended_es_defaults(library, model)
            early_stopping = profile_defaults.get("early_stopping", recommended_early_stopping)
            if supports_validation_defaults:
                validation_fraction = profile_defaults.get("validation_fraction", 0.1)
                n_iter_no_change = profile_defaults.get("n_iter_no_change", 5)

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
        ("tf_learning_rate", tf_learning_rate),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("early_stopping", early_stopping),
        ("validation_fraction", validation_fraction),
        ("n_iter_no_change", n_iter_no_change),
        ("max_iter", max_iter),
        ("n_estimators", n_estimators),
        ("learning_rate", learning_rate),
        ("max_depth", max_depth),
        ("subsample", subsample),
        ("colsample_bytree", colsample_bytree),
        ("xgb_min_child_weight", xgb_min_child_weight),
        ("xgb_reg_lambda", xgb_reg_lambda),
        ("xgb_reg_alpha", xgb_reg_alpha),
        ("xgb_enable_tuning", xgb_enable_tuning),
        ("xgb_tuning_method", xgb_tuning_method),
        ("xgb_cv_folds", xgb_cv_folds),
        ("xgb_cv_scoring", xgb_cv_scoring),
        ("xgb_cv_n_iter", xgb_cv_n_iter),
        ("xgb_cv_n_jobs", xgb_cv_n_jobs),
        ("c", c),
        ("solver", solver),
        ("logistic_penalty", logistic_penalty),
        ("logistic_class_weight", logistic_class_weight),
        ("logistic_enable_tuning", logistic_enable_tuning),
        ("logistic_tuning_method", logistic_tuning_method),
        ("logistic_cv_folds", logistic_cv_folds),
        ("logistic_cv_scoring", logistic_cv_scoring),
        ("logistic_cv_n_iter", logistic_cv_n_iter),
        ("logistic_cv_n_jobs", logistic_cv_n_jobs),
        ("rf_n_estimators", rf_n_estimators),
        ("rf_max_depth", rf_max_depth),
        ("rf_min_samples_split", rf_min_samples_split),
        ("rf_min_samples_leaf", rf_min_samples_leaf),
        ("rf_min_weight_fraction_leaf", rf_min_weight_fraction_leaf),
        ("rf_max_leaf_nodes", rf_max_leaf_nodes),
        ("rf_min_impurity_decrease", rf_min_impurity_decrease),
        ("rf_max_features", rf_max_features),
        ("rf_bootstrap", rf_bootstrap),
        ("rf_max_samples", rf_max_samples),
        ("rf_ccp_alpha", rf_ccp_alpha),
        ("rf_n_jobs", rf_n_jobs),
        ("rf_enable_tuning", rf_enable_tuning),
        ("rf_tuning_method", rf_tuning_method),
        ("rf_cv_folds", rf_cv_folds),
        ("rf_cv_scoring", rf_cv_scoring),
        ("rf_cv_n_iter", rf_cv_n_iter),
        ("rf_cv_n_jobs", rf_cv_n_jobs),
        ("lr_penalty", lr_penalty),
        ("lr_alpha", lr_alpha),
        ("lr_fit_intercept", lr_fit_intercept),
        ("lr_l1_ratio", lr_l1_ratio),
        ("lr_enable_tuning", lr_enable_tuning),
        ("lr_tuning_method", lr_tuning_method),
        ("lr_cv_folds", lr_cv_folds),
        ("lr_cv_scoring", lr_cv_scoring),
        ("lr_cv_n_iter", lr_cv_n_iter),
        ("lr_cv_n_jobs", lr_cv_n_jobs),
        ("tf_enable_tuning", tf_enable_tuning),
        ("tf_tuning_method", tf_tuning_method),
        ("tf_cv_scoring", tf_cv_scoring),
        ("tf_cv_n_iter", tf_cv_n_iter),
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
        display_key = _resolved_display_key(key)
        print(f"  {display_key}={_stringify_setting(value)}")

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
        cmd.extend(["--default-early-stopping", "true" if early_stopping else "false"])
    if _supports_validation_n_iter_defaults(library, model, task):
        cmd.extend(["--default-validation-fraction", str(float(validation_fraction))])
        cmd.extend(["--default-n-iter-no-change", str(int(n_iter_no_change))])

    if max_iter is not None:
        cmd.extend(["--default-max-iter", str(int(max_iter))])

    if n_estimators is not None:
        cmd.extend(["--default-n-estimators", str(int(n_estimators))])
    if learning_rate is not None:
        cmd.extend(["--default-learning-rate", str(float(learning_rate))])
    if max_depth is not None:
        cmd.extend(["--default-max-depth", str(int(max_depth))])
    if subsample is not None:
        cmd.extend(["--default-subsample", str(float(subsample))])
    if colsample_bytree is not None:
        cmd.extend(["--default-colsample-bytree", str(float(colsample_bytree))])
    if xgb_min_child_weight is not None:
        cmd.extend(["--default-xgb-min-child-weight", str(float(xgb_min_child_weight))])
    if xgb_reg_lambda is not None:
        cmd.extend(["--default-xgb-reg-lambda", str(float(xgb_reg_lambda))])
    if xgb_reg_alpha is not None:
        cmd.extend(["--default-xgb-reg-alpha", str(float(xgb_reg_alpha))])

    if c is not None:
        cmd.extend(["--default-c", str(float(c))])
    if solver is not None:
        cmd.extend(["--default-solver", str(solver)])
    if logistic_penalty is not None:
        cmd.extend(["--default-logistic-penalty", str(logistic_penalty)])
    if logistic_class_weight is not None:
        cmd.extend(["--default-logistic-class-weight", str(logistic_class_weight)])
    if logistic_enable_tuning is not None:
        cmd.extend(["--default-logistic-enable-tuning", "true" if logistic_enable_tuning else "false"])
    if logistic_tuning_method is not None:
        cmd.extend(["--default-logistic-tuning-method", str(logistic_tuning_method)])
    if logistic_cv_folds is not None:
        cmd.extend(["--default-logistic-cv-folds", str(int(logistic_cv_folds))])
    if logistic_cv_scoring is not None:
        cmd.extend(["--default-logistic-cv-scoring", str(logistic_cv_scoring)])
    if logistic_tuning_method == "random" and logistic_cv_n_iter is not None:
        cmd.extend(["--default-logistic-cv-n-iter", str(int(logistic_cv_n_iter))])
    if logistic_cv_n_jobs is not None:
        cmd.extend(["--default-logistic-cv-n-jobs", str(int(logistic_cv_n_jobs))])

    if rf_n_estimators is not None:
        cmd.extend(["--default-rf-n-estimators", str(int(rf_n_estimators))])
    if rf_max_depth is not None:
        cmd.extend(["--default-rf-max-depth", str(rf_max_depth).strip().lower()])
    if rf_min_samples_split is not None:
        cmd.extend(["--default-rf-min-samples-split", str(int(rf_min_samples_split))])
    if rf_min_samples_leaf is not None:
        cmd.extend(["--default-rf-min-samples-leaf", str(int(rf_min_samples_leaf))])
    if rf_min_weight_fraction_leaf is not None:
        cmd.extend(["--default-rf-min-weight-fraction-leaf", str(float(rf_min_weight_fraction_leaf))])
    if rf_max_leaf_nodes is not None:
        cmd.extend(["--default-rf-max-leaf-nodes", str(rf_max_leaf_nodes).strip().lower()])
    if rf_min_impurity_decrease is not None:
        cmd.extend(["--default-rf-min-impurity-decrease", str(float(rf_min_impurity_decrease))])
    if rf_max_features is not None:
        cmd.extend(["--default-rf-max-features", str(rf_max_features).strip().lower()])
    if rf_bootstrap is not None:
        cmd.extend(["--default-rf-bootstrap", "true" if rf_bootstrap else "false"])
    if rf_max_samples is not None:
        cmd.extend(["--default-rf-max-samples", str(rf_max_samples).strip().lower()])
    if rf_ccp_alpha is not None:
        cmd.extend(["--default-rf-ccp-alpha", str(float(rf_ccp_alpha))])
    if rf_n_jobs is not None:
        cmd.extend(["--default-rf-n-jobs", str(rf_n_jobs).strip().lower()])
    if rf_enable_tuning is not None:
        cmd.extend(["--default-rf-enable-tuning", "true" if rf_enable_tuning else "false"])
    if rf_tuning_method is not None:
        cmd.extend(["--default-rf-tuning-method", str(rf_tuning_method)])
    if rf_cv_folds is not None:
        cmd.extend(["--default-rf-cv-folds", str(int(rf_cv_folds))])
    if rf_cv_scoring is not None:
        cmd.extend(["--default-rf-cv-scoring", str(rf_cv_scoring)])
    if rf_tuning_method == "random" and rf_cv_n_iter is not None:
        cmd.extend(["--default-rf-cv-n-iter", str(int(rf_cv_n_iter))])
    if rf_cv_n_jobs is not None:
        cmd.extend(["--default-rf-cv-n-jobs", str(int(rf_cv_n_jobs))])

    if xgb_enable_tuning is not None:
        cmd.extend(["--default-xgb-enable-tuning", "true" if xgb_enable_tuning else "false"])
    if xgb_tuning_method is not None:
        cmd.extend(["--default-xgb-tuning-method", str(xgb_tuning_method)])
    if xgb_cv_folds is not None:
        cmd.extend(["--default-xgb-cv-folds", str(int(xgb_cv_folds))])
    if xgb_cv_scoring is not None:
        cmd.extend(["--default-xgb-cv-scoring", str(xgb_cv_scoring)])
    if xgb_cv_n_iter is not None:
        cmd.extend(["--default-xgb-cv-n-iter", str(int(xgb_cv_n_iter))])
    if xgb_cv_n_jobs is not None:
        cmd.extend(["--default-xgb-cv-n-jobs", str(int(xgb_cv_n_jobs))])

    if lr_penalty is not None:
        cmd.extend(["--default-lr-penalty", str(lr_penalty)])
    if lr_alpha is not None:
        cmd.extend(["--default-lr-alpha", str(float(lr_alpha))])
    if lr_fit_intercept is not None:
        cmd.extend(["--default-lr-fit-intercept", "true" if lr_fit_intercept else "false"])
    if lr_l1_ratio is not None:
        cmd.extend(["--default-lr-l1-ratio", str(float(lr_l1_ratio))])
    if lr_enable_tuning is not None:
        cmd.extend(["--default-lr-enable-tuning", "true" if lr_enable_tuning else "false"])
    if lr_tuning_method is not None:
        cmd.extend(["--default-lr-tuning-method", str(lr_tuning_method)])
    if lr_cv_folds is not None:
        cmd.extend(["--default-lr-cv-folds", str(int(lr_cv_folds))])
    if lr_cv_scoring is not None:
        cmd.extend(["--default-lr-cv-scoring", str(lr_cv_scoring)])
    if lr_tuning_method == "random" and lr_cv_n_iter is not None:
        cmd.extend(["--default-lr-cv-n-iter", str(int(lr_cv_n_iter))])
    if lr_cv_n_jobs is not None:
        cmd.extend(["--default-lr-cv-n-jobs", str(int(lr_cv_n_jobs))])

    # Add TensorFlow-only flags where necessary
    if library == "tensorflow":
        # These variables are guaranteed non-None here due to the prompts above.
        cmd.extend(["--optimizer", optimizer])
        cmd.extend(["--learning_rate", str(float(tf_learning_rate))])
        cmd.extend(["--epochs", str(int(epochs))])
        cmd.extend(["--batch_size", str(int(batch_size))])
        if tf_enable_tuning is not None:
            cmd.extend(["--default-tf-enable-tuning", "true" if tf_enable_tuning else "false"])
        if tf_tuning_method is not None:
            cmd.extend(["--default-tf-tuning-method", str(tf_tuning_method)])
        if tf_cv_scoring is not None:
            cmd.extend(["--default-tf-cv-scoring", str(tf_cv_scoring)])
        if tf_cv_n_iter is not None:
            cmd.extend(["--default-tf-cv-n-iter", str(int(tf_cv_n_iter))])

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