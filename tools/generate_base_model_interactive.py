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
TENSORFLOW_MODELS = ["dense_nn", "cnn"]

# Valid tasks per model/library (based on your flag combinations)
TASKS_BY_LIBRARY_MODEL = {
    ("scikit-learn", "linear_regression"): ["regression"],
    ("scikit-learn", "logistic_regression"): ["binary_classification", "multiclass_classification"],
    ("scikit-learn", "random_forest"): ["regression", "binary_classification", "multiclass_classification"],
    ("tensorflow", "dense_nn"): ["regression", "binary_classification", "multiclass_classification"],
    ("tensorflow", "cnn"): ["regression", "binary_classification", "multiclass_classification"],
    # xgboost has no --model in your interface; task applies directly to library
    ("xgboost", None): ["regression", "binary_classification", "multiclass_classification"],
}

XGBOOST_BOOSTERS = ["gbtree", "gblinear", "dart"]

# Only meaningful for TensorFlow models (gradient-based training)
TENSORFLOW_OPTIMIZERS = ["adam", "sgd", "rmsprop", "adagrad", "adamw"]

STARTER_DATASETS_BY_TASK = {
    "regression": ["ames_housing.csv", "california_housing.csv", "insurance.csv"],
    "binary_classification": ["adult_income.csv", "breast_cancer_wisconsin.csv", "titanic.csv"],
    "multiclass_classification": ["car_evaluation.csv", "iris.csv", "mushrooms.csv"],
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


def _supports_early_stopping_defaults(library: str, model: str | None, task: str) -> bool:
    if task == "regression":
        return False
    if library == "xgboost":
        return True
    return library == "scikit-learn" and model in {"logistic_regression", "random_forest"}


def _supports_default_max_iter(library: str, model: str | None, task: str) -> bool:
    return library == "scikit-learn" and model == "logistic_regression" and task in {
        "binary_classification",
        "multiclass_classification",
    }


def _recommended_es_defaults(library: str, model: str | None) -> tuple[bool, float, int]:
    if library == "xgboost":
        return True, 0.1, 20
    if library == "scikit-learn" and model in {"logistic_regression", "random_forest"}:
        return True, 0.1, 5
    return True, 0.1, 5


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    generator_path = script_dir / "generate_base_model.py"

    if not generator_path.exists():
        print(f"Could not find generator script at: {generator_path}", file=sys.stderr)
        return 1

    library = questionary.select(
        "Select library:",
        choices=["scikit-learn", "xgboost", "tensorflow"],
        use_shortcuts=True,
        style=CUSTOM_STYLE,
    ).ask()

    if library is None:
        print("Cancelled.")
        return 0

    model = None
    if library == "scikit-learn":
        model = questionary.select(
            "Select scikit-learn model:",
            choices=SKLEARN_MODELS,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()
        if model is None:
            print("Cancelled.")
            return 0

    if library == "tensorflow":
        model = questionary.select(
            "Select tensorflow model:",
            choices=TENSORFLOW_MODELS,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()
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
        task = questionary.select(
            "Select task:",
            choices=task_choices,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()

        if task is None:
            print("Cancelled.")
            return 0

    # Optional xgboost booster selection
    booster = None
    if library == "xgboost":
        booster = questionary.select(
            "Default xgboost booster:",
            choices=XGBOOST_BOOSTERS,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()

        if booster is None:
            print("Cancelled.")
            return 0

    # Optional starter dataset selection (task-aware)
    starter_dataset = None
    starter_choices = STARTER_DATASETS_BY_TASK.get(task, [])
    if starter_choices:
        starter_dataset = questionary.select(
            "Select starter template dataset:",
            choices=starter_choices,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()

        if starter_dataset is None:
            print("Cancelled.")
            return 0

    # TensorFlow training knobs (ONLY where necessary)
    optimizer = None
    learning_rate = None
    epochs = None
    batch_size = None

    if library == "tensorflow":
        optimizer = questionary.select(
            "Select optimizer:",
            choices=TENSORFLOW_OPTIMIZERS,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()
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
            "Enter epochs (e.g., 10):",
            default="10",
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

    if _supports_default_max_iter(library, model, task):
        default_max_iter = _ask_text(
            "Default max iterations for template --max-iter:",
            default="1000",
            validate_fn=lambda s: True if (_is_int(s) and int(s) > 0) else "Must be a positive integer",
        )
        if default_max_iter is None:
            print("Cancelled.")
            return 0
        default_max_iter = int(default_max_iter)

    if _supports_early_stopping_defaults(library, model, task):
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

    models_dir = script_dir.parent / "models"
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

    if starter_dataset is not None:
        cmd.extend(["--starter-dataset", starter_dataset])

    if _supports_early_stopping_defaults(library, model, task):
        cmd.extend(["--default-early-stopping", "true" if default_early_stopping else "false"])
        cmd.extend(["--default-validation-fraction", str(float(default_validation_fraction))])
        cmd.extend(["--default-n-iter-no-change", str(int(default_n_iter_no_change))])

    if default_max_iter is not None:
        cmd.extend(["--default-max-iter", str(int(default_max_iter))])

    # Add TensorFlow-only flags where necessary
    if library == "tensorflow":
        # These variables are guaranteed non-None here due to the prompts above.
        cmd.extend(["--optimizer", optimizer])
        cmd.extend(["--learning_rate", str(float(learning_rate))])
        cmd.extend(["--epochs", str(int(epochs))])
        cmd.extend(["--batch_size", str(int(batch_size))])

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