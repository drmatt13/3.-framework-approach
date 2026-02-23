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

CUSTOM_STYLE = Style.from_dict(
    {
        "qmark": "fg:#f8b808 bold",  # Question mark
        "question": "bold",  # Question text
        "answer": "fg:#3fb0f0 bold",  # Selected answer after choice
        "pointer": "fg:#f8b808 bold",  # Arrow pointer (>)
        "highlighted": "fg:#ffffff bg:#222222 bold",  # Highlighted option in menu
        "selected": "fg:#00ff00",  # Selected checkbox item (if used)
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
            "Select xgboost booster (optional):",
            choices=["(skip)"] + XGBOOST_BOOSTERS,
            use_shortcuts=True,
            style=CUSTOM_STYLE,
        ).ask()

        if booster is None:
            print("Cancelled.")
            return 0

        if booster == "(skip)":
            booster = None

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

    name = _ask_text(
        "Enter output name (no .py):",
        validate_fn=lambda s: True if s.strip() else "Name cannot be empty",
    )

    if name is None:
        print("Cancelled.")
        return 0

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

    if booster is not None:
        cmd.extend(["--booster", booster])

    # Add TensorFlow-only flags where necessary
    if library == "tensorflow":
        # These variables are guaranteed non-None here due to the prompts above.
        cmd.extend(["--optimizer", optimizer])
        cmd.extend(["--learning_rate", str(float(learning_rate))])
        cmd.extend(["--epochs", str(int(epochs))])
        cmd.extend(["--batch_size", str(int(batch_size))])

    print("\nRunning:")
    print("  " + " ".join(cmd) + "\n")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())