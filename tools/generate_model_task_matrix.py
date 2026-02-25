import subprocess
import sys
from datetime import datetime
from pathlib import Path

from generate_base_model import STARTER_DATASETS_BY_FAMILY, XGBOOST_BOOSTERS


SKLEARN_MODEL_TASKS = {
    "linear_regression": ["regression"],
    "logistic_regression": ["binary_classification", "multiclass_classification"],
    "random_forest": ["regression", "binary_classification", "multiclass_classification"],
}

XGBOOST_TASKS = ["regression", "binary_classification", "multiclass_classification"]


def _dataset_stem(dataset_name: str) -> str:
    return dataset_name.replace(".csv", "")


def _run_command(command: list[str], cwd: Path) -> int:
    print("\nRunning:")
    print("  " + " ".join(command))
    result = subprocess.run(command, cwd=cwd)
    return result.returncode


def _run_generated_model(model_script: Path, python_executable: str, cwd: Path) -> int:
    run_command = [python_executable, str(model_script), "--save-model=true"]
    return _run_command(run_command, cwd)


def main() -> int:
    tools_dir = Path(__file__).resolve().parent
    workspace_root = tools_dir.parent
    models_dir = workspace_root / "models"
    generator_path = tools_dir / "generate_base_model.py"

    if not generator_path.exists():
        print(f"Could not find generator script at: {generator_path}", file=sys.stderr)
        return 1

    python_executable = sys.executable
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running scikit-learn generation + training matrix across all starter datasets...")
    for model_name, tasks in SKLEARN_MODEL_TASKS.items():
        for task in tasks:
            datasets = STARTER_DATASETS_BY_FAMILY[task]
            for dataset_name in datasets:
                output_name = f"scikit-learn_{model_name}_{task}_{_dataset_stem(dataset_name)}_{run_stamp}"
                generate_command = [
                    python_executable,
                    str(generator_path),
                    "--library",
                    "scikit-learn",
                    "--model",
                    model_name,
                    "--task",
                    task,
                    "--starter-dataset",
                    dataset_name,
                    "--name",
                    output_name,
                ]

                if _run_command(generate_command, workspace_root) != 0:
                    print("\nStopped due to generation failure.", file=sys.stderr)
                    return 1

                model_script = models_dir / f"{output_name}.py"
                if _run_generated_model(model_script, python_executable, workspace_root) != 0:
                    print("\nStopped due to command failure.", file=sys.stderr)
                    return 1

    print("\nRunning xgboost generation + training matrix across all starter datasets and boosters...")
    for task in XGBOOST_TASKS:
        datasets = STARTER_DATASETS_BY_FAMILY[task]
        for dataset_name in datasets:
            for booster in sorted(XGBOOST_BOOSTERS):
                output_name = f"xgboost_{task}_{_dataset_stem(dataset_name)}_{booster}_{run_stamp}"
                generate_command = [
                    python_executable,
                    str(generator_path),
                    "--library",
                    "xgboost",
                    "--task",
                    task,
                    "--booster",
                    booster,
                    "--starter-dataset",
                    dataset_name,
                    "--name",
                    output_name,
                ]

                if _run_command(generate_command, workspace_root) != 0:
                    print("\nStopped due to generation failure.", file=sys.stderr)
                    return 1

                model_script = models_dir / f"{output_name}.py"
                if _run_generated_model(model_script, python_executable, workspace_root) != 0:
                    print("\nStopped due to command failure.", file=sys.stderr)
                    return 1

    # ---------------------------------------------------------
    # tensorflow (not invoked yet; uncomment when templates/workflow are ready)
    # Defaulted training options used below:
    #   --optimizer adam
    #   --learning_rate 0.001
    #   --epochs 10
    #   --batch_size 32
    # ---------------------------------------------------------
    # Dense NN
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "dense_nn", "--task", "regression", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_dense_regression_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "dense_nn", "--task", "binary_classification", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_dense_binary_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "dense_nn", "--task", "multiclass_classification", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_dense_multiclass_{run_stamp}"]

    # CNN
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "cnn", "--task", "regression", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_cnn_regression_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "cnn", "--task", "binary_classification", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_cnn_binary_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "tensorflow", "--model", "cnn", "--task", "multiclass_classification", "--optimizer", "adam", "--learning_rate", "0.001", "--epochs", "10", "--batch_size", "32", "--name", f"tensorflow_cnn_multiclass_{run_stamp}"]

    print("\nDone. Generated and ran all scikit-learn and xgboost model/task/dataset combinations.")
    print("tensorflow commands are left commented in this script for later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
