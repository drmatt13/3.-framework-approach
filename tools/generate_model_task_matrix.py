import subprocess
import sys
from datetime import datetime
from pathlib import Path


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

    # ---------------------------------------------------------
    # scikit-learn (invoked)
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

    sklearn_jobs = [
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "linear_regression",
            "--task",
            "regression",
            ],
            f"scikit-learn_linear_regression_{run_stamp}",
        ),
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "logistic_regression",
            "--task",
            "binary_classification",
            ],
            f"scikit-learn_logistic_binary_{run_stamp}",
        ),
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "logistic_regression",
            "--task",
            "multiclass_classification",
            ],
            f"scikit-learn_logistic_multiclass_{run_stamp}",
        ),
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "random_forest",
            "--task",
            "regression",
            ],
            f"scikit-learn_random_forest_regression_{run_stamp}",
        ),
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "random_forest",
            "--task",
            "binary_classification",
            ],
            f"scikit-learn_random_forest_binary_{run_stamp}",
        ),
        (
            [
            python_executable,
            str(generator_path),
            "--library",
            "scikit-learn",
            "--model",
            "random_forest",
            "--task",
            "multiclass_classification",
            ],
            f"scikit-learn_random_forest_multiclass_{run_stamp}",
        ),
    ]

    print("Running scikit-learn generation + training matrix...")
    for command, model_name in sklearn_jobs:
        generate_command = command + ["--name", model_name]
        print("\nRunning:")
        print("  " + " ".join(generate_command))
        result = subprocess.run(generate_command, cwd=workspace_root)
        if result.returncode != 0:
            print("\nStopped due to generation failure.", file=sys.stderr)
            return result.returncode

        model_script = models_dir / f"{model_name}.py"
        run_command = [python_executable, str(model_script), "--save-model=true"]
        print("\nRunning:")
        print("  " + " ".join(run_command))
        result = subprocess.run(run_command, cwd=workspace_root)
        if result.returncode != 0:
            print("\nStopped due to command failure.", file=sys.stderr)
            return result.returncode

    # ---------------------------------------------------------
    # xgboost (not invoked yet; uncomment when templates/workflow are ready)
    # ---------------------------------------------------------
    # [python_executable, str(generator_path), "--library", "xgboost", "--task", "regression", "--name", f"xgboost_regression_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "xgboost", "--task", "binary_classification", "--name", f"xgboost_binary_{run_stamp}"]
    # [python_executable, str(generator_path), "--library", "xgboost", "--task", "multiclass_classification", "--name", f"xgboost_multiclass_{run_stamp}"]
    # Optional booster examples:
    # [python_executable, str(generator_path), "--library", "xgboost", "--task", "regression", "--booster", "gbtree", "--name", f"xgboost_regression_gbtree_{run_stamp}"]

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

    print("\nDone. Generated and ran one model for each scikit-learn task combination.")
    print("xgboost/tensorflow commands are left commented in this script for later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
