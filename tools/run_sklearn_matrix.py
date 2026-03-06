from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MatrixCase:
	name: str
	generate_args: list[str]
	train_args: list[str]


def _run_command(command: list[str], cwd: Path) -> tuple[int, str, str]:
	completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
	return completed.returncode, completed.stdout, completed.stderr


def main() -> int:
	workspace_root = Path(__file__).resolve().parents[1]
	venv_python = workspace_root / ".venv" / "Scripts" / "python.exe"
	python_executable = str(venv_python) if venv_python.exists() else sys.executable
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	models_dir = workspace_root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	cases: list[MatrixCase] = [
		MatrixCase(
			name=f"matrix_lr_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "linear_regression",
				"--task", "regression",
				"--starter-dataset", "insurance.csv",
				"--lr-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_lr_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "linear_regression",
				"--task", "regression",
				"--starter-dataset", "insurance.csv",
				"--lr-enable-tuning", "true",
				"--lr-penalty", "auto",
				"--lr-tuning-method", "random",
				"--lr-cv-folds", "3",
				"--lr-cv-n-iter", "4",
				"--lr-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_log_bin_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "logistic_regression",
				"--task", "binary_classification",
				"--starter-dataset", "breast_cancer_wisconsin.csv",
				"--logistic-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_log_bin_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "logistic_regression",
				"--task", "binary_classification",
				"--starter-dataset", "breast_cancer_wisconsin.csv",
				"--logistic-enable-tuning", "true",
				"--logistic-tuning-method", "random",
				"--logistic-cv-folds", "3",
				"--logistic-cv-n-iter", "4",
				"--logistic-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_log_multi_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "logistic_regression",
				"--task", "multiclass_classification",
				"--starter-dataset", "iris.csv",
				"--logistic-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_log_multi_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "logistic_regression",
				"--task", "multiclass_classification",
				"--starter-dataset", "iris.csv",
				"--logistic-enable-tuning", "true",
				"--logistic-tuning-method", "random",
				"--logistic-cv-folds", "3",
				"--logistic-cv-n-iter", "4",
				"--logistic-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_reg_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "regression",
				"--starter-dataset", "insurance.csv",
				"--rf-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_reg_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "regression",
				"--starter-dataset", "insurance.csv",
				"--rf-enable-tuning", "true",
				"--rf-tuning-method", "random",
				"--rf-cv-folds", "3",
				"--rf-cv-n-iter", "4",
				"--rf-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_bin_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "binary_classification",
				"--starter-dataset", "breast_cancer_wisconsin.csv",
				"--rf-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_bin_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "binary_classification",
				"--starter-dataset", "breast_cancer_wisconsin.csv",
				"--rf-enable-tuning", "true",
				"--rf-tuning-method", "random",
				"--rf-cv-folds", "3",
				"--rf-cv-n-iter", "4",
				"--rf-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_multi_direct_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "multiclass_classification",
				"--starter-dataset", "iris.csv",
				"--rf-enable-tuning", "false",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
		MatrixCase(
			name=f"matrix_rf_multi_tuned_{timestamp}",
			generate_args=[
				"--library", "scikit-learn",
				"--model", "random_forest",
				"--task", "multiclass_classification",
				"--starter-dataset", "iris.csv",
				"--rf-enable-tuning", "true",
				"--rf-tuning-method", "random",
				"--rf-cv-folds", "3",
				"--rf-cv-n-iter", "4",
				"--rf-cv-n-jobs", "-1",
			],
			train_args=["--save-model=true", "--verbose", "0", "--artifact-name-mode", "short"],
		),
	]

	summary: dict[str, object] = {
		"timestamp": timestamp,
		"workspace": str(workspace_root),
		"cases": [],
	}

	for case in cases:
		model_script_path = models_dir / f"{case.name}.py"
		generate_command = [
			python_executable,
			str(workspace_root / "tools" / "generate_model.py"),
			"--name",
			case.name,
			"--output",
			str(model_script_path),
			*case.generate_args,
		]
		generate_returncode, generate_stdout, generate_stderr = _run_command(generate_command, workspace_root)
		case_result: dict[str, object] = {
			"case": case.name,
			"generate": {
				"returncode": generate_returncode,
				"stdout": generate_stdout[-4000:],
				"stderr": generate_stderr[-4000:],
			},
		}
		if generate_returncode != 0:
			summary["cases"].append(case_result)
			continue

		train_command = [
			python_executable,
			str(model_script_path),
			*case.train_args,
		]
		train_returncode, train_stdout, train_stderr = _run_command(train_command, workspace_root)
		case_result["train"] = {
			"returncode": train_returncode,
			"stdout": train_stdout[-4000:],
			"stderr": train_stderr[-4000:],
		}
		summary["cases"].append(case_result)

	audit_command = [python_executable, str(workspace_root / "tools" / "audit_artifacts.py")]
	audit_returncode, audit_stdout, audit_stderr = _run_command(audit_command, workspace_root)
	summary["audit"] = {
		"returncode": audit_returncode,
		"stdout": audit_stdout[-4000:],
		"stderr": audit_stderr[-4000:],
	}

	analysis_dir = workspace_root / "artifacts" / "analysis"
	analysis_dir.mkdir(parents=True, exist_ok=True)
	summary_path = analysis_dir / "sklearn_matrix_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	failed_cases = [item for item in summary["cases"] if item.get("generate", {}).get("returncode") != 0 or item.get("train", {}).get("returncode") != 0]
	print(f"Matrix summary: {summary_path}")
	print(f"Total cases: {len(cases)}")
	print(f"Failed cases: {len(failed_cases)}")
	print(f"Audit return code: {audit_returncode}")
	return 1 if failed_cases or audit_returncode != 0 else 0


if __name__ == "__main__":
	raise SystemExit(main())
