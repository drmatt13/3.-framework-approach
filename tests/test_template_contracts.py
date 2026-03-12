from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON_EXECUTABLE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

TEMPLATE_FILES = [
    REPO_ROOT / "tools" / "model_templates" / "scikit-learn_linear_regression_template.py",
    REPO_ROOT / "tools" / "model_templates" / "scikit-learn_logistic_regression_template.py",
    REPO_ROOT / "tools" / "model_templates" / "scikit-learn_random_forest_classification_template.py",
    REPO_ROOT / "tools" / "model_templates" / "scikit-learn_random_forest_regression_template.py",
    REPO_ROOT / "tools" / "model_templates" / "tensorflow_dense_neural_network_classification_template.py",
    REPO_ROOT / "tools" / "model_templates" / "tensorflow_dense_neural_network_regression_template.py",
    REPO_ROOT / "tools" / "model_templates" / "xgboost_classification_template.py",
    REPO_ROOT / "tools" / "model_templates" / "xgboost_regression_template.py",
]

REQUIRED_SHARED_HELPER_TOKENS = [
    "_initialize_tuning_summary(",
    "_build_tuning_summary(",
    "_build_training_control(",
    "seed_control = _set_deterministic_seeds(",
]

REQUIRED_CORE_CLI_FLAGS = [
    'parser.add_argument("--name"',
    'parser.add_argument("--artifact-name-mode"',
    'parser.add_argument("--save-model"',
    'parser.add_argument("--verbose"',
    'parser.add_argument("--metric-decimals"',
]

REQUIRED_ARTIFACT_CONTRACT_TOKENS = [
    "_validate_artifact_contract(",
    '"fit_summary": {',
    '"training_control":',
    '"tuning":',
    '"artifacts":',
]


class TemplateContractTests(unittest.TestCase):
    def test_templates_use_shared_helpers(self) -> None:
        for template_path in TEMPLATE_FILES:
            template_text = template_path.read_text(encoding="utf-8")
            with self.subTest(template=str(template_path)):
                for token in REQUIRED_SHARED_HELPER_TOKENS:
                    self.assertIn(token, template_text)

    def test_generated_models_include_contract_sections(self) -> None:
        generation_cases = [
            (
                "gen_sk_lr_reg",
                [
                    "--library", "scikit-learn",
                    "--model", "linear_regression",
                    "--task", "regression",
                    "--starter-dataset", "insurance.csv",
                    "--lr-enable-tuning", "false",
                ],
            ),
            (
                "gen_sk_log_bin",
                [
                    "--library", "scikit-learn",
                    "--model", "logistic_regression",
                    "--task", "binary_classification",
                    "--starter-dataset", "breast_cancer_wisconsin.csv",
                    "--logistic-enable-tuning", "true",
                    "--logistic-tuning-method", "bayesian",
                    "--logistic-cv-n-iter", "2",
                    "--logistic-cv-folds", "3",
                ],
            ),
            (
                "gen_sk_rf_reg",
                [
                    "--library", "scikit-learn",
                    "--model", "random_forest",
                    "--task", "regression",
                    "--starter-dataset", "insurance.csv",
                    "--rf-enable-tuning", "true",
                    "--rf-tuning-method", "bayesian",
                    "--rf-cv-n-iter", "2",
                    "--rf-cv-folds", "3",
                ],
            ),
            (
                "gen_xgb_bin",
                [
                    "--library", "xgboost",
                    "--task", "binary_classification",
                    "--starter-dataset", "breast_cancer_wisconsin.csv",
                    "--xgb-enable-tuning", "true",
                    "--xgb-tuning-method", "bayesian",
                    "--xgb-cv-n-iter", "2",
                    "--xgb-cv-folds", "3",
                    "--n-estimators", "80",
                ],
            ),
            (
                "gen_tf_mc",
                [
                    "--library", "tensorflow",
                    "--model", "dense_nn",
                    "--task", "multiclass_classification",
                    "--starter-dataset", "iris.csv",
                    "--optimizer", "adam",
                    "--learning_rate", "0.001",
                    "--tf-enable-tuning", "true",
                    "--tf-tuning-method", "bayesian",
                    "--tf-cv-n-iter", "2",
                    "--epochs", "8",
                    "--batch_size", "16",
                    "--early-stopping", "true",
                    "--n-iter-no-change", "2",
                    "--validation-fraction", "0.2",
                ],
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            for case_name, generate_args in generation_cases:
                model_output_path = temp_dir_path / f"{case_name}.py"
                command = [
                    PYTHON_EXECUTABLE,
                    str(REPO_ROOT / "tools" / "generate_model.py"),
                    "--name", case_name,
                    "--output", str(model_output_path),
                    *generate_args,
                ]
                completed = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"Generation failed for {case_name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
                )

                model_text = model_output_path.read_text(encoding="utf-8")
                with self.subTest(generated_model=case_name):
                    for token in REQUIRED_SHARED_HELPER_TOKENS:
                        self.assertIn(token, model_text)
                    for token in REQUIRED_CORE_CLI_FLAGS:
                        self.assertIn(token, model_text)
                    for token in REQUIRED_ARTIFACT_CONTRACT_TOKENS:
                        self.assertIn(token, model_text)


if __name__ == "__main__":
    unittest.main()
