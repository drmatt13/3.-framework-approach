# Hyperparameter Tuning Integration Reference

This workflow captures a reusable pattern for introducing template-level hyperparameter tuning support while preserving deterministic model generation and backward compatibility.

## Scope of This Iteration

Applied for:
- `scikit-learn` + `linear_regression`, `logistic_regression`, `random_forest`
- `xgboost` + `classification`, `regression`
- `tensorflow` + `dense_nn` (`classification`, `regression`)

Not included:
- `tensorflow` + `cnn`

Updated files:
- [model_init.py](model_init.py)
- [tools/generate_model.py](tools/generate_model.py)

Related template dependency:
- [tools/model_templates/scikit-learn_linear_regression_template.py](tools/model_templates/scikit-learn_linear_regression_template.py)
- [tools/model_templates/scikit-learn_logistic_regression_template.py](tools/model_templates/scikit-learn_logistic_regression_template.py)
- [tools/model_templates/scikit-learn_random_forest_classification_template.py](tools/model_templates/scikit-learn_random_forest_classification_template.py)
- [tools/model_templates/scikit-learn_random_forest_regression_template.py](tools/model_templates/scikit-learn_random_forest_regression_template.py)
- [tools/model_templates/xgboost_classification_template.py](tools/model_templates/xgboost_classification_template.py)
- [tools/model_templates/xgboost_regression_template.py](tools/model_templates/xgboost_regression_template.py)
- [tools/model_templates/tensorflow_dense_neural_network_classification_template.py](tools/model_templates/tensorflow_dense_neural_network_classification_template.py)
- [tools/model_templates/tensorflow_dense_neural_network_regression_template.py](tools/model_templates/tensorflow_dense_neural_network_regression_template.py)

## What Was Updated (and Why)

### 1) Interactive defaults in `model_init.py`

Added non-CNN tuning defaults to profile/custom flows:
- linear regression: `default_lr_*`
- logistic regression: `default_logistic_*`
- random forest: `default_rf_*`
- xgboost: `default_xgb_*`
- tensorflow dense: `default_tf_*`

Why:
- Ensures generated model defaults can be configured from interactive setup, not hardcoded in template.
- Keeps behavior aligned with existing default injection style used by other model families.

### 2) Generator support in `tools/generate_model.py`

Added parser + validation + replacement support for non-CNN tuning defaults:
- New generator flags for logistic/rf/xgb/tf dense tuning defaults
- Expanded validation constraints (folds, iteration ranges, scoring constraints)
- Added model/library routing checks for each default family
- Extended default maps and `template_replacements(...)` so placeholders are always populated

Why:
- Creates a complete defaults pipeline: init prompt -> generator args -> replacement values -> rendered model defaults.
- Prevents invalid flag combinations and keeps templates deterministic.

### 3) Template placeholder parameterization

Changed non-CNN templates to use replacement placeholders for tuning defaults and emit consistent `tuning` metadata.

Why:
- Without placeholders, generator-level tuning defaults cannot affect generated model files.

## Generic Rollout Checklist for Other Models

Use this checklist when adding tuning support to another model template:

1. Template defaults
   - Add/confirm tuning runtime flags exist.
   - Replace hardcoded default literals with `{{...}}` placeholders.

2. Generator default map
   - Extend model-specific `DEFAULT_*_PARAMS_BY_TEMPLATE` with tuning default keys.

3. Generator parser
   - Add model-specific `--default-...` tuning flags.

4. Generator validation
   - Add value-range validation.
   - Add model/library routing restrictions.

5. Template replacements
   - Resolve defaults from args or map fallback.
   - Inject all tuning placeholders.

6. Interactive setup (`model_init.py`)
   - Add profile defaults for tuning knobs.
   - Add custom prompts for tuning knobs.
   - Forward new defaults into generator command.

7. Verification
   - Generate with profile defaults.
   - Generate with explicit tuning defaults.
   - Negative test: pass model-specific tuning defaults to wrong model.
   - Confirm rendered output has no unresolved placeholders for new keys.

## Validation Commands (Suggested)

```powershell
python .\model_init.py
```

```powershell
python .\tools\generate_model.py --library xgboost --task regression --name xgb_tune_defaults_test --default-xgb-enable-tuning true --default-xgb-cv-scoring rmse --default-xgb-cv-n-iter 10
```

```powershell
python .\tools\generate_model.py --library tensorflow --model dense_nn --task binary_classification --name tf_tune_defaults_test --optimizer adam --learning_rate 0.001 --epochs 25 --batch_size 32 --default-tf-enable-tuning true --default-tf-cv-scoring f1_macro --default-tf-cv-n-iter 8
```
