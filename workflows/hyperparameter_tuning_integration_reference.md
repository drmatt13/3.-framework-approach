# Hyperparameter Tuning Integration Reference

This workflow captures a reusable pattern for introducing template-level hyperparameter tuning support while preserving deterministic model generation and backward compatibility.

## Scope of This Iteration

Applied for:
- `scikit-learn` + `linear_regression`

Updated files:
- [model_init.py](model_init.py)
- [tools/generate_model.py](tools/generate_model.py)

Related template dependency:
- [tools/model_templates/scikit-learn_linear_regression_template.py](tools/model_templates/scikit-learn_linear_regression_template.py)

## What Was Updated (and Why)

### 1) Interactive defaults in `model_init.py`

Added linear-regression tuning defaults to profile/custom flows:
- `default_lr_enable_tuning`
- `default_lr_tuning_method`
- `default_lr_cv_folds`
- `default_lr_cv_scoring`
- `default_lr_cv_n_iter`
- `default_lr_cv_n_jobs`

Why:
- Ensures generated model defaults can be configured from interactive setup, not hardcoded in template.
- Keeps behavior aligned with existing default injection style used by other model families.

### 2) Generator support in `tools/generate_model.py`

Added parser + validation + replacement support for LR tuning defaults:
- New generator flags: `--default-lr-enable-tuning`, `--default-lr-tuning-method`, `--default-lr-cv-folds`, `--default-lr-cv-scoring`, `--default-lr-cv-n-iter`, `--default-lr-cv-n-jobs`
- Added validation constraints (e.g., folds >= 2, n_iter > 0, l1_ratio range).
- Added scikit-learn model-routing checks so LR tuning defaults are accepted only for linear regression.
- Extended default map and `template_replacements(...)` so placeholders are always populated.

Why:
- Creates a complete defaults pipeline: init prompt -> generator args -> replacement values -> rendered model defaults.
- Prevents invalid flag combinations and keeps templates deterministic.

### 3) Template placeholder parameterization

Changed linear-regression template tuning defaults from hardcoded literals to replacement placeholders.

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
python .\tools\generate_model.py --library scikit-learn --model linear_regression --task regression --name lr_tune_defaults_test --default-lr-enable-tuning true --default-lr-tuning-method random --default-lr-cv-folds 4 --default-lr-cv-scoring mae --default-lr-cv-n-iter 30 --default-lr-cv-n-jobs -1
```
