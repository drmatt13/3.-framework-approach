# Hyperparameter Tuning Artifact Consistency Reference

This reference documents artifact quality rules used for hyperparameter tuning across non-CNN templates (`scikit-learn`, `xgboost`, and `tensorflow dense`) and provides a repeatable rollout pattern for new templates.

## Why This Reference Exists

During tuning integration, artifact export failed with:

- `TypeError: Object of type Ridge is not JSON serializable`

Root Cause:
- `best_params` from CV search included estimator objects (for example, `"regressor": Ridge(...)`) that were written into `metrics.json` / `run.json`.
- JSON serialization failed because estimator instances are not JSON-safe by default.

Additional Issue Found During Validation:
- The non-tuning path did not call `model.fit(...)`, causing `NotFittedError` before artifact export.

## What Was Updated (Non-CNN Templates)

Representative files:
- [tools/model_templates/scikit-learn_linear_regression_template.py](tools/model_templates/scikit-learn_linear_regression_template.py)
- [tools/model_templates/scikit-learn_logistic_regression_template.py](tools/model_templates/scikit-learn_logistic_regression_template.py)
- [tools/model_templates/scikit-learn_random_forest_classification_template.py](tools/model_templates/scikit-learn_random_forest_classification_template.py)
- [tools/model_templates/scikit-learn_random_forest_regression_template.py](tools/model_templates/scikit-learn_random_forest_regression_template.py)
- [tools/model_templates/xgboost_classification_template.py](tools/model_templates/xgboost_classification_template.py)
- [tools/model_templates/xgboost_regression_template.py](tools/model_templates/xgboost_regression_template.py)
- [tools/model_templates/tensorflow_dense_neural_network_classification_template.py](tools/model_templates/tensorflow_dense_neural_network_classification_template.py)
- [tools/model_templates/tensorflow_dense_neural_network_regression_template.py](tools/model_templates/tensorflow_dense_neural_network_regression_template.py)

Changes:
1. Added JSON-safe tuning-parameter sanitization helpers:
   - `_json_safe_param_value(...)`
   - `_json_safe_best_params(...)`
2. In tuning flow:
   - Capture raw search params from `search.best_params_`
   - Keep raw params for `model.set_params(...)`
   - Write sanitized params to artifacts (`best_params`)
3. Restored non-tuning training path:
   - Added `model.fit(X_train, y_train)` when `--enable-tuning=false`
4. Standardized metadata blocks:
   - `training_control`, `selection`, and `tuning` consistently written to `metrics.json` and `run.json`

## Artifact Contract Expectations for Tuning

For consistency with current schema standards:

- `metrics.json`
  - MUST include `training_control`, `selection`, `timing`
  - SHOULD include `tuning` with:
    - `enabled`, `method`, `cv_folds`, `scoring`, `scoring_sklearn`
    - `n_iter` (random only), `n_candidates`
    - `best_score`, `best_score_std`, `best_params`
- `run.json`
  - MUST include training/evaluation context and `artifacts` map
  - SHOULD include mirrored tuning summary and tuned execution params

JSON-Safety Rules:
- Never write estimator objects directly to JSON.
- Convert estimator instances to stable strings (class names).
- Normalize numpy scalars to Python `int`/`float`/`bool`.
- Convert non-finite floats (`NaN`, `Inf`) to `null`.

## Recommended Integration Pattern for Other Templates

When adding tuning to any template:

1. **Keep Train-Time vs Artifact-Time Values Separate**
   - Use raw values for `model.set_params(...)`
   - Use JSON-safe transformed values for artifact payloads

2. **Guarantee Both Execution Paths Are Fit**
   - Tuning path: search -> set best params -> fit
   - Non-tuning path: direct fit

3. **Keep Tuning Metadata Compact and Deterministic**
   - Include only meaningful keys
   - Omit/`null` fields that are not applicable

4. **Mirror Key Fields Across `metrics.json` and `run.json`**
   - Prevent drift between quick-eval and registry metadata

5. **Validate With at Least 3 Scenarios**
   - Non-tuned run
   - Grid search tuned run
   - Random search tuned run

## Verification Checklist (Per Template)

- [ ] Model trains and predicts with `--save-model=true` in non-tuned mode
- [ ] Model trains and predicts with tuned grid search
- [ ] Model trains and predicts with tuned random search
- [ ] `metrics.json` writes successfully
- [ ] `run.json` writes successfully
- [ ] `best_params` contains JSON-safe values only
- [ ] `training_control` and `selection` remain schema-consistent

## Suggested Validation Commands

```powershell
python .\tools\generate_model.py --library scikit-learn --model linear_regression --task regression --name lr_artifact_grid_check --default-lr-enable-tuning true --default-lr-tuning-method grid --default-lr-cv-folds 3 --default-lr-cv-scoring rmse --default-lr-cv-n-jobs -1
```

```powershell
.\.venv\Scripts\python.exe .\models\lr_artifact_grid_check.py --save-model=true
```

```powershell
python .\tools\generate_model.py --library scikit-learn --model linear_regression --task regression --name lr_artifact_notune_check --default-lr-enable-tuning false
```

```powershell
.\.venv\Scripts\python.exe .\models\lr_artifact_notune_check.py --save-model=true
```

```powershell
python .\tools\generate_model.py --library xgboost --task regression --name xgb_artifact_tune_check --default-xgb-enable-tuning true --default-xgb-cv-scoring rmse --default-xgb-cv-n-iter 8
```

```powershell
python .\tools\generate_model.py --library tensorflow --model dense_nn --task regression --name tf_artifact_tune_check --optimizer adam --learning_rate 0.001 --epochs 20 --batch_size 32 --default-tf-enable-tuning true --default-tf-cv-scoring rmse --default-tf-cv-n-iter 6
```
