# Hyperparameter Tuning Artifact Consistency Reference

This reference documents artifact-quality rules discovered while integrating hyperparameter tuning into the `scikit-learn` linear regression template, and provides a repeatable rollout pattern for other templates.

## Why This Reference Exists

During tuning integration, artifact export failed with:

- `TypeError: Object of type Ridge is not JSON serializable`

Root cause:
- `best_params` from CV search included estimator objects (for example, `"regressor": Ridge(...)`) that were written into `metrics.json` / `run.json`.
- JSON serialization failed because estimator instances are not JSON-safe by default.

Additional issue found during validation:
- The non-tuning path did not call `model.fit(...)`, causing `NotFittedError` before artifact export.

## What Was Updated (Linear Regression Template)

File updated:
- [tools/model_templates/scikit-learn_linear_regression_template.py](tools/model_templates/scikit-learn_linear_regression_template.py)

Changes:
1. Added JSON-safe tuning-param sanitization helpers:
   - `_json_safe_param_value(...)`
   - `_json_safe_best_params(...)`
2. In tuning flow:
   - Capture raw search params from `search.best_params_`
   - Keep raw params for `model.set_params(...)`
   - Write sanitized params to artifacts (`best_params`)
3. Restored non-tuning training path:
   - Added `model.fit(X_train, y_train)` when `--enable-tuning=false`

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

JSON safety rules:
- Never write estimator objects directly to JSON.
- Convert estimator instances to stable strings (class names).
- Normalize numpy scalars to Python `int`/`float`/`bool`.
- Convert non-finite floats (`NaN`, `Inf`) to `null`.

## Recommended Integration Pattern for Other Templates

When adding tuning to any template:

1. **Keep train-time vs artifact-time values separate**
   - Use raw values for `model.set_params(...)`
   - Use JSON-safe transformed values for artifact payloads

2. **Guarantee both execution paths are fit**
   - Tuning path: search -> set best params -> fit
   - Non-tuning path: direct fit

3. **Keep tuning metadata compact and deterministic**
   - Include only meaningful keys
   - Omit/`null` fields that are not applicable

4. **Mirror key fields across `metrics.json` and `run.json`**
   - Prevent drift between quick-eval and registry metadata

5. **Validate with at least 3 scenarios**
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
