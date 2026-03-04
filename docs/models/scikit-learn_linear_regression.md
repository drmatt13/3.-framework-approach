# scikit-learn Linear Regression (Generated Models)

This guide documents CLI flags for models generated from the linear regression template.

## Quick Start

Run with default direct parameters:

```powershell
python .\models\model-1.py
```

Run and export artifacts:

```powershell
python .\models\model-1.py --save-model=true --name=my_linear_model
```

## Core Flags

- `--name` (string): model name used for artifact folder naming.
- `--artifact-name-mode` (`full|short`): artifact run folder naming style.
- `--save-model` (`true|false`): when `true`, writes model + metadata artifacts.
- `--random-state` (int): seed used for split and stochastic tuning.
- `--test-size` (float): train/test split ratio (e.g., `0.2`).
- `--verbose` (`0|1|2|auto`): training log verbosity.
- `--metric-decimals` (int): output rounding precision for reported metrics.
- `--enable-tuning` (`true|false`): turns hyperparameter tuning on or off.

## model_init Prompt Behavior

When generating via `python .\model_init.py` for `scikit-learn` + `linear_regression`:

- Profile mode (`Quick|Balanced|Thorough`) applies preset defaults and prints a resolved-default summary before generation.
- `Thorough` is preset to tuning enabled.
- Custom mode asks `Enable hyperparameter tuning` before direct-fit estimator defaults.
- When tuning is enabled in Custom mode, direct-fit estimator defaults are auto-defaulted and omitted from the resolved-default summary.
- In Custom mode, `--cv-n-iter` is asked only when tuning method is `random`.
- In Custom mode, when tuning is enabled, `penalty=none` is not offered.

## Direct Estimator Flags

Use these when you want explicit, fixed model behavior without cross-validation search.

These are applied directly to the estimator when tuning is disabled (`--enable-tuning=false`):

- `--penalty` (`auto|none|l1|l2|elasticnet`). `auto` requires `--enable-tuning=true` and searches all penalty families.
- `--alpha` (float)
- `--fit-intercept` (`true|false`)
- `--l1-ratio` (float, used for `elasticnet`)

Example (direct configuration, no tuning):

```powershell
python .\models\model-1.py --penalty=l2 --alpha=0.1 --fit-intercept=true
```

## Hyperparameter Tuning Flags

Use these when you want the model to search for better parameter values via cross-validation.

When tuning is enabled (`--enable-tuning=true`), configure search behavior with:

- `--tuning-method` (`grid|random`)
- `--penalty=none` is not allowed when `--enable-tuning=true`.
- `--cv-folds` (int, Number of CV folds) 
- `--cv-scoring` (`rmse|mae|r2`)
  - `rmse` -> `neg_root_mean_squared_error`
  - `mae` -> `neg_mean_absolute_error`
  - `r2` -> `r2`
- `--cv-n-iter` (int, : Number of iterations `random search only`)
- `--cv-n-jobs` (int, parallel workers; `-1` uses all cores)

When tuning is enabled, single-value estimator flags are treated as baseline/default context, and the template expands to deterministic search grids by selected penalty.

Example (grid search):

```powershell
python .\models\model-1.py --enable-tuning=true --tuning-method=grid --penalty=elasticnet --cv-folds=5 --cv-scoring=rmse --cv-n-jobs=-1
```

Example (random search):

```powershell
python .\models\model-1.py --enable-tuning=true --tuning-method=random --penalty=l2 --cv-folds=5 --cv-scoring=mae --cv-n-iter=30 --cv-n-jobs=-1
```

## Data Leakage Guardrail

Tuning and CV are run only on `X_train, y_train`.
After CV selects best params, the best estimator is refit on the full training split before test-set evaluation.
`X_test, y_test` are never used for model selection.

## Outputs and Logging

With `--save-model=true`, the run exports:

- `model/model.pkl`
- `preprocess/preprocessor.pkl`
- `eval/metrics.json`
- `eval/predictions_preview.csv`
- `data/input_schema.json`
- `data/target_mapping_schema.json`
- `inference/inference_example.py`
- `run.json`
- `model_registry.csv`

Tuning-enabled runs include CV details (best params, CV scoring, best CV score) in evaluation and run metadata schemas.
