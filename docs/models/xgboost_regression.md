# XGBoost Regression (Generated Models)

This guide documents CLI flags for models generated from the XGBoost
regression template.

## Quick Start

Run with default parameters:

    python .\models\model-1.py

Run and export artifacts:

    python .\models\model-1.py --save-model=true --name=my_xgb_regressor

------------------------------------------------------------------------

## Core Flags

-   `--task` (`regression`)
-   `--name` (string): model name used for artifact folder naming.
-   `--artifact-name-mode` (`full|short`): artifact run folder naming
    style.
-   `--booster` (`gbtree|gblinear|dart`)
-   `--device` (`auto|cpu|gpu`): for this template architecture, `gpu` is downgraded once at startup to CPU to avoid repeated XGBoost CPU/GPU mismatch fallback warnings.
-   `--save-model` (`true|false`)
-   `--random-state` (int)
-   `--test-size` (float)
-   `--early-stopping` (`true|false`)
-   `--validation-fraction` (float)
-   `--n-iter-no-change` (int)
-   `--n-estimators` (int)
-   `--learning-rate` (float)
-   `--max-depth` (int)
-   `--subsample` (float)
-   `--colsample-bytree` (float)
-   `--min-child-weight` (float)
-   `--reg-lambda` (float)
-   `--reg-alpha` (float)
-   `--verbose` (`0|1|2|auto`)
-   `--metric-decimals` (int)
-   `--enable-tuning` (`true|false`)

------------------------------------------------------------------------

## model_init Prompt Behavior

When generating via:

    python .\model_init.py

Selecting `xgboost` + regression:

-   Profile mode (`Quick|Balanced|Thorough`) applies preset defaults and
    prints a resolved-default summary before generation.
-   `Thorough` is preset to tuning enabled.
-   Custom mode prompts tuning details only when you choose
    `Enable hyperparameter tuning = true`.
-   Random search iteration count (`--cv-n-iter`) is requested only when
    tuning is enabled.

------------------------------------------------------------------------

## Direct Estimator Flags

These flags control the estimator directly when tuning is disabled
(`--enable-tuning=false`):

-   `--n-estimators`
-   `--learning-rate`
-   `--max-depth`
-   `--subsample`
-   `--colsample-bytree`
-   `--min-child-weight`
-   `--reg-lambda`
-   `--reg-alpha`

Example (direct configuration, no tuning):

    python .\models\model-1.py --n-estimators=400 --learning-rate=0.05 --max-depth=8 --device=gpu

Device note: if `--device=gpu` is requested, the template emits a single warning before training begins and runs with effective CPU device for train/CV/predict consistency.

------------------------------------------------------------------------

## Hyperparameter Tuning

When tuning is enabled (`--enable-tuning=true`):

-   `--tuning-method` (`random`)
-   `--cv-folds` (int)
-   `--cv-scoring` (`rmse|mae|r2`)
    -   `rmse` -\> `neg_root_mean_squared_error`
    -   `mae` -\> `neg_mean_absolute_error`
    -   `r2` -\> `r2`
-   `--cv-n-iter` (int, number of randomized trials)
-   `--cv-n-jobs` (int, `-1` uses all cores)

Tuning uses randomized candidate trials evaluated via cross-validation.
The best-performing configuration is then refit on the full training
data before final test evaluation.

Example:

    python .\models\model-1.py --enable-tuning=true --tuning-method=random --cv-folds=5 --cv-scoring=rmse --cv-n-iter=30 --cv-n-jobs=-1

------------------------------------------------------------------------

## Data Leakage Guardrail

Cross-validation is performed strictly on `X_train, y_train`. After best
parameters are selected, the estimator is refit on the full training
split. `X_test, y_test` are never used for model selection.

------------------------------------------------------------------------

## Outputs and Logging

With `--save-model=true`, the run exports:

-   `model/model.json`
-   `model/model.pkl`
-   `preprocess/preprocessor.pkl`
-   `eval/metrics.json`
-   `eval/predictions_preview.csv`
-   `data/input_schema.json`
-   `data/target_mapping_schema.json`
-   `inference/inference_example.py`
-   `run.json`
-   `model_registry.csv`

Saved runs include model, preprocessors, metrics, run metadata, and
registry updates. When tuning is enabled, `best_params` and full CV
summary are written under the `tuning` block in `run.json`. Device
metadata includes requested and effective device values so guardrail
downgrades are auditable.
