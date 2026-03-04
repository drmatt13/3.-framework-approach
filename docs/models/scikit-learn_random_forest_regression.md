# scikit-learn Random Forest Regression (Generated Models)

This guide documents CLI flags for models generated from the random
forest regression template.

## Quick Start

Run with default parameters:

    python .\models\model-1.py

Run and export artifacts:

    python .\models\model-1.py --save-model=true --name=my_rf_regressor

------------------------------------------------------------------------

## Core Flags

-   `--task` (`regression`)
-   `--name` (string): model name used for artifact folder naming.
-   `--artifact-name-mode` (`full|short`): artifact run folder naming
    style.
-   `--save-model` (`true|false`): when `true`, writes model + metadata
    artifacts.
-   `--random-state` (int): seed used for split and stochastic tuning.
-   `--test-size` (float): train/test split ratio (e.g., `0.2`).
-   `--verbose` (`0|1|2|auto`): training log verbosity.
-   `--metric-decimals` (int): output rounding precision for reported
    metrics.
-   `--enable-tuning` (`true|false`): turns hyperparameter tuning on or
    off.

------------------------------------------------------------------------

## model_init Prompt Behavior

When generating via:

    python .\model_init.py

Selecting `scikit-learn` + `random_forest` (regression):

-   Profile mode (`Quick|Balanced|Thorough`) applies preset defaults and
    prints a resolved-default summary before generation.
-   `Thorough` is preset to tuning enabled.
-   Custom mode asks `Enable hyperparameter tuning` before direct-fit
    estimator defaults.
-   When tuning is enabled in Custom mode, direct-fit estimator
    defaults are auto-defaulted and omitted from the resolved-default
    summary.
-   In Custom mode, `--cv-n-iter` is prompted only when tuning method is
    `random`.

------------------------------------------------------------------------

## Direct Estimator Flags

These are applied directly to the estimator when tuning is disabled
(`--enable-tuning=false`):

-   `--n-estimators` (int)
-   `--max-depth` (int\|none)
-   `--min-samples-leaf` (int)
-   `--max-features` (`auto|sqrt|log2|float|none`)

Example (direct configuration, no tuning):

    python .\models\model-1.py --n-estimators=300 --max-depth=15 --min-samples-leaf=2

------------------------------------------------------------------------

## Hyperparameter Tuning Flags

When tuning is enabled (`--enable-tuning=true`), configure search
behavior with:

-   `--tuning-method` (`grid|random`)
-   `--cv-folds` (int)
-   `--cv-scoring` (`rmse|mae|r2`)
    -   `rmse` -\> `neg_root_mean_squared_error`
    -   `mae` -\> `neg_mean_absolute_error`
    -   `r2` -\> `r2`
-   `--cv-n-iter` (int, random search only)
-   `--cv-n-jobs` (int, `-1` uses all cores)

Example (grid search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=grid --cv-folds=5 --cv-scoring=rmse --cv-n-jobs=-1

Example (random search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=random --cv-folds=5 --cv-scoring=mae --cv-n-iter=30 --cv-n-jobs=-1

------------------------------------------------------------------------

## Data Leakage Guardrail

Cross-validation runs only on `X_train, y_train`. After best parameters
are selected, the estimator is refit on the full training split.
`X_test, y_test` are never used for model selection.

------------------------------------------------------------------------

## Outputs and Logging

With `--save-model=true`, the run exports:

-   `model/model.pkl`
-   `preprocess/preprocessor.pkl`
-   `eval/metrics.json`
-   `eval/predictions_preview.csv`
-   `data/input_schema.json`
-   `data/target_mapping_schema.json`
-   `inference/inference_example.py`
-   `run.json`
-   `registry.csv`

Tuned runs persist best search details in `run.json` under tuning
metadata.
