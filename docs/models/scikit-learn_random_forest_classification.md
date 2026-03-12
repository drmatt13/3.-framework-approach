# scikit-learn Random Forest Classification (Generated Models)

This guide documents CLI flags for models generated from the random
forest classification template.

## Quick Start

Run with default parameters:

    python .\models\model-1.py

Run and export artifacts:

    python .\models\model-1.py --save-model=true --name=my_rf_classifier

------------------------------------------------------------------------

## Core Flags

-   `--task` (`binary_classification|multiclass_classification`)
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

Selecting `scikit-learn` + `random_forest` (classification):

-   Profile mode (`Quick|Balanced|Thorough`) applies preset defaults and
    prints a resolved-default summary before generation.
-   `Thorough` is preset to tuning enabled.
-   Custom mode asks `Enable hyperparameter tuning` before direct-fit
    estimator defaults.
-   When tuning is enabled in Custom mode, direct-fit estimator
    defaults are auto-defaulted and omitted from the resolved-default
    summary.
-   In Custom mode, `--cv-n-iter` is prompted when tuning method is
    `random` or `bayesian`.

------------------------------------------------------------------------

## Direct Estimator Flags

These are applied directly to the estimator when tuning is disabled
(`--enable-tuning=false`):

-   `--n-estimators` (int)
-   `--max-depth` (int\|none)
-   `--min-samples-split` (int)
-   `--min-samples-leaf` (int)
-   `--min-weight-fraction-leaf` (float)
-   `--max-leaf-nodes` (int\|none)
-   `--min-impurity-decrease` (float)
-   `--max-features` (`auto|sqrt|log2|float|none`)
-   `--bootstrap` (`true|false`)
-   `--max-samples` (int\|float\|none, requires `--bootstrap=true`; `1.0` uses all rows)
-   `--ccp-alpha` (float)
-   `--n-jobs` (int\|none, estimator fit/predict parallelism)

Example (direct configuration, no tuning):

    python .\models\model-1.py --n-estimators=200 --max-depth=12 --min-samples-leaf=2

------------------------------------------------------------------------

## Hyperparameter Tuning Flags

When tuning is enabled (`--enable-tuning=true`), configure search
behavior with:

-   `--tuning-method` (`grid|random|bayesian`)
-   `--cv-folds` (int)
-   `--cv-scoring` (`f1_macro|accuracy|roc_auc_ovr`)
-   `--cv-n-iter` (int, iterations/trials for `random` and `bayesian`)
-   `--cv-n-jobs` (int, CV search parallelism, `-1` uses all cores)

Example (grid search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=grid --cv-folds=5 --cv-scoring=f1_macro --cv-n-jobs=-1

Example (random search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=random --cv-folds=5 --cv-scoring=accuracy --cv-n-iter=25 --cv-n-jobs=-1

Example (bayesian search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=bayesian --cv-folds=5 --cv-scoring=accuracy --cv-n-iter=25 --cv-n-jobs=-1

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
-   `eval/confusion_matrix.csv`
-   `eval/predictions_preview.csv`
-   `data/input_schema.json`
-   `data/target_mapping_schema.json`
-   `inference/inference_example.py`
-   `run.json`
-   `registry.csv`

Tuned runs persist best search details in `run.json` under tuning
metadata.
