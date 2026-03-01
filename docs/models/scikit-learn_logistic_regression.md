# scikit-learn Logistic Regression (Generated Models)

This guide documents CLI flags for models generated from the logistic
regression template.

## Quick Start

Run with default parameters:

``` powershell
python .\models\model-1.py
```

Run and export artifacts:

``` powershell
python .\models\model-1.py --save-model=true --name=my_logistic_model
```

## Core Flags

-   `--task` (`binary_classification|multiclass_classification`):
    classification task type.
-   `--name` (string): model name used for artifact folder naming.
-   `--artifact-name-mode` (`full|short`): artifact run folder naming
    style.
-   `--save-model` (`true|false`): when `true`, writes model + metadata
    artifacts.
-   `--random-state` (int): seed used for split and stochastic tuning.
-   `--test-size` (float): train/test split ratio (e.g., `0.2`).
-   `--max-iter` (int): maximum solver iterations.
-   `--verbose` (`0|1|2|auto`): training log verbosity.
-   `--metric-decimals` (int): output rounding precision for reported
    metrics.
-   `--enable-tuning` (`true|false`): turns hyperparameter tuning on or
    off.

## model_init Prompt Behavior

When generating via `python .\model_init.py` for `scikit-learn` +
`logistic_regression`:

-   Profile mode (`Quick|Balanced|Thorough`) applies preset defaults and
    prints a resolved-default summary before generation.
-   `Thorough` is preset to tuning enabled.
-   Custom mode asks tuning details only when you choose
    `Enable hyperparameter tuning = true`.
-   In Custom mode, `--cv-n-iter` is asked only when tuning method is
    `random`.
-   If you pick a `penalty` that is incompatible with the chosen
    `solver`, the template (or your generator validation) should force a
    compatible combination.

## Direct Estimator Flags

Use these when you want explicit, fixed model behavior without
cross-validation search.

These are applied directly to the estimator when tuning is disabled
(`--enable-tuning=false`):

-   `--penalty` (`none|l1|l2|elasticnet`)
-   `--c` (float, inverse regularization strength; smaller = more
    regularization)
-   `--solver` (`lbfgs|liblinear|newton-cg|newton-cholesky|sag|saga`)
-   `--class-weight` (`none|balanced`)

Example (direct configuration, no tuning):

``` powershell
python .\models\model-1.py --task=binary_classification --penalty=l2 --c=1.0 --solver=lbfgs --class-weight=balanced
```

## Hyperparameter Tuning Flags

Use these when you want the model to search for better parameter values
via cross-validation.

When tuning is enabled (`--enable-tuning=true`), configure search
behavior with:

-   `--tuning-method` (`grid|random`)
-   `--cv-folds` (int, number of CV folds)
-   `--cv-scoring` (`accuracy|f1|f1_macro|roc_auc`)
    -   `accuracy` -\> `accuracy`
    -   `f1` -\> `f1` (binary only; for multiclass, prefer `f1_macro`)
    -   `f1_macro` -\> `f1_macro`
    -   `roc_auc` -\> `roc_auc` (binary only unless using a multiclass
        variant)
-   `--cv-n-iter` (int, number of iterations; random search only)
-   `--cv-n-jobs` (int, parallel workers; `-1` uses all cores)

When tuning is enabled, single-value estimator flags are treated as
baseline/default context, and the template expands to deterministic
search grids (grid search) or sampled parameter sets (random search).

Example (grid search):

``` powershell
python .\models\model-1.py --enable-tuning=true --tuning-method=grid --task=binary_classification --cv-folds=5 --cv-scoring=f1_macro --cv-n-jobs=-1
```

Example (random search):

``` powershell
python .\models\model-1.py --enable-tuning=true --tuning-method=random --task=binary_classification --cv-folds=5 --cv-scoring=roc_auc --cv-n-iter=30 --cv-n-jobs=-1
```

## Data Leakage Guardrail

Tuning and CV are run only on `X_train, y_train`. After CV selects best
parameters, the best estimator is refit on the full training split
before test-set evaluation. `X_test, y_test` are never used for model
selection.

## Outputs and Logging

With `--save-model=true`, the run exports:

-   `model/model.pkl`
-   `preprocess/preprocessor.pkl`
-   `eval/metrics.json`
-   `eval/confusion_matrix.csv`
-   `eval/roc_curve.csv` (binary classification)
-   `eval/predictions_preview.csv`
-   `data/input_schema.json`
-   `data/target_mapping_schema.json`
-   `inference/inference_example.py`
-   `run.json`
-   `model_registry.csv`

Tuning-enabled runs include CV details (best params, CV scoring, best CV
score) in evaluation and run metadata schemas.
