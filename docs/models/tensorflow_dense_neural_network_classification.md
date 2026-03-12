# TensorFlow Dense Neural Network Classification (Generated Models)

This guide documents CLI flags for models generated from the TensorFlow
dense neural network classification template.

## Quick Start

Run with default parameters:

    python .\models\model-1.py

Run and export artifacts:

    python .\models\model-1.py --save-model=true --name=my_tf_classifier

------------------------------------------------------------------------

## Core Flags

-   `--task` (`binary_classification|multiclass_classification`)
-   `--name` (string): model name used for artifact folder naming.
-   `--artifact-name-mode` (`full|short`): artifact run folder naming
    style.
-   `--save-model` (`true|false`): when `true`, writes model + metadata
    artifacts.
-   `--random-state` (int): seed used for dataset split and
    reproducibility.
-   `--test-size` (float): train/test split ratio (e.g., `0.2`).
-   `--optimizer` (`auto|adam|sgd|rmsprop|adagrad|adamw`). `auto` resolves to `adam` for direct-fit; searches all five during tuning.
-   `--learning-rate` (float)
-   `--epochs` (int)
-   `--batch-size` (int)
-   `--early-stopping` (`true|false`)
-   `--validation-fraction` (float)
-   `--n-iter-no-change` (int)
-   `--verbose` (`0|1|2|auto`)
-   `--metric-decimals` (int)
-   `--enable-tuning` (`true|false`)

------------------------------------------------------------------------

## model_init Prompt Behavior

When generating via:

    python .\model_init.py

Selecting `tensorflow` + `dense_nn` (classification):

-   TensorFlow generation does not use `Quick|Balanced|Thorough`
    profile selection.
-   The generator prompts direct training defaults (`optimizer`,
    `learning-rate`, `epochs`, `batch-size`) first, then asks whether to
    enable tuning.
-   Tuning details are prompted only when you choose
    `Enable hyperparameter tuning = true`.
-   Search iteration/trial count (`--cv-n-iter`) is requested when
    tuning is enabled and method is `random` or `bayesian`.

------------------------------------------------------------------------

## Direct Training Flags

These flags control the model directly when tuning is disabled
(`--enable-tuning=false`):

-   `--optimizer`
-   `--learning-rate`
-   `--epochs`
-   `--batch-size`
-   `--early-stopping`
-   `--validation-fraction`
-   `--n-iter-no-change`

Example (direct configuration, no tuning):

    python .\models\model-1.py --optimizer=adam --learning-rate=0.001 --epochs=50 --batch-size=32

------------------------------------------------------------------------

## Hyperparameter Tuning

When tuning is enabled (`--enable-tuning=true`):

-   `--tuning-method` (`grid|random|bayesian`)
-   `--cv-scoring` (`f1_macro`)
-   `--cv-n-iter` (int, number of randomized trials or bayesian trials)

Tuning uses either exhaustive candidate search (`grid`), randomized
candidate trials (`random`), or KerasTuner BayesianOptimization
(`bayesian`) evaluated on a train-derived validation split. The
best-performing configuration is then refit on the full training data
before final test evaluation.

Example (random search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=random --cv-scoring=f1_macro --cv-n-iter=20

Example (bayesian search):

    python .\models\model-1.py --enable-tuning=true --tuning-method=bayesian --cv-scoring=f1_macro --cv-n-iter=20

------------------------------------------------------------------------

## Data Leakage Guardrail

Validation scoring during tuning is performed strictly on
training-derived validation splits. The final evaluation on
`X_test, y_test` occurs only after the best configuration is selected
and refit.

------------------------------------------------------------------------

## Outputs and Logging

With `--save-model=true`, the run exports:

-   `model/model.keras`
-   `preprocess/preprocessor.pkl`
-   `eval/metrics.json`
-   `eval/confusion_matrix.csv`
-   `eval/roc_curve.csv` (binary classification only)
-   `eval/training_history.json`
-   `data/input_schema.json`
-   `data/target_mapping_schema.json`
-   `inference/inference_example.py`
-   `run.json`
-   `registry.csv`

Tuned runs include best trial parameters and validation scores inside
`run.json` under the `tuning` block.

