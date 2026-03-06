from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from libraries.cli_helpers import parse_bool_flag


def round_metric(value, decimals: int = 4):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    rounded = round(float(value), decimals)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def to_dense_float32(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def find_project_root(marker_file: str = "requirements.txt") -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / marker_file).exists():
            return candidate
    return Path(__file__).resolve().parents[1]

def validate_etl_outputs(
    data_path,
    df,
    X,
    y,
    target_column_name,
) -> None:
    if not isinstance(data_path, Path):
        raise TypeError("ETL contract violation: 'data_path' must be a pathlib.Path.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ETL contract violation: 'df' must be a pandas.DataFrame.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("ETL contract violation: 'X' must be a pandas.DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("ETL contract violation: 'y' must be a pandas.Series.")
    if not isinstance(target_column_name, str) or not target_column_name.strip():
        raise TypeError("ETL contract violation: 'target_column_name' must be a non-empty string.")
    if target_column_name not in df.columns:
        raise ValueError("ETL contract violation: 'target_column_name' must exist in df.columns.")
    if target_column_name in X.columns:
        raise ValueError("ETL contract violation: target column must not be present in X.")
    if len(X) != len(y):
        raise ValueError("ETL contract violation: X and y must have identical row counts.")
    if len(y) == 0:
        raise ValueError("ETL contract violation: no rows remain after ETL.")


def post_transform_feature_count(preprocessor, sample_frame: pd.DataFrame) -> int | None:
    try:
        transformed = preprocessor.transform(sample_frame)
        return int(transformed.shape[1])
    except Exception:
        return None


def artifact_map(base_dir: Path, artifacts: dict[str, Path]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, path in artifacts.items():
        if path.exists():
            resolved[key] = str(path.relative_to(base_dir))
    return resolved


def compact_metadata(value):
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            compacted_item = compact_metadata(item)
            if compacted_item is None:
                continue
            if isinstance(compacted_item, (dict, list)) and len(compacted_item) == 0:
                continue
            compacted[key] = compacted_item
        return compacted
    if isinstance(value, list):
        compacted_list = []
        for item in value:
            compacted_item = compact_metadata(item)
            if compacted_item is None:
                continue
            if isinstance(compacted_item, (dict, list)) and len(compacted_item) == 0:
                continue
            compacted_list.append(compacted_item)
        return compacted_list
    return value


def select_estimator_params(params: dict, keys: list[str]) -> dict:
    return {key: params.get(key) for key in keys if key in params}


def json_safe(value):
    if isinstance(value, float):
        if value != value:
            return None
        if value == float("inf") or value == float("-inf"):
            return None
        return value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def infer_target_mapping(original_target: pd.Series | None, encoded_target: pd.Series) -> dict[str, int] | None:
    if original_target is None:
        return None
    if pd.api.types.is_numeric_dtype(original_target):
        return None

    valid_mask = original_target.notna() & encoded_target.notna()
    if not bool(valid_mask.any()):
        return None

    original_values = original_target.loc[valid_mask].astype("string")
    encoded_numeric = pd.to_numeric(encoded_target.loc[valid_mask], errors="coerce")
    if encoded_numeric.isna().any():
        return None
    if not bool((encoded_numeric % 1 == 0).all()):
        return None

    mapping_frame = pd.DataFrame(
        {
            "label": original_values.astype(str),
            "code": encoded_numeric.astype("int64"),
        }
    )

    unique_codes_per_label = mapping_frame.groupby("label")["code"].nunique()
    if (unique_codes_per_label > 1).any():
        return None

    first_code_by_label = mapping_frame.groupby("label")["code"].first().sort_values(kind="stable")
    return {str(label): int(code) for label, code in first_code_by_label.items()}


def _stable_unique_non_null(series: pd.Series) -> list:
    values = []
    seen = set()
    for value in series.dropna().tolist():
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        values.append(value)
    return values


def _to_json_scalar(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric
    return str(value)


def infer_binary_semantics(series: pd.Series) -> dict | None:
    unique_values = _stable_unique_non_null(series)
    if len(unique_values) != 2:
        return None
    values = [_to_json_scalar(value) for value in unique_values]
    return {
        "semantic_type": "binary",
        "values": values,
    }


def write_model_schemas(
    schema_dir: Path,
    X_raw: pd.DataFrame,
    y_model: pd.Series,
    target_column_name: str,
    transformed_features,
    preprocessor=None,
    y_original: pd.Series | None = None,
) -> dict[str, Path]:
    schema_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = []
    for column in X_raw.columns:
        series = X_raw[column]
        column_entry = {
            "name": str(column),
            "dtype": str(series.dtype),
        }
        binary_semantics = infer_binary_semantics(series)
        if binary_semantics is not None:
            column_entry.update(binary_semantics)
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        ) and "values" not in column_entry:
            unique_values = series.dropna().astype("string").unique().tolist()
            column_entry["values"] = [str(value) for value in unique_values]
        feature_columns.append(column_entry)

    input_schema = {
        "feature_columns": feature_columns,
    }
    input_schema_path = schema_dir / "input_schema.json"
    input_schema_path.write_text(json.dumps(input_schema, indent=2), encoding="utf-8")

    artifacts = {
        "input_schema": input_schema_path,
    }

    target_mapping = infer_target_mapping(y_original, y_model)
    target_mapping_schema = {
        "target": str(target_column_name),
        "dtype": str(y_model.dtype),
    }
    if target_mapping:
        target_mapping_schema["mapping"] = [
            {"label": label, "encoded_value": value} for label, value in target_mapping.items()
        ]

    target_mapping_schema_path = schema_dir / "target_mapping_schema.json"
    target_mapping_schema_path.write_text(json.dumps(target_mapping_schema, indent=2), encoding="utf-8")
    artifacts["target_mapping_schema"] = target_mapping_schema_path

    return artifacts


def transformed_feature_names(preprocessor, X_frame: pd.DataFrame) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(name) for name in preprocessor.get_feature_names_out()]
        except Exception:
            pass
    try:
        transformed = preprocessor.transform(X_frame.iloc[:1])
        width = int(transformed.shape[1])
    except Exception:
        width = int(X_frame.shape[1])
    return [f"feature_{index}" for index in range(width)]


def validate_artifact_contract(
    *,
    run_dir: Path,
    artifact_files: dict[str, Path],
    run_metadata: dict,
    metrics: dict,
    required_artifact_keys: list[str],
    warn_only: bool = True,
) -> list[str]:
    warnings: list[str] = []

    missing_keys = [key for key in required_artifact_keys if key not in artifact_files]
    if missing_keys:
        warnings.append(f"Missing artifact keys in map: {missing_keys}")

    missing_files = [key for key, file_path in artifact_files.items() if not file_path.exists()]
    if missing_files:
        warnings.append(f"Artifact files not found on disk: {missing_files}")

    expected_run_sections = {
        "run_id",
        "model_name",
        "timestamp",
        "library",
        "task",
        "algorithm",
        "estimator_class",
        "model_id",
        "dataset",
        "data_split",
        "preprocessing",
        "params",
        "fit_summary",
        "artifacts",
        "versions",
    }
    missing_run_sections = sorted(expected_run_sections - set(run_metadata.keys()))
    if missing_run_sections:
        warnings.append(f"run.json missing top-level sections: {missing_run_sections}")

    expected_metric_sections = {"train", "test", "data_sizes", "primary_metric", "tuning"}
    missing_metric_sections = sorted(expected_metric_sections - set(metrics.keys()))
    if missing_metric_sections:
        warnings.append(f"metrics.json missing sections: {missing_metric_sections}")

    if warnings and not warn_only:
        raise ValueError("Artifact contract validation failed: " + " | ".join(warnings))

    if warnings:
        warning_payload = {
            "warnings": warnings,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval" / "artifact_validation_warnings.json").write_text(
            json.dumps(warning_payload, indent=2),
            encoding="utf-8",
        )

    return warnings


def write_unified_registry_sqlite(
    *,
    project_root: Path,
    run_dir: Path,
    run_metadata: dict,
    metrics: dict,
) -> Path:
    registry_dir = project_root / "artifacts"
    registry_dir.mkdir(parents=True, exist_ok=True)
    db_path = registry_dir / "model_registry.sqlite"

    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        if cursor.fetchone() is not None:
            raise RuntimeError(
                "Detected legacy SQLite schema containing 'runs' table. "
                "This project now uses only classification_metrics and regression_metrics. "
                "Delete artifacts/model_registry.sqlite to continue."
            )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regression_metrics (
                run_id TEXT PRIMARY KEY,
                model_name TEXT,
                timestamp TEXT,
                library TEXT,
                task TEXT,
                algorithm TEXT,
                estimator_class TEXT,
                model_id TEXT,
                dataset_path TEXT,
                dataset_sha256 TEXT,
                dataset_rows INTEGER,
                dataset_columns INTEGER,
                test_size REAL,
                random_state INTEGER,
                n_train INTEGER,
                n_val INTEGER,
                n_test INTEGER,
                tuning_enabled INTEGER,
                tuning_method TEXT,
                cv_best_score REAL,
                primary_metric_name TEXT,
                primary_metric_direction TEXT,
                primary_metric_value REAL,
                run_dir TEXT,
                created_at_utc TEXT,
                mse REAL,
                mae REAL,
                rmse REAL,
                r2 REAL,
                max_error REAL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS classification_metrics (
                run_id TEXT PRIMARY KEY,
                model_name TEXT,
                timestamp TEXT,
                library TEXT,
                task TEXT,
                algorithm TEXT,
                estimator_class TEXT,
                model_id TEXT,
                dataset_path TEXT,
                dataset_sha256 TEXT,
                dataset_rows INTEGER,
                dataset_columns INTEGER,
                test_size REAL,
                random_state INTEGER,
                n_train INTEGER,
                n_val INTEGER,
                n_test INTEGER,
                tuning_enabled INTEGER,
                tuning_method TEXT,
                cv_best_score REAL,
                primary_metric_name TEXT,
                primary_metric_direction TEXT,
                primary_metric_value REAL,
                run_dir TEXT,
                created_at_utc TEXT,
                accuracy REAL,
                balanced_accuracy REAL,
                precision_macro REAL,
                recall_macro REAL,
                f1_macro REAL,
                roc_auc_value REAL,
                pr_auc_value REAL,
                log_loss REAL,
                brier_score REAL,
                support_total INTEGER,
                support_by_class_json TEXT
            )
            """
        )

        for table_name in ("regression_metrics", "classification_metrics"):
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = {str(row[1]) for row in cursor.fetchall()}
            expected_common_columns = {
                "run_id",
                "model_name",
                "timestamp",
                "library",
                "task",
                "algorithm",
                "estimator_class",
                "model_id",
                "dataset_path",
                "dataset_sha256",
                "dataset_rows",
                "dataset_columns",
                "test_size",
                "random_state",
                "n_train",
                "n_val",
                "n_test",
                "tuning_enabled",
                "tuning_method",
                "cv_best_score",
                "primary_metric_name",
                "primary_metric_direction",
                "primary_metric_value",
                "run_dir",
                "created_at_utc",
            }
            if not expected_common_columns.issubset(columns):
                missing_columns = sorted(expected_common_columns - columns)
                raise RuntimeError(
                    f"Detected outdated schema for '{table_name}'. Missing columns: {missing_columns}. "
                    "Delete artifacts/model_registry.sqlite to continue."
                )

        dataset = run_metadata.get("dataset", {}) if isinstance(run_metadata.get("dataset"), dict) else {}
        data_split = run_metadata.get("data_split", {}) if isinstance(run_metadata.get("data_split"), dict) else {}
        sizes = data_split.get("sizes", {}) if isinstance(data_split.get("sizes"), dict) else {}
        tuning = run_metadata.get("tuning", {}) if isinstance(run_metadata.get("tuning"), dict) else {}
        primary_metric = metrics.get("primary_metric", {}) if isinstance(metrics.get("primary_metric"), dict) else {}

        run_insert_common_values = (
            str(run_metadata.get("run_id")),
            str(run_metadata.get("model_name")),
            str(run_metadata.get("timestamp")),
            str(run_metadata.get("library")),
            str(run_metadata.get("task")),
            str(run_metadata.get("algorithm")),
            str(run_metadata.get("estimator_class")),
            str(run_metadata.get("model_id")),
            str(dataset.get("path")) if dataset.get("path") is not None else None,
            str(dataset.get("sha256")) if dataset.get("sha256") is not None else None,
            int(dataset.get("rows")) if dataset.get("rows") is not None else None,
            int(dataset.get("columns")) if dataset.get("columns") is not None else None,
            float(data_split.get("test_size")) if data_split.get("test_size") is not None else None,
            int(data_split.get("random_state")) if data_split.get("random_state") is not None else None,
            int(sizes.get("n_train")) if sizes.get("n_train") is not None else None,
            int(sizes.get("n_val")) if sizes.get("n_val") is not None else None,
            int(sizes.get("n_test")) if sizes.get("n_test") is not None else None,
            1 if bool(tuning.get("enabled")) else 0,
            str(tuning.get("method")) if tuning.get("method") is not None else None,
            float(tuning.get("best_score")) if tuning.get("best_score") is not None else None,
            str(primary_metric.get("name")) if primary_metric.get("name") is not None else None,
            str(primary_metric.get("direction")) if primary_metric.get("direction") is not None else None,
            float(primary_metric.get("value")) if primary_metric.get("value") is not None else None,
            str(run_dir.relative_to(project_root)).replace("\\", "/"),
            datetime.now(timezone.utc).isoformat(),
        )

        task = str(run_metadata.get("task"))
        test_metrics = metrics.get("test", {}) if isinstance(metrics.get("test"), dict) else {}
        if task == "regression":
            cursor.execute(
                """
                INSERT OR REPLACE INTO regression_metrics (
                    run_id, model_name, timestamp, library, task, algorithm, estimator_class, model_id,
                    dataset_path, dataset_sha256, dataset_rows, dataset_columns,
                    test_size, random_state, n_train, n_val, n_test,
                    tuning_enabled, tuning_method, cv_best_score,
                    primary_metric_name, primary_metric_direction, primary_metric_value,
                    run_dir, created_at_utc,
                    mse, mae, rmse, r2, max_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_insert_common_values
                + (
                    float(test_metrics.get("mse")) if test_metrics.get("mse") is not None else None,
                    float(test_metrics.get("mae")) if test_metrics.get("mae") is not None else None,
                    float(test_metrics.get("rmse")) if test_metrics.get("rmse") is not None else None,
                    float(test_metrics.get("r2")) if test_metrics.get("r2") is not None else None,
                    float(test_metrics.get("max_error")) if test_metrics.get("max_error") is not None else None,
                ),
            )
        else:
            roc_auc_value = None
            pr_auc_value = None
            if isinstance(test_metrics.get("roc_auc"), dict):
                roc_auc_value = test_metrics.get("roc_auc", {}).get("value")
            if isinstance(test_metrics.get("pr_auc"), dict):
                pr_auc_value = test_metrics.get("pr_auc", {}).get("value")
            cursor.execute(
                """
                INSERT OR REPLACE INTO classification_metrics (
                    run_id, model_name, timestamp, library, task, algorithm, estimator_class, model_id,
                    dataset_path, dataset_sha256, dataset_rows, dataset_columns,
                    test_size, random_state, n_train, n_val, n_test,
                    tuning_enabled, tuning_method, cv_best_score,
                    primary_metric_name, primary_metric_direction, primary_metric_value,
                    run_dir, created_at_utc,
                    accuracy, balanced_accuracy, precision_macro, recall_macro, f1_macro,
                    roc_auc_value, pr_auc_value, log_loss, brier_score, support_total, support_by_class_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_insert_common_values
                + (
                    float(test_metrics.get("accuracy")) if test_metrics.get("accuracy") is not None else None,
                    float(test_metrics.get("balanced_accuracy")) if test_metrics.get("balanced_accuracy") is not None else None,
                    float(test_metrics.get("precision_macro")) if test_metrics.get("precision_macro") is not None else None,
                    float(test_metrics.get("recall_macro")) if test_metrics.get("recall_macro") is not None else None,
                    float(test_metrics.get("f1_macro")) if test_metrics.get("f1_macro") is not None else None,
                    float(roc_auc_value) if roc_auc_value is not None else None,
                    float(pr_auc_value) if pr_auc_value is not None else None,
                    float(test_metrics.get("log_loss")) if test_metrics.get("log_loss") is not None else None,
                    float(test_metrics.get("brier_score")) if test_metrics.get("brier_score") is not None else None,
                    int(test_metrics.get("support_total")) if test_metrics.get("support_total") is not None else None,
                    json.dumps(test_metrics.get("support_by_class"), ensure_ascii=False)
                    if test_metrics.get("support_by_class") is not None
                    else None,
                ),
            )

        connection.commit()
    finally:
        connection.close()

    return db_path
