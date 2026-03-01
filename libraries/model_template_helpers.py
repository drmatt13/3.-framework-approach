from pathlib import Path
import json

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
