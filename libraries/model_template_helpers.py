from pathlib import Path

import pandas as pd

from libraries.cli_helpers import parse_bool_flag


def round_metric(value, decimals: int = 4):
    return None if value is None else round(float(value), decimals)


def find_project_root(marker_file: str = "requirements.txt") -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / marker_file).exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def validate_etl_outputs(
    project_root,
    data_path,
    df,
    X,
    y,
    target_column_name,
) -> None:
    if not isinstance(project_root, Path):
        raise TypeError("ETL contract violation: 'project_root' must be a pathlib.Path.")
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
