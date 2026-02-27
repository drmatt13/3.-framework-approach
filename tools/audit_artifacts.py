from pathlib import Path
import json
import subprocess
import sys
from collections import Counter

def main() -> int:
    workspace_root = Path.cwd().resolve()
    root = Path("artifacts/models")
    run_dirs: list[Path] = []

    for model_dir in root.glob("*"):
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.glob("*"):
            if run_dir.is_dir():
                run_dirs.append(run_dir)

    run_dirs = sorted(run_dirs)

    results = {
        "total_run_dirs": len(run_dirs),
        "inference": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "failures": [],
        },
        "file_anomalies": [],
        "schema_anomalies": [],
        "distributions": {
            "library": Counter(),
            "task": Counter(),
            "model_id": Counter(),
            "algorithm": Counter(),
            "metric_key_signatures": Counter(),
            "runjson_top_level_signatures": Counter(),
        },
    }

    expected_run_top = {
        "run_id",
        "name",
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

    for run_dir in run_dirs:
        rel_run = run_dir.as_posix()
        run_json = run_dir / "run.json"
        if not run_json.exists():
            results["file_anomalies"].append({"run_dir": rel_run, "missing": ["run.json"]})
            continue

        run_data = None
        artifact_paths: dict[str, str] = {}
        metrics_json = run_dir / "eval" / "metrics.json"
        inference_py = run_dir / "inference" / "inference_example.py"
        if run_json.exists():
            try:
                run_data = json.loads(run_json.read_text(encoding="utf-8"))
                top_sig = tuple(sorted(run_data.keys()))
                results["distributions"]["runjson_top_level_signatures"][str(top_sig)] += 1

                artifacts_map = run_data.get("artifacts")
                if isinstance(artifacts_map, dict):
                    artifact_paths = {
                        str(k): str(v)
                        for k, v in artifacts_map.items()
                        if isinstance(k, str) and isinstance(v, str)
                    }
                else:
                    results["schema_anomalies"].append(
                        {
                            "run_dir": rel_run,
                            "type": "run_json_missing_artifacts_map",
                            "details": "run.json key 'artifacts' must be a dictionary",
                        }
                    )

                if artifact_paths:
                    missing_artifacts = []
                    for artifact_key, artifact_rel_path in artifact_paths.items():
                        artifact_file = run_dir / Path(artifact_rel_path)
                        if not artifact_file.exists():
                            missing_artifacts.append(f"{artifact_key}:{artifact_rel_path}")

                    if missing_artifacts:
                        results["file_anomalies"].append({"run_dir": rel_run, "missing": missing_artifacts})

                    metrics_candidate = artifact_paths.get("eval_metrics") or artifact_paths.get("metrics")
                    if metrics_candidate:
                        metrics_json = run_dir / Path(metrics_candidate)

                    inference_candidate = artifact_paths.get("inference_example")
                    if inference_candidate:
                        inference_py = run_dir / Path(inference_candidate)

                lib = run_data.get("library")
                task = run_data.get("task")
                model_id = run_data.get("model_id")
                algo = run_data.get("algorithm")
                if lib:
                    results["distributions"]["library"][lib] += 1
                if task:
                    results["distributions"]["task"][task] += 1
                if model_id:
                    results["distributions"]["model_id"][model_id] += 1
                if algo:
                    results["distributions"]["algorithm"][algo] += 1

                missing_top = sorted(list(expected_run_top - set(run_data.keys())))
                if missing_top:
                    results["schema_anomalies"].append(
                        {
                            "run_dir": rel_run,
                            "type": "run_json_missing_top_keys",
                            "details": missing_top,
                        }
                    )
            except Exception as ex:
                results["schema_anomalies"].append(
                    {
                        "run_dir": rel_run,
                        "type": "run_json_parse_error",
                        "details": str(ex),
                    }
                )

        if metrics_json.exists():
            try:
                metrics_data = json.loads(metrics_json.read_text(encoding="utf-8"))
                metric_sig = tuple(sorted(metrics_data.keys()))
                results["distributions"]["metric_key_signatures"][str(metric_sig)] += 1

                if run_data is not None:
                    task = run_data.get("task")
                    if task == "regression":
                        if "test" in metrics_data and "rmse" not in metrics_data.get("test", {}):
                            results["schema_anomalies"].append(
                                {
                                    "run_dir": rel_run,
                                    "type": "regression_metrics_missing_rmse",
                                    "details": list(metrics_data.get("test", {}).keys()),
                                }
                            )
                    elif task in {"binary_classification", "multiclass_classification"}:
                        if "test" in metrics_data and "f1_macro" not in metrics_data.get("test", {}):
                            results["schema_anomalies"].append(
                                {
                                    "run_dir": rel_run,
                                    "type": "classification_metrics_missing_f1_macro",
                                    "details": list(metrics_data.get("test", {}).keys()),
                                }
                            )
            except Exception as ex:
                results["schema_anomalies"].append(
                    {
                        "run_dir": rel_run,
                        "type": "metrics_json_parse_error",
                        "details": str(ex),
                    }
                )
        else:
            results["file_anomalies"].append({"run_dir": rel_run, "missing": [str(metrics_json.relative_to(run_dir)).replace("\\", "/")]})

        if inference_py.exists():
            results["inference"]["total"] += 1
            inference_file_rel = inference_py.as_posix()

            try:
                proc = subprocess.run(
                    [sys.executable, inference_file_rel],
                    cwd=str(workspace_root),
                    capture_output=True,
                    text=True,
                )
            except Exception as ex:
                results["inference"]["failed"] += 1
                results["inference"]["failures"].append(
                    {
                        "run_dir": rel_run,
                        "returncode": None,
                        "stdout_tail": "",
                        "stderr_tail": f"subprocess_exception: {ex}",
                    }
                )
                continue

            if proc.returncode == 0:
                results["inference"]["passed"] += 1
            else:
                results["inference"]["failed"] += 1
                results["inference"]["failures"].append(
                    {
                        "run_dir": rel_run,
                        "returncode": proc.returncode,
                        "stdout_tail": proc.stdout[-1200:],
                        "stderr_tail": proc.stderr[-1200:],
                    }
                )
        else:
            results["file_anomalies"].append({"run_dir": rel_run, "missing": [str(inference_py.relative_to(run_dir)).replace("\\", "/")]})

    for k, v in list(results["distributions"].items()):
        if isinstance(v, Counter):
            results["distributions"][k] = dict(v)

    analysis_dir = Path("artifacts/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_json = analysis_dir / "artifact_audit_summary.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Audit file:", out_json.as_posix())
    print("Run dirs:", results["total_run_dirs"])
    print(
        "Inference total/pass/fail:",
        results["inference"]["total"],
        results["inference"]["passed"],
        results["inference"]["failed"],
    )
    print("File anomalies:", len(results["file_anomalies"]))
    print("Schema anomalies:", len(results["schema_anomalies"]))
    print("Libraries:", results["distributions"]["library"])
    print("Tasks:", results["distributions"]["task"])
    print("Model IDs:", results["distributions"]["model_id"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
