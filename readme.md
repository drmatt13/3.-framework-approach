# Artifact schema contract

Model runs now write schema artifacts under `artifacts/models/<model>/<run>/data/`.

## Files

- `input_schema.json`
	- Raw model input contract before preprocessing.
	- Includes each input feature name + dtype.
	- For object/string/categorical-like feature columns, includes distinct observed values in `values`.

- `target_mapping_schema.json`
	- Always written.
	- Includes `target` and `dtype`.
	- Includes label-to-integer `mapping` when a non-numeric target is encoded.

- `inference/inference_example.py` (classification templates)
	- Prints class labels and full class probabilities (`predict_proba`/softmax-style outputs) for multiclass inference.

## Notes

- Previous `data/feature_schema.json` output has been replaced by this `data/` folder schema contract.
- Artifact maps in each run's `run.json` now reference schema files via:
	- `input_schema`
	- `target_mapping_schema`
- Generated files under `models/` are snapshots at generation time.
	- Template/generator updates do not retroactively change existing generated model files.
	- Re-run `tools/generate_model.py` (or `model_init.py`) to regenerate models after framework/template updates.
