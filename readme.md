# Framework Approach

## Run Metadata (`run.json`) Standard

Generated model artifacts now use a compact metadata profile for productivity-focused auditing.

- `schema_version`: `1.0`
- `metadata_profile`: `compact`

### Always Included (required)

- Run identity: `run_id`, `name`, `timestamp`
- Model identity: `library`, `task`, `algorithm`, `estimator_class`, `model_id`
- Data lineage: `dataset.path`, `dataset.sha256`, `dataset.rows`, `dataset.columns`
- Split summary: `data_split.strategy`, `data_split.test_size`, `data_split.random_state`, `data_split.sizes.*`
- Preprocessing summary: `preprocessing.*`
- Parameter summary: `params.estimator_params` plus key run args
- Fit summary: `fit_summary.*`
- Artifact pointers: `artifacts.*`
- Runtime versions: `versions.*`

### Compact/Omit Rules

- Fields with `null` values are removed recursively.
- Empty objects and arrays are removed recursively.
- Optional blocks are only retained when they contain non-empty values after compaction.
- `False`, `0`, and `""` are preserved (only `null` and empty containers are pruned).
- `params.estimator_params` stores a curated, high-signal subset of effective estimator settings (not full framework defaults).
- Artifact paths in `artifacts.*` are relative to the run directory (for portability and shorter metadata).

### Why this profile

This keeps enterprise audit essentials (lineage, reproducibility, model/runtime traceability) while reducing noise from framework default parameters and non-applicable placeholders.
