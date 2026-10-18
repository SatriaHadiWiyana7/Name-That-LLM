# Kaggle deployment — offline inference only

## Model artifact checklist

- Serialized classifier pipeline (joblib for trusted sklearn checkpoints; never unpickle unknown files), tokenizer and weights for transformers.
- Exact 18 labels ordered as `src/name_that_llm/labels.py`.
- `manifest.json` with training metadata, SHA256 checksum, Python/library versions, config, seed, required files.
- Include local-only dependencies if Kaggle image lacks packages; do not use internet during notebook scoring.
- Upload as a private Kaggle Dataset/Model input accessible to the submission notebook (respect Kaggle rules/permissions).

## Minimal working approach

1. Finish `scripts/train_baseline.py` locally.
2. Upload `artifacts/baseline/model.joblib` as private input.
3. Create Kaggle Python notebook from `kaggle/inference_template.py` (copy to notebook cell), editing `ARTIFACT_DIR` to actual mounted private dataset path.
4. Run locally on sample interactive test to smoke-check schema.
5. Run Kaggle notebook with internet OFF; **Save Version > Save & Run All**; submit the notebook version.

## Submission schema invariants

- Input at `/kaggle/input/competitions/name-that-llm/test.csv`.
- Output at `/kaggle/working/submission.parquet`.
- Columns EXACTLY `id`, `model` (id strings preserved).
- All test IDs present exactly once; no nulls; no duplicate IDs; all 18 labels legal.
- Do not assume row count, test example values, topic or target proportions.

## Smoke-test warnings

- The interactive `test.csv` is tiny, not full hidden test; hidden evaluation reruns notebook.
- Runtime <= 9 hours, GPU or CPU; loading all models also counts.
- No external service calls or network dependencies.
- Verify Kaggle input path after attaching dataset.
- No unverified code executed using unknown serialized model weights.
