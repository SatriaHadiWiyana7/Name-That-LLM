# Name That LLM — Top 5 / Top 10 Challenge

**Status:** research scaffold + executable classical baseline; not a proven Top-5 solution.  
**Primary goal:** final private leaderboard **Top 5**. **Minimum aspirational goal:** **Top 10**. Neither is guaranteed.  
**Latest user-reported score reference:** **0.547** (source/split/rank NOT verified; record is NOT an OOF metric).

Competition: https://www.kaggle.com/competitions/name-that-llm

## Start here (for Codex and humans)

1. Read [`AGENTS.md`](AGENTS.md) entirely. This file defines the non-negotiable workflow.
2. Read [`docs/00_PROJECT_BRIEF.md`](docs/00_PROJECT_BRIEF.md), [`docs/01_VERIFIED_DATASET_FINDINGS.md`](docs/01_VERIFIED_DATASET_FINDINGS.md), and [`docs/02_COMPETITION_RULES.md`](docs/02_COMPETITION_RULES.md).
3. Follow the prioritized tasks in [`docs/08_CODEX_TASKBOARD.md`](docs/08_CODEX_TASKBOARD.md).
4. Run tests before editing code and after every milestone.

## Workflow: local training -> Kaggle offline inference

```text
Original competition ZIP (kept private)
      -> data/raw/{train.csv,test.csv}
      -> prompt-grouped cross validation + diagnostics
      -> locally trained reproducible model artifacts
      -> private Kaggle Dataset / Model input
      -> internet-OFF Kaggle notebook (INFERENCE ONLY)
      -> /kaggle/working/submission.parquet
      -> report LB result -> next hypothesis
```

Data and trained checkpoints are **not** included in this scaffold. Use the original dataset ZIP downloaded from your own Kaggle account. Do not publicly re-upload competition data or private competition code outside the permitted rules.

## Quick start (PowerShell or WSL / Linux)

Python 3.11 is recommended for environment compatibility; Python 3.10–3.12 may also work with matching packages.

```bash
python -m venv .venv
# Linux/WSL:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python -m pytest -q

python scripts/prepare_data.py --zip /path/to/name-that-llm-dataset.zip
python scripts/audit_dataset.py
python scripts/train_baseline.py --config configs/baseline.json --run-id local_baseline --reports reports/local_baseline --out artifacts/local_baseline
python scripts/analyze_oof.py --oof reports/local_baseline/baseline_oof.csv --train data/raw/train.csv --output reports/local_baseline/oof_diagnostics.json
python scripts/predict_local.py --model artifacts/local_baseline/model.joblib --test data/raw/test.csv --output reports/local_demo_submission.parquet
python scripts/validate_submission.py --test data/raw/test.csv --submission reports/local_demo_submission.parquet
```

For the Kaggle submission notebook starter see [`kaggle/inference_template.ipynb`](kaggle/inference_template.ipynb). It becomes usable once your private baseline weights have been uploaded and its input path updated.

Choose a new run ID and output directories for each run; existing experiment
outputs are protected against overwrite. Use `--folds` to replay a saved fold
mapping and `--reference-metrics` to record a provenance-checked local delta.

The `test.csv` in an interactive session has only example rows and is NOT the hidden rerun data. **Never** infer hidden leaderboard performance from those few rows.

GPU environment check (install a Blackwell-compatible CUDA-enabled PyTorch separately from https://pytorch.org/get-started/locally/):

```bash
nvidia-smi
python scripts/check_gpu.py
```

## Files to preserve at every milestone

- `reports/baseline_metrics.json`: out-of-fold balanced accuracy, per-class recall, fold metrics.
- `reports/baseline_oof.csv`: OOF predictions and probabilities; never use in public repos.
- `reports/experiments.csv`: comparable local experiments (populate as models evolve).
- `reports/score_updates.csv`: user-provided LB scores and optional ranking / thresholds.
- `artifacts/*/manifest.json`: data digest, seeds, hyperparameters, model metadata.

**Note:** the Python starter baseline has not been proven Top 10; its first job is to establish a measured, grouped-validation reference and unblock Codex.

## Initial measured CPU baseline

The classical baseline was executed on the original dataset in a sandbox CPU environment and obtained **OOF balanced accuracy 0.318566** (5-fold GroupKFold). See [`docs/12_MEASURED_BASELINE.md`](docs/12_MEASURED_BASELINE.md). It is NOT a Kaggle leaderboard score and has not been benchmarked on your RTX 5060 Ti.

## Local verification — 2026-10-18

P0 is complete: **31 unit tests passed**, original ZIP/data hashes match, and two
local 5-fold runs achieved **OOF BA 0.316338553318** with identical probabilities
and serialized model hashes. Same-fold repeat delta is **0.000000**. Historical
fold equivalence is unverified. Saved-model parquet and local Kaggle-template
smoke tests pass. The existing CUDA environment also passed 100 synthetic
forward/backward steps; no transformer was trained.

See [`docs/13_LOCAL_P0_REPRODUCTION.md`](docs/13_LOCAL_P0_REPRODUCTION.md) for
all class recalls, fold metrics, confusion reports, resource measurements,
reproduction commands and remaining encoder/Kaggle checks.
