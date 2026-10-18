# Local P0 audit and reproduction — 2026-10-18 (Asia/Jakarta)

## Preregistered plan (before training)

Hypothesis: the unchanged word+character TF-IDF / logistic-regression baseline
is reproducible locally when the same saved prompt folds, seed and environment
are replayed. Two local runs must give identical predictions, confusion counts,
and balanced accuracy; probability agreement is checked with absolute tolerance
1e-12. This is a reproducibility experiment, not a feature/model improvement.

Historical reference: `reports/baseline_metrics.json`, BA 0.3185657469550087.
Its historical OOF, dataset hash, NumPy version and fold mapping are absent.
Numerical differences against it cannot establish an identical-split improvement.
Preserve that file and the existing experiment/score records.

Protocol: 5-fold GroupKFold(prompt_id), no shuffle, seed 42, exact 18-label
ordering, fresh vectorizers within each training fold. Save the first run's
fold assignments before fitting; replay those assignments in a second run.
Evaluate aggregate and per-fold BA, all class recalls and confusion matrices.
Inspect the lowest recalls and top confusion partners; sample test is only
for schema/inference validation. No hyperparameter tuning or model ablation.

Resource plan: CPU only, two BLAS/OpenMP threads, sequential runs. Inspect
available RAM/disk first, sample process RSS every 50 ms and capture OS process
peak where supported, record timing per fold/CV/final fit, package versions,
thread settings, hardware, exact command and hashes. No CUDA allocation by the
CPU model; GPU peak memory is not measured by this experiment. Baseline expectation
is minutes per run (estimate, not measured). Large transformer training is outside
this P0 experiment; CUDA backward and a small GPU benchmark are prerequisites.

Local files: `data/raw/train.csv` and `test.csv` matched the original ZIP bytes
before changes. `prepare_data.py` extracts a separate private verification copy
to `data/processed/p0_original_zip`, preserving the original local inputs.

Preflight: the initial base-Python pytest command failed collection because the
project package was not installed. A separate `.venv` is being created; its
scikit-learn is fixed to 1.8.0 to match the recorded historical version. This
does not establish full historical environment equivalence. No base Conda/CUDA
environment is modified.

## Measured results

Run `p0_20261018_run1` completed: OOF BA **0.31633855331841904**;
CV 56.950625 s, final refit 15.665067 s, total before manifest write 74.130748 s.
Sampled peak RSS 725184512 bytes; OS process-lifetime peak 733974528 bytes
(0.683567 GiB). Model 11982649 bytes (11.43 MiB). CPU only; VRAM not measured.
Per-fold BA: 0.2962962963, 0.3037037037, 0.3277777778, 0.3351851852,
0.3187100894. Full recalls/confusion/resources are in the run metrics JSON.

Numerical difference to historical BA: -0.0022271936 (-0.222719 percentage
points). This is not an identical-split comparison: historical fold mapping,
dataset hash and complete environment are missing. No cause is established.

Prompt-cluster bootstrap (2000 repeats, seed 42) gives approximate 95% interval
[0.3011044795, 0.3318518519], conditional on fitted OOF models, without retraining.
`oof_diagnostics.json` records class/family/length/fold diagnostics and top pairs.
The repeat `p0_20261018_run2` completed with the same BA, all class recalls,
all fold confusion matrices, and maximum absolute probability difference **0.0**.
The saved fold hash and serialized model hash match. Same-split delta BA is
**0.000000000000** on every fold and overall. This establishes local repeatability,
not exact reproduction of the historical environment.

| Resource | Run 1 | Run 2 |
|---|---:|---:|
| CV seconds | 56.950625 | 60.134572 |
| Final refit seconds | 15.665067 | 14.945798 |
| Serialization seconds | 1.313833 | 1.340738 |
| Total before manifest seconds | 74.130748 | 76.613253 |
| Sampled peak RSS bytes | 725184512 | 722870272 |
| OS process-lifetime peak RSS bytes | 733974528 | 734617600 |
| Model bytes | 11982649 | 11982649 |

The sampled RSS can miss short peaks; OS peak includes the whole process.
CPU experiments did not measure GPU memory. All fold-level resource values are
saved in each `baseline_metrics.json`. The package lock records the actual
Python 3.13.11 CPU environment, including sklearn 1.8.0, NumPy 2.5.4 and SciPy 1.18.1.

## All 18 class recalls (identical in both local runs)

| Label | Recall |
|---|---:|
| claude-fable-5 | 0.114094 |
| claude-opus-4-6 | 0.400000 |
| claude-opus-4-8 | 0.240000 |
| claude-opus-5 | 0.586667 |
| claude-sonnet-5 | 0.280000 |
| gemini-3.1-pro-preview | 0.700000 |
| gemini-3.6-flash | 0.453333 |
| gemini-3.7-flash | 0.360000 |
| gemini-3.8-flash | 0.306667 |
| glm-5.3 | 0.146667 |
| glm-5.3-flash | 0.186667 |
| gpt-5.5 | 0.266667 |
| gpt-5.6-luna | 0.273333 |
| gpt-5.6-sol | 0.140000 |
| gpt-5.6-terra | 0.280000 |
| gpt-6-astra | 0.426667 |
| kimi-k3 | 0.260000 |
| qwen3.8-max | 0.273333 |

The complete 18x18 confusion matrix and all five fold matrices are in
[`baseline_metrics.json`](../reports/p0_20261018_run1/baseline_metrics.json).
The bottom five classes' top confusion targets (counts) are:

| True label | Three main confusion targets |
|---|---|
| claude-fable-5 | claude-sonnet-5:23; claude-opus-5:22; claude-opus-4-8:17 |
| gpt-5.6-sol | gpt-5.6-luna:38; gpt-5.6-terra:28; gpt-6-astra:18 |
| glm-5.3 | claude-opus-5:30; glm-5.3-flash:20; claude-fable-5:12 |
| glm-5.3-flash | glm-5.3:27; claude-opus-5:17; claude-fable-5:13 |
| claude-opus-4-8 | claude-opus-5:21; claude-fable-5:17; claude-opus-4-6:12 |

Largest directional confusion is gemini-3.8-flash -> gemini-3.7-flash (54).
Of 1845 errors, 947 are within-provider and 898 cross-provider. Provider aggregation
is diagnostic only. Each class with recall < 0.30 has top-three partners/FP/FN
recorded in `oof_diagnostics.json`; no family model was trained or promoted.

## Commands and verification

Actual CPU environment setup used `python -m venv .venv`, followed by
`.venv/Scripts/python.exe -m pip install -r requirements-cpu.txt scikit-learn==1.8.0 psutil`
and `.venv/Scripts/python.exe -m pip install -e . --no-deps`. The exact resulting
versions are frozen in `reports/p0_20261018_requirements-lock.txt`; use that lock
for later environment recreation instead of resolving the original broad ranges.

Run from the repository root in PowerShell. Choose new run/output names when
repeating: existing artifacts and reports are never silently overwritten.

```powershell
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:NUMEXPR_NUM_THREADS='2'
$env:PYTHONHASHSEED='42'
.venv/Scripts/python.exe scripts/train_baseline.py --config configs/baseline.json --run-id p0_20261018_run1 --reports reports/p0_20261018_run1 --out artifacts/p0_20261018_run1 --environment-report reports/p0_20261018_environment.json --reference-metrics reports/baseline_metrics.json
.venv/Scripts/python.exe scripts/train_baseline.py --config configs/baseline.json --run-id p0_20261018_run2 --reports reports/p0_20261018_run2 --out artifacts/p0_20261018_run2 --folds reports/p0_20261018_run1/baseline_fold_assignments.csv --environment-report reports/p0_20261018_environment.json --reference-metrics reports/p0_20261018_run1/baseline_metrics.json
.venv/Scripts/python.exe scripts/analyze_oof.py --oof reports/p0_20261018_run1/baseline_oof.csv --train data/raw/train.csv --bootstrap-repeats 2000 --seed 42 --output reports/p0_20261018_run1/oof_diagnostics.json
.venv/Scripts/python.exe scripts/predict_local.py --model artifacts/p0_20261018_run1/model.joblib --test data/raw/test.csv --output reports/p0_20261018_run1/local_demo_submission.parquet
.venv/Scripts/python.exe scripts/validate_submission.py --test data/raw/test.csv --submission reports/p0_20261018_run1/local_demo_submission.parquet
.venv/Scripts/python.exe scripts/verify_reproduction.py --first-manifest artifacts/p0_20261018_run1/manifest.json --second-manifest artifacts/p0_20261018_run2/manifest.json --local-submission reports/p0_20261018_run1/local_demo_submission.parquet --smoke-output reports/p0_20261018_run1/kaggle_template_smoke.parquet --output reports/p0_20261018_reproduction_check.json
```

The independent verifier checked artifact hashes, recomputed every aggregate/fold
metric from OOF, compared both predictions/probabilities and model hashes, then
executed the actual Python Kaggle template with only deployment paths replaced.
Both saved-model CLI and template produced valid matching three-row parquet.
The sample has no evaluation role. Kaggle hidden/offline rerun remains untested.

Final verification: **31 tests passed in 5.08s**; `pip check` passed; all nine CLI
`--help` entrypoints and the read-only score report passed. The historical metrics
SHA256 is unchanged; git-ignore protection for private datasets, models, nested
OOF/fold files and parquet was checked. Commands, outputs and test-file hashes
are saved in `reports/p0_20261018_final_checks.json`.

## CUDA kernel gate after P0

Existing `gpu_test` ran 100 float32 matmul/backward steps, matrix 1024x1024,
10 warmup steps, seed 42. All losses and gradients finite. Timed steps 0.0734684 s;
whole check 4.6000688 s. Peak PyTorch allocated 92277760 bytes (~88.003 MiB),
reserved 113246208 bytes (108 MiB); process OS peak 1058304000 bytes. The CUDA
runtime's broader footprint and other applications are not included in PyTorch
allocated peaks. Source hash, exact command, dependencies and scope are saved in
`reports/p1_20261018_gpu_benchmark.json`.

Actual benchmark driver 610.88 differs from the earlier audit's 610.74; both dated
observations are retained. This work installed no GPU driver or CUDA package.
This is a synthetic compatibility check, not encoder speed/VRAM or model BA.
No transformer was trained; pretrained license and one-fold encoder preflight
are still outstanding.

## Decision and next experiment

P0 is complete, with measured local baseline 0.316338553318. Preserve its folds
and use it for P2 comparisons. The next falsifiable hypothesis is that word and
character features contribute differently to the low-recall classes; compare
word-only and char-only ablations against this combined reference with identical
folds/seed, reporting global/per-fold BA and class changes. These ablations have
not been executed. No private/public leaderboard result or Top 5/Top 10 cutoff
has been established, and 0.547 remains unverified user context.
