# Dated experiment/research changelog

## 2026-10-17 — Initial scaffold

- Added project governance, verified dataset EDA notes, rule checklist, Top 5 / Top 10 goal.
- User reported score `0.547`; ownership and leaderboard split unconfirmed; logged as contextual score only.
- Added runnable GroupKFold 5-fold TF-IDF + logistic regression baseline, scoring, submission validator and score logger.
- No leaderboard claim; no transformer fine-tune performed by this scaffold.

Append subsequent changes with: timestamp, hypothesis, command/config, OOF BA, class changes, hardware/runtime and decision.

## 2026-10-17 — CPU pipeline execution check

- Unit tests: 4 passed. Data audit reproduced 2699 / 150 / 18 and EDA word mean 115.29 under original tokenizer.
- Local CPU 5-fold grouped OOF balanced accuracy: **0.318566**, 58.24 seconds for CV (additional final fit).
- Out-of-fold per-class recall/confusion and checkpoint generated; docs/12 contains hard confusion examples.
- Python environment had no pyarrow, so parquet writing not directly tested; install requirements and verify locally.

## 2026-10-18 — Local P0 preflight

- Read all project docs and historical experiment/score reports. Target remains
  private Top 5 / minimum aspiration Top 10; 0.547 remains unverified context.
- Original base-Python test collection failed because the package was not
  installed. Created a separate `.venv`, installed CPU requirements and the
  editable project; original tests: 4 passed. Added unequal-support BA and
  missing/illegal-label checks plus parquet round-trip; 7 passed at this milestone.
- Local train/test hashes match the original ZIP. Separate private extraction
  and extended aggregate audit reproduce the recorded dataset counts and format
  statistics; see `reports/p0_20261018_dataset_audit.json`.
- Recorded environment inventory in `reports/p0_20261018_environment.json` and
  CPU dependency lock in `reports/p0_20261018_requirements-lock.txt`. No global
  Conda/CUDA installation was changed. Existing `gpu_test` supports sm_120 in
  metadata; a real backward/100-step benchmark remains unexecuted at this point.
- Preregistered two sequential fixed-fold CPU baseline runs in docs/13. No new
  BA or performance improvement is claimed in this preflight entry.

## 2026-10-18 — P0 CPU reproduction, first run

- Hypothesis: unchanged baseline repeats locally on fixed prompt folds/seed.
  `p0_20261018_run1`, configs/baseline.json, GroupKFold5, seed42, two CPU threads.
- Measured OOF BA **0.31633855331841904**; fold BA 0.2962962963 / 0.3037037037 /
  0.3277777778 / 0.3351851852 / 0.3187100894. CV 56.950625s, final fit15.665067s,
  total74.130748s before final manifest writes; sampled peak RSS725184512 bytes,
  OS peak733974528 bytes, artifact11982649 bytes. GPU not used; VRAM unmeasured.
- Historical numerical difference -0.0022271936 is not a verified same-fold
  delta. Historical data/fold hashes and full dependency versions are absent.
- Worst recalls: claude-fable-5 0.114094, gpt-5.6-sol0.140000, glm-5.3 0.146667,
  glm-5.3-flash0.186667, claude-opus-4-8 0.240000. Largest directional confusion:
  gemini-3.8-flash -> gemini-3.7-flash,54. All 18 recalls and confusion matrices
  saved, including each fold; prompt-bootstrap interval [0.3011045,0.3318519].
- Exact command, environment, data/config/source/fold/OOF/model hashes and
  resource boundaries are in `artifacts/p0_20261018_run1/manifest.json`.
  Experiment register appended; next step is the preregistered fixed-fold repeat.

## 2026-10-18 — P0 fixed-fold repeat and inference verification

- `p0_20261018_run2` replayed run1's saved folds and seed42. OOF BA is exactly
  **0.31633855331841904**, same-split delta **+0.000000000000**, all five fold
  BA values and class recalls/confusion counts identical. CV60.134572s,
  total76.613253s before manifest writes, sampled peak RSS722870272 bytes.
  Full per-fold resources and final-fit timing are saved in its manifest.
- Independent verification recomputed all aggregate/fold metrics and checked
  artifact/data hashes. Maximum probability difference **0.0**; model joblib
  SHA256 identical. `reports/p0_20261018_reproduction_check.json` is the record.
- Loaded the actual saved model, predicted sample test, wrote/validated parquet.
  Executed the actual Kaggle Python template locally with only its three paths
  replaced; its predictions match the CLI. This is not a Kaggle hidden rerun or
  a leaderboard measurement. P0 reproducibility is complete.
- Decision: freeze this measured local reference before P2 feature ablations.
  Next hardware hypothesis: existing sm_120 CUDA environment completes 100 finite
  float32 matmul/backward steps. Plan: synthetic1024x1024,10warmup,seed42,
  synchronized timing/VRAM/RSS; no pretrained model or dataset involved.

## 2026-10-18 — P1 synthetic CUDA compatibility benchmark

- Ran existing `gpu_test` Python 3.11.14 / torch 2.9.1+cu128, CUDA 12.8, sm_120.
  Command: `scripts/check_gpu.py --steps 100 --size 1024 --warmup 10 --seed 42
  --output reports/p1_20261018_gpu_benchmark.json` with that environment's Python.
- All 100 float32 forward/backward steps and gradients finite. Timed loop
  0.0734684 s; full check 4.6000688 s; peak allocated 92277760 bytes (88.003 MiB),
  reserved 113246208 bytes (108 MiB); CPU OS lifetime peak 1058304000 bytes.
- Benchmark observed driver 610.88; initial inventory recorded 610.74. Both
  records retained. No driver/CUDA packages were installed by this work.
- Appended synthetic experiment to reports/experiments.csv with BA/delta blank:
  there is no dataset or classifier score for this kernel test. Hardware gate
  passes; encoder license, tokenizer/offload and one-fold benchmark remain open.
  No transformer training or Kaggle submission performed.

## 2026-10-18 — Final P0 verification

- All 31 unit tests passed in 5.08s; dependency consistency and all nine CLI
  help entrypoints passed. Exact results are in reports/p0_20261018_final_checks.json.
- Historical metrics checksum remained unchanged. Private raw/extracted data,
  models, nested OOF/fold CSVs and parquet outputs are ignored by git.
- README, dataset/hardware notes, historical-baseline follow-up, taskboard and
  docs/13 now link the measured local results. No score event was added for the
  unclarified 0.547 reference, and no rank claim was made.

## 2026-10-18 — P2 audit, preregistration and implementation

- Preserved previously uncommitted P0/P1 foundation in local commit f3d219e;
  raw data, OOF/folds and model artifacts remain excluded from Git.
- Audit confirms exact P0 dataset/fold hashes, label order, Python/dependencies,
  thread settings and recomputed BA 0.31633855331841904. No package changes.
- Preregistered two feature deletions plus one exact repeat per variant in
  docs/14 and reports/p2_20261018_plan.json before implementation/training.
- Added word-only/char-only selection with unchanged retained vectorizers and
  classifier; guard rejects config/provenance/environment drift before fitting.
  Added verified OOF comparison with 10000 paired prompt-cluster bootstrap draws.
- Pretraining full suite: 90 passed in 23.89 s. Both new/updated CLI help paths
  pass. No real P2 training has occurred at this entry; results follow separately.

## 2026-10-18 — P2 Word-only main run

- Hypothesis: deleting the character block decreases BA. Measured OOF BA **0.273733532189908**, delta **-0.042605021128511** versus P0. All five folds decrease. Paired prompt-bootstrap 95% delta CI [-0.054814814815, -0.030387770321] (10000 draws, seed 42).
- CV 6.813215s; final fit 1.569287s; total before manifest 9.026583s. OS peak RSS 336162816 bytes; model 3718594 bytes; CPU only.
- Confusion/recalls, full per-fold results, paired comparison, config/manifest/OOF hashes and exact commands saved in reports/p2_20261018_word_run1. Word-only does not qualify to replace P0. Character-only and repeat runs still pending; no tuning.

## 2026-10-18 — P2 Character-only main run

- Hypothesis: deleting the word block decreases BA. Measured OOF BA **0.313375590355456**, delta **-0.002962962962963** versus P0. 2/5 folds improve. Paired prompt-bootstrap 95% delta CI [-0.015935930897, 0.010000125125] (10000 draws, seed 42); interval includes zero, so evidence does not establish a consistent contribution of the word block.
- CV 38.739072s; final fit 10.289701s; total before manifest 50.186846s. OS peak RSS 574623744 bytes; model 8263159 bytes; CPU only.
- No ablation qualifies to replace P0. In the highest measured BA model (P0), all four specified weak classes still have recall<0.30 and related confusion partners persist. Stylometry-only meets the preregistered need criterion as a separate next experiment; not trained in this phase. Exact repeats remain pending.

## 2026-10-18 — P2 Word-only exact repeat

- Same-config/fold/seed repeat BA 0.273733532189908; delta vs P0 -0.042605021128511. CV 6.791762s, final fit 1.575781s, total 8.991421s; OS peak RSS 336605184 bytes.
- Independent verifier confirms all OOF/per-fold metrics, identical predictions/confusion, probability max difference 0.0 and same model SHA256. Saved-model parquet and actual local-path Kaggle template agree. Repeat paired intervals match the first run; this is determinism evidence, not a new validation split. See reports/p2_20261018_word_reproduction_check.json. Character-only repeat follows.

## 2026-10-18 — P2 Character-only exact repeat and final evaluation

- Repeat BA 0.313375590355456; delta vs P0 -0.002962962962963. CV 38.577895s, final fit 10.147940s, total 49.880021s; OS peak RSS 576126976 bytes.
- Independent verifier confirms identical probabilities (max difference 0.0), class/fold confusion/BA and model hash. Local saved-model/template parquet smoke passes. Repeat paired bootstrap results equal the main run, with no new statistical evidence.
- All four trained runs and comparisons logged. Consolidated experiment/fold/class tables and full confusion/delta matrices are in reports/p2_20261018_summary. P0 remains selected. Next recommendation: separately preregister stylometry-only; no transformer or extra feature experiment run.

## 2026-10-18 — P2 final artifact verification

- `reports/p2_20261018_final_checks.json` records passed artifact/data/config/fold
  checks for all four P2 runs, independent metric recomputation, exact repeat
  equality and preservation of the P0 references. Source and tests are unchanged
  since the 90-test pretraining check and implementation commit `83548ac`.
- Verified private raw data, OOF/folds and model files remain ignored by Git.
  Final documentation and aggregate reports are prepared as a separate local
  results commit; no publication or leaderboard submission is performed.

## 2026-10-18 — P3 audit and preregistration

- Read AGENTS, taskboard, experiment log and complete docs13/docs14. Existing
  full suite: 90 passed in 17.71s before implementation.
- reports/p3_20261018_preflight.json verifies all six P0/P2 runs, data/artifact
  hashes, GroupKFold equivalence and unchanged runtime; P0 BA remains exactly
  0.31633855331841904. Free RAM 18.499 GiB and disk 172.848 GiB at audit.
- Preregistered fixed 81-feature stylometry, training-fold StandardScaler and
  logistic regression, two identical runs and 10000 paired prompt-cluster
  draws in docs/15 and immutable reports/p3_20261018_plan.json. No P3 code or
  training exists at this entry. No packages changed and no transformer planned.
- Primary hypothesis: strictly positive recall delta in at least two of four
  weak focus classes. Global promotion separately requires positive delta,
  improvement on at least four folds and positive 95% delta interval.
