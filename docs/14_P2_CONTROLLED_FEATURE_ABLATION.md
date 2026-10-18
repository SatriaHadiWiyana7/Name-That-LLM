# P2 — Controlled Word / Character TF-IDF ablation

Date: 2026-10-18 (Asia/Jakarta). Target remains final private Top 5, minimum
aspiration Top 10. No rank claim; 0.547 remains unverified context.

## Audit and preregistered plan — before implementation/training

Read AGENTS.md, taskboard, experiment/score logs and docs/13. Initial tests:
31 passed in 5.84 s. `reports/p2_20261018_preflight.json` verifies both P0 runs'
artifact hashes and independently recomputes BA **0.31633855331841904**.
The existing `.venv` Python 3.13.11, all dependency versions, BLAS/OpenMP thread
settings and hardware match P0. Available RAM approximately 17.53 GiB and disk
173.26 GiB at audit. No packages will be installed or upgraded.

The previous P0/P1 work was uncommitted. Local prerequisite commit `f3d219e`
preserves that foundation, excluding raw data, OOF, folds and checkpoints.
P2 implementation and results will have separate local commits; no publishing.

Frozen reference: `artifacts/p0_20261018_run1/manifest.json`; original dataset
SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`;
saved fold SHA256 `3357fd3f290aaad4f1e50d9864db729e3803c2b6c62625748ff28408e94cb2bd`.
Use the existing 2699 rows, 150 prompt groups, five saved folds, seed 42 and
the exact LABELS order. No fitting on validation folds or sample test.

| Run | Single falsifiable hypothesis | Only feature change |
|---|---|---|
| Word-only | Removing character features decreases grouped OOF BA relative to combined P0. | Keep only the existing word vectorizer. |
| Character-only | Removing word features decreases grouped OOF BA relative to combined P0. | Keep only the existing character vectorizer. |

Both configurations retain all original word/char settings and classifier
`C=3.0`, `max_iter=1200`, `class_weight=balanced`, default lbfgs, random_state 42.
Inactive vectorizer configuration is retained for audit but its block is not
constructed/fitted. Do not normalize the concatenated output, retune C, change
feature limits, mask punctuation, preprocess texts differently or tune seeds.
Removing a separately normalized block changes the input norm under fixed C;
this estimates controlled deletion, not each standalone model's tuned optimum.

Execution: Word-only main run, Character-only main run, then an exact repeat
of each (four sequential CPU runs total). Use two BLAS/OpenMP threads and
PYTHONHASHSEED=42, sampling RSS at 50 ms plus OS lifetime peak. Record CV/fold/
refit/serialization time, CPU time, model bytes and source/config/data/artifact
hashes. Based on P0, expect minutes and <1 GiB process RSS per run (estimates).
No simultaneous training, GPU or transformer. Output directories are unique;
P0 files and the existing baseline configuration remain untouched.

Each run saves config, manifest, OOF probabilities/predictions, fixed folds and
metrics under `reports/p2_20261018_<word|char>_<run1|run2>/`; models under matching
private `artifacts/` directories. OOF/folds/models remain ignored by Git.
Require repeat BA/predictions/confusion equality and probability max absolute
difference <=1e-12; also compare serialized model hashes.

Evaluation: aggregate and all five fold BA with delta versus P0, all 18 class
recalls and full confusion matrices, error corrections/regressions and top
confusion partners. Explicit focus: claude-fable-5, gpt-5.6-sol, glm-5.3,
glm-5.3-flash, and both directions of gemini-3.8-flash vs gemini-3.7-flash.
Report paired prompt-cluster bootstrap using 10000 repeats, seed 42: draw the
same 150 prompt groups with replacement for both models, pool TP/support per
class, then compute candidate BA minus baseline BA. Report 95% percentile CI
and per-class delta CIs. A repeat missing a class is omitted and counted.
These are exploratory fixed-OOF intervals, not retraining uncertainty or
multiplicity-corrected significance tests. No ensemble weights will be tuned.

Decision rule: treat a candidate as a provisional replacement only if global
delta is positive, at least four of five folds improve, and the paired delta
interval excludes zero on the positive side. Otherwise retain P0 as reference
and report feature contribution without claiming a performance gain. Repeat
runs test determinism, not independent statistical evidence.

After the two main runs, assess stylometry-only as a separate follow-up if at
least two of the four specified weak classes remain below recall 0.30 in the
best observed configuration and their confusion partners persist. A numeric
style model would test count/rate signals not isolated by TF-IDF deletions.
This phase evaluates that need; it does not silently add a third feature study.
Deliver one evidence-based next experiment, with a fresh hypothesis/protocol.

## Results and decision

**Retain combined P0 as the reference.** Word-only loses 4.260502 percentage points of BA, consistently across all five folds. Character-only loses 0.296296 percentage points at the point estimate, but its paired interval includes zero. The latter result does not establish equivalence or a reliable global benefit from the word block. Neither candidate meets the preregistered promotion rule.

Implementation commit: `83548ac`. Full pretraining suite: **90 passed in 23.89 s**. Training code and the two saved variant configuration files were frozen before all four measured runs. Both retained vectorizers and the classifier keep P0 parameters; all runtime/data/fold guards passed.

| Configuration | OOF BA | Delta vs P0 | Paired 95% delta CI | Folds improved |
|---|---:|---:|---|---:|
| P0 combined | 0.316338553318 | reference | not applicable | not applicable |
| Word-only | 0.273733532190 | -0.042605021129 | [-0.054815, -0.030388] | 0/5 |
| Character-only | 0.313375590355 | -0.002962962963 | [-0.015936, 0.010000] | 2/5 |

Each comparison uses the same 10000 draws of 150 complete prompt groups (seed 42), with pooled class TP/support and all 18 class recalls. All 10000 draws are valid; none lacked a class. Intervals condition on these fitted OOF models, exclude retraining variance, and are exploratory without multiplicity correction. Repeats reproduce the same intervals and are not independent evidence. Removing a normalized block also changes feature norm under fixed C; the conclusions apply to this controlled deletion, not tuned standalone optima.

### Scores on the five saved folds

| Fold | P0 BA | Word BA | Word delta | Character BA | Character delta |
|---|---:|---:|---:|---:|---:|
| 0 | 0.296296296 | 0.259259259 | -0.037037037 | 0.311111111 | +0.014814815 |
| 1 | 0.303703704 | 0.261111111 | -0.042592593 | 0.272222222 | -0.031481481 |
| 2 | 0.327777778 | 0.272222222 | -0.055555556 | 0.324074074 | -0.003703704 |
| 3 | 0.335185185 | 0.285185185 | -0.050000000 | 0.329629630 | -0.005555556 |
| 4 | 0.318710089 | 0.290932312 | -0.027777778 | 0.329821201 | +0.011111111 |

Fold 4 contains 539 rows; the others contain 540. Every fold has 30 prompt groups and all 18 labels, with no train/validation prompt overlap. Aggregate BA is recomputed from all OOF class counts, not averaged fold BA.

### All 18 class recalls

| Class | P0 | Word | Word delta | Character | Character delta |
|---|---:|---:|---:|---:|---:|
| claude-fable-5 | 0.114094 | 0.080537 | -0.033557 | 0.114094 | +0.000000 |
| claude-opus-4-6 | 0.400000 | 0.240000 | -0.160000 | 0.413333 | +0.013333 |
| claude-opus-4-8 | 0.240000 | 0.186667 | -0.053333 | 0.240000 | +0.000000 |
| claude-opus-5 | 0.586667 | 0.586667 | +0.000000 | 0.600000 | +0.013333 |
| claude-sonnet-5 | 0.280000 | 0.213333 | -0.066667 | 0.253333 | -0.026667 |
| gemini-3.1-pro-preview | 0.700000 | 0.660000 | -0.040000 | 0.673333 | -0.026667 |
| gemini-3.6-flash | 0.453333 | 0.446667 | -0.006667 | 0.380000 | -0.073333 |
| gemini-3.7-flash | 0.360000 | 0.313333 | -0.046667 | 0.426667 | +0.066667 |
| gemini-3.8-flash | 0.306667 | 0.300000 | -0.006667 | 0.306667 | +0.000000 |
| glm-5.3 | 0.146667 | 0.126667 | -0.020000 | 0.146667 | +0.000000 |
| glm-5.3-flash | 0.186667 | 0.093333 | -0.093333 | 0.153333 | -0.033333 |
| gpt-5.5 | 0.266667 | 0.200000 | -0.066667 | 0.260000 | -0.006667 |
| gpt-5.6-luna | 0.273333 | 0.226667 | -0.046667 | 0.306667 | +0.033333 |
| gpt-5.6-sol | 0.140000 | 0.133333 | -0.006667 | 0.133333 | -0.006667 |
| gpt-5.6-terra | 0.280000 | 0.273333 | -0.006667 | 0.260000 | -0.020000 |
| gpt-6-astra | 0.426667 | 0.386667 | -0.040000 | 0.446667 | +0.020000 |
| kimi-k3 | 0.260000 | 0.260000 | +0.000000 | 0.240000 | -0.020000 |
| qwen3.8-max | 0.273333 | 0.200000 | -0.073333 | 0.286667 | +0.013333 |

All class delta intervals are in [per_class_scores.csv](../reports/p2_20261018_summary/per_class_scores.csv). Class/fold confusion matrices, FP/FN and top-three partners are saved in each comparison JSON.

### Requested weak classes

| Class | Word recall delta [95% CI] | Character recall delta [95% CI] |
|---|---|---|
| claude-fable-5 | -0.033557 [-0.075342, 0.006757] | +0.000000 [-0.053333, 0.053333] |
| gpt-5.6-sol | -0.006667 [-0.053333, 0.033333] | -0.006667 [-0.060000, 0.046667] |
| glm-5.3 | -0.020000 [-0.066667, 0.026667] | +0.000000 [-0.040000, 0.040000] |
| glm-5.3-flash | -0.093333 [-0.153333, -0.040000] | -0.033333 [-0.086667, 0.013333] |

- **claude-fable-5:** TP is 17/149 for P0, 12/149 for Word-only and 17/149 for Character-only. Confusion toward claude-opus-5 increases from 22 to 31 and 26 respectively. Equal P0/character recall masks 8 corrected and 8 regressed examples; the author remains difficult.
- **gpt-5.6-sol:** TP falls from 21/150 to 20/150 in both ablations. Character-only reduces confusion with gpt-5.6-luna from 38 to 28, while gpt-5.6-terra becomes the largest partner (29). The reduced Luna count does not improve overall class recall.
- **glm-5.3:** TP is 22/150, 19/150 and 22/150. Word-only increases confusion with claude-opus-5 from 30 to 37. Character-only has 29 such errors but increases glm-5.3-flash confusion from 20 to 24; ambiguity persists.
- **glm-5.3-flash:** TP falls from 28/150 to 14/150 with Word-only and 23/150 with Character-only. The Word-only recall delta is -0.093333, exploratory CI [-0.153333, -0.040000]. This is the strongest loss among the four focus classes when character features are removed. The character-only delta interval crosses zero.

| True class | P0 top 3 wrong targets (counts) | Word-only | Character-only |
|---|---|---|---|
| claude-fable-5 | claude-sonnet-5: 23; claude-opus-5: 22; claude-opus-4-8: 17 | claude-opus-5: 31; claude-opus-4-8: 19; glm-5.3: 15 | claude-opus-5: 26; claude-sonnet-5: 21; glm-5.3-flash: 18 |
| gpt-5.6-sol | gpt-5.6-luna: 38; gpt-5.6-terra: 28; gpt-6-astra: 18 | gpt-5.6-luna: 37; gpt-5.6-terra: 23; gpt-5.5: 16 | gpt-5.6-terra: 29; gpt-5.6-luna: 28; gpt-6-astra: 21 |
| glm-5.3 | claude-opus-5: 30; glm-5.3-flash: 20; claude-fable-5: 12 | claude-opus-5: 37; glm-5.3-flash: 15; claude-sonnet-5: 12 | claude-opus-5: 29; glm-5.3-flash: 24; claude-fable-5: 15 |
| glm-5.3-flash | glm-5.3: 27; claude-opus-5: 17; claude-fable-5: 13 | claude-opus-5: 27; glm-5.3: 16; claude-fable-5: 15 | glm-5.3: 26; claude-fable-5: 15; claude-opus-5: 14 |

### Gemini 3.8 versus 3.7 confusion

Each row uses all 150 true-class examples; the 2x2 pair is never renormalized.

| Configuration | True class | Correct | Predicted as the other pair member | Wrong to other labels | Recall |
|---|---|---:|---:|---:|---:|
| P0 combined | gemini-3.8-flash | 46 | 54 | 50 | 0.306667 |
| P0 combined | gemini-3.7-flash | 54 | 42 | 54 | 0.360000 |
| Word-only | gemini-3.8-flash | 45 | 50 | 55 | 0.300000 |
| Word-only | gemini-3.7-flash | 47 | 41 | 62 | 0.313333 |
| Character-only | gemini-3.8-flash | 46 | 49 | 55 | 0.306667 |
| Character-only | gemini-3.7-flash | 64 | 41 | 45 | 0.426667 |

Word-only reduces bidirectional pair confusions from 96 to 91 but also reduces correct pair predictions from 100 to 92: errors shift outside the pair. Character-only has 90 pair confusions and 110 correct predictions. Its gemini-3.7 recall gains +0.066667 (exploratory CI [0.013333, 0.120000]), while gemini-3.8 recall is unchanged (CI [-0.066667, 0.066667]). Across all classes, the benefit is offset, including gemini-3.6 recall -0.073333 and gemini-3.1 recall -0.026667. No family-only model or global gain is established.

### Error corrections and regressions

| Candidate | P0 wrong to correct | P0 correct to wrong | Net correct change | Within-family errors | Cross-family errors |
|---|---:|---:|---:|---:|---:|
| Word-only | 109 | 224 | -115 | 951 | 1009 |
| Character-only | 158 | 166 | -8 | 949 | 904 |

P0 has 947 within-family and 898 cross-family errors. Word-only adds 115 errors, of which 111 are cross-family; character features carry substantial provider-distinguishing signal in this fixed setup. Character-only adds 8 errors overall despite correcting 158 other P0 errors. These overlaps are descriptive; no ensemble weights or offsets were fitted on these OOF predictions.

### Full confusion matrices

Rows are true labels, columns predicted labels, in the unchanged LABELS order. Each full matrix totals 2699; each delta matrix has zero row sums. Aggregate and per-fold matrices remain in every run metrics/comparison JSON.

- [Combined P0 matrix](../reports/p2_20261018_summary/p0_combined_confusion_matrix.csv)
- [Word-only matrix](../reports/p2_20261018_summary/word_only_confusion_matrix.csv) and [delta vs P0](../reports/p2_20261018_summary/word_only_confusion_delta_matrix.csv)
- [Character-only matrix](../reports/p2_20261018_summary/character_only_confusion_matrix.csv) and [delta vs P0](../reports/p2_20261018_summary/character_only_confusion_delta_matrix.csv)
- [Machine-readable matrices with labels](../reports/p2_20261018_summary/confusion_matrices.json)

### Runtime, memory and reproducibility

| Run | CV seconds | Final fit seconds | Total before manifest seconds | Sampled peak RSS GiB | OS lifetime peak GiB | Model MiB |
|---|---:|---:|---:|---:|---:|---:|
| p0_20261018_run1 | 56.950625 | 15.665067 | 74.130748 | 0.675381 | 0.683567 | 11.427545 |
| p2_20261018_word_run1 | 6.813215 | 1.569287 | 9.026583 | 0.307858 | 0.313076 | 3.546328 |
| p2_20261018_char_run1 | 38.739072 | 10.289701 | 50.186846 | 0.519062 | 0.535160 | 7.880363 |
| p2_20261018_word_run2 | 6.791762 | 1.575781 | 8.991421 | 0.307446 | 0.313488 | 3.546328 |
| p2_20261018_char_run2 | 38.577895 | 10.147940 | 49.880021 | 0.535702 | 0.536560 | 7.880363 |

All fits are sequential CPU jobs with P0 package versions and two BLAS/OpenMP threads. These are measured local timings, not a dedicated isolated-machine speed benchmark. RSS sampling is every 50 ms; OS peak covers the full process. Total excludes final manifest/register writes. No GPU is used and VRAM is not measured. Exact byte counts, CPU/fold times and serialization times are retained in CSV/JSON manifests.

Both repeats have BA delta **0.0 versus their own first run**, maximum OOF probability difference **0.0**, identical predictions/confusion and identical serialized model SHA256. Paired-bootstrap outputs also match exactly. The repeats are determinism checks, not extra independent validation data. Saved-model parquet and the actual Kaggle Python template with local paths both pass on the sample; no sample score is computed and no hidden Kaggle run is claimed.

- [Word reproduction proof](../reports/p2_20261018_word_reproduction_check.json)
- [Character reproduction proof](../reports/p2_20261018_char_reproduction_check.json)
- [Comparison/resource table](../reports/p2_20261018_summary/experiment_comparison.csv)
- [All run/fold scores](../reports/p2_20261018_summary/per_fold_scores.csv)

### One next experiment: stylometry-only

**Recommend a separately preregistered stylometry-only grouped-CV experiment.** All four specified weak classes remain below 0.30 recall in the highest measured BA model (P0), satisfying the preregistered follow-up criterion. Recurring Claude/GLM and GPT confusions survive both feature deletions. The proposed hypothesis is that explicit punctuation/length/rate features retain useful style information beyond normalized local TF-IDF patterns; this has not been measured by the current ablations.

Predefine a small feature list (text/word/sentence lengths, word-length statistics, em-dash/semicolon/colon/quote/digit/uppercase/Unicode counts and rates) without class-specific tuning. Use the same saved folds/seed and 18-label logistic regression; fit any StandardScaler only on each training fold. Record global/class/fold BA, paired prompt-bootstrap deltas versus P0, correction/regression counts, resource and repeatability. Budget a small CPU pilot and measure its cost; no predicted gain is assumed. Apply the same promotion rule. A lower-BA model with different errors is only evidence for later leakage-safe research, not permission to tune ensemble weights on these evaluation OOF rows.

Stylometry-only, punctuation masking, new split studies and transformers were not run in this phase. The current outcome supports keeping combined P0; it does not establish a private leaderboard score or proximity to Top 5/Top 10.

### Reproduction commands and output locations

Configurations: `configs/p2_word_only.json`, `configs/p2_char_only.json`. Each run retains exact config bytes and an identical manifest copy under its reports directory; model checkpoints remain in ignored artifacts directories. The baseline reference and original P0 files are preserved.

```powershell
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:NUMEXPR_NUM_THREADS='2'
$env:PYTHONHASHSEED='42'
.venv/Scripts/python.exe scripts/train_baseline.py --config configs/p2_word_only.json --run-id p2_20261018_word_run1 --reports reports/p2_20261018_word_run1 --out artifacts/p2_20261018_word_run1 --folds reports/p0_20261018_run1/baseline_fold_assignments.csv --environment-report reports/p2_20261018_preflight.json --reference-metrics reports/p0_20261018_run1/baseline_metrics.json --reference-manifest artifacts/p0_20261018_run1/manifest.json
.venv/Scripts/python.exe scripts/compare_oof.py --baseline-manifest artifacts/p0_20261018_run1/manifest.json --candidate-manifest artifacts/p2_20261018_word_run1/manifest.json --bootstrap-repeats 10000 --seed 42 --output reports/p2_20261018_word_run1/comparison_vs_p0.json
```

Use char config/run paths for Character-only and run2 paths for exact repeats; every actual full command/environment is in its manifest. Existing outputs are intentionally refused; select fresh versioned run names when rerunning. Main-run diagnostics also use `scripts/analyze_oof.py --train data/raw/train.csv` for length diagnostics. All four runs are appended to reports/experiments.csv. Source code is committed, while raw data, OOF/folds and serialized models remain private/ignored.
