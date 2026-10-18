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

## Results

Implementation validation: 90 unit tests passed in 23.89 s. Baseline/default
behavior, retained-block parameter identity, validation vocabulary isolation,
environment/config drift rejection, manifest copies and paired bootstrap
accounting are covered. Exact pretraining output is saved in
`reports/p2_20261018_pretraining_tests.json`.

Pending execution; this section contains no new measured model score yet.
