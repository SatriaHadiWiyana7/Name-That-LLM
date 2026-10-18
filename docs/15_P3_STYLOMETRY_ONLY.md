# P3 — Preregistered stylometry-only experiment

Date: 2026-10-18 (Asia/Jakarta). This plan is written before P3 implementation or
training and will be snapshotted with a hash in reports/p3_20261018_plan.json.
Final private Top 5 is the primary aspiration, Top 10 the minimum aspiration;
0.547 is unverified context. Official reference remains combined P0 OOF balanced
accuracy **0.31633855331841904**. P0/P2 artifacts, configurations and source stay
unchanged. P3 adds separate modules/entrypoints; no transformer is trained.

## Hypothesis and decision, fixed before coding

Primary falsifiable hypothesis: a fixed numeric stylometry model improves OOF
recall over P0 in at least two of claude-fable-5, gpt-5.6-sol, glm-5.3 and
glm-5.3-flash. The count of strictly positive class recall deltas tests this
descriptive hypothesis; paired intervals quantify uncertainty, and a positive
point estimate alone is not a reliable class improvement claim.

Global promotion is a separate gate: positive overall BA delta, at least four
of five positive fold deltas, and a 95% paired bootstrap delta interval entirely
above zero. Otherwise retain P0. Different errors at lower BA are descriptive
complementarity only; no ensemble or class offsets are fitted to these OOF rows.
The model tests standalone style information, not incremental value when added
to TF-IDF. No feature selection, C search, seed search or validation-driven edits.

## Frozen extractor specification (81 numeric features)

Use original Python strings without normalization. Output float64 in a stable
named column order. No vocabulary learned from data, external lexicon, IDs,
prompt IDs or labels as features. Empty inputs yield finite zeros. A rate with
zero denominator is zero. Reject non-string inputs rather than stringify nulls.

1. Text: Unicode code-point length and UTF-8 byte length (2 features).
2. Words: Unicode regex `[^\W\d_]+(?:['’][^\W\d_]+)*`, preserving internal
   straight/curly apostrophes. Word count and token code-point length mean,
   population standard deviation, minimum and maximum (5 features).
3. Sentences: split at runs of `[.!?]+`; count only segments containing at least
   one word token. Sentence count and words-per-sentence mean, population
   standard deviation, minimum and maximum (5 features). This is a deterministic
   heuristic, not a linguistic parser; abbreviations/decimals can over-split.
4. Lexical statistics: casefolded unique word count, type/token ratio, hapax
   token rate (words whose casefolded type occurs once), short-token rate
   (length <=3), long-token rate (length >=7), apostrophe-token rate (contains
   U+0027 or U+2019), and fixed function-word rate (7 features). Denominator
   for all these rates is word count. Fixed casefolded word set:
   `a an the and or but if then because as of to in on at by for from with
   without is are was were be been being it its this that these those i we
   you he she they them our your their`.
5. Character category counts and rates: whitespace (`isspace`), newline
   (U+000A), tab (U+0009), alphabetic (`isalpha`), uppercase (`isupper`), lowercase
   (`islower`), digit (`isdigit`), non-ASCII (code point >127), Unicode category
   P punctuation, S symbol, M mark (11 pairs =22 features). Record Unicode
   database version; categories may overlap by design.
6. Punctuation counts and rates (20 pairs =40 features), in this exact order:
   period `.`, comma `,`, semicolon `;`, colon `:`, exclamation `!`, question `?`,
   hyphen `-`, em dash U+2014, en dash U+2013, apostrophe U+0027, quote U+0022,
   curly single quotes U+2018/U+2019, curly double quotes U+201C/U+201D,
   ellipsis U+2026, parentheses `()`, brackets `[]`, slash `/`, backslash U+005C,
   underscore `_`, bullet U+2022. Count individual matching code points;
   grouped pairs combine their members. ASCII `...` counts three periods,
   not a Unicode ellipsis. Every character/punctuation rate divides by total
   Unicode code-point text length (including whitespace).

## Model, folds and environment

Pipeline: stateless StylometryExtractor -> StandardScaler(with_mean=True,
with_std=True) -> LogisticRegression(C=3.0, max_iter=1200,
class_weight='balanced', solver='lbfgs', tol=1e-4, random_state=42).
Keep all 81 columns, including constants; no clipping, feature weighting or
nonlinear transformations. Fit a fresh complete Pipeline only on each fold's
training rows. Test scaler training-row means/counts and validation isolation.
Record convergence warnings and n_iter per fold/refit; stop for investigation
if nonconvergence occurs, without silently tuning and reusing the run ID.

Reuse exactly 2699 rows, 150 prompt groups, 18 LABELS, P0 five-fold mapping,
seed42, `.venv` and two BLAS/OpenMP threads, PYTHONHASHSEED=42. Dataset SHA256
`40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`;
fold SHA256 `3357fd3f290aaad4f1e50d9864db729e3803c2b6c62625748ff28408e94cb2bd`.
Preflight verifies P0/P2 artifact/data/fold hashes, recomputed scores and current
numerical runtime. No packages installed or upgraded for P3.

## Execution, evaluation and resource plan

Run exactly two sequential identical CPU experiments:
`p3_20261018_style_run1` and `p3_20261018_style_run2`. Run2 is a determinism
check, not independent validation. Each executes all five saved folds and a
final full-data fit for inference. Freeze tested source/config in a Conventional
Commit before both runs. Use new report/artifact directories and exclusive
writes; save the exact config, feature names/definitions, scaler statistics,
source/data/fold/config/model/OOF hashes, command and environment in manifests.

Save ordered OOF probabilities with id, prompt_id, true/predicted label, fold
and model ID. Save all fold/aggregate BA, 18 recalls, full confusion matrices,
top confusion partners, focus classes, Gemini pair and corrections/regressions.
Compare against P0 with the existing paired prompt-cluster bootstrap engine:
10000 draws, seed42, same sampled 150 prompt groups for both models, aggregate
TP/support before averaging 18 recalls, 95% percentile intervals. Count skipped
draws lacking any class. Intervals condition on fitted OOF, exclude retraining
variance and are exploratory without multiplicity correction.

Measure wall/CPU seconds per fold, CV, refit and serialization; sampled peak RSS
at 50ms, OS process-lifetime peak and serialized model bytes. CPU only, GPU not
used and VRAM unmeasured. Planning estimate: <5 minutes and <1 GiB RSS per run;
investigate if elapsed exceeds 10 minutes or RSS exceeds 4 GiB, preserving
partial outputs. These are budget estimates, not measured results.

Run existing tests before edits, new semantic extractor/scaler isolation/guard
tests and full suite before training. Verify repeats: equal BA/predictions/full
confusion, maximum probability difference <=1e-12 and serialized model hashes.
Smoke-test trusted saved-model inference and the existing local-path Kaggle
template on the three-row sample without scoring it. Package custom extractor
source for future offline inference; local smoke is not a Kaggle hidden run.

Update experiment register, this report, taskboard and dated changelog after
each run. Keep row-level OOF/folds and checkpoints private/ignored by Git.
Conclude with one P4 ModernBERT fold0 GPU benchmark recommendation, informed by
P3 outcomes and the existing P1 synthetic CUDA check. Verify official license/
architecture requirements; encoder throughput/VRAM remain unmeasured in P3.

## Implementation milestone — 2026-10-19

The initial plan above is immutable in reports/p3_20261018_plan.json and commit
ce8556b. Run IDs retain its 20261018 prefix; continuation uses the updated client
date. Preflight passed for all six P0/P2 runs; 109 protected existing files were
snapshotted. No P0/P2 source, configuration, report or artifact was modified.

Separate P3 extractor, guarded trainer and comparison entrypoints implement the
81-feature specification. Full pretraining suite: **144 passed in 20.22 s**;
see reports/p3_20261018_pretraining_tests.json. Both new CLI help entrypoints
pass. Tests include exact feature semantics, Unicode, empty/iterator inputs,
training-only scaler statistics, 18-label probabilities, serialization, full
synthetic grouped runs, output protection, provenance drift and nonconvergence.

Each run records scaler means/variances/counts, convergence iterations/warnings
and a deterministic importable source ZIP for future offline inference. Budget
checks occur after completed folds/refit; they do not interrupt an active fit
asynchronously. No competition-data P3 fit has occurred at this milestone.
