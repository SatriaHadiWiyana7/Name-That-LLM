# Multi-stage research roadmap (Top 5 stretch / Top 10 aspiration)

## Phase 0 — Data and split integrity

- Verify schema, exact classes, all prompt coverage, duplicates, language mix, case/punctuation stats and train/test shape.
- Fix `GroupKFold(n_splits=5)` with `prompt_id` and deterministic ordering. Save row-index-to-fold map.
- Check train/validation group disjointness in unit tests.

## Phase 1 — Strong classical baselines (CPU)

- Word TF-IDF (1–2 grams) + regularized logistic regression / LinearSVC.
- Character TF-IDF (char or char_wb; 2–5 grams) + linear classifier.
- Joint word+char sparse vectorizers, inside each fold, preserving punctuation.
- Stylometric numeric signals: e.g. em dash, semicolon, colon, contractions, digits, Unicode variants, function words, word-length distribution, sentence length. Test per-family sensitivity.
- Run ablations: char-only, word-only, combined, punct-only, combined+stylometric; only keep consistent grouped-OOF improvements.

## Phase 2 — Encoder candidates (local GPU)

- `answerdotai/ModernBERT-base` (~149M) and `microsoft/deberta-v3-base` (~184M including embeddings) are **candidates**, not guaranteed winners.
- Preflight licenses and environment support; RTX 5060 Ti Blackwell CUDA compute capability sm_120 may require a recent PyTorch CUDA build.
- Start with single-fold mini-benchmark: seq length 256/384, BF16 if supported, bs 8–16, accumulation 2–4, epochs 3–6, learning rates 1e-5 to 3e-5.
- Record wall clock, peak GPU memory, metrics, loss curves and confusion before committing to 5 folds.
- Evaluate frozen pretrained embeddings as a cheap intermediate model.
- Keep best OOF-proven checkpoints; do not upload optimizer state for inference.

## Phase 3 — Feature and data studies

- Character/POS/function-word features, punctuation-preserving tokenizer, writing-style classification without semantic shortcuts.
- Supervised contrastive: positive=same author, different prompts; negative=different authors, preferably same prompt. Keep train/val prompt groups isolated.
- Hierarchical family -> model classifier versus 18-way direct; quantify error propagation.
- Public, license-compliant external data only after testing basic model. Domain-shift checks essential.
- Domain out-of-distribution validation (only as additional robustness metric; never replace 5-fold GroupKFold).

## Phase 4 — Ensemble and postprocessing

- Save per-model OOF class probabilities ordered by `LABELS`.
- Evaluate simple average first; then constrained weights with **nested CV** or independent held-out OOF folds, not optimizing/evaluating same OOF without safeguards.
- Check per-class recall improvements and balanced accuracy; do not target overall accuracy alone.
- Optional balanced-accuracy calibrated scores / class-dependent offsets, but train calibration separately to avoid optimistic estimates.
- No forced equal predicted class proportions: test mix explicitly unequal.

## Phase 5 — Production Kaggle notebook

- Serialize final weights + tokenizer + preprocessing + label mapping + manifest; freeze and checksum.
- Offline notebook loads Kaggle input artifacts, predicts each hidden row, writes parquet.
- Create minimal fallback classical CPU classifier when transformers not available or time/resources incompatible.
- Full offline rehearsal plus test assertion using local sample data.

## Decision gates

| Gate | Requirement |
|---|---|
| A | 5-fold baseline OOF exists and all prompts disjoint |
| B | Classical errors analyzed per label; features ablated |
| C | One encoder single-fold benchmark runs on RTX 5060 Ti sm_120 |
| D | Transformer beats baseline or complements errors OOF |
| E | Ensemble estimated using leakage-safe procedure |
| F | Kaggle notebook rerun passes all submission validity checks |
| G | Public LB is only one signal; favor stable OOF for final selection |

## Avoid guessing numeric target BA

0.547 is user-supplied context, not a validated target. Rank #5/#10 cutoff must come from actual leaderboard with date/split. When user supplies thresholds, calculate `gap5 = threshold5 - our_public_score`, `gap10 = threshold10 - our_public_score` and report as score points, explicitly label positive/negative. Never assume Top 5 has a fixed balanced accuracy.
