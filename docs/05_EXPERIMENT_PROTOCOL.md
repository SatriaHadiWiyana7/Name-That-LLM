# Strict validation and experiment protocol

## Official objective

Balanced accuracy across `K=18` classes:

`BA = (1/K) * sum_k TP_k / (TP_k + FN_k)`

- Optimize **macro recall across the 18 labels**, not raw accuracy; test priors may differ from train.
- Probability thresholding/offsets should be cross-fitted, not scored on the data used for calibration.

## Split construction

- `GroupKFold(n_splits=5)` on train `prompt_id`.
- Each fold contains ~30 prompts; every model mostly once/prompt; rare `(p002, claude-fable-5)` missing.
- No prompt group overlap between train and validation.
- Fit TF-IDF, scaling, neural layers, style feature selection, calibration only within training portions.
- Reproducible row-to-fold CSV keyed by `id`, with group check and schema validation.

## Metrics and uncertainty

For each run save:
1. OOF BA, per-class recall (all 18), per-fold BA and confusion matrix.
2. Top 20 symmetric / directional confusion partners by count and normalized rate.
3. Performance by passage-length quartile and (if available) validated domain labels.
4. Runtime + memory + library versions and exact model revision.
5. A prompt-cluster bootstrap 95% interval for OOF BA for final candidates (resample entire prompt groups, report approximation caveat).
6. Grouped permutation/ablation for punctuation and length to test reliance on brittle features.

## Error analysis loop

1. Sort 18 labels by OOF recall ascending.
2. For each bottom-5 label, identify top-3 confusion targets and FP/FN asymmetry.
3. Compare against baseline classwise; prioritize folds and topics with systemic errors.
4. Choose a focused hypothesis and one model change; predict expected effect.
5. Run and log. Do not cherry-pick single fold.

## Leakage prevention

- Never vectorize whole corpus prior to CV (transductive TF-IDF needs explicitly separated experiment and rule review).
- Do not train on validation prompt labels, concatenate validation responses, or tune seeds on just one fold.
- Do not use hidden test texts for manual predictions or human labeling.
- Do not infer author from `id` strings.
- Do not infer prompt grouping on hidden test unless method is carefully validated on grouped train-only holdout without leakage.

## Comparing score changes

- BA delta = `new_OOF_BA - baseline_OOF_BA` on identical folds.
- Class recall delta = `new_recall(k) - baseline_recall(k)`.
- Leaderboard delta = `new_public_LB - previous_public_LB` only if same scoring stage.
- Score to Top N threshold = `TopN_cutoff - own_score`, only when both provided and from the same leaderboard/date.
- Always indicate **measured**, **user-reported**, or **estimate**.
