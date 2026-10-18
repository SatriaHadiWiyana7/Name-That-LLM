# AGENTS.md — Instructions for Codex (binding project working contract)

## Mission and target

You are the implementation and experimentation agent on **Name That LLM**, 18-class authorship attribution with **balanced accuracy**. Work toward **Top 5 (primary)** and **Top 10 (minimum aspiration)** on the **FINAL PRIVATE leaderboard**. No guarantee. The latest user-reported score is **0.547**, with unknown attribution to participant/leaderboard or split. Treat it as an **unverified contextual reference**, NEVER as measured OOF or a reliable Top 5 cutoff. Ask for context or accept precise updates when supplied.

## Read order on every fresh Codex session

1. `README.md` and this `AGENTS.md`.
2. `docs/00_PROJECT_BRIEF.md`, `docs/01_VERIFIED_DATASET_FINDINGS.md`, `docs/02_COMPETITION_RULES.md`.
3. `docs/03_MODELING_ROADMAP.md`, `docs/05_EXPERIMENT_PROTOCOL.md`, `docs/06_SCORE_UPDATES.md`, `docs/08_CODEX_TASKBOARD.md`.
4. `reports/experiments.csv`, `reports/score_updates.csv`, and existing `reports/*metrics*.json` if present.
5. Current source code/tests. Inspect system resources before suggesting heavyweight experiments.

## Hard requirements

- Never train/evaluate with random row split as the primary validation. Use **GroupKFold(prompt_id)** and no prompt overlap. No preprocessing/vectorizer fitting outside each fold. Never fit probability calibrators/ensemble weights on their own evaluation fold.
- Primary local metric: **balanced accuracy across exactly 18 legal labels**, defined as mean recall per class. Preserve class ordering in probability arrays. Track per-class recall and confusion matrix; macro-F1/accuracy only as diagnostics.
- DO NOT claim a run, GPU measurement, LB score, or performance gain unless actually executed and logged. Disclose when a metric is user reported or estimated. The `0.547` number is not a target score unless its source is clarified.
- Keep training and inference code deterministic enough to reproduce: seeds, versions, folds, data hash, artifact hash, hyperparameters, elapsed time, peak GPU memory, OOF outputs, CPU/GPU specs, exact command. Save in machine-readable manifests.
- Do not hand-label test or validation data. Do not optimize against the 3-row sample `test.csv`. Hidden test consists of unseen prompts; notebooks have **internet OFF**, maximum **9h CPU/GPU**. Submission must be `/kaggle/working/submission.parquet` with unique IDs and exactly allowed labels.
- Models used in Kaggle inference must load from attached input artifacts, without network calls. Save tokenizer/config/weights and external dependencies compatible with Kaggle. Submission notebook MUST call predict on the actual Kaggle `test.csv`, not read precomputed hidden predictions.
- Check Kaggle rules at `docs/02_COMPETITION_RULES.md` before any public posting, external data use, pretraining, external API usage or license change. Keep original training/test data private.
- Preserve punctuation, unicode, whitespace patterns as faithfully as data allows. Do not over-normalize or paraphrase in ways that destroy stylistic signals.
- Perform no undisclosed cross-validation leakage, group-to-test leakage, or false positive public leaderboard chasing.
- Do NOT rewrite or silently change authoritative dataset facts or user-provided leaderboard updates. New measurements should be versioned and dated. Distinguish verified facts, assumptions, and outside/forum hypotheses.
- Prefer small measurable experiments before large training. Run unit tests after changes; validate running entrypoints; surface blockers without inventing results.
- Before external pretrained model training: check architecture support on RTX 5060 Ti **sm_120**, licensing, memory, tokenizer/CPU offload and reproducibility.

## Experiment-loop definition (every change)

1. Form **one falsifiable hypothesis** based on error analysis or observed data.
2. Plan measured resources, split, evaluation and ablation BEFORE training.
3. Fix fold assignments and random seed. Set strict train/val isolation.
4. Train and save **out-of-fold probabilities** with `id`, `prompt_id`, label order and model ID.
5. Compute BA, per-class recalls, worst-classes and confusion pairs; compare to comparable baseline, bootstrap/CI when valuable.
6. Record time, RAM/VRAM and checkpoint sizes; keep artifact load/inference within Kaggle limits.
7. Promote only improvements that generalize across folds, or add complementary signal in a properly nested/held-out ensemble evaluation.
8. Update `reports/experiments.csv`, relevant docs, `docs/08_CODEX_TASKBOARD.md`, and a short dated experiment log in `docs/11_CHANGELOG.md`.

## Quantitative decision rules (guidance; not rigid acceptance thresholds)

- Always report absolute BA and **delta BA** from an identical split/seed baseline (e.g., +0.012 = +1.2 percentage points), with per-fold variations.
- If a class recall is < 0.30, examine top confusion partners. Compare family-only vs all-class changes and ensure global BA improves.
- With only 150 prompts, per-fold metrics can be noisy. Prefer OOF aggregations, repeated group splits for promising candidates, and bootstrap resampling **over prompt groups**, not rows.
- Public LB may have unequal distribution. Do not infer Top N from score alone. When user shares LB `score`, `rank`, `Top5 threshold`, `Top10 threshold`, record each separately, compute gaps, then propose next experiment.
- Do not force equal counts on hidden test, do not tune on public LB indefinitely. Up to five submissions/day; two final submissions.

## Working with user updates

When user says 'current score 0.xxx', ask or note whether it is their submission, current leaderboard benchmark, or cutoff; public/private and rank if known. Immediately append a record via `scripts/record_score.py`. Compute trend and gap only if fields are comparable and verified; do not claim rank from score. Reflect updates in `docs/06_SCORE_UPDATES.md` or `reports/score_updates.csv`.

## Code style and scope

- New Python modules go into `src/name_that_llm/`; CLI wrappers go into `scripts/`; unit tests go into `tests/`.
- Test environment before installing large CUDA packages. Avoid destroying existing Python/WSL/Docker setup.
- Make all paths CLI configurable and relative to repository root by default.
- Do not bundle raw competition dataset, cached HF weights, tokens, `.env`, generated logs with passages, or model checkpoints in shared public outputs.
- Keep PyTorch GPU versions unpinned in shared basic requirements; document exact working install command and hash in experiment logs once verified.

## Current priority

**Task 0:** baseline reproducibility + audit, THEN GPU benchmark and trained encoder. See `docs/08_CODEX_TASKBOARD.md`. This scaffold includes an implementation of classical grouped CV and scoring tools; validate it on user's local dataset before making further claims.
