# Project brief: competition target and operating approach

## Goal

Compete in Kaggle `Name That LLM — Identify the Model Behind the Text` (https://www.kaggle.com/competitions/name-that-llm). Primary aim **Private LB Top 5**, fallback aspiration **Private LB Top 10**. Prizes as described by organizer, but **placement is not guaranteed**. As of 2026-10-17, user mentioned **0.547** as latest score update. Attribution (own entry vs leaderboard), source and rank remain unknown. Preserve as reported, do not backsolve a ranking.

## Task

Given a self-contained paragraph 40–250 words approximately, predict exactly one of 18 label strings. 2,699 train samples, 150 known training prompts, tiny interactive test, ~4k hidden test with different prompts, balanced accuracy averaged over 18 class recalls, hidden test distribution may be unequal.

## Training architecture

- Train/experiment LOCAL on RTX 5060 Ti 16GB, 32GB RAM, i5-12400F.
- Use grouped CV by prompt_id, save OOF probabilities and calibration/error analysis.
- Upload inference-only serialised checkpoints as private Kaggle dataset/model input.
- Notebook must load packages/weights offline and write `submission.parquet` on hidden rerun.
- Record exact working environment and training steps for winner obligations.

## Anti-goals

- No claim that 0.547 is model performance or cutoff unless clarified.
- No leaderboard or OOF speculation as if measured.
- No per-test manual labeling, reconstructed hidden labels, id-based hacks, or disallowed competition data sharing.
- No blind large-model training before showing value over classical baselines.

## Dates (organizer description; verify if changed)

- Start: 2026-10-14
- Entry / merger deadline: 2027-01-06
- Final submission: 2027-01-13 at 23:59 UTC
- Submission limit: 5/day; up to 2 final submissions.
