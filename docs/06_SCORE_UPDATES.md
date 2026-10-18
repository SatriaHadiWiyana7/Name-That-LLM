# Score updates, gaps, experiment feedback loop

**Initial user report (2026-10-17): `0.547`.** Interpretation currently unresolved; `reports/score_updates.csv` captures it with `split=unknown` and `source=user_reported`.

## When user sends a score update, Codex must

1. Record exact score and **who/what it belongs to** (our submission, best leaderboard score, or Top5/Top10 cutoff), split (public/private), model/experiment ID, timestamp and optional rank.
2. Use `python scripts/record_score.py --score ... --split public --source user_reported --model-id ...` for our submission; for a leaderboard cutoff use `--source leaderboard_cutoff --cutoff-kind top5` etc.
3. Calculate changes only between comparable rows (same subject + split). Distinguish OOF from LB.
4. If current own public score and Top5/Top10 public cutoffs available, compute gaps and estimated BA improvement needed (not guaranteed because leaderboard may shift).
5. Tie leaderboard improvement back to OOF error analysis, not only score, to avoid overfitting.
6. Write 1–3 next experiments with estimated cost, hypothesis, and explicit acceptance criteria.
7. Update `docs/11_CHANGELOG.md` and experiment register on promotion.

## Minimal example

```bash
# After user's own PUBLIC Kaggle submission:
python scripts/record_score.py --score 0.56 --split public --source own_submission --model-id ensemble_v3 --rank 42 --notes "submitted notebook version 3"

# Only when verified: current Top-5 cutoff (not our own score)
python scripts/record_score.py --score 0.71 --split public --source leaderboard_cutoff --cutoff-kind top5 --notes "copied from public LB, YYYY-MM-DD"

python scripts/record_score.py --report
```

Numbers in these example commands are fictional; do not confuse them with competition results.

## Rank target is not a BA threshold

| Goal | Objective | Current status |
|---|---|---|
| Primary | Final private rank <= 5 | Not established |
| Minimum aspiration | Final private rank <= 10 | Not established |
| Score reference | 0.547 | Reported by user; source unknown |

**Never state**: "Score 0.547 is Top 10" without verified leaderboard rank.
