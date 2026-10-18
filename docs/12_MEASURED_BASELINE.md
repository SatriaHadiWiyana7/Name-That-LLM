# Measured initial baseline (NOT Kaggle leaderboard)

**Environment:** tool sandbox CPU, not user's RTX 5060 Ti. Python 3.13, sklearn 1.8.0, `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`. Measured on the original `train.csv` via 5-fold prompt-grouped validation. Not a neural model.

**OOF balanced accuracy = `0.318566`**. Training time for 5 OOF folds: `58.24` seconds, excluding final refit. Raw training passages, checkpoints, and OOF row-level data are intentionally excluded from the downloaded scaffold ZIP; rerun locally to recreate. `reports/baseline_metrics.json` retains class statistics and fold results.

## Fold metrics

| Fold | Val rows | Val prompts | BA | Time (sec) |
|---|---:|---:|---:|---:|
| 0 | 540 | 30 | 0.30000 | 11.87 |
| 1 | 540 | 30 | 0.30370 | 11.33 |
| 2 | 540 | 30 | 0.33148 | 11.94 |
| 3 | 540 | 30 | 0.33889 | 11.37 |
| 4 | 539 | 30 | 0.31871 | 11.7 |

## Bottom-eight class recalls

| Label | Recall |
|---|---:|
| `claude-fable-5` | 0.128 |
| `glm-5.3` | 0.140 |
| `gpt-5.6-sol` | 0.140 |
| `glm-5.3-flash` | 0.193 |
| `claude-opus-4-8` | 0.240 |
| `gpt-5.5` | 0.267 |
| `gpt-5.6-luna` | 0.273 |
| `kimi-k3` | 0.273 |

## Largest directional confusions (from measured OOF)

- `gemini-3.8-flash` => `gemini-3.7-flash`: 54 incorrect
- `gemini-3.7-flash` => `gemini-3.8-flash`: 43 incorrect
- `gpt-5.6-sol` => `gpt-5.6-luna`: 38 incorrect
- `gemini-3.7-flash` => `gemini-3.6-flash`: 36 incorrect
- `gemini-3.6-flash` => `gemini-3.7-flash`: 36 incorrect
- `glm-5.3` => `claude-opus-5`: 30 incorrect

## Next action for Codex

1. Verify baseline reproducibility on the user's local hardware with pinned environment.
2. Implement word-only, char-only and punctuation-ablation runs with the SAME folds.
3. Investigate hard Gemini/GPT family pairs; try same-family specialized models only if global BA improves.
4. Benchmark encoder single fold on actual RTX 5060 Ti; log VRAM/time.
5. Do not compare this OOF number to user-reported `0.547` as though both were comparable leaderboard scores.

**Parquet smoke-test limitation:** this sandbox lacked `pyarrow`; the CSV/OFF CV pipeline, OOF reports and joblib artifact were executed and checked, but parquet writing needs verification after installing `requirements-cpu.txt` on the user's machine.

## 2026-10-18 — Follow-up local result

The historical result above is preserved. Two local fixed-fold runs obtained
BA **0.316338553318**, with identical probabilities and model hashes. Historical
fold mapping/full environment are unavailable, so the difference of -0.0022271936
does not establish an identical-split regression or its cause. Local parquet and
template smoke tests now pass. See [the dated P0 report](13_LOCAL_P0_REPRODUCTION.md).
