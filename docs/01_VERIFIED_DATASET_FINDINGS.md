# Data audit — verified vs hypothesized

**Basis:** Actual uploaded `name-that-llm-dataset.zip`, earlier EDA summary generated on 2026-10-16. EDA measurements are data observations, NOT model performance. For full re-measurement run `python scripts/audit_dataset.py`.

## Verified file schema and counts

| File | Rows | Columns |
|---|---:|---|
| train.csv | 2699 | id, prompt_id, text, model |
| test.csv (interactive) | 3 | id, text |
| sample_submission.parquet | example | id, model |

- Exactly **150 prompt_id**; **18 class labels**; **149 prompts contain 18 author responses**; **1 prompt has 17**; missing pair `(p002, claude-fable-5)`.
- 17 labels have 150 passages; `claude-fable-5` has 149.
- Missing field values: **0** in train; duplicate train IDs: **0**; duplicate exact text: **0**; duplicate (`prompt_id`, `model`): **0**; no train/test ID overlap in interactive rows.
- Regex-derived length: mean **115.29 words**, median **102**, min **36**, max **271**, 12 texts below 40 and 25 above 250 using this tokenizer. The organizer's tokenizer may differ.
- Mean within-prompt max-minus-min length **184.95 words**.
- Punctuation presence across train: em dash **38.98%**, semicolon **38.68%**, colon **52.13%**. The EDA regex did not find newlines/bullets/Markdown wrapper patterns.
- Example em-dash rates (fraction of passages with >=1): `claude-fable-5` **83.2%**, `glm-5.3` **82.0%**, `gpt-5.5` **13.3%**, `gpt-6-astra` **8.7%**, `gemini-3.6-flash` **0.0%**. These are correlations, not attribution rules.
- Sampled TF-IDF word similarity (previous EDA): mean **0.166** for same-prompt pairs and **0.018** for different prompts, under that specific sampling procedure. Do not treat as full pair enumeration.

## What these observations imply

1. Prompt-based cross-validation is obligatory to mimic unseen test prompts.
2. Raw punctuation is potentially informative; never strip Unicode dashes or collapse semantics into plain word tokens only.
3. Prompt grouping is powerful for supervised contrastive training and per-prompt diagnostic comparisons.
4. Mean passage length differences are not sufficient on their own; validate length-based signals on grouped OOF.
5. Learn fingerprints stable ACROSS prompts, not just topic vocabulary.

## Not yet verified / open questions

- Exact generation temperature / decoding setup per model.
- Whether hidden test has repeated prompt groups and whether test ordering is randomized.
- How much models from same provider are distinguishable under unseen prompts.
- Best CV strategy beyond GroupKFold; whether domain holdout extrapolates to hidden prompts.
- Whether inferred forum prompt types/domain labels are correct.
- Whether leaderboard 0.547 is participant's score, a top cutoff, or another metric.

## Data privacy

The project scaffold does not contain competition `train.csv`, `test.csv`, or model checkpoints. Keep downloaded files local/private; do not publish full paragraphs or competition datasets via independent public channels.

## 2026-10-18 — Local remeasurement

The user's local `data/raw/train.csv` and `test.csv` are now present. Byte hashes
match the original Downloads ZIP; a separate extraction was checked privately.
`reports/p0_20261018_dataset_audit.json` confirms 2699 rows, 150 prompts, all 18
labels, the sole missing `(p002, claude-fable-5)` pair, no nulls/duplicate IDs or
texts, and no sample train/test ID overlap. Word lengths (mean 115.29, range
36–271) and em-dash/semicolon/colon presence reproduce the earlier observations.
This is a new dated measurement; the historical findings above are preserved.
