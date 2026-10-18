# Codex task board — ordered tasks and Definition of Done

## P0 — Reproduce existing scaffold baseline

- [x] Inspect dataset, validate legal labels, prompt distribution. Local audit 2026-10-18 matches original ZIP.
- [x] Create local environment; `pytest -q` passes. Separate CPU `.venv`; 31 tests pass.
- [x] `prepare_data.py` extracts user's original competition zip privately. Separate verification extraction preserves raw originals.
- [x] `audit_dataset.py` shows 2699 rows, 18 labels, 150 prompts and prompt `p002` issue.
- [x] `train_baseline.py` runs full 5-fold grouped OOF and saves full-data pipeline/manifest. Both local runs BA 0.316338553318; exact probability/model-hash repeat verified.
- [x] `analyze_oof.py` prints bottom-5 per-class recall and top confusion pairs. JSON includes full matrices/recalls and prompt bootstrap.
- [x] Kaggle sample parquet smoke test passes. Saved-model CLI and actual template (local paths) agree; hidden Kaggle rerun not performed.
- [x] Log actual baseline score and command in `reports/experiments.csv`.

**Done when:** no errors; all folds disjoint; schema tests pass; run is reproducible.

**P0 complete 2026-10-18:** see docs/13, reports/p0_20261018_reproduction_check.json,
both versioned run directories, and artifact manifests. Historical baseline is
preserved; local BA differs from it by -0.0022271936 with historical split
equivalence unverified. Local repeat delta is exactly zero.

## P1 — GPU readiness (RTX 5060 Ti 16GB)

- [x] `nvidia-smi` driver and WSL status known. Initial driver 610.74; later benchmark 610.88. WSL2 distributions stopped during initial audit.
- [x] Python/PyTorch CUDA build supports `sm_120`, backward pass tested. Existing gpu_test / torch 2.9.1+cu128; 100 finite forward/backward steps.
- [x] Record real GPU mem/speed, library versions, reproducibility seed. See reports/p1_20261018_gpu_benchmark.json; synthetic kernel timing, not encoder throughput.
- [ ] Confirm Kaggle-compatible pretrained encoder license.

**Done when:** 100 forward/backward steps run on local GPU with measured peak VRAM/time.

Hardware gate passed 2026-10-18: 100 steps in 0.0734684s after 10 warmup steps,
1024x1024 float32, seed 42, peak allocated 92277760 bytes. Pretrained license and
encoder-specific memory/tokenizer/offload preflight remain open; no transformer trained.

## P2 — Compare stronger classical models

**Controlled feature ablation complete, 2026-10-18.** See docs/14 and
reports/p2_20261018_summary. Exact P0 dataset/folds/environment preserved;
90 tests passed before training. Word-only BA 0.273733532190 (delta -0.042605021129),
char-only BA 0.313375590355 (delta -0.002962962963), reference P0 BA 0.316338553318.
Both exact repeats have identical probabilities, confusion and model hashes.
Word delta CI is negative; character delta CI includes zero. Retain combined P0.

- [x] Compare word-only, char-only and combined on P0 saved folds.
- [x] Save OOF probability matrices/classes, full confusion reports, configs/manifests/hashes and resources.
- [x] Paired prompt-cluster bootstrap, all fold/class deltas, focus-class and Gemini-pair error analysis.
- [x] Verify one exact repeat per ablation plus saved-model/template inference smoke tests.
- [x] Assess stylometry-only: recommended as one separate next experiment; all four specified weak classes remain below 0.30 in P0.
- [ ] Execute separately preregistered stylometry-only CV (not run in this phase).
- [ ] Ablate em dash/punctuation and word-length features.

**Done when:** evidence-based selection and per-class BA deltas are recorded.

This requested Word/Character study meets that definition. The two remaining
feature studies above are future scope. No transformer training performed.

## P3 — Encoder CV

- [ ] ModernBERT-base one-fold benchmark; compare with frozen embeddings.
- [ ] DeBERTa-v3-base if resource/validation suggests.
- [ ] 5-fold full after single-fold gate. Save OOF.

**Done when:** measured OOF BA and ablations, checkpoint budgets, working inference loader.

## P4 — Specialization & ensemble

- [ ] 18-class confusion analysis; same-family hard cases.
- [ ] Prompt-contrastive/hierarchical trial if value supported.
- [ ] Ensemble OOF in leakage-safe nested/cross-fitting selection.

**Done when:** improvements are stable, not only one leaderboard bump.

## P5 — Competition notebook

- [ ] Upload trained artifact(s) private to Kaggle.
- [ ] Notebook infer on variable-length test, internet OFF, <= 9h.
- [ ] Generate valid submission.parquet and register Kaggle score.
- [ ] Full reproducible winner-ready repo and license scan.

**Done when:** submission notebook rerun accepted; leaderboard logged with honest provenance.

## Codex first prompt (copy/paste)

> Read AGENTS.md and all docs 00–08. Validate the scaffold on my local machine without editing the original competition dataset. Check CPU baseline tests first, then inspect the last experiment and score log. Execute P0 tasks in order, report exact OOF balanced accuracy per class with GroupKFold(prompt_id), save artifacts, update taskboard and change log. Never claim Top5/Top10 or compare 0.547 until its leaderboard meaning is clarified. Do not start large transformer training without my approval and real VRAM/time benchmark.
