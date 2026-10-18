# Advanced analyses to request from Codex

## Per-class diagnostic report

Output table: true model, total OOF count, TP, recall, primary confusion target, second confusion target, delta vs previous iteration. Compute BA weighted equally by classes even though class sizes differ slightly.

## Punctuation fingerprint robustness

- Presence and count per 100 words: em dash vs hyphen/en dash, semicolon, colon, quote glyphs, apostrophes, punctuation bursts.
- Global and matched-prompt controls to separate topic effects.
- Train char-only vs char with punctuation masked, record BA delta and classwise effects.
- Compare predictions by length bins and novel prompt domains.

## Cross-family and intra-family analysis

Aggregate providers only for reporting; predictions must remain 18-way:
OpenAI: 5; Google: 4; Anthropic: 5; Qwen: 1; ZAI: 2; Moonshot: 1.
Confusions within the same family are likely harder; verify, do not assert a priori.

## Prompt-contrastive pairs

Positive samples: same model across different prompt IDs. Negative: different model for same prompt ID (hard negatives). Keep prompt groups disjoint across train/validation, including pair construction and mining. Avoid using validation embeddings to build negatives for training. Compare against CE baseline on OOF.

## Statistical testing

Prompt-level (cluster) bootstrap on held-out OOF predictions to estimate variability; class-stratification cannot substitute prompt clustering. Avoid interpreting single 0.005 BA gain as decisive without uncertainty/stability checks.

## LB overfitting warnings

If public BA improves while grouped OOF deteriorates, investigate distribution shift and noise. Public/private model mixes differ; do not choose final entry solely by public score.
