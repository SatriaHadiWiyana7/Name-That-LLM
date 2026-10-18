# Compliance notes — verify against official live rules before final submission

**Primary source:** User-supplied official rules snapshot (`Pasted markdown(2).md`) plus user's competition description. Current canonical rules: https://www.kaggle.com/competitions/name-that-llm/rules . This is a working checklist, not legal advice. If official live rules conflict, the official rules govern.

## Confirmed from supplied rules

- Maximum **5 submissions per day**; up to **2 final submissions**.
- Up to **5 team members**, team merges under deadline/limit rules.
- Competition data may be used to enter subject to safeguards. Do not disseminate competition data to non-participants or publish/re-distribute it contrary to rules.
- External data/models are permitted subject to being reasonably accessible, low-cost (Reasonableness Standard), and appropriate license/permissions.
- Winner must deliver executable **training and inference code**, dependencies/resources and documentation, plus enough detail to reproduce. Winner's code licensing and pretrained/external-model licensing terms require a careful review.
- No private sharing of competition code outside official teams. Public sharing, if chosen, must use permitted Kaggle competition discussion/notebook mechanisms and satisfy licensing conditions.
- No manual labeling/human prediction of validation/test data records.
- Private leaderboard determines final placement, not public leaderboard.

## Competition submission mechanics (description supplied by user)

- Notebook-based code competition, not ordinary CSV upload.
- Local/offline model training is **not expressly prohibited** in reviewed rules; training locally and attaching weights for Kaggle notebook inference is our documented interpretation. Ask organizers for clarification if concerned about any specific pipeline/data source.
- At inference, Kaggle notebook has **internet disabled** and **<=9h CPU or GPU** runtime.
- Test data available from `/kaggle/input/competitions/name-that-llm/test.csv` in evaluation notebook (~4k hidden rows).
- Output to notebook working directory named **`submission.parquet`**, unique every test ID, no null labels, legal exact model strings only.
- Interactive `test.csv` contains examples, not final hidden rerun.

## Release checklist

- [ ] Check exact model/tokenizer checkpoint licenses and code dependencies.
- [ ] Check all external data provenance, public accessibility and costs.
- [ ] Ensure no test hard-coding and no hidden-label manual guesses.
- [ ] Verify Kaggle notebook has no downloads/network calls.
- [ ] Verify generated `submission.parquet` against test IDs.
- [ ] Confirm artifact/private input permissions are correct for notebook rerun.
- [ ] Keep offline reproducibility artifacts and environment manifest.
- [ ] Verify current official rules and notebook submission page before final.
