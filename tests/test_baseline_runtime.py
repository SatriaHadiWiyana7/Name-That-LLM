import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from name_that_llm.baseline import make_baseline, ordered_probabilities
from name_that_llm.io import sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.runtime import reference_comparison, validate_fold_mapping
from name_that_llm.validation import fold_assignments


@pytest.fixture
def synthetic_train():
    return pd.DataFrame([
        {
            "id": f"p{prompt}-{author}",
            "prompt_id": f"p{prompt}",
            "text": f"promptword{prompt} authorword{author} ; distinctive style {author}!",
            "model": label,
        }
        for prompt in range(4) for author, label in enumerate(LABELS)
    ])


@pytest.fixture
def tiny_config():
    return {
        "name": "synthetic_test_only", "random_seed": 42, "n_splits": 2,
        "word": {"ngram_range": [1, 2], "min_df": 1, "max_features": 1000},
        "char": {"analyzer": "char", "ngram_range": [3, 5], "min_df": 1, "max_features": 1500, "lowercase": False},
        "classifier": {"C": 3.0, "max_iter": 1200, "class_weight": "balanced"},
    }


def mapping_for(train):
    mapping = train[["id", "prompt_id"]].copy()
    mapping["fold"] = fold_assignments(train, n_splits=2)
    return mapping


def test_validation_vocabulary_is_unseen_and_model_roundtrip(synthetic_train, tiny_config, tmp_path):
    assigned = fold_assignments(synthetic_train, n_splits=2)
    train = synthetic_train.loc[assigned != 0]
    validation = synthetic_train.loc[assigned == 0]
    model = make_baseline(tiny_config)
    model.fit(train.text.tolist(), train.model)
    word_vectorizer = dict(model.named_steps["features"].transformer_list)["word"]
    for prompt in validation.prompt_id.unique():
        assert f"promptword{prompt[1:]}" not in word_vectorizer.vocabulary_
    probabilities = ordered_probabilities(model, validation.text.tolist())
    assert probabilities.shape == (len(validation), 18)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.named_steps["clf"].random_state == 42
    output = tmp_path / "synthetic.joblib"
    joblib.dump(model, output)
    restored = joblib.load(output)
    np.testing.assert_array_equal(probabilities, ordered_probabilities(restored, validation.text.tolist()))
    np.testing.assert_array_equal(model.predict(validation.text), restored.predict(validation.text))


def test_probability_columns_follow_fixed_labels_when_classifier_order_differs():
    class ReversedClassifier:
        classes_ = LABELS[::-1]

        def predict_proba(self, texts):
            weights = np.arange(1, 19, dtype=float)
            return np.tile(weights / weights.sum(), (len(texts), 1))

    expected = np.arange(18, 0, -1, dtype=float)
    np.testing.assert_allclose(ordered_probabilities(ReversedClassifier(), ["synthetic"])[0], expected / expected.sum())


def test_saved_mapping_aligns_by_id(synthetic_train):
    mapping = mapping_for(synthetic_train)
    shuffled = mapping.sample(frac=1, random_state=23)
    np.testing.assert_array_equal(validate_fold_mapping(synthetic_train, shuffled, 2), mapping.fold)


@pytest.mark.parametrize("fault", ["missing_id", "duplicate_id", "foreign_id", "wrong_prompt", "prompt_leakage", "fractional_fold", "missing_fold", "null_fold"])
def test_saved_mapping_rejects_invalid_assignments(synthetic_train, fault):
    mapping = mapping_for(synthetic_train)
    if fault == "missing_id":
        mapping = mapping.iloc[:-1]
    elif fault == "duplicate_id":
        mapping.loc[0, "id"] = mapping.loc[1, "id"]
    elif fault == "foreign_id":
        mapping.loc[0, "id"] = "foreign"
    elif fault == "wrong_prompt":
        mapping.loc[0, "prompt_id"] = "unknown"
    elif fault == "prompt_leakage":
        mapping.loc[0, "fold"] = 1 - mapping.loc[0, "fold"]
    elif fault == "fractional_fold":
        mapping["fold"] = mapping.fold.astype(float)
        mapping.loc[0, "fold"] = 0.5
    elif fault == "missing_fold":
        mapping["fold"] = 0
    elif fault == "null_fold":
        mapping["fold"] = mapping.fold.astype(float)
        mapping.loc[0, "fold"] = np.nan
    with pytest.raises(ValueError):
        validate_fold_mapping(synthetic_train, mapping, 2)


def test_delta_requires_identical_dataset_folds_and_seed():
    current = {
        "balanced_accuracy": 0.4,
        "dataset_sha256": "data", "fold_assignments_sha256": "folds", "random_seed": 42,
        "folds": [{"fold": 0, "balanced_accuracy": 0.4}],
        "class_recall": {label: 0.4 for label in LABELS},
    }
    historical = {"balanced_accuracy": 0.3}
    result = reference_comparison(current, historical)
    assert result["numerical_difference"] == pytest.approx(0.1)
    assert result["delta_balanced_accuracy"] is None
    assert not result["identical_dataset_folds_seed_verified"]
    comparable = dict(current, balanced_accuracy=0.3)
    result = reference_comparison(current, comparable)
    assert result["delta_balanced_accuracy"] == pytest.approx(0.1)
    assert result["identical_dataset_folds_seed_verified"]
    for key in ("dataset_sha256", "fold_assignments_sha256", "random_seed"):
        mismatch = dict(comparable, **{key: "different"})
        assert reference_comparison(current, mismatch)["delta_balanced_accuracy"] is None


def test_cli_saves_auditable_outputs_and_refuses_overwrite(synthetic_train, tiny_config, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    train_path = tmp_path / "train.csv"
    config_path = tmp_path / "config.json"
    folds_path = tmp_path / "fixed_folds.csv"
    reference_path = tmp_path / "historical.json"
    reports = tmp_path / "reports"
    artifacts = tmp_path / "artifacts"
    register = tmp_path / "experiments.csv"
    synthetic_train.to_csv(train_path, index=False)
    mapping_for(synthetic_train).to_csv(folds_path, index=False)
    config_path.write_text(json.dumps(tiny_config), encoding="utf-8")
    reference_path.write_text('{"balanced_accuracy": 0.3}', encoding="utf-8")
    command = [
        sys.executable, str(repo / "scripts/train_baseline.py"),
        "--train", str(train_path), "--config", str(config_path),
        "--out", str(artifacts), "--reports", str(reports),
        "--run-id", "synthetic_cli_only", "--folds", str(folds_path),
        "--register", str(register), "--reference-metrics", str(reference_path),
    ]
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((reports / "baseline_metrics.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(reports / "baseline_oof.csv")
    assert manifest["labels"] == LABELS
    assert [column for column in oof if column.startswith("p_")] == ["p_" + label for label in LABELS]
    assert set(oof.model_id) == {"synthetic_cli_only"}
    assert manifest["dataset_sha256"] == sha256_file(train_path)
    assert manifest["fold_assignments_sha256"] == sha256_file(folds_path)
    assert manifest["model_sha256"] == sha256_file(artifacts / "model.joblib")
    assert manifest["resources"]["peak_rss_sampled_bytes"] > 0
    assert manifest["resources"]["cpu_seconds"] >= 0
    assert manifest["resources"]["peak_vram_bytes"] is None
    assert manifest["resources"]["gpu_used"] is False
    assert manifest["environment"]["argv"] == command
    for fold in metrics["folds"]:
        assert len(fold["class_recall"]) == 18
        assert np.asarray(fold["confusion_matrix"]).shape == (18, 18)
    with register.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["delta_vs_baseline"] == ""
    before_hash = sha256_file(artifacts / "model.joblib")
    repeated = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0
    assert "Refusing to overwrite" in repeated.stderr
    assert sha256_file(artifacts / "model.joblib") == before_hash
