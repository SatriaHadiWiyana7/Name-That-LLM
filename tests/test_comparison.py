import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from name_that_llm.comparison import (
    align_paired_oof, compare_paired_oof, load_verified_run,
    paired_prompt_cluster_bootstrap, verify_comparable_manifests,
)
from name_that_llm.io import sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.metrics import classification_report


def make_oof():
    rows = []
    for prompt in range(2):
        for index, label in enumerate(LABELS):
            prediction = label if prompt == 0 else LABELS[(index + 1) % len(LABELS)]
            row = {"id": f"{prompt}_{index}", "prompt_id": f"p{prompt}", "fold": prompt,
                   "true_model": label, "pred_model": prediction, "model_id": "synthetic"}
            row.update({"p_" + target: float(target == prediction) for target in LABELS})
            rows.append(row)
    return pd.DataFrame(rows)


def change_predictions(oof, predictions):
    changed = oof.copy()
    changed["pred_model"] = predictions
    for label in LABELS:
        changed["p_" + label] = (changed.pred_model == label).astype(float)
    return changed


def test_pair_alignment_is_order_invariant_and_self_comparison_has_zero_uncertainty():
    baseline = make_oof()
    shuffled = baseline.sample(frac=1, random_state=12)
    actual = compare_paired_oof(baseline, shuffled, 40, 42)
    expected = compare_paired_oof(baseline, baseline, 40, 42)
    assert actual == expected
    delta = actual["paired_prompt_cluster_bootstrap"]["delta_balanced_accuracy"]
    assert list(delta.values()) == [0.0, 0.0, 0.0, 0.0]
    assert np.asarray(actual["confusion_matrix_delta"]).sum() == 0
    assert actual["error_flips"]["overall"]["predictions_changed"] == 0
    assert all(item["delta_balanced_accuracy"] == 0 for item in actual["folds"])


@pytest.mark.parametrize("problem", ["ids", "prompt_id", "true_model", "fold", "no_probabilities", "model_id"])
def test_pairing_rejects_noncomparable_units(problem):
    baseline = make_oof()
    candidate = baseline.copy()
    if problem == "ids":
        candidate.loc[0, "id"] = "foreign"
    elif problem == "prompt_id":
        candidate.loc[candidate.prompt_id == "p0", "prompt_id"] = "p9"
    elif problem == "true_model":
        candidate.loc[0, "true_model"] = LABELS[1]
    elif problem == "fold":
        candidate["fold"] = 1 - candidate.fold
    elif problem == "no_probabilities":
        candidate = candidate[[column for column in candidate if not column.startswith("p_")]]
    else:
        candidate.loc[0, "model_id"] = "mixed_run"
    with pytest.raises(ValueError):
        align_paired_oof(baseline, candidate)


def test_paired_bootstrap_pools_unequal_support_and_uses_the_same_draws():
    baseline = make_oof()
    duplicate = baseline.iloc[[0]].copy()
    duplicate["id"] = "extra"
    baseline = pd.concat([baseline, duplicate], ignore_index=True)
    predictions = [LABELS[(LABELS.index(row.true_model) + 1) % 18] if row.prompt_id == "p0" else row.true_model
                   for row in baseline.itertuples()]
    candidate = change_predictions(baseline, predictions)
    repeats, seed = 50, 19
    result = paired_prompt_cluster_bootstrap(baseline, candidate, repeats, seed)
    rng = np.random.default_rng(seed)
    first, second, class_delta = [], [], []
    for _ in range(repeats):
        good = int((rng.integers(0, 2, size=2) == 0).sum())
        bad = 2 - good
        first_recall = 2 * good / (2 * good + bad)
        second_recall = bad / (2 * good + bad)
        first.append((first_recall + 17 * good / 2) / 18)
        second.append((second_recall + 17 * bad / 2) / 18)
        class_delta.append(second_recall - first_recall)
    expected = np.asarray(second) - np.asarray(first)
    assert result["valid_repeats"] == repeats
    for name, values in (("delta_balanced_accuracy", expected), ("baseline_balanced_accuracy", first),
                         ("candidate_balanced_accuracy", second)):
        assert result[name]["bootstrap_mean"] == pytest.approx(np.mean(values))
        assert result[name]["ci_lower"] == pytest.approx(np.quantile(values, 0.025))
        assert result[name]["ci_upper"] == pytest.approx(np.quantile(values, 0.975))
    assert result["class_recall_delta"][LABELS[0]]["bootstrap_mean"] == pytest.approx(np.mean(class_delta))
    assert result == paired_prompt_cluster_bootstrap(baseline, candidate, repeats, seed)


def test_bootstrap_missing_class_draws_are_counted_not_scored_as_fewer_classes():
    baseline = make_oof()
    baseline = baseline.loc[~((baseline.prompt_id == "p1") & (baseline.true_model == LABELS[0]))]
    repeats, seed = 100, 7
    result = paired_prompt_cluster_bootstrap(baseline, baseline, repeats, seed)
    rng = np.random.default_rng(seed)
    skipped = sum(bool((rng.integers(0, 2, size=2) == 1).all()) for _ in range(repeats))
    assert skipped > 0
    assert result["skipped_missing_class_repeats"] == skipped
    assert result["valid_repeats"] == repeats - skipped
    assert result["delta_balanced_accuracy"]["ci_lower"] == 0


def test_error_flip_accounting_and_gemini_pair_keep_errors_to_other_labels():
    baseline = make_oof()
    candidate = change_predictions(baseline, baseline.true_model)
    pair = ["gemini-3.8-flash", "gemini-3.7-flash"]
    baseline.loc[baseline.true_model.isin(pair), "pred_model"] = "claude-fable-5"
    baseline = change_predictions(baseline, baseline.pred_model)
    # Regress a different class so corrected-minus-regressed differs from corrections.
    candidate.loc[0, "pred_model"] = LABELS[1]
    candidate = change_predictions(candidate, candidate.pred_model)
    result = compare_paired_oof(baseline, candidate, 20, 42)
    counts = result["error_flips"]["overall"]
    assert sum(counts[key] for key in ("both_correct", "corrected", "regressed", "both_wrong_same_prediction", "both_wrong_different_prediction")) == len(baseline)
    weighted_delta = np.mean([(item["corrected"] - item["regressed"]) / item["rows"]
                              for item in result["error_flips"]["per_class"].values()])
    assert result["delta_balanced_accuracy"] == pytest.approx(weighted_delta)
    for row in result["gemini_pair"]["baseline"]["rows"]:
        assert row["confused_with_pair"] == row["correct"] == 0
        assert row["errors_to_other_labels"] == row["support"] == 2
    for row in result["gemini_pair"]["candidate"]["rows"]:
        assert row["recall"] == 1
    assert (np.asarray(result["baseline"]["confusion_matrix"]) + np.asarray(result["confusion_matrix_delta"])).tolist() == result["candidate"]["confusion_matrix"]
    # Aggregate report exposes no passages or evaluation-unit identifiers.
    def check_keys(value):
        if isinstance(value, dict):
            assert not {"id", "text"}.intersection(value)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)
    check_keys(result)


@pytest.fixture
def manifest_fixture(tmp_path):
    oof = make_oof()
    train = oof[["id", "prompt_id", "true_model"]].rename(columns={"true_model": "model"})
    train["text"] = "synthetic only"
    paths = {name: tmp_path / name for name in ("dataset.csv", "config.json", "oof.csv", "folds.csv", "model.bin", "metrics.json", "manifest.json")}
    train.to_csv(paths["dataset.csv"], index=False)
    config = {"name": "synthetic", "random_seed": 42, "n_splits": 2,
              "word": {"min_df": 2}, "char": {"min_df": 2}, "classifier": {"C": 3.0}}
    paths["config.json"].write_text(json.dumps(config), encoding="utf-8")
    oof.to_csv(paths["oof.csv"], index=False)
    oof[["id", "prompt_id", "fold"]].to_csv(paths["folds.csv"], index=False)
    paths["model.bin"].write_bytes(b"synthetic artifact; never loaded")
    metrics = classification_report(oof.true_model, oof.pred_model)
    metrics["folds"] = [{"fold": int(fold), **classification_report(group.true_model, group.pred_model)}
                        for fold, group in oof.groupby("fold")]
    paths["metrics.json"].write_text(json.dumps(metrics), encoding="utf-8")
    artifacts = {name: {"path": str(paths[file]), "sha256": sha256_file(paths[file]), "bytes": paths[file].stat().st_size}
                 for name, file in (("metrics", "metrics.json"), ("oof", "oof.csv"), ("folds", "folds.csv"), ("model", "model.bin"))}
    manifest = {
        "model_id": "synthetic", "labels": LABELS, "probability_columns": ["p_" + label for label in LABELS],
        "dataset_path": str(paths["dataset.csv"]), "dataset_sha256": sha256_file(paths["dataset.csv"]),
        "config": config, "config_path": str(paths["config.json"]), "config_sha256": sha256_file(paths["config.json"]),
        "random_seed": 42, "classifier_random_state": 42, "fold_assignments_sha256": artifacts["folds"]["sha256"],
        "model_sha256": artifacts["model"]["sha256"], "artifacts": artifacts,
        "oof_balanced_accuracy": metrics["balanced_accuracy"],
        "environment": {"python": "synthetic", "dependencies": {"numpy": "synthetic"},
                        "thread_environment": {"OMP_NUM_THREADS": "2"},
                        "threadpools": [{"user_api": "blas", "version": "1", "architecture": "test", "num_threads": 2, "filepath": "a"}],
                        "cpu": {"processor": "synthetic"}, "system_ram_bytes": 1024, "platform": "synthetic"},
    }
    paths["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths, manifest, oof


def test_saved_manifest_verification_checks_actual_bytes_and_recomputed_metrics(manifest_fixture):
    paths, manifest, oof = manifest_fixture
    loaded, frame = load_verified_run(paths["manifest.json"])
    assert loaded == manifest
    pd.testing.assert_frame_equal(frame, oof)
    metrics = json.loads(paths["metrics.json"].read_text())
    metrics["balanced_accuracy"] = 0.99
    paths["metrics.json"].write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size"):
        load_verified_run(paths["manifest.json"])
    artifact = manifest["artifacts"]["metrics"]
    artifact.update(sha256=sha256_file(paths["metrics.json"]), bytes=paths["metrics.json"].stat().st_size)
    paths["manifest.json"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate balanced_accuracy"):
        load_verified_run(paths["manifest.json"])


@pytest.mark.parametrize("field", ["dataset_sha256", "fold_assignments_sha256", "random_seed", "labels", "python", "dependencies", "thread_environment", "threadpools", "classifier"])
def test_comparison_refuses_provenance_or_uncontrolled_config_changes(manifest_fixture, field):
    _, baseline, _ = manifest_fixture
    candidate = copy.deepcopy(baseline)
    if field in ("python", "dependencies", "thread_environment", "threadpools"):
        if field == "threadpools":
            candidate["environment"][field][0]["num_threads"] = 8
        elif field == "dependencies":
            candidate["environment"][field]["numpy"] = "mismatch"
        else:
            candidate["environment"][field] = "mismatch"
    elif field == "classifier":
        candidate["config"][field]["C"] = 7
    else:
        candidate[field] = "mismatch"
    with pytest.raises(ValueError):
        verify_comparable_manifests(baseline, candidate)


def test_comparison_allows_feature_deletion_and_relocated_library_paths(manifest_fixture):
    _, baseline, _ = manifest_fixture
    candidate = copy.deepcopy(baseline)
    candidate["config"].update(name="word_only", feature_groups=["word"])
    candidate["environment"]["threadpools"][0]["filepath"] = "relocated"
    assert verify_comparable_manifests(baseline, candidate)["config_diff_only_name_feature_groups"]


def test_comparison_cli_produces_hashed_report_and_refuses_overwrite(manifest_fixture, tmp_path):
    paths, _, _ = manifest_fixture
    output = tmp_path / "comparison.json"
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / "scripts/compare_oof.py"),
               "--baseline-manifest", str(paths["manifest.json"]), "--candidate-manifest", str(paths["manifest.json"]),
               "--bootstrap-repeats", "20", "--output", str(output)]
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["delta_balanced_accuracy"] == 0
    assert result["environment"]["argv"] == command
    assert result["resources"]["peak_rss_sampled_bytes"] > 0
    assert result["provenance_verification"]["stored_aggregate_and_fold_metrics_recomputed"]
    assert result["input_manifests"]["baseline"]["sha256"] == sha256_file(paths["manifest.json"])
    digest = sha256_file(output)
    repeated = subprocess.run(command, cwd=repo, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "Refusing to overwrite" in repeated.stderr
    assert sha256_file(output) == digest
