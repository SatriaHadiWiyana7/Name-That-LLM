import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from name_that_llm.ablation import validate_controlled_ablation, validate_reference_environment
from name_that_llm.io import sha256_file
from name_that_llm.labels import LABELS


@pytest.fixture
def reference_inputs(tmp_path):
    environment = {
        "python": "3.13.11", "dependencies": {"numpy": "2.5.4", "scikit-learn": "1.8.0"},
        "thread_environment": {"OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "42"},
        "threadpools": [{"internal_api": "openblas", "version": "test-version", "architecture": "Haswell", "num_threads": 2, "filepath": "original.dll"}],
    }
    config = {
        "name": "reference", "random_seed": 42, "n_splits": 5,
        "word": {"min_df": 2, "ngram_range": [1, 2]},
        "char": {"min_df": 2, "ngram_range": [3, 5]},
        "classifier": {"C": 3.0, "max_iter": 1200, "class_weight": "balanced"},
    }
    folds = tmp_path / "folds.csv"
    folds.write_text("id,prompt_id,fold\na,p1,0\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"balanced_accuracy": 0.3}', encoding="utf-8")
    manifest = {
        "experiment_id": "synthetic_reference", "dataset_sha256": "fixed-dataset",
        "fold_assignments_sha256": sha256_file(folds), "random_seed": 42,
        "labels": LABELS, "probability_columns": ["p_" + label for label in LABELS],
        "config": config, "environment": environment,
        "artifacts": {"metrics": {"sha256": sha256_file(metrics)}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    candidate = copy.deepcopy(config)
    candidate.update(name="word_ablation", feature_groups=["word"])
    return {
        "config": candidate, "environment": copy.deepcopy(environment),
        "dataset_sha256": "fixed-dataset", "folds_path": folds,
        "reference_manifest_path": manifest_path, "reference_metrics_path": metrics,
    }


@pytest.mark.parametrize("group", ["word", "char"])
def test_controlled_ablation_accepts_only_feature_removal(reference_inputs, group):
    reference_inputs["config"]["feature_groups"] = [group]
    result = validate_controlled_ablation(**reference_inputs)
    assert result["status"] == "passed_before_fitting"
    assert result["feature_groups"] == [group]
    assert result["reference_manifest_sha256"] == sha256_file(reference_inputs["reference_manifest_path"])


@pytest.mark.parametrize("field", ["random_seed", "n_splits", "word", "char", "classifier", "unrelated"])
def test_controlled_ablation_rejects_changed_hyperparameters(reference_inputs, field):
    reference_inputs["config"][field] = "changed"
    with pytest.raises(ValueError, match="random_seed|config mismatch"):
        validate_controlled_ablation(**reference_inputs)


@pytest.mark.parametrize("groups", [None, [], ["word", "char"], ["word", "word"], ["style"]])
def test_controlled_ablation_rejects_nonablation_groups(reference_inputs, groups):
    reference_inputs["config"]["feature_groups"] = groups
    with pytest.raises(ValueError, match="feature_groups"):
        validate_controlled_ablation(**reference_inputs)


@pytest.mark.parametrize("mismatch", ["dataset", "folds", "metrics", "labels", "missing_folds", "missing_metrics"])
def test_controlled_ablation_rejects_provenance_mismatches(reference_inputs, mismatch):
    if mismatch == "dataset":
        reference_inputs["dataset_sha256"] = "other-dataset"
    elif mismatch == "folds":
        reference_inputs["folds_path"].write_text("id,prompt_id,fold\na,p1,1\n", encoding="utf-8")
    elif mismatch == "metrics":
        reference_inputs["reference_metrics_path"].write_text('{"balanced_accuracy": 0.8}', encoding="utf-8")
    elif mismatch == "labels":
        reference_inputs["labels"] = LABELS[::-1]
    elif mismatch == "missing_folds":
        reference_inputs["folds_path"] = None
    elif mismatch == "missing_metrics":
        reference_inputs["reference_metrics_path"] = None
    with pytest.raises(ValueError):
        validate_controlled_ablation(**reference_inputs)


@pytest.mark.parametrize("field", ["python", "dependencies", "thread_environment", "version", "architecture", "num_threads"])
def test_controlled_ablation_rejects_changed_environment(reference_inputs, field):
    environment = reference_inputs["environment"]
    if field in ("version", "architecture", "num_threads"):
        environment["threadpools"][0][field] = "changed"
    elif field == "dependencies":
        environment[field]["numpy"] = "changed"
    elif field == "thread_environment":
        environment[field]["OMP_NUM_THREADS"] = "8"
    else:
        environment[field] = "changed"
    with pytest.raises(ValueError, match="environment mismatch"):
        validate_controlled_ablation(**reference_inputs)


def test_environment_comparison_ignores_only_installation_paths_and_pool_order(reference_inputs):
    reference = reference_inputs["environment"]
    current = copy.deepcopy(reference)
    current["threadpools"][0]["filepath"] = "another-installation.dll"
    validate_reference_environment(current, reference)
    del current["threadpools"]
    with pytest.raises(ValueError, match="lacks required fields"):
        validate_reference_environment(current, reference)


def test_cli_controlled_ablation_copies_config_and_manifest_and_fails_before_fit(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    train = pd.DataFrame([
        {"id": f"p{prompt}-{author}", "prompt_id": f"p{prompt}",
         "text": f"prompt{prompt} authorword{author} ; testing distinctive style!", "model": label}
        for prompt in range(4) for author, label in enumerate(LABELS)
    ])
    train_path = tmp_path / "train.csv"
    train.to_csv(train_path, index=False)
    config = {
        "name": "synthetic_reference_only", "random_seed": 42, "n_splits": 2,
        "word": {"ngram_range": [1, 2], "min_df": 1, "max_features": 100},
        "char": {"analyzer": "char", "ngram_range": [3, 5], "min_df": 1, "max_features": 100},
        "classifier": {"C": 3.0, "max_iter": 1200, "class_weight": "balanced"},
    }
    config_path = tmp_path / "reference-config.json"
    config_path.write_text(json.dumps(config, indent=3) + "\n", encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONHASHSEED="42")
    register = tmp_path / "register.csv"

    def run(run_id, config_file, extra=()):
        command = [sys.executable, str(repo / "scripts/train_baseline.py"),
                   "--train", str(train_path), "--config", str(config_file),
                   "--out", str(tmp_path / run_id / "artifacts"),
                   "--reports", str(tmp_path / run_id / "reports"),
                   "--register", str(register), "--run-id", run_id, *extra]
        return subprocess.run(command, cwd=repo, env=environment, capture_output=True, text=True, timeout=60)

    reference_run = run("reference", config_path)
    assert reference_run.returncode == 0, reference_run.stderr
    original_manifest = tmp_path / "reference/artifacts/manifest.json"
    assert original_manifest.read_bytes() == (tmp_path / "reference/reports/manifest.json").read_bytes()
    baseline_model_hash = sha256_file(tmp_path / "reference/artifacts/model.joblib")
    candidate = copy.deepcopy(config)
    candidate.update(name="synthetic_word_only", feature_groups=["word"])
    candidate_path = tmp_path / "word-config.json"
    candidate_path.write_text(json.dumps(candidate, indent=4) + "\n", encoding="utf-8")
    extra = ["--reference-manifest", str(original_manifest),
             "--reference-metrics", str(tmp_path / "reference/reports/baseline_metrics.json"),
             "--folds", str(tmp_path / "reference/reports/baseline_fold_assignments.csv")]
    completed = run("word", candidate_path, extra)
    assert completed.returncode == 0, completed.stderr
    manifest_path = tmp_path / "word/reports/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "word/reports/baseline_metrics.json").read_text(encoding="utf-8"))
    assert manifest_path.read_bytes() == (tmp_path / "word/artifacts/manifest.json").read_bytes()
    assert (tmp_path / "word/reports/config.json").read_bytes() == candidate_path.read_bytes()
    assert manifest["artifacts"]["config"]["sha256"] == sha256_file(candidate_path)
    assert manifest["feature_groups"] == metrics["feature_groups"] == ["word"]
    assert manifest["model"] == metrics["model"] == "tfidf_word_logreg"
    assert manifest["controlled_ablation_validation"]["status"] == "passed_before_fitting"
    assert "src/name_that_llm/ablation.py" in manifest["source_sha256"]
    assert "manifest" not in manifest["artifacts"] and "report_manifest" not in manifest["artifacts"]
    with register.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[-1]["model"] == "tfidf_word_logreg"
    repeated = run("word", candidate_path, extra)
    assert repeated.returncode != 0 and "Refusing to overwrite" in repeated.stderr
    candidate["classifier"]["C"] = 99.0
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    rejected = run("invalid", candidate_path, extra)
    assert rejected.returncode != 0 and "config mismatch" in rejected.stderr
    assert not (tmp_path / "invalid").exists()
    assert sha256_file(tmp_path / "reference/artifacts/model.joblib") == baseline_model_hash
