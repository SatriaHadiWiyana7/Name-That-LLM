import copy
import json
from pathlib import Path

import pytest

from name_that_llm.io import sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.stylometry_validation import (
    validate_stylometry_reference, verify_stylometry_comparison,
)


@pytest.fixture
def inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/p3_stylometry.json").read_text(encoding="utf-8"))
    environment = {
        "python": "test", "dependencies": {"numpy": "test"},
        "thread_environment": {"OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "42"},
        "threadpools": [], "cpu": {"physical_cores": 6},
        "system_ram_bytes": 32 * 1024 ** 3, "platform": "test-platform",
    }
    folds = tmp_path / "folds.csv"
    folds.write_text("id,prompt_id,fold\na,p0,0\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"balanced_accuracy": 0.31633855331841904}), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_bytes((root / "reports/p3_20261018_plan.json").read_bytes())
    baseline = {
        "experiment_id": "reference", "dataset_sha256": "same-data",
        "fold_assignments_sha256": sha256_file(folds),
        "random_seed": 42, "classifier_random_state": 42,
        "labels": LABELS, "probability_columns": ["p_" + x for x in LABELS],
        "environment": environment,
        "config": {"n_splits": 5, "classifier": {"C": 3.0, "max_iter": 1200, "class_weight": "balanced"}},
        "artifacts": {"metrics": {"sha256": sha256_file(metrics)}},
        "oof_balanced_accuracy": 0.31633855331841904,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(baseline), encoding="utf-8")
    return {"config": config, "environment": copy.deepcopy(environment),
            "dataset_sha256": "same-data", "folds_path": folds,
            "reference_manifest_path": manifest, "reference_metrics_path": metrics,
            "plan_path": plan}


def test_valid_numeric_representation_change_passes(inputs):
    result = validate_stylometry_reference(**inputs)
    assert result["status"] == "passed_before_fitting"
    assert result["classifier_matches_p0_explicit_defaults"]


@pytest.mark.parametrize("mismatch", ["data", "folds", "metrics", "labels", "probabilities", "seed", "classifier", "python", "hardware", "missing_folds", "plan", "plan_text"])
def test_guard_fails_before_training_on_drift(inputs, mismatch):
    if mismatch == "data":
        inputs["dataset_sha256"] = "different-data"
    elif mismatch == "folds":
        inputs["folds_path"].write_text("changed", encoding="utf-8")
    elif mismatch == "metrics":
        inputs["reference_metrics_path"].write_text("{}", encoding="utf-8")
    elif mismatch == "python":
        inputs["environment"]["python"] = "different"
    elif mismatch == "hardware":
        inputs["environment"]["system_ram_bytes"] = 123
    elif mismatch == "missing_folds":
        inputs["folds_path"] = None
    elif mismatch in {"plan", "plan_text"}:
        plan = json.loads(inputs["plan_path"].read_text(encoding="utf-8"))
        plan["seed" if mismatch == "plan" else "preregistered_text"] = "changed"
        inputs["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    else:
        path = inputs["reference_manifest_path"]
        reference = json.loads(path.read_text(encoding="utf-8"))
        if mismatch == "labels":
            reference["labels"] = LABELS[::-1]
        elif mismatch == "probabilities":
            reference["probability_columns"].reverse()
        elif mismatch == "seed":
            reference["random_seed"] = 99
        elif mismatch == "classifier":
            reference["config"]["classifier"]["C"] = 99
        path.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_stylometry_reference(**inputs)


def candidate_from_inputs(inputs):
    baseline = json.loads(inputs["reference_manifest_path"].read_text(encoding="utf-8"))
    candidate = copy.deepcopy(baseline)
    candidate["config"] = inputs["config"]
    candidate["stylometry_validation"] = validate_stylometry_reference(**inputs)
    return baseline, candidate


def test_p3_comparison_requires_pre_fit_plan_and_same_reference(inputs):
    baseline, candidate = candidate_from_inputs(inputs)
    check = verify_stylometry_comparison(baseline, candidate, inputs["reference_manifest_path"])
    assert check["preregistered_representation_and_scaler_change"]
    candidate["stylometry_validation"]["reference_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="baseline differs"):
        verify_stylometry_comparison(baseline, candidate, inputs["reference_manifest_path"])


def test_p3_comparison_rejects_plan_edit_after_fit(inputs):
    baseline, candidate = candidate_from_inputs(inputs)
    with inputs["plan_path"].open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="plan changed"):
        verify_stylometry_comparison(baseline, candidate, inputs["reference_manifest_path"])
