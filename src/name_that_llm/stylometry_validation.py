"""P3 provenance guards; the existing P0/P2 ablation contract is unchanged."""

import hashlib
import json
from pathlib import Path

from .ablation import validate_reference_environment
from .io import sha256_file
from .labels import LABELS
from .stylometry import validate_stylometry_config


def validate_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(plan["preregistered_text"].encode("utf-8")).hexdigest()
    if digest != plan["document_sha256"]:
        raise ValueError("Preregistered plan text/hash mismatch")
    required = {"status": "preregistered_before_implementation_and_training",
                "feature_count": 81, "bootstrap_repeats": 10000, "seed": 42,
                "reference_ba": 0.31633855331841904}
    for key, value in required.items():
        if plan.get(key) != value:
            raise ValueError(f"Preregistered plan mismatch: {key}")
    return plan


def _check_shared_controls(baseline, candidate):
    for field in ("dataset_sha256", "fold_assignments_sha256", "random_seed",
                  "classifier_random_state", "labels", "probability_columns"):
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"P3 comparison provenance mismatch: {field}")
    if baseline["labels"] != LABELS or baseline["probability_columns"] != ["p_" + x for x in LABELS]:
        raise ValueError("P3 reference label/probability ordering mismatch")
    validate_reference_environment(candidate["environment"], baseline["environment"])
    for field in ("cpu", "system_ram_bytes", "platform"):
        if field not in baseline["environment"] or baseline["environment"][field] != candidate["environment"].get(field):
            raise ValueError(f"P3 hardware mismatch: {field}")
    validate_stylometry_config(candidate["config"])
    if baseline["config"].get("feature_groups", ["word", "char"]) != ["word", "char"]:
        raise ValueError("P3 reference must be combined P0")
    if baseline["config"]["n_splits"] != candidate["config"]["n_splits"]:
        raise ValueError("P3 n_splits mismatch")
    original = dict(baseline["config"]["classifier"])
    original.setdefault("solver", "lbfgs")
    original.setdefault("tol", 1e-4)
    original.setdefault("random_state", baseline["random_seed"])
    current = dict(candidate["config"]["classifier"])
    current.setdefault("random_state", candidate["random_seed"])
    if original != current:
        raise ValueError("P3 classifier differs from P0 beyond explicit defaults")


def validate_stylometry_reference(*, config, environment, dataset_sha256,
                                 folds_path, reference_manifest_path,
                                 reference_metrics_path, plan_path):
    """Reject any changed data/folds/runtime or unplanned config before fitting."""
    if not folds_path or not reference_metrics_path or not plan_path:
        raise ValueError("P3 requires saved folds, reference metrics and preregistered plan")
    validate_stylometry_config(config)
    plan = validate_plan(plan_path)
    reference = json.loads(Path(reference_manifest_path).read_text(encoding="utf-8"))
    if sha256_file(reference_metrics_path) != reference["artifacts"]["metrics"]["sha256"]:
        raise ValueError("P3 reference metrics hash mismatch")
    metrics = json.loads(Path(reference_metrics_path).read_text(encoding="utf-8"))
    if metrics["balanced_accuracy"] != plan["reference_ba"]:
        raise ValueError("P3 reference BA differs from preregistered official P0")
    candidate = {
        "dataset_sha256": dataset_sha256,
        "fold_assignments_sha256": sha256_file(folds_path),
        "random_seed": config["random_seed"],
        "classifier_random_state": config["random_seed"],
        "labels": LABELS, "probability_columns": ["p_" + x for x in LABELS],
        "config": config, "environment": environment,
    }
    _check_shared_controls(reference, candidate)
    return {
        "status": "passed_before_fitting",
        "reference_manifest_path": str(Path(reference_manifest_path).resolve()),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "reference_metrics_sha256": sha256_file(reference_metrics_path),
        "reference_experiment_id": reference["experiment_id"],
        "plan_path": str(Path(plan_path).resolve()), "plan_sha256": sha256_file(plan_path),
        "dataset_folds_seed_labels_environment_verified": True,
        "classifier_matches_p0_explicit_defaults": True,
        "planned_difference": "TF-IDF replaced with fixed numeric stylometry and training-fold StandardScaler",
    }


def verify_stylometry_comparison(baseline, candidate, baseline_manifest_path):
    """P3-specific representation comparison, without relaxing P2 guards."""
    _check_shared_controls(baseline, candidate)
    guard = candidate.get("stylometry_validation", {})
    if guard.get("status") != "passed_before_fitting":
        raise ValueError("P3 candidate lacks successful pre-fit guard")
    if guard.get("reference_manifest_sha256") != sha256_file(baseline_manifest_path):
        raise ValueError("P3 comparison baseline differs from guarded reference")
    if sha256_file(guard["plan_path"]) != guard["plan_sha256"]:
        raise ValueError("P3 plan changed after fitting")
    plan = validate_plan(guard["plan_path"])
    if baseline["oof_balanced_accuracy"] != plan["reference_ba"]:
        raise ValueError("P3 comparison requires official P0 score")
    return {"same_dataset_folds_seed_labels": True,
            "same_python_dependencies_threads_hardware": True,
            "classifier_matches_p0_explicit_defaults": True,
            "preregistered_representation_and_scaler_change": True,
            "plan_sha256": guard["plan_sha256"]}
