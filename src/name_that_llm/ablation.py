"""Fail-closed provenance checks for controlled TF-IDF feature ablations."""

import json
from pathlib import Path

from .io import sha256_file
from .labels import LABELS


def environment_signature(environment: dict) -> dict:
    """Keep numerical-runtime controls; DLL installation paths are immaterial."""
    required = ("python", "dependencies", "thread_environment", "threadpools")
    missing = [key for key in required if key not in environment]
    if missing:
        raise ValueError("Reference environment lacks required fields: " + ", ".join(missing))
    pools = [{key: value for key, value in pool.items() if key != "filepath"}
             for pool in environment["threadpools"]]
    return {
        "python": environment["python"],
        "dependencies": {key.lower().replace("_", "-"): value
                         for key, value in environment["dependencies"].items()},
        "thread_environment": environment["thread_environment"],
        "threadpools": sorted(pools, key=lambda pool: json.dumps(pool, sort_keys=True)),
    }


def validate_reference_environment(current: dict, reference: dict) -> None:
    current_signature = environment_signature(current)
    reference_signature = environment_signature(reference)
    for key in reference_signature:
        if current_signature[key] != reference_signature[key]:
            raise ValueError(f"Controlled ablation environment mismatch: {key}")


def validate_controlled_ablation(
    *, config: dict, environment: dict, dataset_sha256: str,
    folds_path: str | Path | None, reference_manifest_path: str | Path,
    reference_metrics_path: str | Path | None, labels: list[str] | None = None,
) -> dict:
    """Allow removal of one feature group, with every other control fixed.

    Call before any vectorizer/classifier fitting or experiment output creation.
    Dataset and input fold-file hashes identify the exact P0 rows and assignment.
    """
    if not folds_path:
        raise ValueError("Controlled ablation requires --folds from the reference run")
    if not reference_metrics_path:
        raise ValueError("Controlled ablation requires --reference-metrics")
    reference = json.loads(Path(reference_manifest_path).read_text(encoding="utf-8"))
    if dataset_sha256 != reference["dataset_sha256"]:
        raise ValueError("Controlled ablation dataset_sha256 mismatch")
    if sha256_file(folds_path) != reference["fold_assignments_sha256"]:
        raise ValueError("Controlled ablation saved fold SHA256 mismatch")
    if sha256_file(reference_metrics_path) != reference["artifacts"]["metrics"]["sha256"]:
        raise ValueError("Controlled ablation reference metrics SHA256 mismatch")
    if reference["labels"] != (LABELS if labels is None else labels):
        raise ValueError("Controlled ablation label ordering mismatch")
    if reference["probability_columns"] != ["p_" + label for label in reference["labels"]]:
        raise ValueError("Controlled ablation reference probability ordering mismatch")
    if config["random_seed"] != reference["random_seed"]:
        raise ValueError("Controlled ablation random_seed mismatch")
    if reference["config"].get("feature_groups", ["word", "char"]) != ["word", "char"]:
        raise ValueError("Controlled ablation reference must use word and char features")
    groups = config.get("feature_groups")
    if groups not in (["word"], ["char"]):
        raise ValueError("Controlled ablation feature_groups must be ['word'] or ['char']")
    original = {key: value for key, value in reference["config"].items()
                if key not in ("name", "feature_groups")}
    candidate = {key: value for key, value in config.items()
                 if key not in ("name", "feature_groups")}
    changed = [key for key in sorted(set(original) | set(candidate))
               if key not in original or key not in candidate or original[key] != candidate[key]]
    if changed:
        raise ValueError("Controlled ablation config mismatch: " + ", ".join(changed))
    validate_reference_environment(environment, reference["environment"])
    return {
        "status": "passed_before_fitting",
        "reference_manifest_path": str(Path(reference_manifest_path).resolve()),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "reference_metrics_sha256": sha256_file(reference_metrics_path),
        "reference_experiment_id": reference["experiment_id"],
        "feature_groups": groups,
        "only_permitted_config_changes": ["name", "feature_groups"],
        "dataset_folds_seed_labels_environment_verified": True,
    }
