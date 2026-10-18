"""Paired prompt-grouped OOF comparisons, without source passages or row IDs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyze_oof, validate_oof
from .ablation import validate_reference_environment
from .io import sha256_file
from .labels import LABELS
from .metrics import classification_report
from .runtime import validate_fold_mapping


PROBABILITY_COLUMNS = ["p_" + label for label in LABELS]
FOCUS_LABELS = [
    "claude-fable-5", "gpt-5.6-sol", "glm-5.3", "glm-5.3-flash",
    "gemini-3.8-flash", "gemini-3.7-flash",
]
METRIC_FIELDS = ("balanced_accuracy", "class_recall", "class_support", "confusion_matrix", "labels")


def _validate_comparison_oof(oof: pd.DataFrame) -> None:
    validate_oof(oof)
    if [column for column in oof if column.startswith("p_")] != PROBABILITY_COLUMNS:
        raise ValueError("Paired comparison requires all 18 ordered probability columns")
    if "model_id" not in oof or oof.model_id.isna().any() or oof.model_id.nunique() != 1:
        raise ValueError("Paired comparison requires one nonnull model_id per OOF file")
    if not str(oof.model_id.iloc[0]).strip():
        raise ValueError("OOF model_id must not be blank")
    classification_report(oof.true_model, oof.pred_model)


def align_paired_oof(baseline: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require identical evaluation units, aligning candidate rows by unique ID."""
    _validate_comparison_oof(baseline)
    _validate_comparison_oof(candidate)
    if len(baseline) != len(candidate) or set(baseline.id) != set(candidate.id):
        raise ValueError("Paired OOF IDs must match exactly")
    baseline = baseline.reset_index(drop=True)
    candidate = candidate.set_index("id", drop=False).loc[baseline.id].reset_index(drop=True)
    for column in ("prompt_id", "true_model", "fold"):
        if not np.array_equal(baseline[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(f"Paired OOF {column} differs after ID alignment")
    return baseline, candidate


def _interval(values: np.ndarray, point: float) -> dict:
    return {
        "estimate": float(point),
        "bootstrap_mean": float(np.mean(values)) if len(values) else None,
        "ci_lower": float(np.quantile(values, 0.025, method="linear")) if len(values) else None,
        "ci_upper": float(np.quantile(values, 0.975, method="linear")) if len(values) else None,
    }


def paired_prompt_cluster_bootstrap(baseline: pd.DataFrame, candidate: pd.DataFrame,
                                    repeats: int = 10000, seed: int = 42) -> dict:
    """Apply each prompt draw to both models; compute macro recall from pooled counts."""
    if not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("Bootstrap repeats must be a positive integer")
    baseline, candidate = align_paired_oof(baseline, candidate)
    prompt_codes, prompts = pd.factorize(baseline.prompt_id, sort=True)
    label_codes = pd.Categorical(baseline.true_model, categories=LABELS).codes
    support = np.zeros((len(prompts), len(LABELS)), dtype=np.int64)
    baseline_tp = np.zeros_like(support)
    candidate_tp = np.zeros_like(support)
    np.add.at(support, (prompt_codes, label_codes), 1)
    np.add.at(baseline_tp, (prompt_codes, label_codes), (baseline.true_model == baseline.pred_model).astype(int))
    np.add.at(candidate_tp, (prompt_codes, label_codes), (candidate.true_model == candidate.pred_model).astype(int))
    baseline_recalls = baseline_tp.sum(axis=0) / support.sum(axis=0)
    candidate_recalls = candidate_tp.sum(axis=0) / support.sum(axis=0)
    rng = np.random.default_rng(seed)
    baseline_draws, candidate_draws = [], []
    for _ in range(repeats):
        selected = rng.integers(0, len(prompts), size=len(prompts))
        sampled_support = support[selected].sum(axis=0)
        if (sampled_support == 0).any():
            continue
        baseline_draws.append(baseline_tp[selected].sum(axis=0) / sampled_support)
        candidate_draws.append(candidate_tp[selected].sum(axis=0) / sampled_support)
    baseline_draws = np.asarray(baseline_draws, dtype=float).reshape(-1, len(LABELS))
    candidate_draws = np.asarray(candidate_draws, dtype=float).reshape(-1, len(LABELS))
    baseline_ba = baseline_draws.mean(axis=1)
    candidate_ba = candidate_draws.mean(axis=1)
    return {
        "method": "paired percentile prompt-cluster bootstrap of fixed OOF predictions",
        "resampling_unit": "prompt_id", "paired_draws": True,
        "prompt_groups": len(prompts), "groups_drawn_per_repeat": len(prompts),
        "requested_repeats": repeats, "valid_repeats": len(baseline_draws),
        "skipped_missing_class_repeats": repeats - len(baseline_draws),
        "seed": seed, "rng": type(rng.bit_generator).__name__,
        "ci_level": 0.95, "quantile_method": "linear",
        "delta_definition": "candidate minus baseline",
        "baseline_balanced_accuracy": _interval(baseline_ba, baseline_recalls.mean()),
        "candidate_balanced_accuracy": _interval(candidate_ba, candidate_recalls.mean()),
        "delta_balanced_accuracy": _interval(candidate_ba - baseline_ba, candidate_recalls.mean() - baseline_recalls.mean()),
        "class_recall_delta": {
            label: _interval(candidate_draws[:, i] - baseline_draws[:, i], candidate_recalls[i] - baseline_recalls[i])
            for i, label in enumerate(LABELS)
        },
        "caveat": (
            "Approximate intervals conditional on fitted OOF models; no retraining variance. "
            "Intervals are exploratory and not multiplicity corrected. A positive bootstrap "
            "fraction is not a probability of improvement. Replicates lacking any of 18 "
            "legal classes are excluded and counted. Repeated training runs are not "
            "independent statistical evidence; these intervals do not predict leaderboard rank."
        ),
    }


def _error_flips(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    baseline_correct = (baseline.true_model == baseline.pred_model).to_numpy()
    candidate_correct = (candidate.true_model == candidate.pred_model).to_numpy()
    same = (baseline.pred_model == candidate.pred_model).to_numpy()

    def counts(mask):
        return {
            "rows": int(mask.sum()),
            "both_correct": int((mask & baseline_correct & candidate_correct).sum()),
            "corrected": int((mask & ~baseline_correct & candidate_correct).sum()),
            "regressed": int((mask & baseline_correct & ~candidate_correct).sum()),
            "both_wrong_same_prediction": int((mask & ~baseline_correct & ~candidate_correct & same).sum()),
            "both_wrong_different_prediction": int((mask & ~baseline_correct & ~candidate_correct & ~same).sum()),
            "predictions_changed": int((mask & ~same).sum()),
        }

    return {
        "definitions": {"corrected": "baseline wrong, candidate correct", "regressed": "baseline correct, candidate wrong"},
        "overall": counts(np.ones(len(baseline), dtype=bool)),
        "per_class": {label: counts(baseline.true_model.to_numpy() == label) for label in LABELS},
    }


def _metric_delta(baseline: dict, candidate: dict) -> dict:
    return {
        "baseline_balanced_accuracy": baseline["balanced_accuracy"],
        "candidate_balanced_accuracy": candidate["balanced_accuracy"],
        "delta_balanced_accuracy": candidate["balanced_accuracy"] - baseline["balanced_accuracy"],
        "class_recall_delta": {label: candidate["class_recall"][label] - baseline["class_recall"][label] for label in LABELS},
        "confusion_matrix_delta": (np.asarray(candidate["confusion_matrix"]) - np.asarray(baseline["confusion_matrix"])).tolist(),
    }


def _gemini_pair(report: dict) -> dict:
    pair = ["gemini-3.8-flash", "gemini-3.7-flash"]
    matrix = np.asarray(report["confusion_matrix"])
    indices = [LABELS.index(label) for label in pair]
    rows = []
    for i, label in zip(indices, pair):
        j = next(index for index in indices if index != i)
        support = int(matrix[i].sum())
        rows.append({
            "true_model": label, "support": support, "correct": int(matrix[i, i]),
            "recall": float(matrix[i, i] / support),
            "confused_with_pair": int(matrix[i, j]), "confused_with_pair_rate": float(matrix[i, j] / support),
            "errors_to_other_labels": int(support - matrix[i, i] - matrix[i, j]),
        })
    return {"labels": pair, "matrix": matrix[np.ix_(indices, indices)].tolist(), "rows": rows,
            "note": "Rows are true classes, columns predicted classes. Rates use all true-class rows; the pair submatrix is not renormalized."}


def compare_paired_oof(baseline: pd.DataFrame, candidate: pd.DataFrame,
                       repeats: int = 10000, seed: int = 42) -> dict:
    baseline, candidate = align_paired_oof(baseline, candidate)
    baseline_report, candidate_report = analyze_oof(baseline), analyze_oof(candidate)
    bootstrap = paired_prompt_cluster_bootstrap(baseline, candidate, repeats, seed)
    flips = _error_flips(baseline, candidate)
    folds = []
    for fold in sorted(baseline.fold.unique()):
        left, right = baseline.loc[baseline.fold == fold], candidate.loc[candidate.fold == fold]
        first = classification_report(left.true_model, left.pred_model)
        second = classification_report(right.true_model, right.pred_model)
        folds.append({"fold": int(fold), "rows": len(left), "prompts": int(left.prompt_id.nunique()),
                      "baseline": first, "candidate": second, **_metric_delta(first, second),
                      "error_flips": _error_flips(left, right)})
    baseline_errors = {item["model"]: item for item in baseline_report["per_class_errors"]}
    candidate_errors = {item["model"]: item for item in candidate_report["per_class_errors"]}
    return {
        "baseline_model_id": str(baseline.model_id.iloc[0]),
        "candidate_model_id": str(candidate.model_id.iloc[0]),
        "labels": LABELS, "rows": len(baseline), "prompts": int(baseline.prompt_id.nunique()),
        "alignment": "Unique identical IDs; exact prompt_id, true_model and fold equality after ID alignment; ordered probabilities validated",
        "baseline": baseline_report, "candidate": candidate_report,
        **_metric_delta(baseline_report, candidate_report),
        "folds": folds, "error_flips": flips, "paired_prompt_cluster_bootstrap": bootstrap,
        "focus_classes": {
            label: {"baseline": baseline_errors[label], "candidate": candidate_errors[label],
                    "recall_delta": bootstrap["class_recall_delta"][label], "error_flips": flips["per_class"][label]}
            for label in FOCUS_LABELS
        },
        "gemini_pair": {"baseline": _gemini_pair(baseline_report), "candidate": _gemini_pair(candidate_report)},
        "family_diagnostics_note": "Provider aggregation is diagnostic; model comparisons use all 18 class recalls.",
    }


def load_verified_run(manifest_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Verify saved bytes, labels, metadata and every stored aggregate/fold metric."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["labels"] != LABELS or manifest["probability_columns"] != PROBABILITY_COLUMNS:
        raise ValueError("Manifest label/probability ordering differs from LABELS")
    if not {"metrics", "oof", "folds", "model"}.issubset(manifest["artifacts"]):
        raise ValueError("Manifest lacks required metrics/oof/folds/model artifacts")
    for name, artifact in manifest["artifacts"].items():
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"Artifact hash/size mismatch: {name}")
    for name in ("dataset", "config"):
        if sha256_file(manifest[name + "_path"]) != manifest[name + "_sha256"]:
            raise ValueError(f"Manifest {name} hash mismatch")
    config = json.loads(Path(manifest["config_path"]).read_text(encoding="utf-8"))
    if config != manifest["config"] or config["random_seed"] != manifest["random_seed"]:
        raise ValueError("Manifest config or random seed mismatch")
    if manifest["model_sha256"] != manifest["artifacts"]["model"]["sha256"]:
        raise ValueError("Manifest model hash mismatch")
    if manifest["fold_assignments_sha256"] != manifest["artifacts"]["folds"]["sha256"]:
        raise ValueError("Manifest fold hash mismatch")
    oof = pd.read_csv(manifest["artifacts"]["oof"]["path"], dtype={"id": str, "prompt_id": str})
    _validate_comparison_oof(oof)
    if set(oof.model_id) != {manifest["model_id"]}:
        raise ValueError("OOF model_id differs from manifest")
    train = pd.read_csv(manifest["dataset_path"], usecols=["id", "prompt_id", "model"],
                        dtype={"id": str, "prompt_id": str, "model": str})
    mapping = pd.read_csv(manifest["artifacts"]["folds"]["path"], dtype={"id": str, "prompt_id": str})
    assigned = validate_fold_mapping(train, mapping, config["n_splits"])
    if set(train.id) != set(oof.id):
        raise ValueError("OOF IDs differ from dataset")
    aligned = oof.set_index("id").loc[train.id]
    for column, values in (("prompt_id", train.prompt_id), ("true_model", train.model), ("fold", assigned)):
        if not np.array_equal(aligned[column].to_numpy(), np.asarray(values)):
            raise ValueError(f"OOF {column} differs from verified dataset/fold assignment")
    metrics = json.loads(Path(manifest["artifacts"]["metrics"]["path"]).read_text(encoding="utf-8"))
    recomputed = classification_report(oof.true_model, oof.pred_model)
    for field in METRIC_FIELDS:
        if metrics[field] != recomputed[field]:
            raise ValueError(f"Stored aggregate {field} differs from OOF")
    if manifest["oof_balanced_accuracy"] != recomputed["balanced_accuracy"]:
        raise ValueError("Manifest BA differs from OOF")
    if len(metrics["folds"]) != config["n_splits"] or {item["fold"] for item in metrics["folds"]} != set(range(config["n_splits"])):
        raise ValueError("Stored metrics have missing or duplicate folds")
    for fold in metrics["folds"]:
        subset = oof.loc[oof.fold == fold["fold"]]
        measured = classification_report(subset.true_model, subset.pred_model)
        for field in METRIC_FIELDS:
            if fold[field] != measured[field]:
                raise ValueError(f"Stored fold {fold['fold']} {field} differs from OOF")
    return manifest, oof


def verify_comparable_manifests(baseline: dict, candidate: dict) -> dict:
    for field in ("dataset_sha256", "fold_assignments_sha256", "random_seed", "classifier_random_state", "labels", "probability_columns"):
        if field not in baseline or field not in candidate or baseline[field] != candidate[field]:
            raise ValueError(f"Comparison provenance mismatch: {field}")
    validate_reference_environment(candidate["environment"], baseline["environment"])
    for field in ("cpu", "system_ram_bytes", "platform"):
        if field not in baseline["environment"] or field not in candidate["environment"] or baseline["environment"][field] != candidate["environment"][field]:
            raise ValueError(f"Comparison environment mismatch: {field}")
    configs = [{key: value for key, value in manifest["config"].items() if key not in ("name", "feature_groups")}
               for manifest in (baseline, candidate)]
    if configs[0] != configs[1]:
        raise ValueError("Controlled comparison config differs beyond name/feature_groups")
    return {"same_dataset_folds_seed_labels": True, "same_python_dependencies_threads_hardware": True,
            "config_diff_only_name_feature_groups": True,
            "threadpool_path_note": "Library paths may differ; APIs, prefixes, versions, threads, threading layer and architecture must match."}
