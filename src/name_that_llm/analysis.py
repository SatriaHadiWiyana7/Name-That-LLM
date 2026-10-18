"""Aggregate OOF diagnostics; never include source passages in reports."""

import re

import numpy as np
import pandas as pd

from .labels import LABELS
from .metrics import classification_report


WORD_PATTERN = r"\b[\w]+(?:['\u2019\-][\w]+)*\b"
FAMILY_BY_PREFIX = {
    "claude": "Anthropic", "gemini": "Google", "glm": "ZAI",
    "gpt": "OpenAI", "kimi": "Moonshot", "qwen": "Qwen",
}


def validate_oof(oof: pd.DataFrame) -> None:
    """Reject malformed predictions and visible prompt-to-fold leakage."""
    required = ["id", "prompt_id", "true_model", "pred_model", "fold"]
    if not set(required).issubset(oof.columns):
        raise ValueError(f"OOF requires columns: {required}")
    if oof.empty or oof[required].isna().any().any():
        raise ValueError("OOF must contain rows and no null required fields")
    if oof.id.duplicated().any():
        raise ValueError("OOF IDs must be unique")
    if any(oof[column].astype(str).str.strip().eq("").any() for column in required):
        raise ValueError("OOF required fields must not be blank")
    illegal = (set(oof.true_model) | set(oof.pred_model)) - set(LABELS)
    if illegal:
        raise ValueError(f"Illegal OOF labels: {sorted(illegal)}")
    folds = pd.to_numeric(oof.fold, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(folds).all() or (folds < 0).any() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError("OOF fold numbers must be finite nonnegative integers")
    if oof.groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("PROMPT LEAKAGE: a prompt occurs in multiple OOF folds")
    probability_columns = [column for column in oof if column.startswith("p_")]
    if probability_columns:
        if probability_columns != ["p_" + label for label in LABELS]:
            raise ValueError("OOF probability columns must match exactly the LABELS order")
        probabilities = oof[probability_columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("OOF probabilities must be finite values in [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("OOF probabilities must sum to one")
        predicted = np.asarray(LABELS, dtype=object)[probabilities.argmax(axis=1)]
        if not np.array_equal(predicted, oof.pred_model.to_numpy()):
            raise ValueError("OOF pred_model does not match probability argmax")


def subset_report(truth, predictions) -> dict:
    """Withhold 18-class BA when a diagnostic subset lacks any legal class."""
    missing = [label for label in LABELS if label not in set(truth)]
    if not missing:
        return {**classification_report(truth, predictions), "missing_labels": []}
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(truth, predictions, labels=LABELS)
    supports = matrix.sum(axis=1)
    return {
        "balanced_accuracy": None,
        "balanced_accuracy_note": "Undefined here: subset is missing legal classes; do not average present classes",
        "missing_labels": missing,
        "labels": LABELS,
        "class_support": {label: int(supports[i]) for i, label in enumerate(LABELS)},
        "class_recall": {
            label: float(matrix[i, i] / supports[i]) if supports[i] else None
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
    }


def prompt_cluster_bootstrap(oof: pd.DataFrame, repeats: int, seed: int = 42) -> dict:
    """Resample prompt groups, recomputing class recall from pooled counts."""
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive")
    validate_oof(oof)
    classification_report(oof.true_model, oof.pred_model)
    prompt_codes, prompts = pd.factorize(oof.prompt_id, sort=True)
    label_codes = pd.Categorical(oof.true_model, categories=LABELS).codes
    supports = np.zeros((len(prompts), len(LABELS)), dtype=np.int64)
    true_positives = np.zeros_like(supports)
    np.add.at(supports, (prompt_codes, label_codes), 1)
    np.add.at(true_positives, (prompt_codes, label_codes), (oof.true_model == oof.pred_model).astype(int))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        selected = rng.integers(0, len(prompts), size=len(prompts))
        sampled_supports = supports[selected].sum(axis=0)
        if (sampled_supports == 0).any():
            continue
        recalls = true_positives[selected].sum(axis=0) / sampled_supports
        scores.append(float(recalls.mean()))
    return {
        "method": "percentile prompt-cluster bootstrap of fixed OOF predictions",
        "resampling_unit": "prompt_id",
        "prompt_groups": len(prompts),
        "groups_drawn_per_repeat": len(prompts),
        "seed": seed,
        "requested_repeats": repeats,
        "valid_repeats": len(scores),
        "skipped_missing_class_repeats": repeats - len(scores),
        "mean_balanced_accuracy": float(np.mean(scores)) if scores else None,
        "ci_level": 0.95,
        "ci_lower": float(np.quantile(scores, 0.025)) if scores else None,
        "ci_upper": float(np.quantile(scores, 0.975)) if scores else None,
        "minimum_balanced_accuracy": min(scores) if scores else None,
        "maximum_balanced_accuracy": max(scores) if scores else None,
        "caveat": "Approximate uncertainty conditional on fitted OOF models; does not measure retraining variance. Replicates lacking any legal class are excluded and counted.",
    }


def length_quartile_report(oof: pd.DataFrame, train: pd.DataFrame) -> dict:
    required = ["id", "prompt_id", "model", "text"]
    if not set(required).issubset(train.columns) or train[required].isna().any().any():
        raise ValueError("Length diagnostics require complete training id,prompt_id,model,text")
    if train.id.duplicated().any() or set(train.id) != set(oof.id):
        raise ValueError("Length diagnostics require unique training IDs exactly covering OOF")
    aligned = train.set_index("id").loc[oof.id]
    if not np.array_equal(aligned.prompt_id.to_numpy(), oof.prompt_id.to_numpy()):
        raise ValueError("Training prompt IDs do not match OOF")
    if not np.array_equal(aligned.model.to_numpy(), oof.true_model.to_numpy()):
        raise ValueError("Training labels do not match OOF")
    words = aligned.text.map(lambda text: len(re.findall(WORD_PATTERN, text, re.UNICODE)))
    bins, edges = pd.qcut(words, 4, labels=False, retbins=True, duplicates="drop")
    reports = []
    for value in sorted(bins.dropna().unique()):
        mask = bins.to_numpy() == value
        selected = oof.loc[mask]
        reports.append({
            "quartile": int(value) + 1,
            "rows": len(selected),
            "prompts": int(selected.prompt_id.nunique()),
            "word_min": int(words.to_numpy()[mask].min()),
            "word_max": int(words.to_numpy()[mask].max()),
            **subset_report(selected.true_model, selected.pred_model),
        })
    return {
        "word_regex": WORD_PATTERN,
        "binning": "pandas.qcut on all OOF training-text word counts; identical lengths stay together; duplicate edges dropped; diagnostic only",
        "quantile_edges": edges.tolist(),
        "actual_bins": len(reports),
        "unassigned_rows": int(bins.isna().sum()),
        "bins": reports,
    }


def family_report(matrix: np.ndarray) -> dict:
    label_family = {
        label: next(family for prefix, family in FAMILY_BY_PREFIX.items() if label.startswith(prefix))
        for label in LABELS
    }
    families = list(dict.fromkeys(label_family.values()))
    family_matrix = np.zeros((len(families), len(families)), dtype=np.int64)
    for i, truth in enumerate(LABELS):
        for j, prediction in enumerate(LABELS):
            family_matrix[families.index(label_family[truth]), families.index(label_family[prediction])] += matrix[i, j]
    recalls = family_matrix.diagonal() / family_matrix.sum(axis=1)
    total_errors = int(matrix.sum() - matrix.trace())
    within_errors = int(family_matrix.trace() - matrix.trace())
    return {
        "note": "Provider aggregation is diagnostic only; competition metric remains 18-class BA",
        "label_to_family": label_family,
        "families": families,
        "confusion_matrix": family_matrix.tolist(),
        "family_balanced_accuracy": float(recalls.mean()),
        "family_recall": dict(zip(families, recalls.tolist())),
        "within_family_errors": within_errors,
        "cross_family_errors": total_errors - within_errors,
        "within_family_fraction_of_errors": within_errors / total_errors if total_errors else None,
    }


def analyze_oof(oof: pd.DataFrame, train: pd.DataFrame | None = None,
                bootstrap_repeats: int = 0, seed: int = 42) -> dict:
    validate_oof(oof)
    if bootstrap_repeats < 0:
        raise ValueError("Bootstrap repeats must be nonnegative")
    result = classification_report(oof.true_model, oof.pred_model)
    matrix = np.asarray(result["confusion_matrix"])
    supports = matrix.sum(axis=1)
    per_class, directional, symmetric = [], [], []
    for i, label in enumerate(LABELS):
        targets = [
            {"pred_model": prediction, "count": int(matrix[i, j]),
             "rate_of_true_class": float(matrix[i, j] / supports[i])}
            for j, prediction in enumerate(LABELS) if j != i and matrix[i, j]
        ]
        targets.sort(key=lambda item: (-item["count"], item["pred_model"]))
        per_class.append({
            "model": label, "support": int(supports[i]),
            "recall": result["class_recall"][label],
            "true_positive": int(matrix[i, i]),
            "false_negative": int(supports[i] - matrix[i, i]),
            "false_positive": int(matrix[:, i].sum() - matrix[i, i]),
            "top_confusion_targets": targets[:3],
        })
        directional.extend({"true_model": label, **target} for target in targets)
        for j in range(i + 1, len(LABELS)):
            count = int(matrix[i, j] + matrix[j, i])
            if count:
                symmetric.append({
                    "model_a": label, "model_b": LABELS[j], "count": count,
                    "a_to_b_count": int(matrix[i, j]), "b_to_a_count": int(matrix[j, i]),
                    "a_to_b_rate": float(matrix[i, j] / supports[i]),
                    "b_to_a_rate": float(matrix[j, i] / supports[j]),
                    "rate_of_pair_support": float(count / (supports[i] + supports[j])),
                })
    per_class.sort(key=lambda item: (item["recall"], item["model"]))
    directional.sort(key=lambda item: (-item["count"], item["true_model"], item["pred_model"]))
    symmetric.sort(key=lambda item: (-item["count"], item["model_a"], item["model_b"]))
    result.update({
        "rows": len(oof), "prompts": int(oof.prompt_id.nunique()),
        "probabilities_validated": any(column.startswith("p_") for column in oof),
        "validation_scope": "OOF schema/probabilities/prompt-fold consistency only; does not prove vectorizers were fit within training folds",
        "per_class_errors": per_class,
        "worst_five": per_class[:5],
        "top_directional_confusions": directional[:20],
        "top_symmetric_confusions": symmetric[:20],
        "family_diagnostics": family_report(matrix),
        "folds": [
            {"fold": int(fold), "rows": len(group), "prompts": int(group.prompt_id.nunique()),
             **subset_report(group.true_model, group.pred_model)}
            for fold, group in oof.groupby("fold", sort=True)
        ],
    })
    if "model_id" in oof:
        result["model_ids"] = sorted(oof.model_id.dropna().unique().tolist())
    if train is not None:
        result["length_quartiles"] = length_quartile_report(oof, train)
    if bootstrap_repeats:
        result["prompt_cluster_bootstrap"] = prompt_cluster_bootstrap(oof, bootstrap_repeats, seed)
    return result
