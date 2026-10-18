import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from .labels import LABELS


def classification_report(y_true, y_pred) -> dict:
    bad = (set(y_pred) | set(y_true)) - set(LABELS)
    if bad:
        raise ValueError(f"Illegal labels: {bad}")
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=LABELS)
    score = float(balanced_accuracy_score(y_true, y_pred))
    # Contest BA as mean per-class recall; all classes must be present in evaluation.
    if not set(LABELS).issubset(set(y_true)):
        raise ValueError("Evaluation split missing one of 18 labels")
    if not np.isclose(score, float(recalls.mean())):
        raise AssertionError("Mismatched balanced accuracy")
    return {
        "balanced_accuracy": score,
        "class_recall": {label: float(recall) for label, recall in zip(LABELS, recalls)},
        "class_support": {label: int(confusion[k, :].sum()) for k, label in enumerate(LABELS)},
        "confusion_matrix": confusion.tolist(),
        "labels": LABELS,
    }
