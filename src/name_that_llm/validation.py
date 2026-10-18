import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def group_folds(train: pd.DataFrame, n_splits: int = 5):
    cv = GroupKFold(n_splits=n_splits)
    groups = train["prompt_id"].astype(str).to_numpy()
    folds = np.full(len(train), -1, dtype=int)
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(train, train["model"], groups)):
        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise AssertionError("PROMPT LEAKAGE in GroupKFold")
        if np.any(folds[va_idx] != -1):
            raise AssertionError("OOF row assigned twice")
        folds[va_idx] = fold_idx
        yield fold_idx, tr_idx, va_idx
    assert (folds != -1).all(), "Some rows not assigned to a validation fold"


def fold_assignments(train: pd.DataFrame, n_splits: int = 5):
    assigned = np.full(len(train), -1, dtype=int)
    for i, _, va in group_folds(train, n_splits=n_splits):
        assigned[va] = i
    return assigned
