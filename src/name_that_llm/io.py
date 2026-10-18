from pathlib import Path
import hashlib
import pandas as pd
from .labels import LABEL_SET


def load_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "prompt_id": str, "text": str, "model": str})
    expected = ["id", "prompt_id", "text", "model"]
    if any(col not in df for col in expected):
        raise ValueError(f"Missing required training columns: {set(expected) - set(df.columns)}")
    if df[expected].isna().any().any():
        raise ValueError("Training contains nulls in required columns")
    if df.id.duplicated().any() or df[["prompt_id", "model"]].duplicated().any():
        raise ValueError("Duplicate IDs or duplicate (prompt_id, model) pairs")
    labels = set(df.model.unique())
    if labels != LABEL_SET:
        raise ValueError(f"Training labels not exactly 18 legal labels; missing={LABEL_SET-labels}, extra={labels-LABEL_SET}")
    if (df.text.str.len() == 0).any():
        raise ValueError("Empty training passage")
    return df


def load_test(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "text": str})
    if not {"id", "text"}.issubset(df.columns):
        raise ValueError("Test requires id,text columns")
    if df[["id", "text"]].isna().any().any() or df.id.duplicated().any():
        raise ValueError("Null test id/text or duplicate id")
    return df


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()
