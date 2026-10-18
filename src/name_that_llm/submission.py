from pathlib import Path
import pandas as pd
from .labels import LABEL_SET


def validate_submission(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["id", "model"]:
        raise ValueError("Submission must contain exactly ['id', 'model'] columns in this order")
    if submission[["id", "model"]].isna().any().any():
        raise ValueError("Null submission ID/label")
    if submission.id.duplicated().any() or test.id.duplicated().any():
        raise ValueError("Duplicate submission/test ID")
    if len(submission) != len(test) or set(submission.id.astype(str)) != set(test.id.astype(str)):
        raise ValueError("Submitted ids differ from test ids")
    if not set(submission.model).issubset(LABEL_SET):
        raise ValueError(f"Illegal model labels: {set(submission.model)-LABEL_SET}")


def write_submission(test: pd.DataFrame, predictions, output: str | Path) -> pd.DataFrame:
    submission = pd.DataFrame({"id": test["id"].astype(str).to_numpy(), "model": predictions})
    validate_submission(test, submission)
    path = Path(output)
    path.parent.mkdir(exist_ok=True, parents=True)
    submission.to_parquet(path, index=False)
    return submission
