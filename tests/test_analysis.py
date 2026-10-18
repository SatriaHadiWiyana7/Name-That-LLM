import numpy as np
import pandas as pd
import pytest

from name_that_llm.analysis import analyze_oof, prompt_cluster_bootstrap, validate_oof
from name_that_llm.labels import LABELS


def make_oof():
    rows = []
    for prompt in range(2):
        for index, label in enumerate(LABELS):
            prediction = label if prompt == 0 else LABELS[(index + 1) % len(LABELS)]
            row = {"id": f"{prompt}_{index}", "prompt_id": f"p{prompt}", "fold": prompt,
                   "true_model": label, "pred_model": prediction}
            row.update({"p_" + target: float(target == prediction) for target in LABELS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_diagnostics_preserve_confusion_diagonal_and_error_counts():
    oof = make_oof()
    report = analyze_oof(oof)
    matrix = np.asarray(report["confusion_matrix"])
    assert report["balanced_accuracy"] == pytest.approx(0.5)
    assert matrix.trace() == 18
    assert matrix.sum() == len(oof)
    assert report["labels"] == LABELS
    assert len(report["worst_five"]) == 5
    for error in report["per_class_errors"]:
        assert (error["true_positive"], error["false_positive"], error["false_negative"]) == (1, 1, 1)
        assert error["top_confusion_targets"][0]["rate_of_true_class"] == 0.5
    assert report["family_diagnostics"]["within_family_errors"] + report["family_diagnostics"]["cross_family_errors"] == 18


@pytest.mark.parametrize("problem", ["duplicate_id", "leakage", "column_order", "missing_probability", "negative", "nan", "sum", "argmax"])
def test_invalid_oof_is_rejected(problem):
    oof = make_oof()
    if problem == "duplicate_id":
        oof.loc[1, "id"] = oof.loc[0, "id"]
    elif problem == "leakage":
        oof.loc[1, "fold"] = 1
    elif problem == "column_order":
        columns = list(oof.columns)
        columns[-1], columns[-2] = columns[-2], columns[-1]
        oof = oof[columns]
    elif problem == "missing_probability":
        oof = oof.drop(columns="p_" + LABELS[-1])
    elif problem == "negative":
        oof.loc[0, "p_" + LABELS[1]] = -0.1
    elif problem == "nan":
        oof.loc[0, "p_" + LABELS[0]] = np.nan
    elif problem == "sum":
        oof.loc[0, "p_" + LABELS[0]] = 0.5
    else:
        oof.loc[0, "pred_model"] = LABELS[1]
    with pytest.raises(ValueError):
        validate_oof(oof)


def test_bootstrap_draws_entire_prompts_and_recomputes_pooled_class_recall():
    oof = make_oof()
    # Unequal class supports make averaging per-prompt BA incorrect.
    duplicate = oof.iloc[[0]].copy()
    duplicate["id"] = "extra"
    oof = pd.concat([oof, duplicate], ignore_index=True)
    repeats, seed = 30, 19
    result = prompt_cluster_bootstrap(oof, repeats, seed)
    rng = np.random.default_rng(seed)
    expected = []
    for _ in range(repeats):
        good = int((rng.integers(0, 2, size=2) == 0).sum())
        bad = 2 - good
        expected.append((2 * good / (2 * good + bad) + 17 * good / 2) / 18)
    assert result["prompt_groups"] == result["groups_drawn_per_repeat"] == 2
    assert result["valid_repeats"] == repeats
    assert result["mean_balanced_accuracy"] == pytest.approx(np.mean(expected))
    assert result["ci_lower"] == pytest.approx(np.quantile(expected, 0.025))
    assert result["ci_upper"] == pytest.approx(np.quantile(expected, 0.975))
    assert result == prompt_cluster_bootstrap(oof, repeats, seed)


def test_length_quartiles_withhold_ba_for_missing_classes_and_check_alignment():
    oof = make_oof()
    train = oof[["id", "prompt_id", "true_model"]].rename(columns={"true_model": "model"})
    train["text"] = ["word " * (i + 1) for i in range(len(train))]
    report = analyze_oof(oof, train.iloc[::-1])
    assert sum(item["rows"] for item in report["length_quartiles"]["bins"]) == len(oof)
    for item in report["length_quartiles"]["bins"]:
        assert item["balanced_accuracy"] is None
        assert item["missing_labels"]
    assert "text" not in report
    train.loc[0, "model"] = LABELS[1]
    with pytest.raises(ValueError, match="labels do not match"):
        analyze_oof(oof, train)
