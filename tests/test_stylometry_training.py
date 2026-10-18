import argparse
import csv
import json
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from name_that_llm.baseline import ordered_probabilities
from name_that_llm.comparison import load_verified_run
from name_that_llm.io import sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.metrics import classification_report
from name_that_llm.stylometry import FEATURE_NAMES, StylometryExtractor
from name_that_llm import stylometry_training as training


def synthetic_inputs(tmp_path):
    frame = pd.DataFrame([
        {
            "id": f"p{prompt}-{author}", "prompt_id": f"p{prompt}", "model": label,
            "text": ("Word " * (author + 1)) + ("!" * (author % 4)) + f" sample {prompt}.",
        }
        for prompt in range(5) for author, label in enumerate(LABELS)
    ])
    train_path = tmp_path / "train.csv"
    frame.to_csv(train_path, index=False)
    mapping = frame[["id", "prompt_id"]].copy()
    mapping["fold"] = np.repeat(np.arange(5), len(LABELS))
    folds_path = tmp_path / "reference-folds.csv"
    mapping.to_csv(folds_path, index=False)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/p3_stylometry.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    reference = classification_report(frame.model, np.roll(frame.model.to_numpy(), 1))
    reference.update({
        "dataset_sha256": sha256_file(train_path),
        "fold_assignments_sha256": sha256_file(folds_path), "random_seed": 42,
        "folds": [{"fold": i, "balanced_accuracy": reference["balanced_accuracy"]} for i in range(5)],
    })
    reference_path = tmp_path / "reference-metrics.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    plan = tmp_path / "synthetic-plan.json"
    plan.write_text("{}", encoding="utf-8")
    reference_manifest = tmp_path / "reference-manifest.json"
    reference_manifest.write_text("{}", encoding="utf-8")
    return frame, argparse.Namespace(
        train=str(train_path), config=str(config_path), plan=str(plan),
        folds=str(folds_path), reference_metrics=str(reference_path),
        reference_manifest=str(reference_manifest), environment_report=None,
        register=str(tmp_path / "register.csv"), run_id="synthetic-style-1",
        reports=str(tmp_path / "run1" / "reports"), out=str(tmp_path / "run1" / "artifacts"),
    )


def test_full_training_preserves_folds_scaler_isolation_and_repeat_artifacts(tmp_path, monkeypatch):
    frame, args = synthetic_inputs(tmp_path)
    calls = []

    def synthetic_guard(**kwargs):
        calls.append(kwargs)
        return {"status": "passed_before_fitting", "scope": "Synthetic test only; production guard tested separately"}

    monkeypatch.setattr(training, "validate_stylometry_reference", synthetic_guard)
    first = training.run_experiment(args)
    first_manifest = Path(args.out) / "manifest.json"
    verified, first_oof = load_verified_run(first_manifest)
    assert verified == first
    assert first["labels"] == LABELS
    assert first["feature_count"] == 81
    assert first["stylometry_validation"]["status"] == "passed_before_fitting"
    assert first_manifest.read_bytes() == (Path(args.reports) / "manifest.json").read_bytes()
    assert (Path(args.reports) / "config.json").read_bytes() == Path(args.config).read_bytes()
    assert (Path(args.reports) / "baseline_fold_assignments.csv").read_bytes() == Path(args.folds).read_bytes()
    assert calls[0]["dataset_sha256"] == sha256_file(args.train)
    assert calls[0]["plan_path"] == args.plan
    assert len(first["fit_diagnostics"]) == 6
    assert all(not item["convergence_warning"] for item in first["fit_diagnostics"])
    assert all(item["n_iter"] for item in first["fit_diagnostics"])
    assert not (Path(args.reports) / "failure.json").exists()
    for key in ("resources", "cv_resources", "final_fit_resources", "serialization_resources"):
        assert first[key]["wall_seconds"] >= 0
        assert first[key]["cpu_seconds"] >= 0
        assert first[key]["peak_rss_sampled_bytes"] > 0
        assert first[key]["gpu_used"] is False
        assert first[key]["peak_vram_bytes"] is None
    stats = json.loads(Path(first["artifacts"]["scaler_statistics"]["path"]).read_text())
    features = StylometryExtractor().fit_transform(frame.text.tolist())
    for fold_stat in stats["folds"]:
        mask = np.repeat(np.arange(5), len(LABELS)) != fold_stat["fold"]
        assert fold_stat["train_rows"] == fold_stat["n_samples_seen"] == int(mask.sum())
        assert fold_stat["feature_names"] == list(FEATURE_NAMES)
        np.testing.assert_allclose(fold_stat["mean"], features[mask].mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(fold_stat["var"], features[mask].var(axis=0), atol=1e-12)
    assert stats["final_fit"]["n_samples_seen"] == len(frame)
    np.testing.assert_allclose(stats["final_fit"]["mean"], features.mean(axis=0), atol=1e-12)
    loaded = joblib.load(first["artifacts"]["model"]["path"])
    probabilities = ordered_probabilities(loaded, ["unseen sample?!"])
    assert probabilities.shape == (1, len(LABELS))
    with zipfile.ZipFile(first["artifacts"]["inference_source"]["path"]) as archive:
        assert "name_that_llm/stylometry.py" in archive.namelist()
        assert all(name.endswith(".py") for name in archive.namelist())
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    args.run_id = "synthetic-style-2"
    args.reports = str(tmp_path / "run2" / "reports")
    args.out = str(tmp_path / "run2" / "artifacts")
    second = training.run_experiment(args)
    _, second_oof = load_verified_run(Path(args.out) / "manifest.json")
    assert first["model_sha256"] == second["model_sha256"]
    assert first["artifacts"]["inference_source"]["sha256"] == second["artifacts"]["inference_source"]["sha256"]
    pd.testing.assert_frame_equal(first_oof.drop(columns="model_id"), second_oof.drop(columns="model_id"))
    with Path(args.register).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["experiment_id"] for row in rows] == ["synthetic-style-1", "synthetic-style-2"]
    assert all(row["model"] == training.MODEL_KIND for row in rows)


def test_guard_rejection_occurs_before_outputs_and_fit(tmp_path, monkeypatch):
    _, args = synthetic_inputs(tmp_path)

    def reject(**kwargs):
        raise ValueError("synthetic provenance mismatch")

    monkeypatch.setattr(training, "validate_stylometry_reference", reject)
    monkeypatch.setattr(training, "make_stylometry", lambda cfg: pytest.fail("Must not construct a model after failed guard"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        training.run_experiment(args)
    assert not Path(args.out).exists()
    assert not Path(args.reports).exists()
    assert not Path(args.register).exists()


@pytest.mark.parametrize("filename", ["failure.json", "config.json", "baseline_oof.csv"])
def test_partial_output_refused_before_training_data_is_loaded(tmp_path, monkeypatch, filename):
    _, args = synthetic_inputs(tmp_path)
    report_dir = Path(args.reports)
    report_dir.mkdir(parents=True)
    sentinel = report_dir / filename
    sentinel.write_bytes(b"preserved")
    monkeypatch.setattr(training, "load_train", lambda path: pytest.fail("Existing output must be checked first"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        training.run_experiment(args)
    assert sentinel.read_bytes() == b"preserved"


def test_duplicate_register_id_and_output_alias_are_rejected(tmp_path):
    register = tmp_path / "experiments.csv"
    with register.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=training.REGISTER_FIELDS)
        writer.writeheader()
        writer.writerow({"experiment_id": "already-used"})
    with pytest.raises(ValueError, match="already registered"):
        training.reserve_outputs({"one": tmp_path / "new-file"}, register, "already-used")
    with pytest.raises(ValueError, match="distinct"):
        training.reserve_outputs({"one": tmp_path / "same", "two": tmp_path / "same"}, register, "new")


def test_nonconvergence_preserves_failure_resources_and_never_predicts(tmp_path, monkeypatch):
    _, args = synthetic_inputs(tmp_path)
    monkeypatch.setattr(training, "validate_stylometry_reference", lambda **kwargs: {"status": "passed_before_fitting"})

    class NonconvergentModel:
        named_steps = {"clf": SimpleNamespace(n_iter_=np.array([1200]))}

        def fit(self, texts, targets):
            warnings.warn("Synthetic nonconvergence", ConvergenceWarning)
            return self

        def predict_proba(self, texts):
            pytest.fail("Nonconverged model must not predict")

    monkeypatch.setattr(training, "make_stylometry", lambda cfg: NonconvergentModel())
    with pytest.raises(training.StylometryConvergenceError):
        training.run_experiment(args)
    failure = json.loads((Path(args.reports) / "failure.json").read_text())
    assert failure["phase"] == "fold_0"
    assert failure["status"] == "failed_no_promotion"
    assert failure["failed_fit_diagnostics"]["n_iter"] == [1200]
    assert failure["failed_fit_diagnostics"]["convergence_warning"] is True
    assert failure["resources"]["peak_rss_sampled_bytes"] > 0
    assert failure["resources"]["cpu_seconds"] >= 0
    assert not Path(args.register).exists()
    assert not (Path(args.out) / "model.joblib").exists()
    assert not (Path(args.reports) / "baseline_oof.csv").exists()
    assert not (Path(args.reports) / "manifest.json").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        training.run_experiment(args)


def test_scaler_sample_count_mismatch_rejected():
    model = SimpleNamespace(named_steps={"scale": SimpleNamespace(n_samples_seen_=100)})
    with pytest.raises(AssertionError, match="training rows"):
        training.scaler_statistics(model, train_rows=80)


@pytest.mark.parametrize("wall,rss", [(600.001, 1024), (1, 4 * 1024**3 + 1)])
def test_resource_investigation_threshold_stops_further_training(wall, rss):
    training.check_resource_budget(600, 4 * 1024**3)
    with pytest.raises(RuntimeError, match="threshold exceeded"):
        training.check_resource_budget(wall, rss)


def test_cli_requires_saved_folds_and_defaults_to_preregistered_config():
    common = ["--out", "new-artifacts", "--reports", "new-reports", "--run-id", "new-run",
              "--reference-metrics", "p0-metrics", "--reference-manifest", "p0-manifest"]
    with pytest.raises(SystemExit):
        training.parse_args(common)
    args = training.parse_args(common + ["--folds", "p0-folds"])
    assert args.config == "configs/p3_stylometry.json"
    assert args.plan == "reports/p3_20261018_plan.json"
