"""Preregistered stylometry training, isolated from the preserved P0/P2 trainers."""

import argparse
import csv
import json
import random
import subprocess
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .baseline import ordered_probabilities
from .io import load_train, sha256_file
from .labels import LABELS
from .metrics import classification_report
from .runtime import ResourceMonitor, environment_snapshot, reference_comparison, validate_fold_mapping
from .stylometry import FEATURE_NAMES, feature_schema, make_stylometry
from .stylometry_validation import validate_stylometry_reference


REGISTER_FIELDS = [
    "experiment_id", "utc_date", "model", "split", "seed", "oof_balanced_accuracy",
    "delta_vs_baseline", "elapsed_seconds", "peak_vram_gb", "artifact_path", "notes",
]
MODEL_KIND = "stylometry_standard_scaler_logreg"
RESOURCE_BUDGET = {
    "wall_limit_seconds": 600,
    "peak_rss_limit_bytes": 4 * 1024**3,
    "policy": "Check after each completed fold and final refit; no asynchronous interruption of an active fit",
}


class StylometryConvergenceError(RuntimeError):
    """A fitted classifier warned; retain diagnostics but never predict with it."""

    def __init__(self, diagnostics):
        super().__init__("LogisticRegression emitted ConvergenceWarning; stop without tuning or predictions")
        self.diagnostics = diagnostics


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_checked(model, texts, targets) -> dict:
    """Fit once, capture every warning and refuse a nonconverged result."""
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(texts, targets)
    diagnostics = {
        "fit_seconds": round(time.perf_counter() - started, 6),
        "n_iter": np.asarray(model.named_steps["clf"].n_iter_).astype(int).tolist(),
        "warnings": [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in captured
        ],
        "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in captured),
    }
    if diagnostics["convergence_warning"]:
        raise StylometryConvergenceError(diagnostics)
    return diagnostics


def scaler_statistics(model, train_rows: int) -> dict:
    """Persist the fitted scaler's sufficient statistics in feature column order."""
    scaler = model.named_steps["scale"]
    seen = np.asarray(scaler.n_samples_seen_)
    if not np.all(seen == train_rows):
        raise AssertionError("Scaler sample count differs from fold training rows")
    names = list(model.named_steps["features"].get_feature_names_out())
    if names != list(FEATURE_NAMES):
        raise AssertionError("Extractor feature order differs from the frozen schema")
    for values in (scaler.mean_, scaler.var_, scaler.scale_):
        if len(values) != len(names) or not np.isfinite(values).all():
            raise AssertionError("Scaler statistics must be finite and match all feature columns")
    return {
        "train_rows": int(train_rows),
        "n_samples_seen": int(seen) if seen.ndim == 0 else seen.astype(int).tolist(),
        "n_features_in": int(scaler.n_features_in_),
        "feature_names": names,
        "mean": scaler.mean_.tolist(),
        "var": scaler.var_.tolist(),
        "scale": scaler.scale_.tolist(),
        "scope": "Fitted only on this phase's training rows; no validation rows used",
    }


def check_resource_budget(elapsed_seconds: float, peak_rss_bytes: int) -> None:
    if elapsed_seconds > RESOURCE_BUDGET["wall_limit_seconds"] or peak_rss_bytes > RESOURCE_BUDGET["peak_rss_limit_bytes"]:
        raise RuntimeError("Preregistered resource investigation threshold exceeded; stop before further fitting")


def reserve_outputs(paths: dict, register: Path, run_id: str) -> None:
    """Reject any prior output or reused ID before fitting or writing anything."""
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Report and artifact output paths must be distinct")
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite experiment outputs: " + ", ".join(existing))
    if register.exists() and register.stat().st_size:
        with register.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != REGISTER_FIELDS:
                raise ValueError("Experiment register schema differs from the existing CSV schema")
            if any(row["experiment_id"] == run_id for row in reader):
                raise ValueError("Experiment run ID is already registered; choose a unique --run-id")


def source_inventory(repo_root: Path) -> dict:
    paths = sorted(set(repo_root.joinpath("src").rglob("*.py")) | set(repo_root.joinpath("scripts").glob("*.py")))
    return {path.relative_to(repo_root).as_posix(): sha256_file(path) for path in paths}


def package_inference_source(repo_root: Path, destination: Path) -> None:
    """Copy importable Python source with fixed ZIP metadata, no data or weights."""
    package_root = repo_root / "src" / "name_that_llm"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.glob("*.py")):
            entry = zipfile.ZipInfo("name_that_llm/" + path.name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, path.read_bytes())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--config", default="configs/p3_stylometry.json")
    parser.add_argument("--plan", default="reports/p3_20261018_plan.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--folds", required=True, help="Exact saved P0 id,prompt_id,fold CSV")
    parser.add_argument("--register", default="reports/experiments.csv")
    parser.add_argument("--environment-report")
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--reference-manifest", required=True)
    return parser.parse_args(argv)


def run_experiment(args) -> dict:
    script_started = time.perf_counter()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id
    seed, n_splits = int(cfg["random_seed"]), int(cfg["n_splits"])
    reports, out = Path(args.reports), Path(args.out)
    paths = {
        "metrics": reports / "baseline_metrics.json",
        "oof": reports / "baseline_oof.csv",
        "folds": reports / "baseline_fold_assignments.csv",
        "config": reports / "config.json",
        "feature_schema": reports / "feature_schema.json",
        "scaler_statistics": reports / "scaler_statistics.json",
        "model": out / "model.joblib",
        "inference_source": out / "inference_source.zip",
        "manifest": out / "manifest.json",
        "report_manifest": reports / "manifest.json",
        "failure": reports / "failure.json",
    }
    register = Path(args.register)
    reserve_outputs(paths, register, run_id)
    train = load_train(args.train)
    reference = json.loads(Path(args.reference_metrics).read_text(encoding="utf-8"))
    environment = environment_snapshot()
    if args.environment_report:
        environment["external_report"] = {
            "path": str(Path(args.environment_report).resolve()),
            "sha256": sha256_file(args.environment_report),
            "report": json.loads(Path(args.environment_report).read_text(encoding="utf-8")),
        }
    mapping = pd.read_csv(args.folds, dtype={"id": str, "prompt_id": str})
    folds = validate_fold_mapping(train, mapping, n_splits)
    dataset_hash = sha256_file(args.train)
    guard = validate_stylometry_reference(
        config=cfg, environment=environment, dataset_sha256=dataset_hash,
        folds_path=args.folds, reference_manifest_path=args.reference_manifest,
        reference_metrics_path=args.reference_metrics, plan_path=args.plan,
    )
    random.seed(seed)
    np.random.seed(seed)
    reports.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for key, original in (("config", args.config), ("folds", args.folds)):
        with paths[key].open("xb") as stream:
            stream.write(Path(original).read_bytes())
    fold_hash = sha256_file(paths["folds"])
    if fold_hash != sha256_file(args.folds):
        raise ValueError("Persisted fold SHA256 differs from the P0 source")
    write_json(paths["feature_schema"], feature_schema())
    repo_root = Path(__file__).resolve().parents[2]
    source_hashes = source_inventory(repo_root)
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False)
    git_commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    started_utc = datetime.now(timezone.utc).isoformat()
    texts, targets = train.text.to_list(), train.model.to_numpy()
    label_array = np.array(LABELS, dtype=object)
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float64)
    fold_metrics, fold_scalers, fits = [], [], []
    phase = "initialization"

    try:
        with ResourceMonitor() as run_monitor:
            with ResourceMonitor() as cv_monitor:
                for fold in range(n_splits):
                    phase = f"fold_{fold}"
                    validation_rows = np.flatnonzero(folds == fold)
                    training_rows = np.flatnonzero(folds != fold)
                    model = make_stylometry(cfg)
                    with ResourceMonitor() as fold_monitor:
                        diagnostics = fit_checked(model, [texts[index] for index in training_rows], targets[training_rows])
                        fits.append({"phase": phase, **diagnostics})
                        fold_scalers.append({"fold": fold, **scaler_statistics(model, len(training_rows))})
                        predict_started = time.perf_counter()
                        oof[validation_rows] = ordered_probabilities(model, [texts[index] for index in validation_rows])
                        predict_seconds = time.perf_counter() - predict_started
                        result = classification_report(targets[validation_rows], label_array[oof[validation_rows].argmax(axis=1)])
                    resources = fold_monitor.report()
                    result.update({
                        "fold": fold,
                        "train_rows": len(training_rows),
                        "train_prompts": int(train.iloc[training_rows].prompt_id.nunique()),
                        "val_rows": len(validation_rows),
                        "val_prompts": int(train.iloc[validation_rows].prompt_id.nunique()),
                        "seconds": resources["wall_seconds"],
                        "fit_seconds": diagnostics["fit_seconds"],
                        "predict_seconds": round(predict_seconds, 6),
                        "fit_diagnostics": diagnostics,
                        "resources": resources,
                    })
                    fold_metrics.append(result)
                    print(f"Fold {fold}: BA={result['balanced_accuracy']:.9f} time={resources['wall_seconds']:.2f}s peak RSS={resources['peak_rss_sampled_bytes']/1024**3:.3f} GiB n_iter={diagnostics['n_iter']}", flush=True)
                    check_resource_budget(time.perf_counter() - run_monitor.wall_started, max(run_monitor.peak_rss_bytes, resources["peak_rss_sampled_bytes"]))
                    del model
                if not np.isfinite(oof).all():
                    raise AssertionError("Some rows lack valid OOF probabilities")
                predictions = label_array[oof.argmax(axis=1)]
                overall = classification_report(targets, predictions)
            cv_seconds = cv_monitor.report()["wall_seconds"]
            overall.update({
                "folds": fold_metrics,
                "oof_elapsed_seconds": cv_seconds,
                "experiment_id": run_id, "model_id": run_id, "config_name": cfg["name"],
                "feature_groups": ["stylometry"], "model": MODEL_KIND,
                "random_seed": seed, "dataset_sha256": dataset_hash,
                "fold_assignments_sha256": fold_hash,
            })
            overall["reference_comparison"] = reference_comparison(overall, reference)
            overall["reference_comparison"].update({
                "path": str(Path(args.reference_metrics).resolve()),
                "sha256": sha256_file(args.reference_metrics),
            })
            columns = {
                "id": train.id.to_numpy(), "prompt_id": train.prompt_id.to_numpy(),
                "true_model": targets, "pred_model": predictions, "fold": folds,
                "model_id": np.full(len(train), run_id, dtype=object),
            }
            columns.update({"p_" + label: oof[:, index] for index, label in enumerate(LABELS)})
            with paths["oof"].open("x", newline="", encoding="utf-8") as stream:
                pd.DataFrame(columns).to_csv(stream, index=False)
            phase = "final_fit"
            print("Fit FINAL stylometry model on full training data...", flush=True)
            with ResourceMonitor() as final_monitor:
                full_model = make_stylometry(cfg)
                final_diagnostics = fit_checked(full_model, texts, targets)
                fits.append({"phase": phase, **final_diagnostics})
                final_scaler = scaler_statistics(full_model, len(train))
            check_resource_budget(time.perf_counter() - run_monitor.wall_started, max(run_monitor.peak_rss_bytes, final_monitor.report()["peak_rss_sampled_bytes"]))
            write_json(paths["scaler_statistics"], {
                "feature_names": list(FEATURE_NAMES), "folds": fold_scalers,
                "final_fit": final_scaler,
            })
            phase = "serialization"
            with ResourceMonitor() as serialization_monitor:
                with paths["model"].open("xb") as stream:
                    joblib.dump(full_model, stream, compress=3)
            package_inference_source(repo_root, paths["inference_source"])
    except Exception as error:
        failure = {
            "experiment_id": run_id, "status": "failed_no_promotion", "phase": phase,
            "utc_created": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(error).__name__, "message": str(error),
            "fit_diagnostics": fits,
            "failed_fit_diagnostics": getattr(error, "diagnostics", None),
            "completed_fold_metrics": fold_metrics, "completed_fold_scalers": fold_scalers,
            "resources": run_monitor.report(), "environment": environment,
            "dataset_sha256": dataset_hash, "fold_assignments_sha256": fold_hash,
            "config": cfg, "config_sha256": sha256_file(args.config),
            "source_sha256": source_hashes, "git_commit": git_commit,
            "stylometry_validation": guard, "resource_budget": RESOURCE_BUDGET,
            "note": "Partial outputs preserved; choose a fresh run ID after investigation. No hyperparameter retry occurred.",
        }
        write_json(paths["failure"], failure)
        raise

    resources = run_monitor.report()
    resources["scope"] = "OOF training/evaluation/write, full refit, scaler diagnostics, model serialization and inference source packaging; excludes startup and final manifest/register writes"
    timings = {
        "oof_seconds": cv_seconds,
        "oof_fit_seconds": round(sum(item["fit_seconds"] for item in fold_metrics), 6),
        "final_fit_seconds": final_diagnostics["fit_seconds"],
        "serialization_seconds": serialization_monitor.report()["wall_seconds"],
        "total_before_manifest_seconds": round(time.perf_counter() - script_started, 6),
    }
    overall.update({"resources": resources, "timings": timings, "fit_diagnostics": fits,
                    "cv_resources": cv_monitor.report(), "final_fit_resources": final_monitor.report()})
    write_json(paths["metrics"], overall)
    output_artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in paths.items() if name not in ("manifest", "report_manifest", "failure")
    }
    manifest = {
        "experiment_id": run_id, "model_id": run_id, "model": MODEL_KIND,
        "feature_groups": ["stylometry"], "feature_count": len(FEATURE_NAMES),
        "utc_started": started_utc, "utc_created": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash, "dataset_path": str(Path(args.train).resolve()),
        "config": cfg, "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config), "random_seed": seed,
        "classifier_random_state": full_model.named_steps["clf"].random_state,
        "source_sha256": source_hashes, "git_commit": git_commit,
        "labels": LABELS, "probability_columns": ["p_" + label for label in LABELS],
        "environment": environment, "validation": f"{n_splits}-fold saved prompt-grouped mapping",
        "fold_source": str(Path(args.folds).resolve()), "fold_assignments_sha256": fold_hash,
        "oof_balanced_accuracy": overall["balanced_accuracy"],
        "oof_metrics_path": str(paths["metrics"].resolve()),
        "model_sha256": output_artifacts["model"]["sha256"], "artifacts": output_artifacts,
        "resources": resources, "cv_resources": cv_monitor.report(),
        "final_fit_resources": final_monitor.report(),
        "serialization_resources": serialization_monitor.report(), "timings": timings,
        "fit_diagnostics": fits, "stylometry_validation": guard,
        "resource_budget": RESOURCE_BUDGET,
        "reference_comparison": overall["reference_comparison"],
        "serialization": "joblib; LOAD ONLY TRUSTED LOCAL ARTIFACTS",
        "offline_inference": "Attach inference_source.zip plus model.joblib and matching dependencies; extract source or place ZIP on sys.path before joblib.load. No network calls required.",
    }
    write_json(paths["manifest"], manifest)
    write_json(paths["report_manifest"], manifest)
    comparison = overall["reference_comparison"]
    delta = comparison.get("delta_balanced_accuracy")
    entry = {
        "experiment_id": run_id, "utc_date": manifest["utc_created"], "model": MODEL_KIND,
        "split": f"groupkfold{n_splits}_oof", "seed": seed,
        "oof_balanced_accuracy": f"{overall['balanced_accuracy']:.12f}",
        "delta_vs_baseline": "" if delta is None else f"{delta:+.12f}",
        "elapsed_seconds": timings["total_before_manifest_seconds"],
        "peak_vram_gb": "N/A (CPU; GPU memory not measured)",
        "artifact_path": str(paths["model"]),
        "notes": f"Measured local OOF; not leaderboard. Manifest={paths['manifest']}; command={environment['command']}; peak_RSS_bytes={resources['peak_rss_sampled_bytes']}; CV_seconds={cv_seconds}; {comparison['interpretation']}",
    }
    register.parent.mkdir(parents=True, exist_ok=True)
    with register.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTER_FIELDS)
        if register.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(entry)
    print(f"OOF balanced accuracy: {overall['balanced_accuracy']:.12f}", flush=True)
    print(f"Model: {paths['model']} | {output_artifacts['model']['bytes']/1048576:.4f} MiB", flush=True)
    print(f"Manifest: {paths['manifest']}", flush=True)
    return manifest


def main(argv=None):
    run_experiment(parse_args(argv))
