#!/usr/bin/env python3
"""Prompt-grouped OOF baseline with measured resources and a full-train artifact."""

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from name_that_llm.baseline import make_baseline, ordered_probabilities
from name_that_llm.io import load_train, sha256_file
from name_that_llm.labels import LABELS
from name_that_llm.metrics import classification_report
from name_that_llm.runtime import (
    ResourceMonitor,
    environment_snapshot,
    reference_comparison,
    validate_fold_mapping,
)
from name_that_llm.validation import fold_assignments


REGISTER_FIELDS = [
    "experiment_id", "utc_date", "model", "split", "seed", "oof_balanced_accuracy",
    "delta_vs_baseline", "elapsed_seconds", "peak_vram_gb", "artifact_path", "notes",
]


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--out", default="artifacts/baseline")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--run-id", help="Unique experiment/model ID; defaults to config name")
    parser.add_argument("--folds", help="Reuse a complete id,prompt_id,fold CSV")
    parser.add_argument("--register", default="reports/experiments.csv")
    parser.add_argument("--environment-report", help="Optional local hardware audit JSON")
    parser.add_argument("--reference-metrics", help="Reference metrics; missing provenance means historical numerical difference only")
    return parser.parse_args()


def main():
    script_started = time.perf_counter()
    args = parse_args()
    train = load_train(args.train)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = args.run_id or cfg["name"]
    seed = int(cfg["random_seed"])
    n_splits = int(cfg["n_splits"])
    random.seed(seed)
    np.random.seed(seed)
    reports = Path(args.reports)
    out = Path(args.out)
    paths = {
        "metrics": reports / "baseline_metrics.json",
        "oof": reports / "baseline_oof.csv",
        "folds": reports / "baseline_fold_assignments.csv",
        "model": out / "model.joblib",
        "manifest": out / "manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite experiment outputs: " + ", ".join(existing))
    register = Path(args.register)
    if register.exists() and register.stat().st_size:
        with register.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != REGISTER_FIELDS:
                raise ValueError("Experiment register schema differs from the existing CSV schema")
            if any(row["experiment_id"] == run_id for row in reader):
                raise ValueError("Experiment run ID is already registered; choose a unique --run-id")
    reference = None
    if args.reference_metrics:
        reference = json.loads(Path(args.reference_metrics).read_text(encoding="utf-8"))
        if not 0 <= float(reference["balanced_accuracy"]) <= 1:
            raise ValueError("Reference balanced accuracy must be in [0,1]")
    environment = environment_snapshot()
    if args.environment_report:
        environment["external_report"] = {
            "path": str(Path(args.environment_report).resolve()),
            "sha256": sha256_file(args.environment_report),
            "report": json.loads(Path(args.environment_report).read_text(encoding="utf-8")),
        }
    if args.folds:
        mapping = pd.read_csv(args.folds, dtype={"id": str, "prompt_id": str})
    else:
        mapping = train[["id", "prompt_id"]].copy()
        mapping["fold"] = fold_assignments(train, n_splits=n_splits)
    folds = validate_fold_mapping(train, mapping, n_splits)
    reports.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    # Persist the immutable row-to-fold plan BEFORE fitting anything.
    pd.DataFrame({"id": train.id, "prompt_id": train.prompt_id, "fold": folds}).to_csv(paths["folds"], index=False)
    dataset_hash = sha256_file(args.train)
    fold_hash = sha256_file(paths["folds"])
    repo_root = Path(__file__).resolve().parents[1]
    source_paths = sorted(set(repo_root.joinpath("src").rglob("*.py")) | set(repo_root.joinpath("scripts").glob("*.py")))
    source_hashes = {str(path.relative_to(repo_root)).replace("\\", "/"): sha256_file(path) for path in source_paths}
    started_utc = datetime.now(timezone.utc).isoformat()
    texts = train.text.astype(str).to_list()
    targets = train.model.to_numpy()
    label_array = np.array(LABELS, dtype=object)
    oof = np.full((len(train), len(LABELS)), np.nan, dtype=np.float64)
    fold_metrics = []

    with ResourceMonitor() as run_monitor:
        cv_started = time.perf_counter()
        for fold in range(n_splits):
            validation_rows = np.flatnonzero(folds == fold)
            training_rows = np.flatnonzero(folds != fold)
            model = make_baseline(cfg)
            with ResourceMonitor() as fold_monitor:
                fit_started = time.perf_counter()
                model.fit([texts[index] for index in training_rows], targets[training_rows])
                fit_seconds = time.perf_counter() - fit_started
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
                "fit_seconds": round(fit_seconds, 6),
                "predict_seconds": round(predict_seconds, 6),
                "resources": resources,
            })
            fold_metrics.append(result)
            print(f"Fold {fold}: BA={result['balanced_accuracy']:.9f} time={resources['wall_seconds']:.2f}s peak RSS={resources['peak_rss_sampled_bytes']/1024**3:.3f} GiB", flush=True)
            del model
        if not np.isfinite(oof).all():
            raise AssertionError("Some rows lack valid OOF probabilities")
        predictions = label_array[oof.argmax(axis=1)]
        overall = classification_report(targets, predictions)
        cv_seconds = time.perf_counter() - cv_started
        overall.update({
            "folds": fold_metrics,
            "oof_elapsed_seconds": round(cv_seconds, 6),
            "experiment_id": run_id,
            "model_id": run_id,
            "config_name": cfg["name"],
            "random_seed": seed,
            "dataset_sha256": dataset_hash,
            "fold_assignments_sha256": fold_hash,
        })
        if reference is not None:
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
        for index, label in enumerate(LABELS):
            columns["p_" + label] = oof[:, index]
        pd.DataFrame(columns).to_csv(paths["oof"], index=False)

        print("Fit FINAL model on full training data...", flush=True)
        with ResourceMonitor() as final_monitor:
            full_model = make_baseline(cfg)
            final_started = time.perf_counter()
            full_model.fit(texts, targets)
            final_fit_seconds = time.perf_counter() - final_started
        with ResourceMonitor() as serialization_monitor:
            joblib.dump(full_model, paths["model"], compress=3)

    resources = run_monitor.report()
    resources["scope"] = "OOF training/evaluation, OOF CSV write, final refit and model serialization; excludes startup and final manifest/register writes"
    timings = {
        "oof_seconds": round(cv_seconds, 6),
        "oof_fit_seconds": round(sum(item["fit_seconds"] for item in fold_metrics), 6),
        "final_fit_seconds": round(final_fit_seconds, 6),
        "serialization_seconds": serialization_monitor.report()["wall_seconds"],
        "total_before_manifest_seconds": round(time.perf_counter() - script_started, 6),
    }
    overall["resources"] = resources
    overall["timings"] = timings
    write_json(paths["metrics"], overall)
    output_artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in paths.items() if name != "manifest"
    }
    manifest = {
        "experiment_id": run_id,
        "model_id": run_id,
        "utc_started": started_utc,
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "dataset_path": str(Path(args.train).resolve()),
        "config": cfg,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "random_seed": seed,
        "classifier_random_state": full_model.named_steps["clf"].random_state,
        "source_sha256": source_hashes,
        "labels": LABELS,
        "probability_columns": ["p_" + label for label in LABELS],
        "environment": environment,
        "validation": f"{n_splits}-fold GroupKFold(prompt_id)" if not args.folds else f"{n_splits}-fold saved prompt-grouped mapping",
        "fold_source": str(Path(args.folds).resolve()) if args.folds else "GroupKFold, shuffle=False",
        "fold_assignments_sha256": fold_hash,
        "oof_balanced_accuracy": overall["balanced_accuracy"],
        "oof_metrics_path": str(paths["metrics"].resolve()),
        "model_sha256": output_artifacts["model"]["sha256"],
        "artifacts": output_artifacts,
        "resources": resources,
        "final_fit_resources": final_monitor.report(),
        "serialization_resources": serialization_monitor.report(),
        "timings": timings,
        "serialization": "joblib; LOAD ONLY TRUSTED LOCAL ARTIFACTS",
    }
    if "reference_comparison" in overall:
        manifest["reference_comparison"] = overall["reference_comparison"]
    write_json(paths["manifest"], manifest)
    comparison = overall.get("reference_comparison", {})
    comparable_delta = comparison.get("delta_balanced_accuracy")
    entry = {
        "experiment_id": run_id,
        "utc_date": manifest["utc_created"],
        "model": "tfidf_word_char_logreg",
        "split": f"groupkfold{n_splits}_oof",
        "seed": seed,
        "oof_balanced_accuracy": f"{overall['balanced_accuracy']:.12f}",
        "delta_vs_baseline": "" if comparable_delta is None else f"{comparable_delta:+.12f}",
        "elapsed_seconds": timings["total_before_manifest_seconds"],
        "peak_vram_gb": "N/A (CPU; GPU memory not measured)",
        "artifact_path": str(paths["model"]),
        "notes": f"Measured local OOF; not leaderboard. Manifest={paths['manifest']}; command={environment['command']}; peak_RSS_bytes={resources['peak_rss_sampled_bytes']}; CV_seconds={timings['oof_seconds']}" + (f"; {comparison['interpretation']}" if comparison else ""),
    }
    register.parent.mkdir(parents=True, exist_ok=True)
    with register.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REGISTER_FIELDS)
        if register.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(entry)
    print(f"OOF balanced accuracy: {overall['balanced_accuracy']:.12f}", flush=True)
    print(f"Model: {paths['model']} | {output_artifacts['model']['bytes']/1048576:.2f} MiB", flush=True)
    print(f"Manifest: {paths['manifest']}", flush=True)


if __name__ == "__main__":
    main()
