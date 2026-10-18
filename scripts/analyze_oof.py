#!/usr/bin/env python3
"""Validate OOF outputs and save aggregate diagnostics without passages."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import name_that_llm.analysis as analysis
from name_that_llm.io import load_train, sha256_file
from name_that_llm.runtime import ResourceMonitor, environment_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="reports/baseline_oof.csv")
    parser.add_argument("--output", help="New JSON path for aggregate diagnostics; existing files are preserved")
    parser.add_argument("--train", help="Optional training CSV for word-length quartile diagnostics")
    parser.add_argument("--bootstrap-repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repeats < 0:
        parser.error("--bootstrap-repeats must be nonnegative")
    if args.output and Path(args.output).exists():
        parser.error("--output already exists; choose a new report path")
    with ResourceMonitor() as monitor:
        oof = pd.read_csv(args.oof, dtype={"id": str, "prompt_id": str})
        train = load_train(args.train) if args.train else None
        result = analysis.analyze_oof(oof, train, args.bootstrap_repeats, args.seed)
    source_dir = Path(analysis.__file__).parent
    result.update({
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "oof_path": str(Path(args.oof).resolve()),
        "oof_sha256": sha256_file(args.oof),
        "dataset_sha256": sha256_file(args.train) if args.train else None,
        "code_sha256": {
            str(path): sha256_file(path)
            for path in [Path(__file__), *[source_dir / name for name in
                         ("analysis.py", "metrics.py", "labels.py", "io.py", "runtime.py")]]
        },
        "environment": environment_snapshot(),
        "resources": monitor.report(),
    })
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(f"OOF balanced accuracy: {result['balanced_accuracy']:.6f}")
    print("\nWorst classes:")
    print(pd.DataFrame(result["per_class_errors"])[["model", "recall", "support"]].head(8).to_string(index=False))
    print("\nLargest directional confusions:")
    for pair in result["top_directional_confusions"]:
        print(f"{pair['true_model']:25} -> {pair['pred_model']:25} {pair['count']}")
    if args.bootstrap_repeats:
        bootstrap = result["prompt_cluster_bootstrap"]
        print(f"\nPrompt-cluster bootstrap 95% CI: {bootstrap['ci_lower']} to {bootstrap['ci_upper']} ({bootstrap['valid_repeats']} valid repeats)")
    if args.output:
        print(f"Diagnostics: {args.output}")


if __name__ == "__main__":
    main()
