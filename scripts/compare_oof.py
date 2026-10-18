#!/usr/bin/env python3
"""Verify paired run provenance and report prompt-cluster bootstrap feature ablation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import name_that_llm.comparison as comparison
from name_that_llm.io import sha256_file
from name_that_llm.runtime import ResourceMonitor, environment_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="New JSON path; existing outputs are preserved")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite comparison output")
    if args.bootstrap_repeats <= 0:
        parser.error("--bootstrap-repeats must be positive")
    with ResourceMonitor() as monitor:
        baseline, baseline_oof = comparison.load_verified_run(args.baseline_manifest)
        candidate, candidate_oof = comparison.load_verified_run(args.candidate_manifest)
        provenance = comparison.verify_comparable_manifests(baseline, candidate)
        result = comparison.compare_paired_oof(baseline_oof, candidate_oof, args.bootstrap_repeats, args.seed)
    source_dir = Path(comparison.__file__).parent
    result.update({
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "provenance_verification": {**provenance, "artifact_hashes_verified": True, "stored_aggregate_and_fold_metrics_recomputed": True},
        "input_manifests": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                   "oof_sha256": manifest["artifacts"]["oof"]["sha256"],
                   "dataset_sha256": manifest["dataset_sha256"], "fold_assignments_sha256": manifest["fold_assignments_sha256"]}
            for name, path, manifest in (("baseline", args.baseline_manifest, baseline), ("candidate", args.candidate_manifest, candidate))
        },
        "code_sha256": {str(path): sha256_file(path) for path in [Path(__file__), *[source_dir / name for name in
                         ("comparison.py", "ablation.py", "analysis.py", "metrics.py", "labels.py", "io.py", "runtime.py")]]},
        "environment": environment_snapshot(), "resources": monitor.report(),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    interval = result["paired_prompt_cluster_bootstrap"]["delta_balanced_accuracy"]
    print(f"Candidate OOF BA: {result['candidate_balanced_accuracy']:.12f}")
    print(f"Paired delta: {result['delta_balanced_accuracy']:+.12f}; 95% CI [{interval['ci_lower']}, {interval['ci_upper']}]")
    print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
