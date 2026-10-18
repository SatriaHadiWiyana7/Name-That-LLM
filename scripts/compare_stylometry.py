#!/usr/bin/env python3
"""P3 paired prompt bootstrap with its own preregistered provenance guard."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from name_that_llm.comparison import load_verified_run, compare_paired_oof
from name_that_llm.io import sha256_file
from name_that_llm.runtime import ResourceMonitor, environment_snapshot
from name_that_llm.stylometry_validation import verify_stylometry_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite P3 comparison output")
    with ResourceMonitor() as monitor:
        baseline, baseline_oof = load_verified_run(args.baseline_manifest)
        candidate, candidate_oof = load_verified_run(args.candidate_manifest)
        provenance = verify_stylometry_comparison(baseline, candidate, args.baseline_manifest)
        result = compare_paired_oof(baseline_oof, candidate_oof, repeats=10000, seed=42)
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), *sorted((root / "src/name_that_llm").glob("*.py"))]
    result.update({
        "utc_created": datetime.now(timezone.utc).isoformat(),
        "provenance_verification": {**provenance, "artifact_hashes_verified": True,
                                    "stored_aggregate_and_fold_metrics_recomputed": True},
        "input_manifests": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
                   "oof_sha256": manifest["artifacts"]["oof"]["sha256"],
                   "dataset_sha256": manifest["dataset_sha256"],
                   "fold_assignments_sha256": manifest["fold_assignments_sha256"]}
            for name, path, manifest in (("baseline", args.baseline_manifest, baseline),
                                          ("candidate", args.candidate_manifest, candidate))
        },
        "code_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256_file(p) for p in source_paths},
        "environment": environment_snapshot(), "resources": monitor.report(),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    interval = result["paired_prompt_cluster_bootstrap"]["delta_balanced_accuracy"]
    print(f"P3 BA {result['candidate_balanced_accuracy']:.12f}; delta {result['delta_balanced_accuracy']:+.12f}")
    print(f"Paired 95% delta CI [{interval['ci_lower']}, {interval['ci_upper']}]")


if __name__ == "__main__":
    main()
