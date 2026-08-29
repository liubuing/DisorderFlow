#!/usr/bin/env python
"""Re-evaluate frozen BFN scores with candidate-sensitive state compatibility.

The original independent BFN protocol used context-only ipTM, which has almost
no within-structure candidate variation. This post-failure diagnostic reuses
the already stored model outputs; it neither reruns the model nor replaces the
failed frozen ipTM endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_idp_ensemble_candidates_bfn import summarize
from scripts.score_idp_ensemble_matched_candidates import leave_one_out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(config_path, output_path=None):
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    input_path = ROOT / config["input"]["path"]
    if sha256(input_path) != config["input"]["sha256"]:
        raise ValueError("Frozen BFN score artifact hash mismatch")
    source = json.loads(input_path.read_text(encoding="ascii"))
    methods = list(source["method_summary"])
    component_ids = [row["component_id"] for row in source["component_scores"]]
    metric = "state_compatibility"
    rows = []
    ranges = {}
    for component in source["component_scores"]:
        native = np.asarray([row[metric] for row in component["native_scores"]])
        values = []
        for method in component["methods"]:
            sequences = [row["sequence"] for row in method["candidates"]]
            matrix = np.asarray([
                [score[metric] for score in candidate["conformer_scores"]]
                for candidate in method["candidates"]
            ], dtype=np.float64)
            values.extend(matrix.ravel().tolist())
            rows.extend(leave_one_out(
                component["component_id"], method["method"], sequences, matrix,
                native, config["endpoint"]["ensemble_aggregation"],
            ))
        # The range audit answers whether the endpoint can rank candidates at
        # all; statistical gates below answer whether that ranking helps.
        ranges[component["component_id"]] = max(values) - min(values)
    method_summary = summarize(
        rows, methods, component_ids, config["statistics"],
        config["development_gates"],
    )
    output = {
        "schema_version": 1,
        "status": (
            "state_compatibility_development_gate_passed"
            if all(row["passed"] for row in method_summary.values())
            else "state_compatibility_development_gate_failed"
        ),
        "classification": config["classification"],
        "endpoint": config["endpoint"],
        "provenance": {
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": sha256(config_path),
            "input": config["input"],
        },
        "candidate_score_range_by_component": ranges,
        "median_candidate_score_range": float(np.median(list(ranges.values()))),
        "method_summary": method_summary,
        "leave_one_conformer_out": rows,
        "decision": (
            "eligible_to_freeze_for_new_untouched_panel"
            if all(row["passed"] for row in method_summary.values())
            else "do_not_promote_state_compatibility_endpoint"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output_path = Path(output_path or ROOT / config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_bfn_state_compatibility_dev_v1.yml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = analyze(
        ROOT / args.config, ROOT / args.output if args.output else None
    )
    print(json.dumps({
        "status": output["status"],
        "median_candidate_score_range": output["median_candidate_score_range"],
        "method_summary": output["method_summary"],
        "decision": output["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
