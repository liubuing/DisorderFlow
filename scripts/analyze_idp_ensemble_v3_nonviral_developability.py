#!/usr/bin/env python
"""Attach explicitly heuristic developability proxies to nonviral candidates."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.state_specificity_scorer import (
    immunogenicity_proxy,
    sequence_complexity,
)


def analyze(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite developability proxy: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    subset = json.loads(
        (ROOT / config["inputs"]["subset"]).read_text(encoding="ascii")
    )
    qc = json.loads(
        (ROOT / config["inputs"]["sequence_qc"]).read_text(encoding="ascii")
    )
    native = {
        component_id: component["native_h3"]
        for component_id, component in subset["components"].items()
    }
    thresholds = config["thresholds"]
    records = []
    for row in qc["records"]:
        sequence = row["sequence"]
        complexity = sequence_complexity(sequence)
        immunogenicity = immunogenicity_proxy(sequence)
        introduced_cysteine = "C" in sequence and "C" not in native[row["component_id"]]
        flags = []
        if complexity < thresholds["minimum_sequence_complexity"]:
            flags.append("low_sequence_complexity")
        if immunogenicity > thresholds["maximum_immunogenicity_proxy"]:
            flags.append("high_immunogenicity_proxy")
        if introduced_cysteine:
            flags.append("introduced_cysteine")
        records.append({
            "component_id": row["component_id"],
            "target": row["target"],
            "arm": row["arm"],
            "substitution_bucket": row["substitution_bucket"],
            "sequence": sequence,
            "sequence_complexity": round(complexity, 6),
            "immunogenicity_proxy": round(immunogenicity, 6),
            "introduced_cysteine": introduced_cysteine,
            "proxy_flags": flags,
            "proxy_pass": not flags,
        })
    arm_summary = {}
    for arm in ("ensemble", "single_state"):
        rows = [row for row in records if row["arm"] == arm]
        arm_summary[arm] = {
            "candidate_count": len(rows),
            "proxy_pass_count": sum(row["proxy_pass"] for row in rows),
            "low_complexity_count": sum(
                "low_sequence_complexity" in row["proxy_flags"] for row in rows
            ),
            "high_immunogenicity_proxy_count": sum(
                "high_immunogenicity_proxy" in row["proxy_flags"] for row in rows
            ),
            "introduced_cysteine_count": sum(
                row["introduced_cysteine"] for row in rows
            ),
            "median_sequence_complexity": statistics.median(
                row["sequence_complexity"] for row in rows
            ),
            "median_immunogenicity_proxy": statistics.median(
                row["immunogenicity_proxy"] for row in rows
            ),
        }
    payload = {
        "schema_version": 1,
        "status": "nonviral_developability_proxy_complete",
        "classification": config["classification"],
        "config": str(config_path),
        "candidate_count": len(records),
        "proxy_pass_count": sum(row["proxy_pass"] for row in records),
        "flagged_count": sum(not row["proxy_pass"] for row in records),
        "arm_summary": arm_summary,
        "records": records,
        "decision": "use_as_risk_flags_only_not_arm_ranking",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/benchmarks/idp_ensemble_v3_nonviral_developability.yml"),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = analyze(config_path, ROOT / config["output"])
    print(json.dumps({
        "status": result["status"],
        "candidates": result["candidate_count"],
        "proxy_pass": result["proxy_pass_count"],
        "flagged": result["flagged_count"],
        "arm_summary": result["arm_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
