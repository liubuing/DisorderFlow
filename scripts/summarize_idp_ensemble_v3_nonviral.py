#!/usr/bin/env python
"""Summarize nonviral sequence-QC warnings and arm divergence without claims."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def warning_count(row):
    return sum(bool(row[key]) for key in (
        "n_linked_glycosylation_motifs",
        "deamidation_motifs",
        "isomerization_motifs",
        "oxidation_residues",
    ))


def hamming(left, right):
    if len(left) != len(right):
        raise ValueError("Matched H3 sequences have different lengths")
    return sum(a != b for a, b in zip(left, right, strict=True))


def summarize(analysis_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite nonviral summary: {output}")
    analysis = json.loads(analysis_path.read_text(encoding="ascii"))
    records = analysis["records"]
    arm_summary = {}
    for arm in ("ensemble", "single_state"):
        rows = [row for row in records if row["arm"] == arm]
        arm_summary[arm] = {
            "candidate_count": len(rows),
            "hard_qc_pass_count": sum(row["qc_pass"] for row in rows),
            "glycosylation_warning_count": sum(
                bool(row["n_linked_glycosylation_motifs"]) for row in rows
            ),
            "deamidation_warning_count": sum(
                bool(row["deamidation_motifs"]) for row in rows
            ),
            "isomerization_warning_count": sum(
                bool(row["isomerization_motifs"]) for row in rows
            ),
            "oxidation_warning_count": sum(
                bool(row["oxidation_residues"]) for row in rows
            ),
            "median_warning_categories": statistics.median(
                warning_count(row) for row in rows
            ),
            "median_hydrophobic_fraction": statistics.median(
                row["hydrophobic_fraction"] for row in rows
            ),
            "median_charge_proxy": statistics.median(
                row["charge_proxy"] for row in rows
            ),
        }
    matched = defaultdict(dict)
    for row in records:
        key = (row["component_id"], row["substitution_bucket"])
        matched[key][row["arm"]] = row
    contrasts = []
    for (component_id, bucket), arms in sorted(matched.items()):
        if set(arms) != {"ensemble", "single_state"}:
            raise RuntimeError("Matched arm pair is incomplete")
        ensemble = arms["ensemble"]
        single = arms["single_state"]
        contrasts.append({
            "component_id": component_id,
            "substitution_bucket": bucket,
            "ensemble_single_hamming_distance": hamming(
                ensemble["sequence"], single["sequence"]
            ),
            "ensemble_warning_categories": warning_count(ensemble),
            "single_state_warning_categories": warning_count(single),
            "warning_category_difference": (
                warning_count(ensemble) - warning_count(single)
            ),
        })
    payload = {
        "schema_version": 1,
        "status": "nonviral_computational_summary_complete",
        "classification": "nonviral_general_antibody_computational_exploration",
        "analysis": str(analysis_path),
        "component_count": len({row["component_id"] for row in records}),
        "target_count": len({row["target"] for row in records}),
        "candidate_count": len(records),
        "arm_summary": arm_summary,
        "matched_bucket_contrasts": contrasts,
        "median_cross_arm_hamming_distance": statistics.median(
            row["ensemble_single_hamming_distance"] for row in contrasts
        ),
        "insufficient_for_confirmation": True,
        "decision": "do_not_rank_arms_without_external_binding_measurements",
        "claim_boundary": (
            "Exploratory nonviral sequence-QC and arm-divergence summary on "
            "three components; no binding, ensemble-improvement, efficacy, "
            "safety, IDP-specific, or therapeutic claim"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.analysis, args.output)
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "candidates": result["candidate_count"],
        "median_cross_arm_hamming_distance": result[
            "median_cross_arm_hamming_distance"
        ],
        "arm_summary": result["arm_summary"],
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
