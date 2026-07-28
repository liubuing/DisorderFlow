#!/usr/bin/env python
"""Apply frozen native-relative gates to the complete alpha-synuclein redesign panel."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/alpha_synuclein_redesign_validation_v2"


def rows(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def mean(records, field):
    return float(np.mean([float(row[field]) for row in records]))


def main():
    folds = rows(BASE / "fold_evidence/initial_guess_aggregate.csv")
    refinement = rows(BASE / "refinement/complex_refinement_audit.csv")
    constructs = {row["construct_id"]: row for row in rows(BASE / "redesign_constructs.csv")}
    by_id = defaultdict(list)
    for row in refinement:
        by_id[row["construct_id"]].append(row)
    fold_by_id = {row["construct_id"]: row for row in folds}
    native = fold_by_id["8B9V_native"]
    native_refined = by_id["8B9V_native"]
    native_refined_retention = mean(native_refined, "contact_retention")
    results = []
    for construct_id, fold in sorted(fold_by_id.items()):
        if construct_id == "8B9V_native":
            continue
        refined = by_id[construct_id]
        delta_refined = mean(refined, "contact_retention") - native_refined_retention
        zero_clash = sum(int(row["after_severe_clash_pairs_lt_1_5A"]) == 0 for row in refined)
        checks = {
            "delta_pae": float(fold["delta_pae_vs_native"]) <= 0.0,
            "delta_contact_retention": float(fold["delta_contact_retention_vs_native"]) >= 0.0,
            "delta_antigen_pose_rmsd": float(fold["delta_antigen_rmsd_vs_native"]) <= 0.0,
            "delta_refined_contact_retention": delta_refined >= 0.0,
            "refined_zero_clash_15_of_15": zero_clash == 15,
        }
        results.append({
            "construct_id": construct_id,
            "n_mutations": constructs[construct_id].get("n_mutations", ""),
            "applied_mutations": constructs[construct_id].get("applied_mutations", ""),
            "delta_pae": fold["delta_pae_vs_native"],
            "delta_contact_retention": fold["delta_contact_retention_vs_native"],
            "delta_antigen_pose_rmsd": fold["delta_antigen_rmsd_vs_native"],
            "delta_refined_contact_retention": round(delta_refined, 5),
            "refined_zero_clash_models": zero_clash,
            **{f"check_{key}": value for key, value in checks.items()},
            "final_status": "computational_hit" if all(checks.values()) else "no_hit",
        })
    with open(BASE / "alpha_synuclein_candidate_gates.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    hits = [row for row in results if row["final_status"] == "computational_hit"]
    closest = sorted(results, key=lambda row: sum(not row[key] for key in row if key.startswith("check_")))[:5]
    summary = {
        "schema_version": "alphasyn.redesign_validation.v2",
        "status": "pass" if hits else "no_hit",
        "candidates": len(results), "models": len(refinement),
        "refinement_pass_models": sum(row["refinement_status"] == "refinement_pass" for row in refinement),
        "computational_hits": [row["construct_id"] for row in hits],
        "closest_candidates_not_promoted": [row["construct_id"] for row in closest],
        "frozen_gates": {
            "delta_pae_max": 0.0, "delta_contact_retention_min": 0.0,
            "delta_antigen_pose_rmsd_max": 0.0, "delta_refined_contact_retention_min": 0.0,
            "required_refined_zero_clash_models": 15,
        },
        "claim_boundary": "No candidate is promoted unless all five pre-registered computational gates pass.",
    }
    json.dump(summary, open(BASE / "redesign_validation_summary.json", "w", encoding="utf-8"), indent=2)
    print(f"Alpha-syn redesign validation: {summary}")


if __name__ == "__main__":
    main()
