#!/usr/bin/env python
"""Rank fixed paratope candidates across the audited A-beta pose panel."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from ensemble_pose_transfer import fixed_paratope_contact_map  # noqa: E402
from state_contact_scorer import extract_contact_map, score_sequence_on_contact_map  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/statecontrast/abeta_position_effect_v3_4ref_loro.yml")
    parser.add_argument("--pose-audit", default="outputs/abeta_multiconf_pose_panel_v1/pose_panel_audit.json")
    parser.add_argument("--candidates", default="outputs/statecontrast_domain_abeta_position_4ref_loro_v1/v4_position_effect/statecontrast_effect_guided_v4_scores.csv")
    parser.add_argument("--out", default="outputs/abeta_multiconf_candidate_scoring_v1")
    parser.add_argument("--top-per-reference", type=int, default=25)
    return parser.parse_args()


def load_candidates(path):
    unique = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["reference_pdb"], row["sequence"])
            current = unique.get(key)
            if current is None or float(row.get("statecontrast_score", 0)) > float(current.get("statecontrast_score", 0)):
                unique[key] = row
    return list(unique.values())


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(PROJECT_ROOT / args.pose_audit, encoding="utf-8") as handle:
        audit = json.load(handle)
    if audit["status"] != "pass":
        raise SystemExit("Pose-panel audit has not passed")

    references = {row["pdb"]: row for row in config["references"]}
    poses = defaultdict(list)
    for row in audit["entries"]:
        if row["status"] == "pass":
            poses[row["reference_pdb"]].append(row)

    maps = {}
    native_scores = {}
    for reference_id, reference in references.items():
        template_map = extract_contact_map(
            str(PROJECT_ROOT / reference["complex_pdb"]),
            peptide_chain=reference["peptide_chain"],
        )
        maps[reference_id] = [
            fixed_paratope_contact_map(PROJECT_ROOT / pose["pose_pdb"], template_map)
            for pose in poses[reference_id]
        ]
        native_scores[reference_id] = [
            score_sequence_on_contact_map(template_map["paratope_sequence"], contact_map)["state_contact_score"]
            for contact_map in maps[reference_id]
        ]

    rows = []
    for candidate in load_candidates(PROJECT_ROOT / args.candidates):
        reference_id = candidate["reference_pdb"]
        if reference_id not in maps:
            continue
        values = [
            score_sequence_on_contact_map(candidate["sequence"], contact_map)["state_contact_score"]
            for contact_map in maps[reference_id]
        ]
        native = native_scores[reference_id]
        deltas = [value - baseline for value, baseline in zip(values, native)]
        expected = audit["conformers"]
        coverage = len(values) / expected
        p25 = float(np.percentile(values, 25))
        delta_p25 = float(np.percentile(deltas, 25))
        robust_score = 0.45 * p25 + 0.25 * min(values) + 0.20 * float(np.mean(values)) - 0.10 * float(np.std(values))
        robust_score *= coverage
        candidate_uid = f"{reference_id}_{hashlib.sha256(candidate['sequence'].encode('ascii')).hexdigest()[:12]}"
        rows.append({
            "candidate_uid": candidate_uid,
            "reference_pdb": reference_id,
            "candidate_id": candidate["candidate_id"],
            "sequence": candidate["sequence"],
            "generator": candidate.get("generator", ""),
            "n_mutations": candidate.get("n_mutations", ""),
            "statecontrast_score": candidate.get("statecontrast_score", ""),
            "pose_coverage": round(coverage, 4),
            "n_scored_conformers": len(values),
            "ensemble_min": round(min(values), 4),
            "ensemble_p25": round(p25, 4),
            "ensemble_mean": round(float(np.mean(values)), 4),
            "ensemble_std": round(float(np.std(values)), 4),
            "native_delta_p25": round(delta_p25, 4),
            "native_delta_min": round(min(deltas), 4),
            "multiconf_robust_score": round(robust_score, 6),
            "per_conformer_scores": ";".join(f"{value:.4f}" for value in values),
            "pose_semantics": "template_transferred_computational_hypothesis",
        })

    ranked_all = []
    selected = []
    for reference_id in sorted(references):
        group = sorted(
            (row for row in rows if row["reference_pdb"] == reference_id),
            key=lambda row: (row["multiconf_robust_score"], row["native_delta_p25"]),
            reverse=True,
        )
        for rank, row in enumerate(group, 1):
            row["multiconf_rank_within_reference"] = rank
            row["selected_top_per_reference"] = rank <= args.top_per_reference
            ranked_all.append(row)
            if row["selected_top_per_reference"]:
                selected.append(row)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "multiconf_rank_within_reference", "selected_top_per_reference", "candidate_uid",
        "reference_pdb", "candidate_id", "sequence",
        "generator", "n_mutations", "statecontrast_score", "pose_coverage",
        "n_scored_conformers", "ensemble_min", "ensemble_p25", "ensemble_mean",
        "ensemble_std", "native_delta_p25", "native_delta_min", "multiconf_robust_score",
        "per_conformer_scores", "pose_semantics",
    ]
    with open(out_dir / "multiconf_candidate_ranking.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "multiconf_candidate_ranking_all.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked_all)
    summary = {
        "status": "pass" if selected and all(row["pose_coverage"] >= 0.8 for row in selected) else "fail",
        "input_unique_candidates": len(rows),
        "selected": len(selected),
        "references": len({row["reference_pdb"] for row in selected}),
        "pose_panel_pass_fraction": audit["pass_fraction"],
        "ranking": "coverage-penalized worst-case/p25/mean ensemble compatibility",
        "claim_boundary": "computational pose hypotheses; no affinity or binding claim",
    }
    with open(out_dir / "multiconf_candidate_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "multiconf_candidate_report.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Multi-Conformation Candidate Ranking\n\n")
        handle.write(f"Status: `{summary['status']}`; selected: {len(selected)}; references: {summary['references']}.\n\n")
        handle.write("Scores use only audited template-transferred pose hypotheses and are penalized for missing conformers.\n\n")
        handle.write("| Ref | Rank | Candidate | Coverage | Min | P25 | Mean | Std | Native delta P25 | Robust score |\n")
        handle.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in selected:
            handle.write(
                f"| {row['reference_pdb']} | {row['multiconf_rank_within_reference']} | {row['candidate_id']} | "
                f"{row['pose_coverage']} | {row['ensemble_min']} | {row['ensemble_p25']} | "
                f"{row['ensemble_mean']} | {row['ensemble_std']} | {row['native_delta_p25']} | "
                f"{row['multiconf_robust_score']} |\n"
            )
    print(f"Multi-conformation scoring: {len(rows)} candidates; selected {len(selected)}; status={summary['status']}")
    if summary["status"] != "pass":
        raise SystemExit("Multi-conformation candidate gate failed")


if __name__ == "__main__":
    main()
