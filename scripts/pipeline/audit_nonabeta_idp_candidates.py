#!/usr/bin/env python
"""Audit non-A-beta IDP candidates and prepare a family-isolated Fv fold panel."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from Bio.Align import PairwiseAligner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline"))

from analyze_shortlist_variable_regions import find_variable_region  # noqa: E402
from score_shortlist_developability import score_sequence  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="outputs/non_abeta_idp_ensemble_design_v1/ensemble_designed_candidates.csv",
    )
    parser.add_argument("--out", default="outputs/non_abeta_idp_candidate_audit_v1")
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument("--max-risk", type=float, default=0.55)
    parser.add_argument("--family-identity", type=float, default=0.90)
    return parser.parse_args()


def mutation_set(row):
    return set(filter(None, row["applied_mutations"].split(";")))


def mutation_distance(left, right):
    return len(mutation_set(left) ^ mutation_set(right))


def sequence_identity(left, right):
    if not left or not right:
        return 0.0
    aligner = PairwiseAligner(mode="global", match_score=1.0, mismatch_score=0.0)
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(left, right)[0]
    indices = alignment.indices
    matches = sum(
        left[left_index] == right[right_index]
        for left_index, right_index in zip(indices[0], indices[1])
        if left_index >= 0 and right_index >= 0
    )
    return matches / indices.shape[1]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.candidates, newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    audited = []
    for row in candidates:
        heavy = row["heavy_sequence"]
        light = row["light_sequence"]
        vh = find_variable_region(heavy, "heavy")
        vl = find_variable_region(light, "light")
        heavy_score = score_sequence(heavy)
        light_score = score_sequence(light)
        vh_score = score_sequence(vh["variable_seq"])
        vl_score = score_sequence(vl["variable_seq"])
        candidate_flags = set(filter(None, row.get("introduced_flags", "").split(";")))
        candidate_flags.update(filter(None, vh_score["flags"].split(";")))
        candidate_flags.update(filter(None, vl_score["flags"].split(";")))
        full_risk = max(heavy_score["sequence_risk"], light_score["sequence_risk"])
        boundaries_pass = vh["boundary_status"] == vl["boundary_status"] == "j_motif_found"
        passes = (
            row.get("passes_design_gate", "").lower() == "true"
            and full_risk <= args.max_risk
            and not candidate_flags
            and boundaries_pass
        )
        audited.append({
            **row,
            "antibody_family": row["reference_pdb"],
            "heavy_length": len(heavy),
            "light_length": len(light),
            "vh_length": len(vh["variable_seq"]),
            "vl_length": len(vl["variable_seq"]),
            "vh_boundary_status": vh["boundary_status"],
            "vl_boundary_status": vl["boundary_status"],
            "heavy_risk_audit": heavy_score["sequence_risk"],
            "light_risk_audit": light_score["sequence_risk"],
            "combined_risk_audit": full_risk,
            "vh_flags": vh_score["flags"],
            "vl_flags": vl_score["flags"],
            "candidate_specific_flags_audit": ";".join(sorted(candidate_flags)),
            "developability_status": "developability_pass" if passes else "developability_review",
            "vh_sequence": vh["variable_seq"],
            "vl_sequence": vl["variable_seq"],
        })

    by_family = defaultdict(list)
    for row in audited:
        by_family[(row["target"], row["antibody_family"])].append(row)
    selected = []
    for family, rows in sorted(by_family.items()):
        target = family[0]
        min_delta = -0.01 if target == "tau" else 0.0
        min_p25 = -0.005 if target == "tau" else 0.0
        eligible = [
            row for row in rows
            if row["developability_status"] == "developability_pass"
            and float(row["native_delta_min"]) >= min_delta
            and float(row["native_delta_p25"]) >= min_p25
        ]
        eligible.sort(key=lambda row: (
            float(row["native_delta_p25"]), float(row["native_delta_min"]),
            float(row["multiconf_robust_score"]),
        ), reverse=True)
        if not eligible:
            raise SystemExit(f"{family}: no objective-passing developability-pass candidate")
        family_selected = [eligible.pop(0)]
        while eligible and len(family_selected) < args.per_family:
            choice = max(eligible, key=lambda row: (
                min(mutation_distance(row, chosen) for chosen in family_selected),
                float(row["native_delta_p25"]), float(row["native_delta_min"]),
            ))
            family_selected.append(choice)
            eligible.remove(choice)
        if len(family_selected) != args.per_family:
            raise SystemExit(f"{family}: selected {len(family_selected)}, expected {args.per_family}")
        for rank, row in enumerate(family_selected, 1):
            selected.append({**row, "family_panel_rank": rank})

    cross_family = []
    families = sorted(by_family)
    for index, left_family in enumerate(families):
        for right_family in families[index + 1:]:
            max_identity = max(
                (sequence_identity(left["vh_sequence"], right["vh_sequence"])
                 + sequence_identity(left["vl_sequence"], right["vl_sequence"])) / 2.0
                for left in by_family[left_family] for right in by_family[right_family]
            )
            cross_family.append({
                "left_target": left_family[0],
                "left_family": left_family[1],
                "right_target": right_family[0],
                "right_family": right_family[1],
                "max_paired_variable_identity": round(max_identity, 4),
                "identity_threshold": args.family_identity,
                "family_isolated": max_identity < args.family_identity,
            })

    write_csv(out_dir / "candidate_developability_audit.csv", audited)
    write_csv(out_dir / "family_isolation_audit.csv", cross_family)
    write_csv(out_dir / "fv_fold_panel.csv", selected)
    with open(out_dir / "fv_fold_queue.fasta", "w", encoding="ascii") as handle:
        for row in selected:
            handle.write(
                f">{row['candidate_uid']}|target={row['target']}|family={row['antibody_family']}|Fv_HL_pair\n"
                f"{row['vh_sequence']}:{row['vl_sequence']}\n"
            )

    target_counts = Counter(row["target"] for row in audited)
    pass_counts = Counter(
        row["target"] for row in audited if row["developability_status"] == "developability_pass"
    )
    summary = {
        "schema_version": "nonabeta.idp_candidate_audit.v1",
        "status": "pass" if (
            all(pass_counts[target] >= args.per_family for target in target_counts)
            and all(row["family_isolated"] for row in cross_family)
        ) else "partial",
        "input_candidates": len(audited),
        "developability_pass": sum(pass_counts.values()),
        "by_target": {
            target: {"input": target_counts[target], "developability_pass": pass_counts[target]}
            for target in sorted(target_counts)
        },
        "fold_panel_selected": len(selected),
        "per_family": args.per_family,
        "family_isolation": cross_family,
        "claim_boundary": "Heuristic sequence developability and sequence-identity family isolation; experimental assays remain required.",
    }
    with open(out_dir / "candidate_audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Non-A-beta candidate audit: status={summary['status']} "
        f"developability_pass={summary['developability_pass']}/{len(audited)} "
        f"fold_panel={len(selected)}"
    )
    if summary["status"] != "pass":
        raise SystemExit("Non-A-beta candidate audit partial")


if __name__ == "__main__":
    main()
