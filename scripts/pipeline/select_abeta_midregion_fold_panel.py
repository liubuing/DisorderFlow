#!/usr/bin/env python
"""Select a diverse, all-conformer-positive middle-region Fv fold panel."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_shortlist_variable_regions import find_variable_region  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="outputs/abeta_midregion_ensemble_design_v1/ensemble_designed_candidates.csv")
    parser.add_argument("--out", default="outputs/abeta_midregion_fold_panel_v1")
    parser.add_argument("--per-reference", type=int, default=2)
    parser.add_argument("--relative-p25-floor", type=float, default=0.90)
    return parser.parse_args()


def mutations(row):
    return set(filter(None, row["applied_mutations"].split(";")))


def mutation_distance(left, right):
    return len(mutations(left) ^ mutations(right))


def select_group(rows, count, relative_floor):
    eligible = [row for row in rows if float(row["native_delta_min"]) > 0]
    if not eligible:
        return []
    best_p25 = max(float(row["native_delta_p25"]) for row in eligible)
    pool = [
        row for row in eligible
        if float(row["native_delta_p25"]) >= best_p25 * relative_floor
    ]
    pool.sort(key=lambda row: (
        float(row["native_delta_p25"]), float(row["native_delta_min"]),
        float(row["multiconf_robust_score"]),
    ), reverse=True)
    selected = [pool.pop(0)]
    while pool and len(selected) < count:
        next_row = max(pool, key=lambda row: (
            min(mutation_distance(row, chosen) for chosen in selected),
            float(row["native_delta_p25"]), float(row["native_delta_min"]),
        ))
        selected.append(next_row)
        pool.remove(next_row)
    return selected


def main():
    args = parse_args()
    with open(args.candidates, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_reference = defaultdict(list)
    for row in rows:
        by_reference[row["reference_pdb"]].append(row)
    selected = []
    for reference_id in sorted(by_reference):
        group = select_group(by_reference[reference_id], args.per_reference, args.relative_p25_floor)
        if len(group) != args.per_reference:
            raise SystemExit(f"{reference_id}: selected {len(group)}, expected {args.per_reference}")
        for rank, row in enumerate(group, 1):
            selected.append({
                **row,
                "construct_id": row["candidate_uid"],
                "candidate_id": row["candidate_uid"],
                "panel_rank_within_reference": rank,
                "selection_reason": "all_conformer_positive_then_mutation_diversity",
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0])
    with open(out_dir / "midregion_fold_panel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "midregion_fv_fold_queue.fasta", "w", encoding="ascii") as handle:
        for row in selected:
            heavy = find_variable_region(row["heavy_sequence"], "heavy")["variable_seq"]
            light = find_variable_region(row["light_sequence"], "light")["variable_seq"]
            handle.write(
                f">{row['construct_id']}|candidate_uid={row['candidate_uid']}|Fv_HL_pair\n"
                f"{heavy}:{light}\n"
            )
    summary = {
        "status": "pass",
        "selected": len(selected),
        "per_reference": args.per_reference,
        "selection": "all native_delta_min > 0, near-best P25, greedy mutation diversity",
        "references": {
            reference_id: [row["candidate_uid"] for row in selected if row["reference_pdb"] == reference_id]
            for reference_id in sorted(by_reference)
        },
    }
    with open(out_dir / "midregion_fold_panel_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "midregion_fold_panel_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Middle-Region Fv Fold Panel\n\n")
        handle.write(f"Selected: {len(selected)}; status: `{summary['status']}`.\n\n")
        handle.write("| Ref | Panel rank | UID | Mutations | Delta P25 | Delta min |\n")
        handle.write("|---|---:|---|---|---:|---:|\n")
        for row in selected:
            handle.write(
                f"| {row['reference_pdb']} | {row['panel_rank_within_reference']} | "
                f"{row['candidate_uid']} | {row['applied_mutations']} | "
                f"{float(row['native_delta_p25']):.4f} | {float(row['native_delta_min']):.4f} |\n"
            )
    print(f"Middle-region fold panel: selected={len(selected)}")


if __name__ == "__main__":
    main()
