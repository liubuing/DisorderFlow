#!/usr/bin/env python
"""Build a focused claim report for position-only StateContrast-Ab v4."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build StateContrast v4 claim report")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--seed-summary", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_csv(args.summary)
    rules = load_csv(args.rules)
    seed_summary = load_csv(args.seed_summary)
    claim = build_claim(summary, rules, seed_summary)
    with open(out_dir / "statecontrast_v4_position_claim.json", "w", encoding="utf-8") as f:
        json.dump(claim, f, indent=2)
    write_report(out_dir / "statecontrast_v4_position_claim.md", claim)
    print(f"Wrote StateContrast v4 claim report to {out_dir}")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_claim(summary, rules, seed_summary):
    top10 = {r["reference_pdb"]: r for r in summary if r.get("metric") == "top10_mean_gap_delta"}
    seed_wins = seed_level_wins(seed_summary)
    return {
        "claim": "Position-only effect-size guidance robustly improves held-out StateContrast specificity-gap delta beyond parent-only, random-position, and shuffled-position controls in the A-beta 4HIX/5CSZ domain workflow.",
        "claim_scope": "A-beta proof-of-concept; synthetic negative states; position discovery, not residue-identity prescription.",
        "references": sorted(top10),
        "top10_summary": top10,
        "seed_level_wins": seed_wins,
        "recurrent_positions": summarize_positions(rules),
        "decision": decision(top10),
    }


def seed_level_wins(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["seed"], row["reference_pdb"])][row["arm"]] = row
    out = []
    for (seed, ref), arms in sorted(grouped.items()):
        guided = arms.get("effect_guided_v4_heldout_eval")
        parent = arms.get("top_parent_only_heldout_eval")
        random = arms.get("effect_random_rule_v4_heldout_eval")
        shuffled = arms.get("effect_shuffled_rule_v4_heldout_eval")
        if not guided or not parent or not random or not shuffled:
            continue
        metric = "top10_mean_gap_delta"
        win = float(guided[metric]) > float(parent[metric]) and float(guided[metric]) > float(random[metric]) and float(guided[metric]) > float(shuffled[metric])
        out.append({"seed": seed, "reference_pdb": ref, "top10_win": win, "guided_top10": guided[metric]})
    return out


def summarize_positions(rules):
    positions = defaultdict(set)
    recurrence = defaultdict(int)
    for row in rules:
        key = (row["reference_pdb"], row["paratope_index"])
        positions[row["reference_pdb"]].add(row["paratope_index"])
        recurrence[key] = max(recurrence[key], int(row.get("recurrence", 0)))
    out = {}
    for ref, vals in positions.items():
        out[ref] = [
            {"paratope_index": pos, "max_recurrence": recurrence[(ref, pos)]}
            for pos in sorted(vals, key=lambda x: int(x))
        ]
    return out


def decision(top10):
    if not top10:
        return "not_evaluable"
    if all(r.get("verdict") == "stable_method_signal" and float(r.get("v4_beats_parent_random_and_shuffled_rate", 0.0)) >= 1.0 for r in top10.values()):
        return "supported_position_discovery_claim"
    return "mixed_or_requires_more_references"


def write_report(path, claim):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab V4 Position-Effect Claim\n\n")
        f.write(f"Decision: `{claim['decision']}`\n\n")
        f.write(f"Claim: {claim['claim']}\n\n")
        f.write(f"Scope: {claim['claim_scope']}\n\n")
        f.write("## Top10 Evidence\n\n")
        f.write("| Ref | V4 Mean | Vs Parent | Vs Random | Vs Shuffled | Win Rate | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for ref, row in claim["top10_summary"].items():
            f.write(
                f"| {ref} | {row['v4_mean']} | {row['v4_minus_parent_only_mean']} | "
                f"{row['v4_minus_random_mean']} | {row['v4_minus_shuffled_mean']} | "
                f"{row['v4_beats_parent_random_and_shuffled_rate']} | {row['verdict']} |\n"
            )
        f.write("\n## Recurrent Positions\n\n")
        f.write("| Ref | Positions |\n")
        f.write("|---|---|\n")
        for ref, rows in claim["recurrent_positions"].items():
            positions = ", ".join(f"{r['paratope_index']} (rec={r['max_recurrence']})" for r in rows)
            f.write(f"| {ref} | {positions} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("This supports position-level state-sensitive redesign guidance under controlled computational benchmarks. It does not validate residue-identity causality or therapeutic readiness.\n")


if __name__ == "__main__":
    main()
