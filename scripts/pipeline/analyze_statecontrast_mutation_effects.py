#!/usr/bin/env python
"""Estimate mutation/position effect sizes for StateContrast-Ab candidates."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze StateContrast mutation effect sizes")
    parser.add_argument("--scores", required=True, help="statecontrast_candidate_scores.csv or held-out score CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metric", default="specificity_gap_delta")
    parser.add_argument("--arm", default="", help="Optional arm filter for held-out score CSVs")
    parser.add_argument("--min-support", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.scores)
    if args.arm:
        rows = [r for r in rows if r.get("arm") == args.arm]
    mutation_rows, position_rows = mutation_effects(rows, metric=args.metric, min_support=args.min_support)
    write_csv(out_dir / "statecontrast_mutation_effects.csv", mutation_rows)
    write_csv(out_dir / "statecontrast_position_effects.csv", position_rows)
    summary = {
        "input_rows": len(rows),
        "arm": args.arm,
        "metric": args.metric,
        "min_support": args.min_support,
        "mutation_effect_rows": len(mutation_rows),
        "position_effect_rows": len(position_rows),
        "top_mutation_effects": mutation_rows[:10],
        "top_position_effects": position_rows[:10],
    }
    with open(out_dir / "statecontrast_mutation_effect_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "statecontrast_mutation_effect_report.md", summary)
    print(f"Wrote mutation effect attribution to {out_dir}")
    print(f"Mutation effects={len(mutation_rows)} position effects={len(position_rows)}")


def mutation_effects(rows, metric="specificity_gap_delta", min_support=5):
    parsed = []
    for row in rows:
        if row.get("candidate_id") == "native":
            continue
        try:
            value = float(row.get(metric, 0.0))
        except ValueError:
            continue
        mutations = parse_mutations(row.get("mutations", ""))
        parsed.append({**row, "_metric": value, "_mutations": mutations})
    by_ref = defaultdict(list)
    for row in parsed:
        by_ref[row.get("reference_pdb", "")].append(row)
    mutation_out = []
    position_out = []
    for ref, ref_rows in by_ref.items():
        mutation_keys = sorted({mutation_key(m) for r in ref_rows for m in r["_mutations"]})
        for key in mutation_keys:
            present = [r["_metric"] for r in ref_rows if key in {mutation_key(m) for m in r["_mutations"]}]
            absent = [r["_metric"] for r in ref_rows if key not in {mutation_key(m) for m in r["_mutations"]}]
            effect = effect_row(ref, key, present, absent, min_support, kind="mutation")
            if effect:
                native, pos, new = key.split(":")
                effect.update({"native_aa": native, "paratope_index": pos, "new_aa": new})
                mutation_out.append(effect)
        position_keys = sorted({m["pos"] for r in ref_rows for m in r["_mutations"]})
        for pos in position_keys:
            present = [r["_metric"] for r in ref_rows if any(m["pos"] == pos for m in r["_mutations"])]
            absent = [r["_metric"] for r in ref_rows if not any(m["pos"] == pos for m in r["_mutations"])]
            effect = effect_row(ref, str(pos), present, absent, min_support, kind="position")
            if effect:
                effect.update({"paratope_index": pos})
                position_out.append(effect)
    mutation_out.sort(key=lambda r: (float(r["mean_delta"]), int(r["present_n"])), reverse=True)
    position_out.sort(key=lambda r: (float(r["mean_delta"]), int(r["present_n"])), reverse=True)
    return mutation_out, position_out


def parse_mutations(text):
    out = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        match = MUT_RE.match(token)
        if not match:
            continue
        native, pos, new = match.groups()
        out.append({"native": native, "pos": int(pos), "new": new})
    return out


def mutation_key(mutation):
    return f"{mutation['native']}:{mutation['pos']}:{mutation['new']}"


def effect_row(reference_pdb, key, present, absent, min_support, kind):
    if len(present) < min_support or len(absent) < min_support:
        return None
    present_mean = sum(present) / len(present)
    absent_mean = sum(absent) / len(absent)
    delta = present_mean - absent_mean
    return {
        "reference_pdb": reference_pdb,
        "effect_type": kind,
        "effect_key": key,
        "present_n": len(present),
        "absent_n": len(absent),
        "present_mean": round(present_mean, 4),
        "absent_mean": round(absent_mean, 4),
        "mean_delta": round(delta, 4),
        "present_median": round(statistics.median(present), 4),
        "absent_median": round(statistics.median(absent), 4),
        "median_delta": round(statistics.median(present) - statistics.median(absent), 4),
    }


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast Mutation Effect Attribution\n\n")
        f.write(f"Metric: {summary['metric']}\n\n")
        f.write(f"Minimum support: {summary['min_support']}\n\n")
        f.write("## Top Mutation Effects\n\n")
        f.write("| Ref | Mutation | N | Mean Delta | Present Mean | Absent Mean |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in summary["top_mutation_effects"]:
            f.write(f"| {r['reference_pdb']} | {r['native_aa']}{r['paratope_index']}{r['new_aa']} | {r['present_n']} | {r['mean_delta']} | {r['present_mean']} | {r['absent_mean']} |\n")
        f.write("\n## Top Position Effects\n\n")
        f.write("| Ref | Position | N | Mean Delta | Present Mean | Absent Mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in summary["top_position_effects"]:
            f.write(f"| {r['reference_pdb']} | {r['paratope_index']} | {r['present_n']} | {r['mean_delta']} | {r['present_mean']} | {r['absent_mean']} |\n")
        f.write("\nInterpretation: positive deltas indicate candidates carrying that mutation/position change score higher than candidates without it. This is associative, not causal, unless validated under held-out generation controls.\n")


if __name__ == "__main__":
    main()
