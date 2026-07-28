#!/usr/bin/env python
"""Summarize high-gap vs low-gap contact-variable enrichments."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize StateContrast contact variables")
    parser.add_argument("--enrichment", required=True, help="statecontrast_high_low_enrichment.csv")
    parser.add_argument("--scores", required=True, help="statecontrast_candidate_scores.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-abs-delta", type=float, default=0.20)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    enrichment = load_csv(args.enrichment)
    scores = load_csv(args.scores)
    delta_rows = compute_deltas(enrichment)
    signal_rows = [r for r in delta_rows if abs(float(r["fraction_delta"])) >= args.min_abs_delta]
    signal_rows.sort(key=lambda r: abs(float(r["fraction_delta"])), reverse=True)
    volume_rows = aggregate_by_volume(delta_rows)
    chemistry_rows = aggregate_by_chemistry(delta_rows)
    write_csv(out_dir / "statecontrast_variable_deltas.csv", delta_rows)
    write_csv(out_dir / "statecontrast_variable_signals.csv", signal_rows)
    write_csv(out_dir / "statecontrast_volume_class_summary.csv", volume_rows)
    write_csv(out_dir / "statecontrast_chemistry_class_summary.csv", chemistry_rows)
    summary = {
        "candidate_rows": len(scores),
        "delta_rows": len(delta_rows),
        "signal_rows": len(signal_rows),
        "min_abs_delta": args.min_abs_delta,
        "top_signals": signal_rows[:10],
        "top_gap_candidates": top_gap_candidates(scores),
    }
    with open(out_dir / "statecontrast_variable_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(out_dir / "statecontrast_variable_report.md", summary, volume_rows, chemistry_rows)
    print(f"Wrote StateContrast variable summary to {out_dir}")
    print(f"Signals={len(signal_rows)} delta_rows={len(delta_rows)}")


def compute_deltas(rows):
    grouped = defaultdict(dict)
    for r in rows:
        key = (
            r["reference_pdb"], r["paratope_index"], r["chain"], r["resid"],
            r["volume_class"], r["chemistry_class"],
        )
        grouped[key][r["group"]] = float(r["fraction"])
    out = []
    for key, values in grouped.items():
        high = values.get("high_gap", 0.0)
        low = values.get("low_gap", 0.0)
        ref, pidx, chain, resid, vol, chem = key
        out.append({
            "reference_pdb": ref,
            "paratope_index": pidx,
            "chain": chain,
            "resid": resid,
            "volume_class": vol,
            "chemistry_class": chem,
            "high_fraction": round(high, 4),
            "low_fraction": round(low, 4),
            "fraction_delta": round(high - low, 4),
            "direction": "high_gap_enriched" if high > low else "low_gap_enriched" if low > high else "no_change",
        })
    return out


def aggregate_by_volume(delta_rows):
    return aggregate_class(delta_rows, "volume_class")


def aggregate_by_chemistry(delta_rows):
    return aggregate_class(delta_rows, "chemistry_class")


def aggregate_class(delta_rows, field):
    values = defaultdict(list)
    for r in delta_rows:
        values[(r["reference_pdb"], r[field])].append(float(r["fraction_delta"]))
    out = []
    for (ref, cls), vals in sorted(values.items()):
        out.append({
            "reference_pdb": ref,
            field: cls,
            "mean_delta": round(sum(vals) / len(vals), 4),
            "max_abs_delta": round(max(abs(v) for v in vals), 4),
            "n_positions": len(vals),
        })
    return out


def top_gap_candidates(scores):
    rows = sorted(scores, key=lambda r: float(r.get("specificity_gap_delta", 0.0)), reverse=True)
    return [
        {
            "reference_pdb": r["reference_pdb"],
            "candidate_id": r["candidate_id"],
            "specificity_gap_delta": r.get("specificity_gap_delta"),
            "positive_score": r.get("positive_score"),
            "negative_max_score": r.get("negative_max_score"),
            "mutations": r.get("mutations"),
        }
        for r in rows[:10]
    ]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summary, volume_rows, chemistry_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast Variable Attribution Report\n\n")
        f.write(f"Signals with abs(delta) >= {summary['min_abs_delta']}: {summary['signal_rows']}\n\n")
        f.write("## Top High-Gap Variable Signals\n\n")
        f.write("| Ref | Pos | Residue | Volume | Chemistry | High | Low | Delta | Direction |\n")
        f.write("|---|---:|---|---|---|---:|---:|---:|---|\n")
        for r in summary["top_signals"]:
            f.write(
                f"| {r['reference_pdb']} | {r['paratope_index']} | {r['chain']}:{r['resid']} | "
                f"{r['volume_class']} | {r['chemistry_class']} | {r['high_fraction']} | "
                f"{r['low_fraction']} | {r['fraction_delta']} | {r['direction']} |\n"
            )
        f.write("\n## Top Gap Candidates\n\n")
        f.write("| Ref | Candidate | Gap Delta | Positive | Negative Max | Mutations |\n")
        f.write("|---|---|---:|---:|---:|---|\n")
        for r in summary["top_gap_candidates"]:
            f.write(
                f"| {r['reference_pdb']} | {r['candidate_id']} | {r['specificity_gap_delta']} | "
                f"{r['positive_score']} | {r['negative_max_score']} | {r['mutations']} |\n"
            )
        f.write("\nInterpretation: positive deltas mark classes enriched among high specificity-gap candidates; negative deltas mark classes depleted from high-gap candidates.\n")


if __name__ == "__main__":
    main()
