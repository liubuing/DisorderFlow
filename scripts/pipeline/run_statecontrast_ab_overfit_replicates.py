#!/usr/bin/env python
"""Run multi-seed StateContrast-Ab overfit-control replicates."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUN = PROJECT_ROOT / "scripts" / "pipeline" / "run_statecontrast_ab_overfit_controls.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run replicated StateContrast-Ab overfit controls")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="31,41,51,61,71")
    parser.add_argument("--samples-per-ref", type=int, default=150)
    parser.add_argument("--v3-samples-per-ref", type=int, default=150)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--min-abs-delta", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    replicate_rows = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        run_single(seed, seed_dir, args)
        rows = load_csv(seed_dir / "statecontrast_overfit_summary.csv")
        for row in rows:
            replicate_rows.append({"seed": seed, **row})
    aggregate_rows = aggregate_replicates(replicate_rows)
    write_csv(out_dir / "statecontrast_overfit_replicate_rows.csv", replicate_rows)
    write_csv(out_dir / "statecontrast_overfit_replicate_summary.csv", aggregate_rows)
    with open(out_dir / "statecontrast_overfit_replicate_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "summaries": aggregate_rows}, f, indent=2)
    write_report(out_dir / "statecontrast_overfit_replicate_report.md", seeds, aggregate_rows)
    print(f"Wrote replicated StateContrast-Ab overfit controls to {out_dir}")
    print(f"Seeds={len(seeds)} replicate_rows={len(replicate_rows)}")


def run_single(seed, seed_dir, args):
    cmd = [
        sys.executable,
        str(SINGLE_RUN),
        "--samples-per-ref", str(args.samples_per_ref),
        "--v3-samples-per-ref", str(args.v3_samples_per_ref),
        "--max-mutations", str(args.max_mutations),
        "--min-abs-delta", str(args.min_abs_delta),
        "--seed", str(seed),
        "--out", str(seed_dir),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def aggregate_replicates(rows):
    by_ref = {}
    for row in rows:
        by_ref.setdefault(row["reference_pdb"], []).append(row)
    out = []
    for ref, ref_rows in sorted(by_ref.items()):
        for metric in ("best_gap_delta", "top10_mean_gap_delta", "top5pct_mean_gap_delta", "median_gap_delta"):
            out.append(summarize_metric(ref, ref_rows, metric))
    return out


def summarize_metric(reference_pdb, rows, metric):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = float(row[metric])
    guided_vals = []
    v2_deltas = []
    parent_deltas = []
    random_deltas = []
    shuffled_deltas = []
    pass_count = 0
    evaluable = 0
    for seed, arms in sorted(by_seed.items()):
        guided = arms.get("guided_v3_heldout_eval")
        v2 = arms.get("v2_discovery_pool_heldout_eval")
        parent = arms.get("top_parent_only_heldout_eval")
        random_arm = arms.get("random_rule_v3_heldout_eval")
        shuffled = arms.get("shuffled_rule_v3_heldout_eval")
        if guided is None or v2 is None or parent is None or random_arm is None or shuffled is None:
            continue
        evaluable += 1
        guided_vals.append(guided)
        v2_deltas.append(guided - v2)
        parent_deltas.append(guided - parent)
        random_deltas.append(guided - random_arm)
        shuffled_deltas.append(guided - shuffled)
        if guided > parent and guided > random_arm and guided > shuffled:
            pass_count += 1
    return {
        "reference_pdb": reference_pdb,
        "metric": metric,
        "n_seeds": evaluable,
        "guided_mean": mean(guided_vals),
        "guided_median": median(guided_vals),
        "guided_minus_v2_mean": mean(v2_deltas),
        "guided_minus_parent_only_mean": mean(parent_deltas),
        "guided_minus_random_mean": mean(random_deltas),
        "guided_minus_shuffled_mean": mean(shuffled_deltas),
        "guided_beats_parent_random_and_shuffled_rate": round(pass_count / max(1, evaluable), 4),
        "verdict": replicate_verdict(pass_count, evaluable, parent_deltas, random_deltas, shuffled_deltas),
    }


def replicate_verdict(pass_count, evaluable, parent_deltas, random_deltas, shuffled_deltas):
    if evaluable == 0:
        return "not_evaluable"
    win_rate = pass_count / evaluable
    if win_rate >= 0.8 and mean(parent_deltas) > 0 and mean(random_deltas) > 0 and mean(shuffled_deltas) > 0:
        return "stable_method_signal"
    if win_rate >= 0.6 and mean(parent_deltas) > 0 and (mean(random_deltas) > 0 or mean(shuffled_deltas) > 0):
        return "partial_signal"
    return "mixed_or_overfit_risk"


def mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else ""


def median(vals):
    return round(statistics.median(vals), 4) if vals else ""


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


def write_report(path, seeds, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab Replicated Overfit-Control Report\n\n")
        f.write(f"Seeds: {', '.join(str(s) for s in seeds)}\n\n")
        f.write("Win-rate is the fraction of seeds where guided_v3 exceeds top-parent-only, random-rule, and shuffled-rule baselines for the metric.\n\n")
        f.write("| Ref | Metric | Seeds | Guided Mean | Guided-V2 | Guided-Parent | Guided-Random | Guided-Shuffled | Win Rate | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['reference_pdb']} | {r['metric']} | {r['n_seeds']} | {r['guided_mean']} | "
                f"{r['guided_minus_v2_mean']} | {r['guided_minus_parent_only_mean']} | {r['guided_minus_random_mean']} | {r['guided_minus_shuffled_mean']} | "
                f"{r['guided_beats_parent_random_and_shuffled_rate']} | {r['verdict']} |\n"
            )
        f.write("\nInterpretation: a stable method signal requires high win-rate plus positive mean margins over parent-only, random, and shuffled controls.\n")


if __name__ == "__main__":
    main()
