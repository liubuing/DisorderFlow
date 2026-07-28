#!/usr/bin/env python
"""Build a concise final StateContrast-Ab method report from benchmark outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build StateContrast-Ab final report")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mvp-summary", default="outputs/statecontrast_ab_mvp_v2/statecontrast_summary.json")
    parser.add_argument("--class-summary", default="outputs/statecontrast_stable_attribution_v1/statecontrast_stable_attribution_summary.csv")
    parser.add_argument("--mutation-summary", default="outputs/statecontrast_effect_guided_v4_mutation_only_v1/statecontrast_effect_guided_v4_summary.csv")
    parser.add_argument("--position-summary", default="outputs/statecontrast_effect_guided_v4_position_only_v2/statecontrast_effect_guided_v4_summary.csv")
    parser.add_argument("--position-rules", default="outputs/statecontrast_effect_guided_v4_position_only_v2/statecontrast_effect_guided_v4_rules.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mvp = load_json(args.mvp_summary)
    class_rows = load_csv(args.class_summary)
    mutation_rows = load_csv(args.mutation_summary)
    position_rows = load_csv(args.position_summary)
    position_rules = load_csv(args.position_rules)
    report = out_dir / "statecontrast_ab_final_method_report.md"
    write_report(report, mvp, class_rows, mutation_rows, position_rows, position_rules)
    machine = {
        "mvp": mvp,
        "class_attribution_top10": metric_rows(class_rows, "top10_mean_gap_delta"),
        "mutation_only_top10": metric_rows(mutation_rows, "top10_mean_gap_delta"),
        "position_only_top10": metric_rows(position_rows, "top10_mean_gap_delta"),
        "position_rules": summarize_rules(position_rules),
    }
    with open(out_dir / "statecontrast_ab_final_method_report.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, indent=2)
    print(f"Wrote StateContrast-Ab final method report to {report}")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_rows(rows, metric):
    return [r for r in rows if r.get("metric") == metric]


def summarize_rules(rows):
    by_ref = {}
    for row in rows:
        by_ref.setdefault(row["reference_pdb"], set()).add(str(row["paratope_index"]))
    return {ref: sorted(vals, key=lambda x: int(x)) for ref, vals in by_ref.items()}


def write_report(path, mvp, class_rows, mutation_rows, position_rows, position_rules):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab Final Method Report\n\n")
        f.write("## Executive Conclusion\n\n")
        f.write(
            "StateContrast-Ab is best supported as a contrastive, state-sensitive paratope-position discovery framework. "
            "The contact-aware scorer and held-out negative-state controls support contrastive parent selection. "
            "Class-level and mutation-identity attribution remain weak or mixed. "
            "Position-level mutation effect-size guidance is the first redesign strategy that robustly beats parent-only, random, and shuffled controls across both A-beta references.\n\n"
        )
        f.write("## Core Scoring Result\n\n")
        f.write("| Reference | Best Candidate | Best Gap Delta | Native Gap | Variants |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in mvp.get("summaries", []):
            f.write(
                f"| {row['reference_pdb']} | {row['best_candidate']} | {row['best_gap_delta']} | "
                f"{row['native_gap']} | {row['variants']} |\n"
            )
        f.write("\nInterpretation: raw specificity gaps can be negative because synthetic negative states are deliberately strong; gap_delta is the correct MVP ranking metric.\n\n")

        f.write("## Attribution Control Ladder\n\n")
        f.write("| Method Layer | Result | Interpretation |\n")
        f.write("|---|---|---|\n")
        f.write("| High/low class enrichment | mixed_or_overfit_risk | Local class signals exist, but they do not reliably beat parent/random/shuffled controls. |\n")
        f.write("| Mutation-only effect rules | partial for 4HIX, weak for 5CSZ | Specific mutation identity rules are not yet robust across references. |\n")
        f.write("| Position-only effect rules | stable_method_signal on both references | Recurrent state-sensitive positions are the strongest validated redesign signal. |\n\n")

        write_metric_table(f, "Class/Stable Attribution Top10", class_rows, "stable")
        write_metric_table(f, "Mutation-Only V4 Top10", mutation_rows, "v4")
        write_metric_table(f, "Position-Only V4 Top10", position_rows, "v4")

        f.write("## Position-Only Rules\n\n")
        f.write("These are the recurrent effect-size positions used by the strongest current method layer.\n\n")
        f.write("| Reference | Recurrent Positions |\n")
        f.write("|---|---|\n")
        for ref, positions in summarize_rules(position_rules).items():
            f.write(f"| {ref} | {', '.join(positions)} |\n")

        f.write("\n## Supported Claims\n\n")
        f.write("- StateContrast-Ab can score antibody paratopes by positive-vs-negative IDP state contrast using contact topology.\n")
        f.write("- On A-beta 4HIX/5CSZ references, contrastive parent selection improves held-out specificity-gap delta over the native/reference baseline.\n")
        f.write("- Recurrent mutation effect-size positions provide robust guidance beyond parent-only, random-position, and shuffled-position controls.\n")
        f.write("- The strongest current method unit is state-sensitive position discovery, not residue-identity prescription.\n\n")

        f.write("## Unsupported Claims\n\n")
        f.write("- Do not claim general IDP antibody design across targets; current reliable references are only 4HIX and 5CSZ.\n")
        f.write("- Do not claim class-enrichment or mutation-identity attribution is validated as causal.\n")
        f.write("- Do not claim synthesis-ready, therapeutic-ready, or fold-validated antibodies.\n")
        f.write("- Do not use ColabFold/AF2 as main evidence for IDP binding specificity.\n\n")

        f.write("## Next Validation\n\n")
        f.write("- Add more IDP antibody references beyond A-beta N-terminal examples.\n")
        f.write("- Replace synthetic negative states with experimentally or simulation-derived conformational ensembles.\n")
        f.write("- Evaluate whether position-only rules transfer leave-one-reference-out between 4HIX and 5CSZ.\n")
        f.write("- Experimentally test top parent-only and position-guided candidates side by side.\n")


def write_metric_table(f, title, rows, prefix):
    top10 = metric_rows(rows, "top10_mean_gap_delta")
    f.write(f"## {title}\n\n")
    f.write("| Reference | Mean | Vs V2 | Vs Parent | Vs Random | Vs Shuffled | Win Rate | Verdict |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for row in top10:
        mean_key = f"{prefix}_mean"
        f.write(
            f"| {row['reference_pdb']} | {row.get(mean_key, '')} | "
            f"{row.get(f'{prefix}_minus_v2_mean', '')} | "
            f"{row.get(f'{prefix}_minus_parent_only_mean', '')} | "
            f"{row.get(f'{prefix}_minus_random_mean', '')} | "
            f"{row.get(f'{prefix}_minus_shuffled_mean', '')} | "
            f"{row.get(f'{prefix}_beats_parent_random_and_shuffled_rate', '')} | {row.get('verdict', '')} |\n"
        )
    f.write("\n")


if __name__ == "__main__":
    main()
