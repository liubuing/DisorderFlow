#!/usr/bin/env python
"""Evaluate leave-seed-out stable StateContrast-Ab attribution rules."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "pipeline"
sys.path.insert(0, str(SCRIPT_DIR))

from contact_variable_attribution import (  # noqa: E402
    compare_high_low_gap,
    compile_guidance_rules,
    generate_attribution_guided_variants,
)
from idp_state_ensemble import build_negative_states, positive_state  # noqa: E402
from run_statecontrast_ab_overfit_controls import (  # noqa: E402
    enrichment_deltas,
    matched_random_rules,
    native_variant,
    rule_rows,
    select_top_parents,
    shuffled_position_rules,
    summarize_reference,
)
from state_contact_scorer import extract_contact_map, generate_contact_guided_variants  # noqa: E402
from state_contrast_scorer import rank_state_contrast_candidates  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run leave-seed-out stable attribution benchmark")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="31,41,51,61,71")
    parser.add_argument("--samples-per-ref", type=int, default=150)
    parser.add_argument("--v3-samples-per-ref", type=int, default=150)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--min-abs-delta", type=float, default=0.20)
    parser.add_argument("--min-recurrence", type=int, default=2)
    parser.add_argument("--heldout-seed-offset", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    refs = load_refs(args.whitelist)
    discovery = build_discovery_contexts(refs, seeds, args)
    all_scores = []
    all_rules = []
    all_stable_signals = []
    all_summaries = []
    for seed in seeds:
        for ref in refs:
            ref_name = ref["pdb"]
            ctx = discovery[(seed, ref_name)]
            training_signals = [r for (s, rname), rows in discovery_signal_items(discovery) if s != seed and rname == ref_name for r in rows]
            stable_signals = stable_signal_rows(training_signals, ref_name, min_recurrence=args.min_recurrence)
            all_stable_signals.extend({"heldout_seed": seed, **r} for r in stable_signals)
            stable_rules = compile_guidance_rules(stable_signals, min_abs_delta=0.0).get(ref_name, {})
            rng = random.Random(seed)
            random_rules = matched_random_rules(stable_rules, len(ctx["cmap"]["paratope_sequence"]), rng)
            shuffled_rules = shuffled_position_rules(stable_rules, len(ctx["cmap"]["paratope_sequence"]), rng)
            for arm, rules in (("stable_attribution_v3", stable_rules), ("stable_random_rule_v3", random_rules), ("stable_shuffled_rule_v3", shuffled_rules)):
                all_rules.extend({"seed": seed, **r} for r in rule_rows(ref_name, arm, rules))
            parents = select_top_parents(ctx["discovery_ranked"], top_fraction=0.20)
            arms = {
                "v2_discovery_pool_heldout_eval": ctx["v2_variants"],
                "top_parent_only_heldout_eval": generate_attribution_guided_variants(
                    ctx["cmap"], parents, {}, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                    seed=seed + 404, reference_pdb=ref_name,
                ),
                "stable_attribution_v3_heldout_eval": generate_attribution_guided_variants(
                    ctx["cmap"], parents, stable_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                    seed=seed + 505, reference_pdb=ref_name,
                ),
                "stable_random_rule_v3_heldout_eval": generate_attribution_guided_variants(
                    ctx["cmap"], parents, random_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                    seed=seed + 606, reference_pdb=ref_name,
                ),
                "stable_shuffled_rule_v3_heldout_eval": generate_attribution_guided_variants(
                    ctx["cmap"], parents, shuffled_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                    seed=seed + 707, reference_pdb=ref_name,
                ),
            }
            seed_ref_scores = []
            for arm, variants in arms.items():
                ranked = rank_state_contrast_candidates(variants, ctx["pos"], ctx["heldout_neg"])
                for row in ranked:
                    score_row = {
                        "seed": seed,
                        "reference_pdb": ref_name,
                        "arm": arm,
                        **{k: v for k, v in row.items() if k not in ("positive_states", "negative_states")},
                    }
                    all_scores.append(score_row)
                    seed_ref_scores.append(score_row)
            summary_rows = summarize_stable_reference(seed, ref_name, seed_ref_scores, len(stable_rules), len(stable_signals))
            all_summaries.extend(summary_rows)
    aggregate_rows = aggregate_stable_replicates(all_summaries)
    write_csv(out_dir / "statecontrast_stable_attribution_scores.csv", all_scores)
    write_csv(out_dir / "statecontrast_stable_attribution_rules.csv", all_rules)
    write_csv(out_dir / "statecontrast_stable_attribution_signals.csv", all_stable_signals)
    write_csv(out_dir / "statecontrast_stable_attribution_seed_summary.csv", all_summaries)
    write_csv(out_dir / "statecontrast_stable_attribution_summary.csv", aggregate_rows)
    with open(out_dir / "statecontrast_stable_attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "summaries": aggregate_rows}, f, indent=2)
    write_report(out_dir / "statecontrast_stable_attribution_report.md", seeds, aggregate_rows)
    print(f"Wrote stable attribution benchmark to {out_dir}")
    print(f"Seeds={len(seeds)} score_rows={len(all_scores)}")


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("abeta_references", [])


def build_discovery_contexts(refs, seeds, args):
    contexts = {}
    for seed in seeds:
        for ref_idx, ref in enumerate(refs):
            ref_name = ref["pdb"]
            cmap = extract_contact_map(ref["path"], peptide_chain=ref.get("peptide_chain"))
            pos = [positive_state(f"{ref_name}_positive_bound", cmap, "known antibody-bound positive state")]
            discovery_neg = build_negative_states(cmap, f"{ref_name}_discovery", seed=seed + ref_idx)
            heldout_neg = build_negative_states(cmap, f"{ref_name}_heldout", seed=seed + args.heldout_seed_offset + ref_idx)
            v2_variants = generate_contact_guided_variants(cmap, n=args.samples_per_ref, max_mutations=args.max_mutations, seed=seed + ref_idx)
            v2_variants.append(native_variant(cmap))
            discovery_ranked = rank_state_contrast_candidates(v2_variants, pos, discovery_neg)
            attr = compare_high_low_gap(discovery_ranked, cmap, top_fraction=0.20)
            signals = enrichment_deltas(ref_name, attr["enrichment_rows"], min_abs_delta=args.min_abs_delta)
            contexts[(seed, ref_name)] = {
                "cmap": cmap,
                "pos": pos,
                "heldout_neg": heldout_neg,
                "v2_variants": v2_variants,
                "discovery_ranked": discovery_ranked,
                "signals": [{"seed": seed, **r} for r in signals],
            }
    return contexts


def discovery_signal_items(discovery):
    for key, ctx in discovery.items():
        yield key, ctx["signals"]


def stable_signal_rows(signal_rows, reference_pdb, min_recurrence=2):
    grouped = defaultdict(list)
    for row in signal_rows:
        direction = "high_gap_enriched" if float(row["fraction_delta"]) > 0 else "low_gap_enriched"
        key = (row["paratope_index"], row["chain"], row["resid"], row["volume_class"], row["chemistry_class"], direction)
        grouped[key].append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        seeds = sorted({int(r["seed"]) for r in rows})
        if len(seeds) < min_recurrence:
            continue
        pidx, chain, resid, vol, chem, direction = key
        mean_delta = sum(float(r["fraction_delta"]) for r in rows) / len(rows)
        out.append({
            "reference_pdb": reference_pdb,
            "paratope_index": pidx,
            "chain": chain,
            "resid": resid,
            "volume_class": vol,
            "chemistry_class": chem,
            "fraction_delta": round(mean_delta, 4),
            "direction": direction,
            "recurrence": len(seeds),
            "training_seeds": ";".join(str(s) for s in seeds),
        })
    out.sort(key=lambda r: (r["recurrence"], abs(float(r["fraction_delta"]))), reverse=True)
    return out


def summarize_stable_reference(seed, reference_pdb, rows, n_rules, n_signals):
    summaries = summarize_reference(reference_pdb, rows, n_rules, n_signals)
    mapped = []
    for row in summaries:
        if row["arm"] == "guided_v3_heldout_eval":
            row["arm"] = "stable_attribution_v3_heldout_eval"
        mapped.append({"seed": seed, **row})
    add_stable_verdict(mapped)
    return mapped


def add_stable_verdict(rows):
    by_arm = {r["arm"]: r for r in rows}
    guided = by_arm.get("stable_attribution_v3_heldout_eval")
    parent = by_arm.get("top_parent_only_heldout_eval")
    random_arm = by_arm.get("stable_random_rule_v3_heldout_eval")
    shuffled = by_arm.get("stable_shuffled_rule_v3_heldout_eval")
    if not guided or not parent or not random_arm or not shuffled:
        for row in rows:
            row["stable_control_verdict"] = "not_evaluable"
        return
    top10_pass = guided["top10_mean_gap_delta"] > parent["top10_mean_gap_delta"] and guided["top10_mean_gap_delta"] > random_arm["top10_mean_gap_delta"] and guided["top10_mean_gap_delta"] > shuffled["top10_mean_gap_delta"]
    top5_pass = guided["top5pct_mean_gap_delta"] > parent["top5pct_mean_gap_delta"] and guided["top5pct_mean_gap_delta"] > random_arm["top5pct_mean_gap_delta"] and guided["top5pct_mean_gap_delta"] > shuffled["top5pct_mean_gap_delta"]
    verdict = "passes_top_metrics" if top10_pass and top5_pass else "mixed_or_overfit_risk"
    for row in rows:
        row["stable_control_verdict"] = verdict


def aggregate_stable_replicates(rows):
    out = []
    for ref in sorted({r["reference_pdb"] for r in rows}):
        ref_rows = [r for r in rows if r["reference_pdb"] == ref]
        for metric in ("best_gap_delta", "top10_mean_gap_delta", "top5pct_mean_gap_delta", "median_gap_delta"):
            out.append(summarize_stable_metric(ref, ref_rows, metric))
    return out


def summarize_stable_metric(reference_pdb, rows, metric):
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["arm"]] = float(row[metric])
    stable_vals = []
    v2_deltas = []
    parent_deltas = []
    random_deltas = []
    shuffled_deltas = []
    wins = 0
    for _, arms in sorted(by_seed.items()):
        stable = arms.get("stable_attribution_v3_heldout_eval")
        v2 = arms.get("v2_discovery_pool_heldout_eval")
        parent = arms.get("top_parent_only_heldout_eval")
        random_arm = arms.get("stable_random_rule_v3_heldout_eval")
        shuffled = arms.get("stable_shuffled_rule_v3_heldout_eval")
        if stable is None or v2 is None or parent is None or random_arm is None or shuffled is None:
            continue
        stable_vals.append(stable)
        v2_deltas.append(stable - v2)
        parent_deltas.append(stable - parent)
        random_deltas.append(stable - random_arm)
        shuffled_deltas.append(stable - shuffled)
        if stable > parent and stable > random_arm and stable > shuffled:
            wins += 1
    n = len(stable_vals)
    return {
        "reference_pdb": reference_pdb,
        "metric": metric,
        "n_seeds": n,
        "stable_mean": mean(stable_vals),
        "stable_median": median(stable_vals),
        "stable_minus_v2_mean": mean(v2_deltas),
        "stable_minus_parent_only_mean": mean(parent_deltas),
        "stable_minus_random_mean": mean(random_deltas),
        "stable_minus_shuffled_mean": mean(shuffled_deltas),
        "stable_beats_parent_random_and_shuffled_rate": round(wins / max(1, n), 4),
        "verdict": stable_verdict(wins, n, parent_deltas, random_deltas, shuffled_deltas),
    }


def stable_verdict(wins, n, parent_deltas, random_deltas, shuffled_deltas):
    if n == 0:
        return "not_evaluable"
    win_rate = wins / n
    if win_rate >= 0.8 and mean(parent_deltas) > 0 and mean(random_deltas) > 0 and mean(shuffled_deltas) > 0:
        return "stable_method_signal"
    if win_rate >= 0.6 and mean(parent_deltas) > 0:
        return "partial_signal"
    return "mixed_or_overfit_risk"


def mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else ""


def median(vals):
    return round(statistics.median(vals), 4) if vals else ""


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, seeds, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab Stable Attribution Report\n\n")
        f.write(f"Seeds: {', '.join(str(s) for s in seeds)}\n\n")
        f.write("Stable rules are built leave-seed-out from recurrent discovery signals in the other seeds.\n\n")
        f.write("| Ref | Metric | Seeds | Stable Mean | Stable-V2 | Stable-Parent | Stable-Random | Stable-Shuffled | Win Rate | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['reference_pdb']} | {r['metric']} | {r['n_seeds']} | {r['stable_mean']} | "
                f"{r['stable_minus_v2_mean']} | {r['stable_minus_parent_only_mean']} | {r['stable_minus_random_mean']} | {r['stable_minus_shuffled_mean']} | "
                f"{r['stable_beats_parent_random_and_shuffled_rate']} | {r['verdict']} |\n"
            )
        f.write("\nInterpretation: stable attribution is convincing only if it beats parent-only, random-rule, and shuffled-rule baselines across seeds.\n")


if __name__ == "__main__":
    main()
