#!/usr/bin/env python
"""Run StateContrast-Ab attribution-guided redesign with overfit controls."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from contact_variable_attribution import (  # noqa: E402
    AA,
    chemistry_class,
    compare_high_low_gap,
    compile_guidance_rules,
    generate_attribution_guided_variants,
    residues_for_classes,
    volume_class,
)
from idp_state_ensemble import build_negative_states, positive_state  # noqa: E402
from state_contact_scorer import extract_contact_map, generate_contact_guided_variants  # noqa: E402
from state_contrast_scorer import rank_state_contrast_candidates  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run StateContrast-Ab overfit controls")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-per-ref", type=int, default=300)
    parser.add_argument("--v3-samples-per-ref", type=int, default=300)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--heldout-seed-offset", type=int, default=10000)
    parser.add_argument("--min-abs-delta", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_refs(args.whitelist)
    all_scores = []
    all_rule_rows = []
    all_signal_rows = []
    all_summaries = []
    for ref_idx, ref in enumerate(refs):
        rng = random.Random(args.seed + ref_idx)
        cmap = extract_contact_map(ref["path"], peptide_chain=ref.get("peptide_chain"))
        pos = [positive_state(f"{ref['pdb']}_positive_bound", cmap, "known antibody-bound positive state")]
        discovery_neg = build_negative_states(cmap, f"{ref['pdb']}_discovery", seed=args.seed + ref_idx)
        heldout_neg = build_negative_states(cmap, f"{ref['pdb']}_heldout", seed=args.seed + args.heldout_seed_offset + ref_idx)

        v2_variants = generate_contact_guided_variants(cmap, n=args.samples_per_ref, max_mutations=args.max_mutations, seed=args.seed + ref_idx)
        v2_variants.append(native_variant(cmap))
        discovery_ranked = rank_state_contrast_candidates(v2_variants, pos, discovery_neg)
        discovery_attr = compare_high_low_gap(discovery_ranked, cmap, top_fraction=0.20)
        signal_rows = enrichment_deltas(ref["pdb"], discovery_attr["enrichment_rows"], min_abs_delta=args.min_abs_delta)
        all_signal_rows.extend(signal_rows)
        real_rules = compile_guidance_rules(signal_rows, min_abs_delta=args.min_abs_delta).get(ref["pdb"], {})
        random_rules = matched_random_rules(real_rules, len(cmap["paratope_sequence"]), rng)
        shuffled_rules = shuffled_position_rules(real_rules, len(cmap["paratope_sequence"]), rng)
        for arm, rules in (("guided_v3", real_rules), ("random_rule_v3", random_rules), ("shuffled_rule_v3", shuffled_rules)):
            all_rule_rows.extend(rule_rows(ref["pdb"], arm, rules))

        parents = select_top_parents(discovery_ranked, top_fraction=0.20)
        arms = {
            "v2_discovery_pool_heldout_eval": v2_variants,
            "top_parent_only_heldout_eval": generate_attribution_guided_variants(
                cmap, parents, {}, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                seed=args.seed + 404 + ref_idx, reference_pdb=ref["pdb"],
            ),
            "guided_v3_heldout_eval": generate_attribution_guided_variants(
                cmap, parents, real_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                seed=args.seed + 101 + ref_idx, reference_pdb=ref["pdb"],
            ),
            "random_rule_v3_heldout_eval": generate_attribution_guided_variants(
                cmap, parents, random_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                seed=args.seed + 202 + ref_idx, reference_pdb=ref["pdb"],
            ),
            "shuffled_rule_v3_heldout_eval": generate_attribution_guided_variants(
                cmap, parents, shuffled_rules, n=args.v3_samples_per_ref, max_mutations=args.max_mutations,
                seed=args.seed + 303 + ref_idx, reference_pdb=ref["pdb"],
            ),
        }
        ref_scores = []
        for arm, variants in arms.items():
            ranked = rank_state_contrast_candidates(variants, pos, heldout_neg)
            for r in ranked:
                row = {
                    "reference_pdb": ref["pdb"],
                    "arm": arm,
                    "peptide_sequence": cmap["peptide_sequence"],
                    **{k: v for k, v in r.items() if k not in ("positive_states", "negative_states")},
                }
                all_scores.append(row)
                ref_scores.append(row)
        all_summaries.extend(summarize_reference(ref["pdb"], ref_scores, len(real_rules), len(signal_rows)))

    score_fields = [
        "reference_pdb", "arm", "statecontrast_rank", "candidate_id", "sequence", "mutations", "n_mutations",
        "generator", "parent_candidate_id", "rule_hits", "positive_score", "negative_max_score", "negative_mean_score",
        "specificity_gap", "native_specificity_gap", "specificity_gap_delta", "statecontrast_score", "peptide_sequence",
    ]
    write_csv(out_dir / "statecontrast_overfit_control_scores.csv", all_scores, score_fields)
    write_csv(out_dir / "statecontrast_overfit_signal_deltas.csv", all_signal_rows)
    write_csv(out_dir / "statecontrast_overfit_rules.csv", all_rule_rows)
    write_csv(out_dir / "statecontrast_overfit_summary.csv", all_summaries)
    with open(out_dir / "statecontrast_overfit_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summaries": all_summaries}, f, indent=2)
    write_report(out_dir / "statecontrast_overfit_control_report.md", all_summaries)
    print(f"Wrote StateContrast-Ab overfit controls to {out_dir}")
    print(f"References={len(refs)} score_rows={len(all_scores)}")


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("abeta_references", [])


def native_variant(contact_map):
    return {"candidate_id": "native", "sequence": contact_map["paratope_sequence"], "mutations": "", "n_mutations": 0, "generator": "native_reference"}


def select_top_parents(rows, top_fraction=0.20):
    n = max(1, int(len(rows) * top_fraction))
    return rows[:n]


def enrichment_deltas(reference_pdb, enrichment_rows, min_abs_delta=0.20):
    grouped = {}
    for r in enrichment_rows:
        key = (r["paratope_index"], r["chain"], r["resid"], r["volume_class"], r["chemistry_class"])
        grouped.setdefault(key, {})[r["group"]] = float(r["fraction"])
    out = []
    for key, values in grouped.items():
        high = values.get("high_gap", 0.0)
        low = values.get("low_gap", 0.0)
        delta = round(high - low, 4)
        if abs(delta) < min_abs_delta:
            continue
        pidx, chain, resid, vol, chem = key
        out.append({
            "reference_pdb": reference_pdb,
            "paratope_index": pidx,
            "chain": chain,
            "resid": resid,
            "volume_class": vol,
            "chemistry_class": chem,
            "high_fraction": round(high, 4),
            "low_fraction": round(low, 4),
            "fraction_delta": delta,
            "direction": "high_gap_enriched" if delta > 0 else "low_gap_enriched",
        })
    out.sort(key=lambda r: abs(float(r["fraction_delta"])), reverse=True)
    return out


def matched_random_rules(real_rules, sequence_len, rng):
    rules = {}
    used = set()
    for _, rule in real_rules.items():
        pos = random_unused_position(sequence_len, used, rng)
        used.add(pos)
        rules[pos] = random_rule_like(rule, rng)
    return rules


def shuffled_position_rules(real_rules, sequence_len, rng):
    positions = list(real_rules.keys())
    if len(positions) == 1 and sequence_len > 1:
        old_pos = positions[0]
        choices = [i for i in range(sequence_len) if i != old_pos]
        return {rng.choice(choices): copy_rule(real_rules[old_pos])}
    shuffled = positions[:]
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == positions:
        shuffled = shuffled[1:] + shuffled[:1]
    return {new_pos: copy_rule(real_rules[old_pos]) for old_pos, new_pos in zip(positions, shuffled) if 0 <= new_pos < sequence_len}


def random_unused_position(sequence_len, used, rng):
    choices = [i for i in range(sequence_len) if i not in used]
    return rng.choice(choices or list(range(sequence_len)))


def random_rule_like(rule, rng):
    def rand_classes(classes):
        out = []
        for c in classes:
            vol, chem = random_nonempty_class(rng)
            out.append({"volume_class": vol, "chemistry_class": chem, "fraction_delta": c.get("fraction_delta", 0.0)})
        return out
    return {"preferred": rand_classes(rule.get("preferred", [])), "avoid": rand_classes(rule.get("avoid", [])), "chain": "random", "resid": ""}


def random_nonempty_class(rng):
    classes = sorted({(volume_class(aa), chemistry_class(aa)) for aa in AA if residues_for_classes(volume_class(aa), chemistry_class(aa))})
    return rng.choice(classes)


def copy_rule(rule):
    return {
        "preferred": [dict(c) for c in rule.get("preferred", [])],
        "avoid": [dict(c) for c in rule.get("avoid", [])],
        "chain": rule.get("chain", ""),
        "resid": rule.get("resid", ""),
    }


def rule_rows(reference_pdb, arm, rules):
    rows = []
    for pos, rule in sorted(rules.items()):
        for direction in ("preferred", "avoid"):
            for cls in rule.get(direction, []):
                rows.append({
                    "reference_pdb": reference_pdb,
                    "arm": arm,
                    "paratope_index": pos + 1,
                    "chain": rule.get("chain", ""),
                    "resid": rule.get("resid", ""),
                    "direction": direction,
                    "volume_class": cls["volume_class"],
                    "chemistry_class": cls["chemistry_class"],
                    "fraction_delta": cls.get("fraction_delta", ""),
                })
    return rows


def summarize_reference(reference_pdb, rows, n_rules, n_signals):
    arms = sorted({r["arm"] for r in rows})
    out = []
    for arm in arms:
        vals = sorted([float(r["specificity_gap_delta"]) for r in rows if r["arm"] == arm], reverse=True)
        if not vals:
            continue
        top_n = max(1, min(10, len(vals)))
        top_pct_n = max(1, int(len(vals) * 0.05))
        out.append({
            "reference_pdb": reference_pdb,
            "arm": arm,
            "n_candidates": len(vals),
            "n_rules": n_rules,
            "n_signals": n_signals,
            "best_gap_delta": round(vals[0], 4),
            "top10_mean_gap_delta": round(sum(vals[:top_n]) / top_n, 4),
            "top5pct_mean_gap_delta": round(sum(vals[:top_pct_n]) / top_pct_n, 4),
            "median_gap_delta": round(statistics.median(vals), 4),
        })
    add_verdicts(out)
    return out


def add_verdicts(rows):
    by_arm = {r["arm"]: r for r in rows}
    guided = by_arm.get("guided_v3_heldout_eval")
    parent = by_arm.get("top_parent_only_heldout_eval")
    random_arm = by_arm.get("random_rule_v3_heldout_eval")
    shuffled = by_arm.get("shuffled_rule_v3_heldout_eval")
    if not guided or not parent or not random_arm or not shuffled:
        for r in rows:
            r["overfit_control_verdict"] = "not_evaluable"
        return
    top10_pass = guided["top10_mean_gap_delta"] > parent["top10_mean_gap_delta"] and guided["top10_mean_gap_delta"] > random_arm["top10_mean_gap_delta"] and guided["top10_mean_gap_delta"] > shuffled["top10_mean_gap_delta"]
    top5_pass = guided["top5pct_mean_gap_delta"] > parent["top5pct_mean_gap_delta"] and guided["top5pct_mean_gap_delta"] > random_arm["top5pct_mean_gap_delta"] and guided["top5pct_mean_gap_delta"] > shuffled["top5pct_mean_gap_delta"]
    median_pass = guided["median_gap_delta"] > parent["median_gap_delta"] and guided["median_gap_delta"] > random_arm["median_gap_delta"] and guided["median_gap_delta"] > shuffled["median_gap_delta"]
    verdict = "passes_top_metrics" if top10_pass and top5_pass else "mixed_or_overfit_risk"
    if top10_pass and top5_pass and median_pass:
        verdict = "passes_top_and_median_metrics"
    for r in rows:
        r["overfit_control_verdict"] = verdict


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries):
    by_ref = {}
    for s in summaries:
        by_ref.setdefault(s["reference_pdb"], []).append(s)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab Overfit-Control Report\n\n")
        f.write("Discovery attribution and final scoring are separated by using different synthetic negative-state seeds. Guided v3 is compared against matched random-rule and position-shuffled-rule baselines.\n\n")
        for ref, rows in sorted(by_ref.items()):
            f.write(f"## {ref}\n\n")
            f.write("| Arm | N | Rules | Best Delta | Top10 Mean | Top5% Mean | Median |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(
                    f"| {r['arm']} | {r['n_candidates']} | {r['n_rules']} | {r['best_gap_delta']} | "
                    f"{r['top10_mean_gap_delta']} | {r['top5pct_mean_gap_delta']} | {r['median_gap_delta']} |\n"
                )
            f.write("\n")
            f.write(f"Verdict: {rows[0].get('overfit_control_verdict', 'not_evaluable')}\n\n")
        f.write("Interpretation rule: same-discovery improvement is not sufficient. A method signal requires guided_v3 held-out metrics to exceed top_parent_only, random_rule_v3, and shuffled_rule_v3 under matched candidate counts.\n")


if __name__ == "__main__":
    main()
