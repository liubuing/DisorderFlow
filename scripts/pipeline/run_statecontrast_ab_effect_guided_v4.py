#!/usr/bin/env python
"""Run leave-seed-out mutation effect-size guided StateContrast-Ab v4."""
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

from contact_variable_attribution import mutation_string  # noqa: E402
from idp_state_ensemble import build_negative_states, positive_state  # noqa: E402
from run_statecontrast_ab_overfit_controls import (  # noqa: E402
    matched_random_rules,
    native_variant,
    select_top_parents,
    shuffled_position_rules,
)
from run_statecontrast_ab_stable_attribution import mean, median  # noqa: E402
from state_contact_scorer import AA, extract_contact_map, generate_contact_guided_variants  # noqa: E402
from state_contrast_scorer import rank_state_contrast_candidates  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run effect-size guided v4 benchmark")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="31,41,51,61,71")
    parser.add_argument("--samples-per-ref", type=int, default=150)
    parser.add_argument("--v4-samples-per-ref", type=int, default=150)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--min-effect", type=float, default=0.004)
    parser.add_argument("--min-recurrence", type=int, default=2)
    parser.add_argument("--rule-mode", choices=("mixed", "mutation", "position"), default="mixed")
    parser.add_argument("--heldout-seed-offset", type=int, default=10000)
    parser.add_argument(
        "--validation-mode",
        choices=("leave_seed_out", "leave_reference_out"),
        default="leave_seed_out",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    refs = load_refs(args.whitelist)
    contexts = build_contexts(refs, seeds, args)
    all_rules = []
    all_scores = []
    all_seed_summaries = []
    for seed in seeds:
        for ref in refs:
            ref_name = ref["pdb"]
            ctx = contexts[(seed, ref_name)]
            if args.validation_mode == "leave_reference_out":
                rules = select_transferable_position_rules(contexts, refs, ref_name, ctx["cmap"], args)
            else:
                training_ranked = [row for (s, rname), c in contexts.items() if s != seed and rname == ref_name for row in c["discovery_ranked"]]
                rules = select_effect_rules(training_ranked, ref_name, args)
            all_rules.extend({"seed": seed, **r} for r in rules)
            parent_rows = select_top_parents(ctx["discovery_ranked"], top_fraction=0.20)
            rng = random.Random(seed)
            random_rules = random_effect_rules_like(rules, len(ctx["cmap"]["paratope_sequence"]), rng)
            shuffled_rules = shuffle_effect_rule_positions(rules, len(ctx["cmap"]["paratope_sequence"]), rng)
            arms = {
                "v2_discovery_pool_heldout_eval": ctx["v2_variants"],
                "top_parent_only_heldout_eval": generate_effect_guided_variants(ctx["cmap"], parent_rows, [], args.v4_samples_per_ref, args.max_mutations, seed + 100, ref_name),
                "effect_guided_v4_heldout_eval": generate_effect_guided_variants(ctx["cmap"], parent_rows, rules, args.v4_samples_per_ref, args.max_mutations, seed + 200, ref_name),
                "effect_random_rule_v4_heldout_eval": generate_effect_guided_variants(ctx["cmap"], parent_rows, random_rules, args.v4_samples_per_ref, args.max_mutations, seed + 300, ref_name),
                "effect_shuffled_rule_v4_heldout_eval": generate_effect_guided_variants(ctx["cmap"], parent_rows, shuffled_rules, args.v4_samples_per_ref, args.max_mutations, seed + 400, ref_name),
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
            all_seed_summaries.extend(summarize_seed_ref(seed, ref_name, seed_ref_scores, len(rules)))
    aggregate = aggregate_v4(all_seed_summaries)
    write_csv(out_dir / "statecontrast_effect_guided_v4_rules.csv", all_rules)
    write_csv(out_dir / "statecontrast_effect_guided_v4_scores.csv", all_scores)
    write_csv(out_dir / "statecontrast_effect_guided_v4_seed_summary.csv", all_seed_summaries)
    write_csv(out_dir / "statecontrast_effect_guided_v4_summary.csv", aggregate)
    with open(out_dir / "statecontrast_effect_guided_v4_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "rule_mode": args.rule_mode, "validation_mode": args.validation_mode, "summaries": aggregate}, f, indent=2)
    write_report(out_dir / "statecontrast_effect_guided_v4_report.md", seeds, aggregate, args.rule_mode, args.validation_mode)
    print(f"Wrote effect-guided v4 benchmark to {out_dir}")
    print(f"Seeds={len(seeds)} score_rows={len(all_scores)}")


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("abeta_references", [])


def build_contexts(refs, seeds, args):
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
            for row in discovery_ranked:
                row["seed"] = seed
                row["reference_pdb"] = ref_name
            contexts[(seed, ref_name)] = {"cmap": cmap, "pos": pos, "heldout_neg": heldout_neg, "v2_variants": v2_variants, "discovery_ranked": discovery_ranked}
    return contexts


def stable_effect_rules(rows, reference_pdb, min_support=4, min_effect=0.004, min_recurrence=2):
    seed_groups = rows_by_seed(rows)
    recurrent = defaultdict(list)
    for seed, seed_rows in seed_groups.items():
        effects = mutation_effects_for_rows(seed_rows, min_support=min_support)
        for effect in effects:
            if effect["mean_delta"] >= min_effect:
                key = (effect["pos"], effect["native"], effect["new"])
                recurrent[key].append({"seed": seed, **effect})
    out = []
    for key, vals in recurrent.items():
        if len({v["seed"] for v in vals}) < min_recurrence:
            continue
        pos, native, new = key
        out.append({
            "reference_pdb": reference_pdb,
            "rule_type": "mutation",
            "paratope_index": pos,
            "native_aa": native,
            "new_aa": new,
            "mean_delta": round(sum(v["mean_delta"] for v in vals) / len(vals), 4),
            "recurrence": len({v["seed"] for v in vals}),
            "support": sum(v["support"] for v in vals),
            "training_seeds": ";".join(str(v["seed"]) for v in sorted(vals, key=lambda x: x["seed"])),
        })
    out.sort(key=lambda r: (r["recurrence"], r["mean_delta"], r["support"]), reverse=True)
    return out


def select_effect_rules(training_ranked, ref_name, args):
    mutation_rules = stable_effect_rules(
        training_ranked,
        ref_name,
        min_support=args.min_support,
        min_effect=args.min_effect,
        min_recurrence=args.min_recurrence,
    )
    position_rules = stable_position_rules(
        training_ranked,
        ref_name,
        min_support=args.min_support,
        min_effect=args.min_effect,
        min_recurrence=args.min_recurrence,
    )
    if args.rule_mode == "mutation":
        return mutation_rules
    if args.rule_mode == "position":
        return position_rules
    return mutation_rules or position_rules


def stable_position_rules(rows, reference_pdb, min_support=4, min_effect=0.004, min_recurrence=2):
    seed_groups = rows_by_seed(rows)
    recurrent = defaultdict(list)
    for seed, seed_rows in seed_groups.items():
        effects = position_effects_for_rows(seed_rows, min_support=min_support)
        for effect in effects:
            if effect["mean_delta"] >= min_effect:
                recurrent[effect["pos"]].append({"seed": seed, **effect})
    out = []
    for pos, vals in recurrent.items():
        if len({v["seed"] for v in vals}) < min_recurrence:
            continue
        out.append({
            "reference_pdb": reference_pdb,
            "rule_type": "position",
            "paratope_index": pos,
            "native_aa": "",
            "new_aa": "",
            "mean_delta": round(sum(v["mean_delta"] for v in vals) / len(vals), 4),
            "recurrence": len({v["seed"] for v in vals}),
            "support": sum(v["support"] for v in vals),
            "training_seeds": ";".join(str(v["seed"]) for v in sorted(vals, key=lambda x: x["seed"])),
        })
    out.sort(key=lambda r: (r["recurrence"], r["mean_delta"], r["support"]), reverse=True)
    return out


def chain_role(ref, chain):
    if chain == ref.get("heavy_chain"):
        return "heavy"
    if chain == ref.get("light_chain"):
        return "light"
    return None


def transferable_position_effects(contexts, refs, heldout_ref, args):
    refs_by_name = {ref["pdb"]: ref for ref in refs}
    evidence = defaultdict(list)
    for (seed, ref_name), ctx in contexts.items():
        if ref_name == heldout_ref:
            continue
        ref = refs_by_name[ref_name]
        residues = ctx["cmap"]["paratope_residues"]
        for effect in position_effects_for_rows(ctx["discovery_ranked"], args.min_support):
            if effect["mean_delta"] < args.min_effect:
                continue
            residue = residues[effect["pos"] - 1]
            role = chain_role(ref, residue["chain"])
            if role:
                evidence[(role, int(residue["resid"]))].append({
                    "reference_pdb": ref_name,
                    "seed": seed,
                    **effect,
                })
    return evidence


def select_transferable_position_rules(contexts, refs, heldout_ref, heldout_cmap, args):
    evidence = transferable_position_effects(contexts, refs, heldout_ref, args)
    heldout = next(ref for ref in refs if ref["pdb"] == heldout_ref)
    local_positions = {}
    for index, residue in enumerate(heldout_cmap["paratope_residues"], start=1):
        role = chain_role(heldout, residue["chain"])
        if role:
            local_positions[(role, int(residue["resid"]))] = index

    rules = []
    for key, values in evidence.items():
        source_refs = sorted({value["reference_pdb"] for value in values})
        if len(source_refs) < args.min_recurrence or key not in local_positions:
            continue
        role, resid = key
        rules.append({
            "reference_pdb": heldout_ref,
            "rule_type": "position",
            "paratope_index": local_positions[key],
            "native_aa": "",
            "new_aa": "",
            "mean_delta": round(sum(value["mean_delta"] for value in values) / len(values), 4),
            "recurrence": len(source_refs),
            "support": sum(value["support"] for value in values),
            "training_seeds": ";".join(str(seed) for seed in sorted({value["seed"] for value in values})),
            "normalized_position": f"{role}:{resid}",
            "source_references": ";".join(source_refs),
        })
    rules.sort(key=lambda row: (row["recurrence"], row["mean_delta"], row["support"]), reverse=True)
    return rules


def rows_by_seed(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row.get("seed", 0))].append(row)
    return groups


def mutation_effects_for_rows(rows, min_support=4):
    parsed = [(float(r["specificity_gap_delta"]), parse_mutations(r.get("mutations", ""))) for r in rows if r.get("candidate_id") != "native"]
    keys = sorted({(m["pos"], m["native"], m["new"]) for _, muts in parsed for m in muts})
    out = []
    for key in keys:
        present = [v for v, muts in parsed if any((m["pos"], m["native"], m["new"]) == key for m in muts)]
        absent = [v for v, muts in parsed if not any((m["pos"], m["native"], m["new"]) == key for m in muts)]
        if len(present) < min_support or len(absent) < min_support:
            continue
        pos, native, new = key
        out.append({"pos": pos, "native": native, "new": new, "support": len(present), "mean_delta": sum(present) / len(present) - sum(absent) / len(absent)})
    return out


def position_effects_for_rows(rows, min_support=4):
    parsed = [(float(r["specificity_gap_delta"]), parse_mutations(r.get("mutations", ""))) for r in rows if r.get("candidate_id") != "native"]
    keys = sorted({m["pos"] for _, muts in parsed for m in muts})
    out = []
    for pos in keys:
        present = [v for v, muts in parsed if any(m["pos"] == pos for m in muts)]
        absent = [v for v, muts in parsed if not any(m["pos"] == pos for m in muts)]
        if len(present) < min_support or len(absent) < min_support:
            continue
        out.append({"pos": pos, "support": len(present), "mean_delta": sum(present) / len(present) - sum(absent) / len(absent)})
    return out


def parse_mutations(text):
    out = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if len(token) < 3:
            continue
        native, new = token[0], token[-1]
        try:
            pos = int(token[1:-1])
        except ValueError:
            continue
        out.append({"native": native, "pos": pos, "new": new})
    return out


def generate_effect_guided_variants(contact_map, parent_rows, rules, n, max_mutations, seed, reference_pdb):
    rng = random.Random(seed)
    native = contact_map["paratope_sequence"]
    parents = list(parent_rows) or []
    parents.append({"candidate_id": "native", "sequence": native})
    seen = {native}
    variants = []
    attempts = 0
    while len(variants) < n and attempts < n * 1000:
        attempts += 1
        parent = rng.choice(parents)
        seq = list(parent["sequence"])
        active_rules = list(rules)[:max_mutations]
        hits = apply_effect_rules(seq, native, active_rules, rng)
        explore_positions(seq, native, active_rules, max_mutations, rng, hits)
        s = "".join(seq)
        if s in seen or s == native:
            continue
        seen.add(s)
        variants.append({
            "candidate_id": f"v4_{len(variants) + 1:04d}",
            "sequence": s,
            "mutations": mutation_string(native, s),
            "n_mutations": sum(1 for a, b in zip(native, s) if a != b),
            "generator": "effect_guided_v4",
            "parent_candidate_id": parent.get("candidate_id", ""),
            "reference_pdb": reference_pdb,
            "rule_hits": ";".join(hits),
        })
    fallback_attempts = 0
    while len(variants) < n and fallback_attempts < n * 1000:
        fallback_attempts += 1
        parent = rng.choice(parents)
        seq = list(parent["sequence"])
        hits = []
        explore_positions(seq, native, [], max_mutations, rng, hits)
        s = "".join(seq)
        if s in seen or s == native:
            continue
        seen.add(s)
        variants.append({
            "candidate_id": f"v4_{len(variants) + 1:04d}",
            "sequence": s,
            "mutations": mutation_string(native, s),
            "n_mutations": sum(1 for a, b in zip(native, s) if a != b),
            "generator": "effect_guided_v4_fallback_parent_explore",
            "parent_candidate_id": parent.get("candidate_id", ""),
            "reference_pdb": reference_pdb,
            "rule_hits": ";".join(hits),
        })
    return variants


def apply_effect_rules(seq, native, rules, rng):
    hits = []
    for rule in rules:
        pos = int(rule["paratope_index"]) - 1
        if pos < 0 or pos >= len(seq):
            continue
        if rule["rule_type"] == "mutation" and rule.get("new_aa"):
            seq[pos] = rule["new_aa"]
            hits.append(f"effect_mutation:{pos + 1}:{rule['new_aa']}")
        elif rule["rule_type"] == "position":
            choices = [aa for aa in AA if aa != seq[pos] and aa != "C"]
            seq[pos] = rng.choice(choices)
            hits.append(f"effect_position:{pos + 1}:{seq[pos]}")
    return hits


def explore_positions(seq, native, rules, max_mutations, rng, hits):
    current = sum(1 for a, b in zip(seq, native) if a != b)
    budget = max(0, max_mutations - current)
    if budget <= 0:
        return
    protected = {int(r["paratope_index"]) - 1 for r in rules}
    choices = [i for i in range(len(seq)) if i not in protected]
    rng.shuffle(choices)
    for pos in choices[:rng.randint(1, max(1, budget))]:
        aas = [aa for aa in AA if aa != seq[pos] and aa != "C"]
        seq[pos] = rng.choice(aas)
        hits.append(f"explore:{pos + 1}:{seq[pos]}")


def random_effect_rules_like(rules, sequence_len, rng):
    out = []
    for rule in rules:
        pos = rng.randint(1, sequence_len)
        new_aa = rng.choice([aa for aa in AA if aa != "C"])
        out.append({**rule, "rule_type": "mutation", "paratope_index": pos, "native_aa": "", "new_aa": new_aa})
    return out


def shuffle_effect_rule_positions(rules, sequence_len, rng):
    out = []
    for rule in rules:
        old = int(rule["paratope_index"])
        choices = [i for i in range(1, sequence_len + 1) if i != old]
        out.append({**rule, "paratope_index": rng.choice(choices or [old])})
    return out


def summarize_seed_ref(seed, reference_pdb, rows, n_rules):
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        vals = sorted([float(r["specificity_gap_delta"]) for r in rows if r["arm"] == arm], reverse=True)
        top_n = max(1, min(10, len(vals)))
        top_pct_n = max(1, int(len(vals) * 0.05))
        out.append({
            "seed": seed,
            "reference_pdb": reference_pdb,
            "arm": arm,
            "n_candidates": len(vals),
            "n_rules": n_rules,
            "best_gap_delta": round(vals[0], 4),
            "top10_mean_gap_delta": round(sum(vals[:top_n]) / top_n, 4),
            "top5pct_mean_gap_delta": round(sum(vals[:top_pct_n]) / top_pct_n, 4),
            "median_gap_delta": round(statistics.median(vals), 4),
        })
    return out


def aggregate_v4(rows):
    out = []
    for ref in sorted({r["reference_pdb"] for r in rows}):
        ref_rows = [r for r in rows if r["reference_pdb"] == ref]
        for metric in ("best_gap_delta", "top10_mean_gap_delta", "top5pct_mean_gap_delta", "median_gap_delta"):
            out.append(summarize_v4_metric(ref, ref_rows, metric))
    return out


def summarize_v4_metric(reference_pdb, rows, metric):
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["arm"]] = float(row[metric])
    vals = []
    v2_d, parent_d, random_d, shuffled_d = [], [], [], []
    wins = 0
    for _, arms in by_seed.items():
        guided = arms.get("effect_guided_v4_heldout_eval")
        v2 = arms.get("v2_discovery_pool_heldout_eval")
        parent = arms.get("top_parent_only_heldout_eval")
        random_arm = arms.get("effect_random_rule_v4_heldout_eval")
        shuffled = arms.get("effect_shuffled_rule_v4_heldout_eval")
        if guided is None or v2 is None or parent is None or random_arm is None or shuffled is None:
            continue
        vals.append(guided)
        v2_d.append(guided - v2)
        parent_d.append(guided - parent)
        random_d.append(guided - random_arm)
        shuffled_d.append(guided - shuffled)
        if guided > parent and guided > random_arm and guided > shuffled:
            wins += 1
    n = len(vals)
    return {
        "reference_pdb": reference_pdb,
        "metric": metric,
        "n_seeds": n,
        "v4_mean": mean(vals),
        "v4_median": median(vals),
        "v4_minus_v2_mean": mean(v2_d),
        "v4_minus_parent_only_mean": mean(parent_d),
        "v4_minus_random_mean": mean(random_d),
        "v4_minus_shuffled_mean": mean(shuffled_d),
        "v4_beats_parent_random_and_shuffled_rate": round(wins / max(1, n), 4),
        "verdict": v4_verdict(wins, n, parent_d, random_d, shuffled_d),
    }


def v4_verdict(wins, n, parent_d, random_d, shuffled_d):
    if n == 0:
        return "not_evaluable"
    win_rate = wins / n
    if win_rate >= 0.8 and mean(parent_d) > 0 and mean(random_d) > 0 and mean(shuffled_d) > 0:
        return "stable_method_signal"
    if win_rate >= 0.6 and mean(parent_d) > 0:
        return "partial_signal"
    return "mixed_or_overfit_risk"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, seeds, rows, rule_mode, validation_mode="leave_seed_out"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab Effect-Guided V4 Report\n\n")
        f.write(f"Seeds: {', '.join(str(s) for s in seeds)}\n\n")
        f.write(f"Rule mode: {rule_mode}\n\n")
        f.write(f"Validation mode: {validation_mode}\n\n")
        if validation_mode == "leave_reference_out":
            f.write("Rules are learned only from other references and transferred by normalized heavy/light-chain PDB residue position.\n\n")
        else:
            f.write("Rules are learned leave-seed-out from recurrent positive mutation/position effect sizes.\n\n")
        f.write("| Ref | Metric | Seeds | V4 Mean | V4-V2 | V4-Parent | V4-Random | V4-Shuffled | Win Rate | Verdict |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['reference_pdb']} | {r['metric']} | {r['n_seeds']} | {r['v4_mean']} | "
                f"{r['v4_minus_v2_mean']} | {r['v4_minus_parent_only_mean']} | {r['v4_minus_random_mean']} | {r['v4_minus_shuffled_mean']} | "
                f"{r['v4_beats_parent_random_and_shuffled_rate']} | {r['verdict']} |\n"
            )
        f.write("\nInterpretation: v4 is convincing only if effect-guided variants beat parent-only, random-effect, and shuffled-effect controls across seeds.\n")


if __name__ == "__main__":
    main()
