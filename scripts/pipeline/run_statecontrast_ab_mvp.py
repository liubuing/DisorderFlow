#!/usr/bin/env python
"""Run StateContrast-Ab MVP on A-beta reference contact maps."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from contact_variable_attribution import compare_high_low_gap  # noqa: E402
from idp_state_ensemble import build_negative_states, positive_state  # noqa: E402
from state_contact_scorer import extract_contact_map, generate_contact_guided_variants  # noqa: E402
from state_contrast_scorer import rank_state_contrast_candidates  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run StateContrast-Ab MVP")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples-per-ref", type=int, default=500)
    parser.add_argument("--max-mutations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_refs(args.whitelist)
    all_scores = []
    all_attr = []
    all_enrich = []
    summaries = []
    for ref_idx, ref in enumerate(refs):
        cmap = extract_contact_map(ref["path"], peptide_chain=ref.get("peptide_chain"))
        pos = [positive_state(f"{ref['pdb']}_positive_bound", cmap, "known antibody-bound positive state")]
        neg = build_negative_states(cmap, ref["pdb"], seed=args.seed + ref_idx)
        variants = generate_contact_guided_variants(
            cmap,
            n=args.samples_per_ref,
            max_mutations=args.max_mutations,
            seed=args.seed + ref_idx,
        )
        variants.append({
            "candidate_id": "native",
            "sequence": cmap["paratope_sequence"],
            "mutations": "",
            "n_mutations": 0,
            "generator": "native_reference",
        })
        ranked = rank_state_contrast_candidates(variants, pos, neg)
        for r in ranked:
            all_scores.append({
                "reference_pdb": ref["pdb"],
                "peptide_sequence": cmap["peptide_sequence"],
                **{k: v for k, v in r.items() if k not in ("positive_states", "negative_states")},
            })
        attr = compare_high_low_gap(ranked, cmap, top_fraction=0.20)
        for r in attr["feature_rows"]:
            all_attr.append({"reference_pdb": ref["pdb"], **r})
        for r in attr["enrichment_rows"]:
            all_enrich.append({"reference_pdb": ref["pdb"], **r})
        summaries.append({
            "reference_pdb": ref["pdb"],
            "variants": len(variants),
            "best_candidate": ranked[0]["candidate_id"],
            "best_gap": ranked[0]["specificity_gap"],
            "best_gap_delta": ranked[0]["specificity_gap_delta"],
            "native_gap": next(r["specificity_gap"] for r in ranked if r["candidate_id"] == "native"),
            "n_high_gap_for_attribution": attr["n_high"],
            "n_low_gap_for_attribution": attr["n_low"],
        })

    score_fields = [
        "reference_pdb", "statecontrast_rank", "candidate_id", "sequence", "mutations", "n_mutations",
        "generator", "positive_score", "negative_max_score", "negative_mean_score",
        "specificity_gap", "native_specificity_gap", "specificity_gap_delta", "statecontrast_score", "peptide_sequence",
    ]
    write_csv(out_dir / "statecontrast_candidate_scores.csv", all_scores, score_fields)
    write_csv(out_dir / "statecontrast_contact_features.csv", all_attr)
    write_csv(out_dir / "statecontrast_high_low_enrichment.csv", all_enrich)
    with open(out_dir / "statecontrast_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2)
    write_report(out_dir / "statecontrast_ab_mvp_report.md", summaries, all_scores)
    print(f"Wrote StateContrast-Ab MVP to {out_dir}")
    print(f"References={len(refs)} candidates={len(all_scores)}")


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("abeta_references", [])


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries, scores):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# StateContrast-Ab MVP Report\n\n")
        f.write("This MVP treats IDP antibody design as positive-vs-negative state discrimination, not sequence epitope affinity alone. Negative states are synthetic contact-topology perturbations used for computational contrast.\n\n")
        f.write("| Ref | Variants | Best Candidate | Best Gap | Native Gap | Gap Delta |\n")
        f.write("|---|---:|---|---:|---:|---:|\n")
        for s in summaries:
            f.write(f"| {s['reference_pdb']} | {s['variants']} | {s['best_candidate']} | {s['best_gap']} | {s['native_gap']} | {s['best_gap_delta']} |\n")
        f.write("\nTop state-contrast candidates optimize improvement over native gap: gap_delta = candidate_gap - native_gap. Raw gaps can be negative when synthetic negative states are deliberately strong.\n")


if __name__ == "__main__":
    main()
