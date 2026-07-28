#!/usr/bin/env python
"""Generate minimal 5CSZ VH N-glycosylation rescue variants.

The problematic motif is the 5CSZ heavy-chain variable-region A:52-A:54 NAS
motif. This script edits the already selected paratope candidate sequence at
contact-map positions corresponding to A:52 or A:54, then rescoring with the
same contact-aware scorer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules"))

from state_contact_scorer import extract_contact_map, score_contact_guided_variants  # noqa: E402


RESCUE_OPTIONS = [
    ("A", 52, "N52H", "H"),
    ("A", 52, "N52Q", "Q"),
    ("A", 52, "N52D", "D"),
    ("A", 54, "S54A", "A"),
    ("A", 54, "S54F", "F"),
    ("A", 54, "S54Y", "Y"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Rescue 5CSZ N-glycosylation motifs")
    parser.add_argument("--selected", required=True, help="selected_contact_guided_library.csv")
    parser.add_argument("--review", required=True, help="synthesis_candidate_shortlist_v3/synthesis_candidate_shortlist_v2.csv")
    parser.add_argument("--whitelist", default=str(PROJECT_ROOT / "configs" / "idp" / "abeta_reference_whitelist.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-retention", type=float, default=0.98)
    return parser.parse_args()


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_refs(path):
    with open(path, encoding="utf-8") as f:
        return {r["pdb"]: r for r in (yaml.safe_load(f) or {}).get("abeta_references", [])}


def paratope_index_by_residue(contact_map):
    return {
        (str(r["chain"]), int(r["resid"])): i
        for i, r in enumerate(contact_map["paratope_residues"])
    }


def has_nglyco_motif(seq, idx_by_residue=None):
    import re
    if re.search(r"N[^P][ST]", seq):
        return True
    # A:54 is followed by G/T in the full 5CSZ VH sequence but only A:54 is in
    # the contact-map paratope string, so the paratope-only regex cannot see NGT.
    if idx_by_residue:
        idx54 = idx_by_residue.get(("A", 54))
        if idx54 is not None and seq[idx54] == "N":
            return True
    return False


def mutate(seq, idx, aa):
    return seq[:idx] + aa + seq[idx + 1:]


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        (r["reference_pdb"], r["candidate_id"]): r
        for r in load_csv(args.selected)
    }
    review_rows = [
        r for r in load_csv(args.review)
        if r["reference_pdb"] == "5CSZ" and r["shortlist_v2_status"] == "developability_review"
    ]
    refs = load_refs(args.whitelist)
    ref = refs["5CSZ"]
    cmap = extract_contact_map(ref["path"], peptide_chain=ref.get("peptide_chain"))
    idx_by_residue = paratope_index_by_residue(cmap)

    rescue_rows = []
    for row in review_rows:
        selected_row = selected[(row["reference_pdb"], row["candidate_id"])]
        base_seq = selected_row["sequence"]
        variants = []
        variant_meta = []
        for chain, resid, label, aa in RESCUE_OPTIONS:
            idx = idx_by_residue.get((chain, resid))
            if idx is None or base_seq[idx] == aa:
                continue
            seq = mutate(base_seq, idx, aa)
            if has_nglyco_motif(seq, idx_by_residue):
                continue
            variants.append({"candidate_id": f"{row['candidate_id']}_{label}", "sequence": seq})
            variant_meta.append({"rescue_label": label, "rescue_mutation": f"{chain}:{resid}:{base_seq[idx]}>{aa}"})
        scored = score_contact_guided_variants(cmap, variants)
        meta_by_id = {v["candidate_id"]: m for v, m in zip(variants, variant_meta)}
        for s in scored:
            meta = meta_by_id[s["candidate_id"]]
            rescue_rows.append({
                "source_construct_id": row["construct_id"],
                "source_candidate_id": row["candidate_id"],
                "rescue_candidate_id": s["candidate_id"],
                "source_sequence": base_seq,
                "rescue_sequence": s["sequence"],
                "source_contact_retention": row["contact_retention"],
                "rescue_contact_retention": s["contact_retention"],
                "rescue_candidate_score": s["candidate_score"],
                "rescue_hotspot_score": s["hotspot_score"],
                "rescue_mutation": meta["rescue_mutation"],
                "rescue_label": meta["rescue_label"],
                "passes_rescue_filter": s["contact_retention"] >= args.min_retention and s["passes_filters"],
                "filter_failures": s["filter_failures"],
            })
    rescue_rows.sort(key=lambda r: (r["passes_rescue_filter"], float(r["rescue_candidate_score"]), float(r["rescue_contact_retention"])), reverse=True)
    for i, r in enumerate(rescue_rows, 1):
        r["rescue_rank"] = i

    fields = [
        "rescue_rank", "source_construct_id", "source_candidate_id", "rescue_candidate_id",
        "rescue_mutation", "rescue_label", "source_contact_retention", "rescue_contact_retention",
        "rescue_candidate_score", "rescue_hotspot_score", "passes_rescue_filter",
        "filter_failures", "source_sequence", "rescue_sequence",
    ]
    with open(out_dir / "5csz_nglyco_rescue_candidates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rescue_rows)
    with open(out_dir / "5csz_nglyco_rescue_candidates.fasta", "w", encoding="utf-8") as f:
        for r in rescue_rows:
            if str(r["passes_rescue_filter"]) == "True":
                f.write(
                    f">{r['rescue_candidate_id']}|source={r['source_construct_id']}|"
                    f"mutation={r['rescue_mutation']}|retention={r['rescue_contact_retention']}\n"
                    f"{r['rescue_sequence']}\n"
                )
    summary = {
        "review_inputs": len(review_rows),
        "rescues_generated": len(rescue_rows),
        "rescues_passing": sum(1 for r in rescue_rows if r["passes_rescue_filter"]),
        "min_retention": args.min_retention,
        "contact_map": {**{k: v for k, v in cmap.items() if k != "contacts"}, "contacts": [asdict(c) for c in cmap["contacts"]]},
    }
    with open(out_dir / "5csz_nglyco_rescue_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "5csz_nglyco_rescue_report.md", "w", encoding="utf-8") as f:
        f.write("# 5CSZ N-Glycosylation Rescue Candidates\n\n")
        f.write(f"Review inputs: {summary['review_inputs']}; generated: {summary['rescues_generated']}; passing: {summary['rescues_passing']}\n\n")
        f.write("| Rank | Source | Rescue | Mutation | Retention | Score | Pass |\n")
        f.write("|---:|---|---|---|---:|---:|---|\n")
        for r in rescue_rows[:30]:
            f.write(
                f"| {r['rescue_rank']} | {r['source_construct_id']} | {r['rescue_candidate_id']} | "
                f"{r['rescue_mutation']} | {r['rescue_contact_retention']} | "
                f"{r['rescue_candidate_score']} | {r['passes_rescue_filter']} |\n"
            )
    print(f"Wrote 5CSZ N-glyco rescue candidates to {out_dir}")
    print(f"Inputs={len(review_rows)} generated={len(rescue_rows)} pass={summary['rescues_passing']}")


if __name__ == "__main__":
    main()
