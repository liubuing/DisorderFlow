#!/usr/bin/env python
"""Create a clean full-chain candidate shortlist from construct drafts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthesis candidate shortlist draft")
    parser.add_argument("--constructs", required=True, help="full_chain_constructs.csv")
    parser.add_argument("--construct-fasta", required=True, help="full_chain_constructs.fasta")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def load_constructs(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["selection_rank"] = int(r["selection_rank"])
        r["n_mutations"] = int(r["n_mutations"])
        r["candidate_score"] = float(r["candidate_score"])
        r["contact_retention"] = float(r["contact_retention"])
    return rows


def load_fasta(path):
    records = {}
    header = None
    seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header:
        records[header] = "".join(seq)
    return records


def fasta_records_for_construct(records, construct_id):
    out = []
    prefix = f"{construct_id}|"
    for header, seq in records.items():
        if header.startswith(prefix):
            out.append((header, seq))
    out.sort(key=lambda x: ("|light|" in x[0], x[0]))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    constructs = load_constructs(args.constructs)
    fasta = load_fasta(args.construct_fasta)
    clean = [r for r in constructs if not r.get("construct_risk_flags") and not r.get("warnings")]
    clean.sort(key=lambda r: (r["selection_rank"], -r["candidate_score"]))
    selected = clean[:args.top]

    fields = [
        "shortlist_rank", "construct_id", "selection_rank", "reference_pdb", "candidate_id",
        "heavy_chain", "light_chain", "heavy_length", "light_length", "n_mutations",
        "applied_mutations", "candidate_score", "contact_retention", "status", "required_side_evidence",
    ]
    side_evidence = [
        "ANARCI_or_Chothia_numbering",
        "Fv_or_VHVL_fold_stability_side_check",
        "side_chain_rebuild_or_repack",
        "developability_filter",
        "positive_negative_state_specificity_check",
    ]
    rows = []
    for i, r in enumerate(selected, 1):
        rows.append({
            **r,
            "shortlist_rank": i,
            "status": "draft_not_synthesis_ready",
            "required_side_evidence": ";".join(side_evidence),
        })

    with open(out_dir / "synthesis_candidate_shortlist.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "synthesis_candidate_shortlist.fasta", "w", encoding="utf-8") as f:
        for r in rows:
            for header, seq in fasta_records_for_construct(fasta, r["construct_id"]):
                f.write(f">shortlist_rank={r['shortlist_rank']}|{header}\n{seq}\n")

    mutation_rows = []
    for r in rows:
        for m in r["applied_mutations"].split(";") if r["applied_mutations"] else []:
            mutation_rows.append({
                "shortlist_rank": r["shortlist_rank"],
                "construct_id": r["construct_id"],
                "reference_pdb": r["reference_pdb"],
                "candidate_id": r["candidate_id"],
                "mutation": m,
            })
    with open(out_dir / "synthesis_candidate_mutations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["shortlist_rank", "construct_id", "reference_pdb", "candidate_id", "mutation"])
        writer.writeheader()
        writer.writerows(mutation_rows)

    summary = {
        "input_constructs": len(constructs),
        "clean_constructs": len(clean),
        "selected": len(rows),
        "excluded_flagged": len(constructs) - len(clean),
        "status": "draft_not_synthesis_ready",
        "required_side_evidence": side_evidence,
    }
    with open(out_dir / "synthesis_candidate_shortlist_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "synthesis_candidate_shortlist_report.md", "w", encoding="utf-8") as f:
        f.write("# Synthesis Candidate Shortlist Draft\n\n")
        f.write(f"Input constructs: {len(constructs)}; clean: {len(clean)}; selected: {len(rows)}; excluded flagged: {len(constructs) - len(clean)}\n\n")
        f.write("Status: `draft_not_synthesis_ready`\n\n")
        f.write("Required side evidence before synthesis:\n")
        for item in side_evidence:
            f.write(f"- `{item}`\n")
        f.write("\n| Shortlist | Construct | Ref | Mutations | Score | Retention |\n")
        f.write("|---:|---|---|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['shortlist_rank']} | {r['construct_id']} | {r['reference_pdb']} | "
                f"{r['n_mutations']} | {r['candidate_score']} | {r['contact_retention']} |\n"
            )

    print(f"Wrote synthesis candidate shortlist draft to {out_dir}")
    print(f"Input={len(constructs)} clean={len(clean)} selected={len(rows)} excluded_flagged={len(constructs) - len(clean)}")


if __name__ == "__main__":
    main()
