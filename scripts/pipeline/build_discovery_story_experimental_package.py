#!/usr/bin/env python
"""Build an experimental package for the design-variable discovery story.

This package deliberately avoids calling candidates synthesis-ready. The goal is
to send a small experimental draft set, measure binding/state specificity, and
then mine hit/non-hit differences for emergent design variables.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build discovery-story experimental package")
    parser.add_argument("--shortlist", required=True, help="synthesis_candidate_shortlist_v2.csv")
    parser.add_argument("--full-fasta", required=True, help="synthesis_candidate_shortlist.fasta")
    parser.add_argument("--anarci", required=True, help="anarci_chothia_construct_summary.csv")
    parser.add_argument("--fold-evidence", default="", help="fv_colabfold_evidence.csv, optional")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def load_csv(path):
    if not path:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_fasta(path):
    records = defaultdict(dict)
    header = None
    seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    store(records, header, "".join(seq))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
    if header:
        store(records, header, "".join(seq))
    return records


def store(records, header, seq):
    parts = header.split("|")
    construct = next((p for p in parts if "_cg_" in p), None)
    chain_type = next((p for p in parts if p in ("heavy", "light")), None)
    if construct and chain_type:
        records[construct][chain_type] = {"header": header, "seq": seq}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shortlist = load_csv(args.shortlist)
    shortlist.sort(key=lambda r: int(r["shortlist_v2_rank"]))
    selected = shortlist[:args.top]
    fasta = load_fasta(args.full_fasta)
    anarci = {r["construct_id"]: r for r in load_csv(args.anarci)}
    fold = {r["construct_id"]: r for r in load_csv(args.fold_evidence)} if args.fold_evidence else {}

    rows = []
    fasta_lines = []
    for r in selected:
        cid = r["construct_id"]
        h = fasta.get(cid, {}).get("heavy", {})
        l = fasta.get(cid, {}).get("light", {})
        ar = anarci.get(cid, {})
        fr = fold.get(cid, {})
        evidence_gaps = [
            "final_Fv_fold_JSON_or_PDB_not_available",
            "side_chain_rebuild_or_repack_not_done",
            "experimental_binding_not_done",
            "monomer_oligomer_fibril_specificity_not_done",
        ]
        rows.append({
            "experimental_rank": len(rows) + 1,
            "construct_id": cid,
            "reference_pdb": r["reference_pdb"],
            "candidate_id": r["candidate_id"],
            "candidate_score": r["candidate_score"],
            "contact_retention": r["contact_retention"],
            "n_mutations": r["n_mutations"],
            "applied_mutations": r["applied_mutations"],
            "anarci_chothia_status": ar.get("numbering_status", "missing"),
            "heavy_chain_type": ar.get("heavy_chain_type", ""),
            "light_chain_type": ar.get("light_chain_type", ""),
            "fold_sidecheck_status": fr.get("fold_sidecheck_status", "not_finalized"),
            "experimental_status": "experimental_draft_not_synthesis_ready_claim",
            "purpose": "discover_emergent_state_specificity_variables",
            "evidence_gaps": ";".join(evidence_gaps),
        })
        if h:
            fasta_lines.extend([f">{cid}|heavy|experimental_rank={len(rows)}", h["seq"]])
        if l:
            fasta_lines.extend([f">{cid}|light|experimental_rank={len(rows)}", l["seq"]])

    write_csv(out_dir / "top10_experimental_draft_constructs.csv", rows)
    with open(out_dir / "top10_experimental_draft_constructs.fasta", "w", encoding="utf-8") as f:
        f.write("\n".join(fasta_lines) + "\n")
    write_bli_template(out_dir / "binding_screen_template.csv", rows)
    write_specificity_template(out_dir / "state_specificity_screen_template.csv", rows)
    write_story(out_dir / "discovery_story_plan.md", rows)
    summary = {
        "selected": len(rows),
        "status": "experimental_draft_not_synthesis_ready_claim",
        "core_claim": "pipeline_as_discovery_instrument_for_emergent_design_variables",
        "must_not_claim": ["synthesis_ready", "therapeutic_ready", "fold_validated"],
    }
    with open(out_dir / "experimental_package_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote discovery experimental package to {out_dir}")
    print(f"Selected={len(rows)}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_bli_template(path, rows):
    fields = [
        "construct_id", "expression_yield_mg_L", "purity_percent", "abeta_peptide", "kon", "koff",
        "kd_nM", "response_RU_or_nm", "binding_call", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"construct_id": r["construct_id"], "abeta_peptide": "DAEFRH_or_DAEFRHDSGY_or_KLVFFAED"})


def write_specificity_template(path, rows):
    fields = [
        "construct_id", "monomer_kd_nM", "oligomer_kd_nM", "fibril_kd_nM",
        "oligomer_vs_monomer_gap", "fibril_vs_monomer_gap", "state_specificity_call", "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({"construct_id": r["construct_id"]})


def write_story(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Discovery Story Experimental Plan\n\n")
        f.write("Core narrative: known antibodies provide contact rules; the pipeline generates constrained variants; experiments identify hits; hit/non-hit contrasts reveal emergent variables such as side-chain volume at state-sensitive contacts.\n\n")
        f.write("## Corrected Status\n\n")
        f.write("- ANARCI/Chothia numbering is complete for the selected VH/VL sequences.\n")
        f.write("- ColabFold smoke runs are not fold-pass evidence. No final Fv fold JSON/PDB is currently available for the MSA run.\n")
        f.write("- These constructs are experimental drafts, not synthesis-ready or therapeutic-ready claims.\n\n")
        f.write("## Top Constructs\n\n")
        f.write("| Rank | Construct | Ref | Mutations | Contact Retention | Purpose |\n")
        f.write("|---:|---|---|---:|---:|---|\n")
        for r in rows:
            f.write(
                f"| {r['experimental_rank']} | {r['construct_id']} | {r['reference_pdb']} | "
                f"{r['n_mutations']} | {r['contact_retention']} | {r['purpose']} |\n"
            )
        f.write("\n## Required Experimental Readout\n\n")
        f.write("1. Expression and purification success/failure.\n")
        f.write("2. BLI/SPR binding against target peptide or aggregate prep.\n")
        f.write("3. Monomer vs oligomer vs fibril specificity.\n")
        f.write("4. Hit/non-hit contact feature contrast to discover emergent design variables.\n")


if __name__ == "__main__":
    main()
