#!/usr/bin/env python
"""Join ensemble scores to exact full-chain, developability, and fold evidence."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_shortlist_variable_regions import find_variable_region  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", default="outputs/abeta_multiconf_candidate_scoring_v1/multiconf_candidate_ranking_all.csv")
    parser.add_argument("--source-candidates", default="outputs/abeta_contact_guided_candidates_v1/contact_guided_candidates_all.csv")
    parser.add_argument("--constructs", default="outputs/abeta_full_chain_constructs_v2/full_chain_constructs.csv")
    parser.add_argument("--construct-fasta", default="outputs/abeta_full_chain_constructs_v2/full_chain_constructs.fasta")
    parser.add_argument("--developability", default="outputs/synthesis_candidate_developability_fixed_export_v1/shortlist_developability.csv")
    parser.add_argument("--fold", default="outputs/abeta_multiconf_fold_evidence_v1/fv_colabfold_evidence.csv")
    parser.add_argument("--repack", default="outputs/abeta_multiconf_fv_repack_v1/fv_repack_audit.json")
    parser.add_argument("--out", default="outputs/abeta_multiconf_integrated_evidence_v1")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_fasta(path):
    records = {}
    header = None
    sequence = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    records[header] = "".join(sequence)
                header = line[1:]
                sequence = []
            elif line:
                sequence.append(line)
    if header:
        records[header] = "".join(sequence)
    return records


def main():
    args = parse_args()
    ranking = read_csv(args.ranking)
    sources = read_csv(args.source_candidates)
    constructs = read_csv(args.constructs)
    developability = {row["construct_id"]: row for row in read_csv(args.developability)}
    fold = {row["construct_id"]: row for row in read_csv(args.fold)}
    with open(args.repack, encoding="utf-8") as handle:
        repack_report = json.load(handle)
    repack = {row["construct_id"]: row for row in repack_report.get("records", [])}
    fasta = read_fasta(args.construct_fasta)

    source_sequence = {
        (row["reference_pdb"], row["candidate_id"]): row["sequence"] for row in sources
    }
    ranking_by_sequence = {
        (row["reference_pdb"], row["sequence"]): row for row in ranking
    }
    rows = []
    for construct in constructs:
        key = (construct["reference_pdb"], construct["candidate_id"])
        sequence = source_sequence.get(key)
        ensemble = ranking_by_sequence.get((construct["reference_pdb"], sequence)) if sequence else None
        if ensemble is None:
            continue
        construct_id = construct["construct_id"]
        heavy = next((seq for header, seq in fasta.items() if header.startswith(f"{construct_id}|heavy|")), "")
        light = next((seq for header, seq in fasta.items() if header.startswith(f"{construct_id}|light|")), "")
        dev = developability.get(construct_id, {})
        fold_row = fold.get(construct_id, {})
        repack_row = repack.get(construct_id, {})
        fold_status = fold_row.get("fold_sidecheck_status", "pending_not_run")
        dev_status = dev.get("developability_status", "pending_not_run")
        repack_status = repack_row.get("status", "pending_not_run")
        evidence_status = (
            "computational_side_evidence_complete_wetlab_pending"
            if dev_status == "developability_pass"
            and fold_status == "fold_sidecheck_pass"
            and repack_status == "repack_pass"
            else "computational_side_evidence_incomplete"
        )
        rows.append({
            "candidate_uid": ensemble["candidate_uid"],
            "construct_id": construct_id,
            "reference_pdb": construct["reference_pdb"],
            "candidate_id": construct["candidate_id"],
            "paratope_sequence": sequence,
            "heavy_sequence": heavy,
            "light_sequence": light,
            "multiconf_rank_within_reference": ensemble["multiconf_rank_within_reference"],
            "multiconf_robust_score": ensemble["multiconf_robust_score"],
            "pose_coverage": ensemble["pose_coverage"],
            "ensemble_min": ensemble["ensemble_min"],
            "ensemble_p25": ensemble["ensemble_p25"],
            "native_delta_p25": ensemble["native_delta_p25"],
            "developability_status": dev_status,
            "developability_risk": dev.get("combined_developability_risk", ""),
            "candidate_specific_flags": dev.get("introduced_or_candidate_specific_flags", ""),
            "fold_sidecheck_status": fold_status,
            "fold_mode": fold_row.get("mode_label", ""),
            "mean_plddt": fold_row.get("mean_plddt", ""),
            "iptm": fold_row.get("iptm", ""),
            "repack_status": repack_status,
            "repack_backbone_rmsd": repack_row.get("backbone_rmsd_angstrom", ""),
            "repack_interface_contact_retention": repack_row.get("interface_contact_retention", ""),
            "repack_global_clashes": repack_row.get("after_geometry", {}).get(
                "nonlocal_heavy_atom_clashes_lt_1_5A", ""
            ),
            "repacked_pdb": repack_row.get("repacked_pdb", ""),
            "integrated_evidence_status": evidence_status,
            "claim_boundary": "computational prioritization only; not synthesis-ready binding evidence",
        })
    rows.sort(key=lambda row: (
        row["developability_status"] != "developability_pass",
        -float(row["multiconf_robust_score"]),
    ))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(out_dir / "integrated_candidate_evidence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "partial" if rows else "fail",
        "exact_sequence_matched_constructs": len(rows),
        "developability_pass": sum(row["developability_status"] == "developability_pass" for row in rows),
        "fold_pass": sum(row["fold_sidecheck_status"] == "fold_sidecheck_pass" for row in rows),
        "fold_reviewed_nonpassing": sum(
            row["fold_sidecheck_status"] not in ("fold_sidecheck_pass", "pending_not_run")
            for row in rows
        ),
        "fold_pending": sum(row["fold_sidecheck_status"] == "pending_not_run" for row in rows),
        "fold_pending_or_nonpassing": sum(row["fold_sidecheck_status"] != "fold_sidecheck_pass" for row in rows),
        "repack_pass": sum(row["repack_status"] == "repack_pass" for row in rows),
        "computational_side_evidence_complete": sum(
            row["integrated_evidence_status"]
            == "computational_side_evidence_complete_wetlab_pending"
            for row in rows
        ),
        "identity_key": "reference_pdb + full paratope sequence",
        "claim_boundary": "fold smoke outputs are not accepted as fold-pass evidence",
    }
    with open(out_dir / "integrated_evidence_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "integrated_evidence_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Integrated Multi-Conformation Evidence\n\n")
        handle.write(f"Exact construct matches: {len(rows)}; developability pass: {summary['developability_pass']}; fold pass: {summary['fold_pass']}.\n\n")
        handle.write("Candidate identity uses reference plus full paratope sequence; repeated generator IDs are not trusted.\n\n")
        handle.write("| Construct | Ref | Ensemble rank | Robust | Delta P25 | Dev | Fold | Repack |\n")
        handle.write("|---|---|---:|---:|---:|---|---|---|\n")
        for row in rows:
            handle.write(
                f"| {row['construct_id']} | {row['reference_pdb']} | {row['multiconf_rank_within_reference']} | "
                f"{row['multiconf_robust_score']} | {row['native_delta_p25']} | "
                f"{row['developability_status']} | {row['fold_sidecheck_status']} | "
                f"{row['repack_status']} |\n"
            )
    with open(out_dir / "fv_fold_queue_developability_pass.fasta", "w", encoding="ascii") as handle:
        for row in rows:
            if row["developability_status"] != "developability_pass":
                continue
            heavy_variable = find_variable_region(row["heavy_sequence"], "heavy")["variable_seq"]
            light_variable = find_variable_region(row["light_sequence"], "light")["variable_seq"]
            handle.write(
                f">{row['construct_id']}|candidate_uid={row['candidate_uid']}|Fv_HL_pair\n"
                f"{heavy_variable}:{light_variable}\n"
            )
    print(f"Integrated exact-sequence evidence for {len(rows)} constructs")


if __name__ == "__main__":
    main()
