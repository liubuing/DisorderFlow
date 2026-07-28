#!/usr/bin/env python
"""Summarize computation-complete A-beta candidates across epitope regions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nterm", default="outputs/abeta_multiconf_integrated_evidence_v1/integrated_candidate_evidence.csv")
    parser.add_argument("--mid-design", default="outputs/abeta_midregion_fold_panel_v1/midregion_fold_panel.csv")
    parser.add_argument("--mid-fold", default="outputs/abeta_midregion_fold_evidence_v1/fv_colabfold_evidence.csv")
    parser.add_argument("--mid-repack", default="outputs/abeta_midregion_fv_repack_v1/fv_repack_audit.json")
    parser.add_argument("--out", default="outputs/abeta_cross_epitope_maturity_v1")
    return parser.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    nterm = [
        row for row in read_csv(args.nterm)
        if row["integrated_evidence_status"]
        == "computational_side_evidence_complete_wetlab_pending"
    ]
    mid_design = {row["candidate_uid"]: row for row in read_csv(args.mid_design)}
    mid_fold = {row["construct_id"]: row for row in read_csv(args.mid_fold)}
    with open(args.mid_repack, encoding="utf-8") as handle:
        mid_repack = {
            row["construct_id"]: row for row in json.load(handle).get("records", [])
        }

    rows = []
    for row in nterm:
        rows.append({
            "candidate_uid": row["candidate_uid"],
            "reference_pdb": row["reference_pdb"],
            "epitope_region": "N_terminal_1_6",
            "generator": "contact_guided_then_multiconf_ranked",
            "native_delta_p25": float(row["native_delta_p25"]),
            "native_delta_min": "",
            "full_chain_risk": float(row["developability_risk"]),
            "fold_mean_plddt": float(row["mean_plddt"]),
            "fold_iptm": float(row["iptm"]),
            "repack_status": row["repack_status"],
            "repack_backbone_rmsd": float(row["repack_backbone_rmsd"]),
            "repack_global_clashes": int(row["repack_global_clashes"]),
            "computational_status": "complete_wetlab_pending",
        })
    for uid, design in mid_design.items():
        fold = mid_fold.get(uid, {})
        repack = mid_repack.get(uid, {})
        complete = (
            fold.get("fold_sidecheck_status") == "fold_sidecheck_pass"
            and repack.get("status") == "repack_pass"
        )
        rows.append({
            "candidate_uid": uid,
            "reference_pdb": design["reference_pdb"],
            "epitope_region": "middle_13_24",
            "generator": design["generator"],
            "native_delta_p25": float(design["native_delta_p25"]),
            "native_delta_min": float(design["native_delta_min"]),
            "full_chain_risk": float(design["full_chain_risk"]),
            "fold_mean_plddt": float(fold["mean_plddt"]) if fold else "",
            "fold_iptm": float(fold["iptm"]) if fold else "",
            "repack_status": repack.get("status", "pending_not_run"),
            "repack_backbone_rmsd": repack.get("backbone_rmsd_angstrom", ""),
            "repack_global_clashes": repack.get("after_geometry", {}).get(
                "nonlocal_heavy_atom_clashes_lt_1_5A", ""
            ),
            "computational_status": (
                "complete_wetlab_pending" if complete else "computational_incomplete"
            ),
        })
    complete = [row for row in rows if row["computational_status"] == "complete_wetlab_pending"]
    references = sorted({row["reference_pdb"] for row in complete})
    regions = sorted({row["epitope_region"] for row in complete})
    status = (
        "target_internal_cross_epitope_computational_validation"
        if len(references) >= 3 and len(regions) >= 2 else "single_epitope_computational_validation"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(out_dir / "cross_epitope_candidate_evidence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "abeta.cross_epitope_maturity.v1",
        "status": status,
        "computation_complete_candidates": len(complete),
        "references": references,
        "epitope_regions": regions,
        "direct_ensemble_conditioned_candidates": sum(
            row["generator"] == "direct_multiconf_beam_v1" for row in complete
        ),
        "claim_boundary": (
            "Target-internal A-beta cross-epitope computation only; no experimental binding "
            "and no generalization claim to other IDP targets."
        ),
        "next_computational_gap": (
            "Add at least two non-A-beta IDP targets with homolog-isolated references and "
            "the same pose, ensemble, fold, and repack gates."
        ),
    }
    with open(out_dir / "cross_epitope_maturity.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "cross_epitope_maturity.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Cross-Epitope Computational Maturity\n\n")
        handle.write(f"Status: `{status}`.\n\n")
        handle.write(
            f"Computation-complete candidates: {len(complete)} across {len(references)} references "
            f"and {len(regions)} epitope regions.\n\n"
        )
        handle.write("| UID | Ref | Region | Generator | Delta P25 | Delta min | pLDDT | ipTM | Repack |\n")
        handle.write("|---|---|---|---|---:|---:|---:|---:|---|\n")
        for row in complete:
            handle.write(
                f"| {row['candidate_uid']} | {row['reference_pdb']} | {row['epitope_region']} | "
                f"{row['generator']} | {row['native_delta_p25']} | {row['native_delta_min']} | "
                f"{row['fold_mean_plddt']} | {row['fold_iptm']} | {row['repack_status']} |\n"
            )
        handle.write("\nThis is target-internal cross-epitope evidence, not cross-IDP validation.\n")
    print(f"Cross-epitope maturity: {status}; complete={len(complete)}")


if __name__ == "__main__":
    main()
