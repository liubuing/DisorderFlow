#!/usr/bin/env python
"""Audit independent Tau and alpha-synuclein antibody families for LOFO readiness."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parents[2]
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
FAMILIES = [
    ("tau", "MN423", "2V17", "H", "L", "A", "TDHGAE", "published", "10.1016/j.febslet.2007.11.067"),
    ("tau", "Tau5", "4TQE", "H", "L", "A", "LPTPPTREPKKVAVVR", "pdb_only", ""),
    ("tau", "DC8E8", "5MP3", "A", "B", "C", "GSKDNIKHVPGGGSVQIVYKPVDLSKVTSK", "pdb_only", ""),
    ("tau", "gosuranemab", "6PXR", "H", "L", "A", "AGTYGLGD", "published", "10.1016/j.nbd.2020.105120"),
    ("tau", "Tau2r3", "6LRA", "H", "L", "C", "VQIINK", "published", "10.1002/1873-3468.13791"),
    ("alpha_synuclein", "NbSyn2", "2X6M", "A", "", "B", "GYQDYEPEA", "published", "10.1016/j.jmb.2010.07.001"),
    ("alpha_synuclein", "BIIB054", "6CT7", "A", "B", "S", "MDVFMKGLSK", "published", "10.1016/j.nbd.2018.10.016"),
    ("alpha_synuclein", "Lu_AF82422", "8B9V", "H", "L", "A", "AEGILEDMPVD", "pdb_only", ""),
]


def pdb_path(pdb_id):
    candidates = [
        ROOT / f"data/non_abeta_idp/{pdb_id}.pdb",
        ROOT / f"way4_s0_v2/data/pdbs/{pdb_id}.pdb",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def chain_sequence(path, chain_id):
    model = next(iter(PDBParser(QUIET=True).get_structure(path.stem, str(path))))
    if chain_id not in model:
        return ""
    return "".join(AA3.get(residue.resname, "") for residue in model[chain_id])


def main():
    rows = []
    for target, family, pdb_id, heavy, light, antigen, epitope, citation, doi in FAMILIES:
        path = pdb_path(pdb_id)
        antigen_sequence = chain_sequence(path, antigen) if path.exists() else ""
        exact = epitope in antigen_sequence or antigen_sequence in epitope
        rows.append({
            "target": target, "family": family, "pdb_id": pdb_id,
            "heavy_chain": heavy, "light_chain": light, "antigen_chain": antigen,
            "epitope": epitope, "pdb_present": path.exists(),
            "antigen_chain_present": bool(antigen_sequence), "epitope_compatible": exact,
            "citation_status": citation, "doi": doi,
            "pdb_path": path.relative_to(ROOT).as_posix() if path.exists() else "",
            "family_structure_status": "usable" if path.exists() and antigen_sequence and exact else "review",
        })
    out = ROOT / "outputs/non_abeta_idp_family_panel_v1"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "family_panel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_target = {}
    for target in sorted({row["target"] for row in rows}):
        target_rows = [row for row in rows if row["target"] == target]
        usable = [row for row in target_rows if row["family_structure_status"] == "usable"]
        by_target[target] = {
            "independent_families": len(target_rows),
            "usable_structural_families": len(usable),
            "family_ids": [row["family"] for row in usable],
            "lofo_data_ready": len(usable) >= 3,
            "lofo_design_complete": False,
        }
    summary = {
        "schema_version": "nonabeta.family_panel.v1",
        "status": "data_ready" if all(row["lofo_data_ready"] for row in by_target.values()) else "partial",
        "by_target": by_target,
        "claim_boundary": (
            "Independent experimental family structures are available. A leave-one-family-out design benchmark "
            "has not been completed because family identity is confounded with non-overlapping epitope regions."
        ),
        "excluded_misannotations": {
            "6CBV": "anti-BRIL Fab, not anti-Tau",
            "6H1F": "anti-gelsolin nanobody, not anti-alpha-synuclein",
            "6H3R": "SMAD2-DNA complex, no antibody",
        },
    }
    with open(out / "family_panel_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Family panel: {summary['status']} {by_target}")


if __name__ == "__main__":
    main()
