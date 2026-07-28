#!/usr/bin/env python
"""Prepare sequence-consistent VH:VL:IDP initial guesses from strict experimental poses."""
from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
BACKBONE = {"N", "CA", "C", "O"}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def residue_lines(path, chain_id):
    residues = []
    lookup = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[21:22].strip() != chain_id:
            continue
        key = (line[22:26], line[26:27])
        if key not in lookup:
            lookup[key] = []
            residues.append(lookup[key])
        lookup[key].append(line)
    return residues


def build_guess(source, output, row, heavy_chain, light_chain):
    sequences = [row["vh_sequence"], row["vl_sequence"], row["antigen_sequence"]]
    source_chains = [heavy_chain, light_chain, "P"]
    output_chains = ["A", "B", "C"]
    lines = []
    serial = 1
    for sequence, source_chain, output_chain in zip(sequences, source_chains, output_chains):
        residues = residue_lines(source, source_chain)
        if len(residues) < len(sequence):
            raise ValueError(
                f"{source}: chain {source_chain} has {len(residues)} residues, needs {len(sequence)}"
            )
        for resid, (amino_acid, atom_lines) in enumerate(zip(sequence, residues), 1):
            for line in atom_lines:
                atom_name = line[12:16].strip()
                if atom_name not in BACKBONE:
                    continue
                record = "ATOM  "
                rewritten = (
                    f"{record}{serial:5d}{line[11:17]}{AA1_TO_3[amino_acid]:>3} "
                    f"{output_chain}{resid:4d} {line[27:]}"
                )
                lines.append(rewritten)
                serial += 1
        lines.append("TER")
    lines.append("END")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    panel = load_csv(
        PROJECT_ROOT / "outputs/non_abeta_idp_complex_fold_panel_v1/complex_fold_panel.csv"
    )
    with open(
        PROJECT_ROOT / "outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)
    with open(
        PROJECT_ROOT / "outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json",
        encoding="utf-8",
    ) as handle:
        poses = json.load(handle)
    target_config = {row["target"]: row for row in manifest["targets"]}
    priority = {
        "5MP3_native", "5MP3_cbefe653f673", "5MP3_59c6c0a35345",
        "8B9V_native", "8B9V_01d0ed9adb40", "8B9V_498c3ede07c8",
    }
    rows = []
    out_dir = PROJECT_ROOT / "outputs/non_abeta_idp_initial_guess_panel_v1"
    for construct in panel:
        if construct["construct_id"] not in priority:
            continue
        target = target_config[construct["target"]]
        target_poses = sorted(
            (
                row for row in poses["entries"]
                if row["target"] == construct["target"] and row.get("selected_for_panel")
            ),
            key=lambda row: row["conformer"],
        )
        for pose in target_poses:
            output = (
                out_dir / "initial_guesses" / construct["construct_id"]
                / f"conformer{pose['conformer']}.pdb"
            )
            build_guess(
                PROJECT_ROOT / pose["pose_pdb"], output, construct,
                target["heavy_chain"], target["light_chain"],
            )
            rows.append({
                "construct_id": construct["construct_id"],
                "target": construct["target"],
                "antibody_family": construct["antibody_family"],
                "construct_type": construct["construct_type"],
                "conformer": pose["conformer"],
                "initial_guess_pdb": output.relative_to(PROJECT_ROOT).as_posix(),
                "source_pose_pdb": pose["pose_pdb"],
                "vh_sequence": construct["vh_sequence"],
                "vl_sequence": construct["vl_sequence"],
                "antigen_sequence": construct["antigen_sequence"],
            })
    with open(out_dir / "initial_guess_panel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "nonabeta.initial_guess_panel.v1",
        "status": "pass" if len(rows) == 30 else "partial",
        "constructs": len({row["construct_id"] for row in rows}),
        "conformers_per_construct": 5,
        "jobs": len(rows),
        "coordinates": "strict pose-derived backbone-only VH:VL:IDP, sequence-consistent",
    }
    with open(out_dir / "initial_guess_panel_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Initial-guess panel: status={summary['status']} jobs={len(rows)}")


if __name__ == "__main__":
    main()
