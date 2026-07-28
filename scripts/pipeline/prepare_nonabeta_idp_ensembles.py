#!/usr/bin/env python
"""Prepare diverse experimental epitope ensembles for tau and alpha-synuclein."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


TARGETS = {
    "tau": {
        "source_pdb": "data/non_abeta_idp/2MZ7.pdb",
        "source_id": "2MZ7",
        "chain": "A",
        "epitope": "KHVPGGGSV",
        "condition": "tau_267_312_microtubule_bound_solution_nmr",
        "reference_pdb": "way4_s0_v2/data/pdbs/5MP3.pdb",
        "reference_id": "5MP3",
        "reference_peptide_chain": "C",
        # 5MP3 contains two Fab-tau copies: A/B/C and H/L/D. Keep all three
        # chains from the same biological copy when trimming the reference.
        "heavy_chain": "A",
        "light_chain": "B",
    },
    "alpha_synuclein": {
        "source_pdb": "data/non_abeta_idp/2KKW.pdb",
        "source_id": "2KKW",
        "chain": "A",
        "epitope": "EGILEDMPVD",
        "condition": "slas_micelle_bound_nmr_epr_ensemble",
        "reference_pdb": "data/non_abeta_idp/8B9V.pdb",
        "reference_id": "8B9V",
        "reference_peptide_chain": "A",
        "heavy_chain": "H",
        "light_chain": "L",
    },
}


class EpitopeSelect(Select):
    def __init__(self, chain_id, residue_ids):
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.id in self.residue_ids


class ComplexCoreSelect(Select):
    def __init__(self, antibody_chains, peptide_chain, peptide_residue_ids):
        self.antibody_chains = set(antibody_chains)
        self.peptide_chain = peptide_chain
        self.peptide_residue_ids = set(peptide_residue_ids)

    def accept_chain(self, chain):
        return chain.id in self.antibody_chains or chain.id == self.peptide_chain

    def accept_residue(self, residue):
        chain = residue.get_parent().id
        return chain in self.antibody_chains or (
            chain == self.peptide_chain and residue.id in self.peptide_residue_ids
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/non_abeta_idp/ensembles_v1")
    parser.add_argument("--manifest-out", default="outputs/non_abeta_idp_ensemble_prep_v1")
    parser.add_argument("--conformers", type=int, default=5)
    return parser.parse_args()


def sequence_residues(chain):
    residues = [residue for residue in chain if residue.resname in AA3_TO_1 and "CA" in residue]
    return "".join(AA3_TO_1[residue.resname] for residue in residues), residues


def kabsch_rmsd(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left_centered = left - left.mean(axis=0)
    right_centered = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(left_centered.T @ right_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    delta = left_centered @ rotation - right_centered
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def farthest_models(coordinates, count):
    n = len(coordinates)
    distances = np.zeros((n, n), dtype=float)
    for left in range(n):
        for right in range(left + 1, n):
            distances[left, right] = distances[right, left] = kabsch_rmsd(
                coordinates[left], coordinates[right]
            )
    medoid = int(np.argmin(distances.mean(axis=1)))
    selected = [medoid]
    while len(selected) < min(count, n):
        candidate = max(
            (index for index in range(n) if index not in selected),
            key=lambda index: min(distances[index, chosen] for chosen in selected),
        )
        selected.append(candidate)
    return selected, distances


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def prepare_target(name, config, output_root, count):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(name, config["source_pdb"])
    model_rows = []
    coords = []
    for model_index, model in enumerate(structure, 1):
        sequence, residues = sequence_residues(model[config["chain"]])
        start = sequence.find(config["epitope"])
        if start < 0:
            raise ValueError(f"{name} model {model_index} lacks epitope {config['epitope']}")
        epitope_residues = residues[start:start + len(config["epitope"])]
        coords.append(np.asarray([residue["CA"].coord for residue in epitope_residues]))
        model_rows.append({
            "model_index": model_index,
            "epitope_start_resid": epitope_residues[0].id[1],
            "epitope_end_resid": epitope_residues[-1].id[1],
            "residue_ids": [residue.id for residue in epitope_residues],
        })
    selected, distances = farthest_models(coords, count)
    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for rank, model_zero_index in enumerate(selected):
        model = list(structure)[model_zero_index]
        row = model_rows[model_zero_index]
        output_path = target_dir / f"{name}_conformer{rank}_model{row['model_index']}.pdb"
        io = PDBIO()
        io.set_structure(model)
        io.save(str(output_path), EpitopeSelect(config["chain"], row["residue_ids"]))
        entries.append({
            "conformer": rank,
            "source_model": row["model_index"],
            "pdb": output_path.as_posix(),
            "sha256": sha256(output_path),
            "epitope_start_resid": row["epitope_start_resid"],
            "epitope_end_resid": row["epitope_end_resid"],
            "minimum_rmsd_to_other_selected": min(
                distances[model_zero_index, other] for other in selected if other != model_zero_index
            ),
        })
    reference_structure = parser.get_structure(f"{name}_reference", config["reference_pdb"])
    reference_model = next(iter(reference_structure))
    reference_sequence, reference_residues = sequence_residues(
        reference_model[config["reference_peptide_chain"]]
    )
    reference_start = reference_sequence.find(config["epitope"])
    if reference_start < 0:
        raise ValueError(
            f"Reference {config['reference_id']} lacks exact epitope {config['epitope']}: "
            f"{reference_sequence}"
        )
    reference_core_residues = reference_residues[
        reference_start:reference_start + len(config["epitope"])
    ]
    trimmed_reference = target_dir / f"{config['reference_id']}_{name}_core_complex.pdb"
    io = PDBIO()
    io.set_structure(reference_model)
    io.save(str(trimmed_reference), ComplexCoreSelect(
        [config["heavy_chain"], config["light_chain"]],
        config["reference_peptide_chain"],
        [residue.id for residue in reference_core_residues],
    ))
    return {
        "target": name,
        **config,
        "source_sha256": sha256(config["source_pdb"]),
        "source_models": len(model_rows),
        "trimmed_reference_pdb": trimmed_reference.as_posix(),
        "trimmed_reference_sha256": sha256(trimmed_reference),
        "selection": "CA Kabsch farthest-point sampling initialized at ensemble medoid",
        "selected_conformers": entries,
    }


def main():
    args = parse_args()
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    records = [prepare_target(name, config, output_root, args.conformers) for name, config in TARGETS.items()]
    report = {
        "schema_version": "nonabeta.idp_ensemble.v1",
        "status": "pass" if all(len(row["selected_conformers"]) == args.conformers for row in records) else "fail",
        "targets": records,
        "off_state_semantics": {
            "tau": {
                "structures": ["5O3L", "6QJH"],
                "evidence": "experimental fibril geometry",
                "binding_label": "not_available",
            },
            "alpha_synuclein": {
                "structures": ["1XQ8", "2N0A", "8B9V"],
                "evidence": "experimental average/fibril geometry and independent C-terminal antibody complex",
                "binding_label": "not_available_for_6CT7_candidates",
            },
        },
        "claim_boundary": "Off-state structures are geometry/provenance controls, not experimental negatives.",
    }
    manifest_dir = Path(args.manifest_out)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "ensemble_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Prepared non-A-beta ensembles: {len(records)} targets; status={report['status']}")
    if report["status"] != "pass":
        raise SystemExit("Ensemble preparation failed")


if __name__ == "__main__":
    main()
