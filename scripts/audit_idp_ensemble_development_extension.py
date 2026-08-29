#!/usr/bin/env python
"""Audit a frozen local antibody-IDP development-extension registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import yaml
from anarcii import Anarcii
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
MODIFIED_RESIDUES = {"MSE": "M", "PCA": "E", "SEP": "S", "TPO": "T", "PTR": "Y"}
CHOTHIA_H3 = (93, 102)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_residues(chain):
    rows = []
    for residue in chain:
        if "CA" not in residue:
            continue
        aa = seq1(residue.resname, custom_map=MODIFIED_RESIDUES)
        if aa not in AA:
            continue
        rows.append({
            "aa": aa,
            "resname": residue.resname,
            "residue": residue,
        })
    return rows


def number_heavy_domains(components, structures):
    model = Anarcii(
        seq_type="antibody", mode="accuracy", batch_size=16,
        cpu=True, ncpu=1, verbose=False,
    )
    sequences = {
        component["component_id"]: "".join(
            row["aa"] for row in structures[component["component_id"]]["heavy"]
        )
        for component in components
    }
    numbered = model.number(sequences)
    numbered = model.to_scheme("chothia")
    results = {}
    for component_id, row in numbered.items():
        if not row or row.get("error"):
            results[component_id] = {"error": row.get("error") if row else "no result"}
            continue
        query_index = int(row["query_start"])
        h3_indices = []
        h3_sequence = []
        for (position, _insertion), aa in row["numbering"]:
            if aa == "-":
                continue
            if CHOTHIA_H3[0] <= int(position) <= CHOTHIA_H3[1]:
                h3_indices.append(query_index)
                h3_sequence.append(aa)
            query_index += 1
        results[component_id] = {
            "chain_type": row["chain_type"],
            "query_start": int(row["query_start"]),
            "query_end": int(row["query_end"]),
            "h3_indices_zero_based": h3_indices,
            "h3_positions_1_indexed": [min(h3_indices) + 1, max(h3_indices) + 1]
            if h3_indices else [],
            "h3_sequence": "".join(h3_sequence),
        }
    return results


def contact_metrics(heavy, antigen, h3_indices, cutoff):
    contacting_positions = set()
    contact_pairs = 0
    minimum_distance = float("inf")
    for h3_index in h3_indices:
        heavy_atoms = [atom for atom in heavy[h3_index]["residue"] if atom.element != "H"]
        for antigen_row in antigen:
            antigen_atoms = [
                atom for atom in antigen_row["residue"] if atom.element != "H"
            ]
            if not heavy_atoms or not antigen_atoms:
                continue
            distances = np.linalg.norm(
                np.asarray([atom.coord for atom in heavy_atoms])[:, None, :]
                - np.asarray([atom.coord for atom in antigen_atoms])[None, :, :],
                axis=-1,
            )
            distance = float(distances.min())
            minimum_distance = min(minimum_distance, distance)
            if distance <= cutoff:
                contacting_positions.add(h3_index)
                contact_pairs += 1
    return {
        "minimum_h3_peptide_heavy_atom_distance": minimum_distance,
        "contacting_h3_positions": len(contacting_positions),
        "h3_peptide_residue_contact_pairs": contact_pairs,
    }


def audit_component(component, structures, numbering, gates):
    component_id = component["component_id"]
    structure = structures[component_id]
    heavy = structure["heavy"]
    light = structure["light"]
    antigen = structure["antigen"]
    numbered = numbering[component_id]
    checks = {
        "heavy_chain_length": len(heavy) >= int(gates["minimum_heavy_chain_residues"]),
        "light_chain_length": len(light) >= int(gates["minimum_light_chain_residues"]),
        "peptide_length": int(gates["peptide_length_range"][0])
        <= len(antigen) <= int(gates["peptide_length_range"][1]),
        "expected_peptide": "".join(row["aa"] for row in antigen)
        == component["expected_observed_peptide"],
        "heavy_chain_role": numbered.get("chain_type") == "H",
        "h3_length": int(gates["h3_length_range"][0])
        <= len(numbered.get("h3_sequence", "")) <= int(gates["h3_length_range"][1]),
    }
    contacts = contact_metrics(
        heavy, antigen, numbered.get("h3_indices_zero_based", []),
        float(gates["h3_contact_cutoff_angstrom"]),
    ) if numbered.get("h3_indices_zero_based") else {
        "minimum_h3_peptide_heavy_atom_distance": float("inf"),
        "contacting_h3_positions": 0,
        "h3_peptide_residue_contact_pairs": 0,
    }
    checks["h3_peptide_contact"] = contacts["contacting_h3_positions"] >= int(
        gates["minimum_contacting_h3_positions"]
    )
    return {
        "component_id": component_id,
        "target": component["target"],
        "antibody": component["antibody"],
        "antibody_lineage_cluster": component["antibody_lineage_cluster"],
        "epitope_cluster": component["epitope_cluster"],
        "reference_pdb": component["reference_pdb"],
        "reference_sha256": sha256(ROOT / component["reference_pdb"]),
        "chain_roles": {
            "heavy": component["heavy_chain"],
            "light": component["light_chain"],
            "antigen": component["antigen_chain"],
        },
        "chain_lengths": {
            "heavy": len(heavy), "light": len(light), "antigen": len(antigen),
        },
        "antigen_sequence": "".join(row["aa"] for row in antigen),
        "modified_antigen_residues": [
            row["resname"] for row in antigen if row["resname"] in MODIFIED_RESIDUES
        ],
        "numbering": numbered,
        "contacts": contacts,
        "ensemble_plan": component["ensemble_plan"],
        "sensitivity_only": bool(component.get("sensitivity_only", False)),
        "checks": checks,
        "eligible": all(checks.values()),
    }


def write_csv(path, rows):
    fields = [
        "component_id", "target", "antibody", "antibody_lineage_cluster",
        "epitope_cluster", "eligible", "sensitivity_only", "antigen_sequence",
        "h3_sequence", "contacting_h3_positions", "reference_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{field: row.get(field) for field in fields},
                "h3_sequence": row["numbering"].get("h3_sequence", ""),
                "contacting_h3_positions": row["contacts"]["contacting_h3_positions"],
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/idp_ensemble_development_extension_registry_v1.yml",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pdb_parser = PDBParser(QUIET=True)
    structures = {}
    load_errors = []
    for component in config["components"]:
        try:
            path = ROOT / component["reference_pdb"]
            model = next(iter(pdb_parser.get_structure(component["component_id"], str(path))))
            structures[component["component_id"]] = {
                "heavy": chain_residues(model[component["heavy_chain"]]),
                "light": chain_residues(model[component["light_chain"]]),
                "antigen": chain_residues(model[component["antigen_chain"]]),
            }
        except Exception as error:  # noqa: BLE001
            load_errors.append({"component_id": component["component_id"], "error": str(error)})
    loaded = [
        component for component in config["components"]
        if component["component_id"] in structures
    ]
    numbering = number_heavy_domains(loaded, structures)
    rows = [
        audit_component(component, structures, numbering, config["structural_gates"])
        for component in loaded
    ]
    duplicate_lineages = sorted({
        row["antibody_lineage_cluster"] for row in rows
        if sum(
            other["antibody_lineage_cluster"] == row["antibody_lineage_cluster"]
            for other in rows
        ) > 1
    })
    output = {
        "schema_version": 1,
        "status": "registry_structurally_ready"
        if len(rows) == len(config["components"]) and all(row["eligible"] for row in rows)
        else "registry_requires_revision",
        "classification": config["classification"],
        "provenance": {
            "config": args.config,
            "config_sha256": sha256(config_path),
            "anarcii_version": importlib.metadata.version("anarcii"),
            "numbering_scheme": "chothia",
        },
        "counts": {
            "registered": len(config["components"]),
            "loaded": len(rows),
            "eligible": sum(row["eligible"] for row in rows),
            "sensitivity_only": sum(row["sensitivity_only"] for row in rows),
        },
        "duplicate_antibody_lineage_clusters": duplicate_lineages,
        "epitope_cluster_counts": {
            cluster: sum(row["epitope_cluster"] == cluster for row in rows)
            for cluster in sorted({row["epitope_cluster"] for row in rows})
        },
        "load_errors": load_errors,
        "components": rows,
        "confirmation_reservation": config["confirmation_reservation"],
        "claim_boundary": config["claim_boundary"],
    }
    output_path = ROOT / config["output"]["audit"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    write_csv(ROOT / config["output"]["component_csv"], rows)
    print(json.dumps({
        "status": output["status"],
        "counts": output["counts"],
        "load_errors": load_errors,
        "failed": [
            {"component_id": row["component_id"], "checks": row["checks"]}
            for row in rows if not row["eligible"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
