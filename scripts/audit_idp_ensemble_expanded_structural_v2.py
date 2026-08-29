#!/usr/bin/env python
"""Run ANARCII H3 localization and H3-antigen contact admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from anarcii import Anarcii
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_rows(chain):
    rows = []
    for residue in chain:
        if "CA" not in residue or not is_aa(residue, standard=True):
            continue
        atoms = np.asarray([
            atom.coord for atom in residue if atom.element != "H"
        ], dtype=float)
        rows.append({
            "aa": seq1(residue.resname),
            "resid": residue.id[1],
            "icode": residue.id[2].strip(),
            "resname": residue.resname,
            "atoms": atoms,
        })
    return rows


def number_h3(sequences, residue_range):
    model = Anarcii(
        seq_type="antibody", mode="accuracy", batch_size=24,
        cpu=True, ncpu=1, verbose=False,
    )
    numbered = model.number(sequences)
    numbered = model.to_scheme("chothia")
    results = {}
    for component_id, row in numbered.items():
        if not row or row.get("error"):
            results[component_id] = {
                "ready": False,
                "error": row.get("error") if row else "no_result",
            }
            continue
        query_index = int(row["query_start"])
        indices, sequence = [], []
        for (position, _insertion), amino_acid in row["numbering"]:
            if amino_acid == "-":
                continue
            if residue_range[0] <= int(position) <= residue_range[1]:
                indices.append(query_index)
                sequence.append(amino_acid)
            query_index += 1
        results[component_id] = {
            "ready": bool(indices),
            "chain_type": row["chain_type"],
            "query_start": int(row["query_start"]),
            "query_end": int(row["query_end"]),
            "h3_indices_zero_based": indices,
            "h3_sequence": "".join(sequence),
        }
    return results


def contact_metrics(heavy_rows, antigen_rows, h3_indices, cutoff):
    antigen_atoms = np.concatenate([
        row["atoms"] for row in antigen_rows if len(row["atoms"])
    ], axis=0)
    if len(antigen_atoms) == 0:
        return {
            "contacting_h3_positions": [],
            "contact_pair_count": 0,
            "minimum_distance_angstrom": None,
        }
    tree = cKDTree(antigen_atoms)
    contacting, pair_count = [], 0
    minimum = float("inf")
    for h3_position, heavy_index in enumerate(h3_indices):
        atoms = heavy_rows[heavy_index]["atoms"]
        if len(atoms) == 0:
            continue
        distances, _ = tree.query(atoms, k=1)
        local_minimum = float(np.min(distances))
        minimum = min(minimum, local_minimum)
        neighbors = tree.query_ball_point(atoms, cutoff)
        local_pairs = sum(len(row) for row in neighbors)
        if local_pairs:
            contacting.append(h3_position)
            pair_count += local_pairs
    return {
        "contacting_h3_positions": contacting,
        "contact_pair_count": pair_count,
        "minimum_distance_angstrom": (
            round(minimum, 6) if np.isfinite(minimum) else None
        ),
    }


def audit(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite structural admission: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    panel_path = ROOT / config["inputs"]["panel"]
    coordinate_audit_path = ROOT / config["inputs"]["coordinate_audit"]
    if sha256(panel_path) != config["inputs"]["panel_sha256"]:
        raise RuntimeError("Balanced panel hash changed after structural freeze")
    if sha256(coordinate_audit_path) != config["inputs"]["coordinate_audit_sha256"]:
        raise RuntimeError("Coordinate audit hash changed after structural freeze")
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    coordinate_audit = json.loads(
        coordinate_audit_path.read_text(encoding="ascii")
    )
    coordinate_rows = {
        row["component_id"]: row for row in coordinate_audit["components"]
    }
    coordinate_dir = ROOT / config["inputs"]["coordinate_directory"]
    parser = MMCIFParser(QUIET=True)
    structures, sequences = {}, {}
    for component in panel["components"]:
        component_id = component["component_id"]
        path = coordinate_dir / f"{component['pdb_id'].upper()}.cif"
        model = next(iter(parser.get_structure(component_id, str(path))))
        heavy = chain_rows(model[component["heavy_chain"]])
        antigen = []
        for chain_id in coordinate_rows[component_id]["typed_antigen_chains"]:
            antigen.extend(chain_rows(model[chain_id]))
        structures[component_id] = {
            "path": path,
            "heavy": heavy,
            "antigen": antigen,
        }
        sequences[component_id] = "".join(row["aa"] for row in heavy)
    numbering = number_h3(sequences, config["h3"]["residue_range_inclusive"])
    rows = []
    for component in panel["components"]:
        component_id = component["component_id"]
        numbered = numbering[component_id]
        h3_indices = numbered.get("h3_indices_zero_based", [])
        contacts = contact_metrics(
            structures[component_id]["heavy"],
            structures[component_id]["antigen"],
            h3_indices,
            config["contact"]["cutoff_angstrom"],
        ) if numbered.get("ready") else {
            "contacting_h3_positions": [],
            "contact_pair_count": 0,
            "minimum_distance_angstrom": None,
        }
        h3_length = len(numbered.get("h3_sequence", ""))
        checks = {
            "coordinate_ready": coordinate_rows[component_id]["coordinate_ready"],
            "anarcii_h3_ready": numbered.get("ready") is True,
            "heavy_chain_type": numbered.get("chain_type") == "H",
            "h3_length_in_range": (
                config["h3"]["length_range"][0]
                <= h3_length <= config["h3"]["length_range"][1]
            ),
            "h3_antigen_contact": (
                len(contacts["contacting_h3_positions"])
                >= config["contact"]["minimum_contacting_h3_positions"]
            ),
        }
        rows.append({
            "component_id": component_id,
            "pdb_id": component["pdb_id"],
            "target": component["target"],
            "target_category": component["target_category"],
            "lineage_proxy": component["lineage_proxy"],
            "heavy_chain": component["heavy_chain"],
            "light_chain": component["light_chain"],
            "antigen_chains": coordinate_rows[component_id][
                "typed_antigen_chains"
            ],
            "coordinate_path": str(structures[component_id]["path"]),
            "coordinate_sha256": sha256(structures[component_id]["path"]),
            "heavy_sequence": sequences[component_id],
            "antigen_residue_count": len(structures[component_id]["antigen"]),
            "numbering": numbered,
            "contacts": contacts,
            "checks": checks,
            "structurally_admitted": all(checks.values()),
            "exposure_classification": "exposed_development_only",
        })
    admitted = [row for row in rows if row["structurally_admitted"]]
    payload = {
        "schema_version": 1,
        "status": "expanded_structural_admission_complete",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "component_count": len(rows),
        "structurally_admitted_count": len(admitted),
        "blocked_count": len(rows) - len(admitted),
        "admitted_target_count": len({row["target"] for row in admitted}),
        "admitted_lineage_proxy_count": len({
            row["lineage_proxy"] for row in admitted
        }),
        "components": rows,
        "future_confirmation_eligible_count": 0,
        "decision": "build_poses_for_structurally_admitted_development_components",
        "claim_boundary": config["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_structural.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = audit(config_path, ROOT / config["output"])
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "admitted": result["structurally_admitted_count"],
        "blocked": result["blocked_count"],
        "targets": result["admitted_target_count"],
        "lineages": result["admitted_lineage_proxy_count"],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
