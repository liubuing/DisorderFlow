#!/usr/bin/env python
"""Freeze the structural panel and audit experimental pose-source coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_sequence(chain):
    return "".join(
        seq1(residue.resname)
        for residue in chain
        if "CA" in residue and is_aa(residue, standard=True)
    )


def audit(config_path, panel_output, audit_output):
    for path in (panel_output, audit_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite pose-source artifact: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    admission_path = ROOT / config["inputs"]["structural_admission"]
    if sha256(admission_path) != config["inputs"]["structural_admission_sha256"]:
        raise RuntimeError("Structural admission changed after pose-source freeze")
    admission = json.loads(admission_path.read_text(encoding="ascii"))
    registry_path = ROOT / config["inputs"]["registry"]
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    coordinate_dir = ROOT / config["inputs"]["coordinates"]
    admitted = [
        row for row in admission["components"] if row["structurally_admitted"]
    ]
    structural_panel = {
        "schema_version": 1,
        "status": "expanded_structural_development_panel_frozen",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "structural_admission": str(admission_path),
        "structural_admission_sha256": sha256(admission_path),
        "component_count": len(admitted),
        "target_count": len({row["target"] for row in admitted}),
        "lineage_proxy_count": len({row["lineage_proxy"] for row in admitted}),
        "components": admitted,
        "excluded_components": [
            {
                "component_id": row["component_id"],
                "failed_checks": [
                    key for key, value in row["checks"].items() if not value
                ],
            }
            for row in admission["components"] if not row["structurally_admitted"]
        ],
        "future_confirmation_eligible_count": 0,
        "claim_boundary": config["claim_boundary"],
    }

    registry_groups = {}
    for row in registry["records"]:
        key = (row["target"], row["lineage_proxy"])
        registry_groups.setdefault(key, set()).add(row["pdb_id"])
    parser = MMCIFParser(QUIET=True)
    source_rows = []
    for component in admitted:
        path = coordinate_dir / f"{component['pdb_id'].upper()}.cif"
        model = next(iter(parser.get_structure(component["component_id"], str(path))))
        sequences = {
            chain.id: chain_sequence(chain) for chain in model
        }
        heavy_sequence = sequences[component["heavy_chain"]]
        antigen_sequences = {
            sequences[chain] for chain in component["antigen_chains"]
            if chain in sequences and sequences[chain]
        }
        heavy_copies = sum(
            sequence == heavy_sequence for sequence in sequences.values()
        )
        antigen_copies = sum(
            sequence in antigen_sequences for sequence in sequences.values()
        )
        assembly_complex_copies = min(heavy_copies, antigen_copies)
        donor_pdbs = sorted(registry_groups[
            (component["target"], component["lineage_proxy"])
        ])
        experimental_sources = len(donor_pdbs) + max(
            0, assembly_complex_copies - 1
        )
        minimum = config["requirements"]["minimum_experimental_pose_sources"]
        source_rows.append({
            "component_id": component["component_id"],
            "target": component["target"],
            "lineage_proxy": component["lineage_proxy"],
            "reference_pdb_id": component["pdb_id"],
            "same_lineage_target_pdb_ids": donor_pdbs,
            "unique_same_lineage_target_pdb_count": len(donor_pdbs),
            "same_assembly_heavy_sequence_copies": heavy_copies,
            "same_assembly_antigen_sequence_copies": antigen_copies,
            "same_assembly_complex_copy_count": assembly_complex_copies,
            "estimated_experimental_pose_source_count": experimental_sources,
            "experimental_pose_source_ready": experimental_sources >= minimum,
            "restrained_sampling_required": experimental_sources < minimum,
        })
    target_sampling = Counter(
        row["target"] for row in source_rows if row["restrained_sampling_required"]
    )
    pose_audit = {
        "schema_version": 1,
        "status": "expanded_pose_source_audit_complete",
        "classification": config["classification"],
        "structural_panel": str(panel_output),
        "structural_panel_sha256": None,
        "component_count": len(source_rows),
        "experimental_pose_source_ready_count": sum(
            row["experimental_pose_source_ready"] for row in source_rows
        ),
        "restrained_sampling_required_count": sum(
            row["restrained_sampling_required"] for row in source_rows
        ),
        "target_sampling_required_counts": dict(sorted(target_sampling.items())),
        "components": source_rows,
        "future_confirmation_eligible_count": 0,
        "decision": "materialize_donor_structures_then_sample_remaining_components",
        "claim_boundary": config["claim_boundary"],
    }
    panel_output.parent.mkdir(parents=True, exist_ok=True)
    panel_output.write_text(
        json.dumps(structural_panel, indent=2) + "\n", encoding="ascii"
    )
    pose_audit["structural_panel_sha256"] = sha256(panel_output)
    audit_output.write_text(json.dumps(pose_audit, indent=2) + "\n", encoding="ascii")
    return structural_panel, pose_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_pose_sources.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    panel, source = audit(
        config_path,
        ROOT / config["outputs"]["structural_panel"],
        ROOT / config["outputs"]["pose_source_audit"],
    )
    print(json.dumps({
        "structural_panel_status": panel["status"],
        "components": panel["component_count"],
        "targets": panel["target_count"],
        "experimental_pose_source_ready": source[
            "experimental_pose_source_ready_count"
        ],
        "restrained_sampling_required": source[
            "restrained_sampling_required_count"
        ],
        "target_sampling_required": source[
            "target_sampling_required_counts"
        ],
        "future_confirmation_eligible": source[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
