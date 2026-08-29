#!/usr/bin/env python
"""Extract reference/donor PDB poses and sampling-compatible contact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBIO, MMCIFParser, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = set(chains)

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, residue):
        return is_aa(residue, standard=True)


def extract_pose(parser, coordinate_path, chains, output_path):
    structure = parser.get_structure(output_path.stem, str(coordinate_path))
    model = next(iter(structure))
    missing = [chain for chain in chains if chain not in {row.id for row in model}]
    if missing:
        raise ValueError(f"Configured pose chains are missing: {missing}")
    mapping = dict(zip(chains, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", strict=False))
    canonical_structure = Structure(output_path.stem)
    canonical_model = Model(0)
    canonical_structure.add(canonical_model)
    for old_chain in chains:
        new_chain = Chain(mapping[old_chain])
        canonical_model.add(new_chain)
        for residue in model[old_chain]:
            if is_aa(residue, standard=True):
                new_chain.add(residue.copy())
    io = PDBIO()
    io.set_structure(canonical_structure)
    io.save(str(output_path), ChainSelect(mapping.values()))
    return mapping


def reference_contact_residues(parser, reference, h3_indices, cutoff):
    model = next(iter(parser.get_structure(
        reference["coordinate_sha256"], reference["coordinate_path"]
    )))
    heavy = [
        residue for residue in model[reference["heavy_chain"]]
        if "CA" in residue and is_aa(residue, standard=True)
    ]
    antigen = [
        residue for chain_id in reference["antigen_chains"]
        for residue in model[chain_id]
        if "CA" in residue and is_aa(residue, standard=True)
    ]
    h3_atoms = np.asarray([
        atom.coord for index in h3_indices
        for atom in heavy[index] if atom.element != "H"
    ], dtype=float)
    contacts = []
    for residue in antigen:
        atoms = np.asarray([
            atom.coord for atom in residue if atom.element != "H"
        ], dtype=float)
        if len(atoms) and np.min(np.linalg.norm(
            h3_atoms[:, None, :] - atoms[None, :, :], axis=-1
        )) <= cutoff:
            key = [residue.id[1], residue.resname]
            if key not in contacts:
                contacts.append(key)
    return contacts


def extract(config_path, pose_dir, manifest_path, admission_path, readiness_path):
    for path in (manifest_path, admission_path, readiness_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite pose extraction artifact: {path}")
    if pose_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pose directory: {pose_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    structural_path = ROOT / config["inputs"]["structural_panel"]
    scorer_path = ROOT / config["inputs"]["scorer_inputs"]
    donor_path = ROOT / config["inputs"]["donor_audit"]
    if sha256(structural_path) != config["inputs"]["structural_panel_sha256"]:
        raise RuntimeError("Structural panel hash changed")
    if sha256(scorer_path) != config["inputs"]["scorer_inputs_sha256"]:
        raise RuntimeError("Scorer input manifest hash changed")
    if sha256(donor_path) != config["inputs"]["donor_audit_sha256"]:
        raise RuntimeError("Donor audit hash changed")
    structural = json.loads(structural_path.read_text(encoding="ascii"))
    structural_rows = {
        row["component_id"]: row for row in structural["components"]
    }
    scorer = json.loads(scorer_path.read_text(encoding="ascii"))
    parser = MMCIFParser(QUIET=True)
    pose_dir.mkdir(parents=True, exist_ok=False)
    manifest_components, admission_components, readiness_components = [], [], []
    try:
        for component in scorer["components"]:
            component_id = component["component_id"]
            component_dir = pose_dir / component_id
            component_dir.mkdir()
            reference = component["reference"]
            reference_pdb_id = Path(reference["coordinate_path"]).stem.casefold()
            sources = [{
                "pdb_id": reference_pdb_id,
                "coordinate_path": reference["coordinate_path"],
                "heavy_chain": reference["heavy_chain"],
                "light_chain": reference["light_chain"],
                "antigen_chains": reference["antigen_chains"],
                "source": "own_entry",
            }]
            sources.extend({
                **donor,
                "source": "independent_experimental_structure_same_lineage_and_target",
            } for donor in component["accepted_experimental_donors"]
            if donor["pdb_id"] != reference_pdb_id)
            poses, seen_hashes = [], set()
            for index, source in enumerate(sources):
                chains = [source["heavy_chain"]]
                if source.get("light_chain"):
                    chains.append(source["light_chain"])
                chains.extend(source["antigen_chains"])
                output = component_dir / f"pose_{index}_{source['pdb_id'].upper()}.pdb"
                mapping = extract_pose(
                    parser, source["coordinate_path"], chains, output
                )
                digest = sha256(output)
                if digest in seen_hashes:
                    output.unlink()
                    continue
                seen_hashes.add(digest)
                poses.append({
                    "pose_id": output.stem,
                    "entry_id": source["pdb_id"],
                    "source": source["source"],
                    "path": output.resolve().relative_to(ROOT).as_posix(),
                    "coordinate_sha256": digest,
                    "antibody_chains": [
                        mapping[chain] for chain in (
                            source["heavy_chain"], source.get("light_chain")
                        ) if chain
                    ],
                    "antigen_chains": [
                        mapping[chain] for chain in source["antigen_chains"]
                    ],
                    "source_chain_mapping": mapping,
                    "accepted": True,
                })
            contact_residues = reference_contact_residues(
                parser,
                reference,
                structural_rows[component_id]["numbering"][
                    "h3_indices_zero_based"
                ],
                config["contact"]["cutoff_angstrom"],
            )
            if not contact_residues:
                raise RuntimeError(
                    f"Structurally admitted component lacks contact residues: {component_id}"
                )
            manifest_components.append({
                "component_id": component_id,
                "poses": poses,
                "accepted_pose_count": len(poses),
            })
            admission_components.append({
                "component_id": component_id,
                "target": component["target"],
                "lineage_proxy": component["lineage_proxy"],
                "contact_residues": contact_residues,
            })
            readiness_components.append({
                "component_id": component_id,
                "h3_ready": True,
                "h3_sequence": reference["h3_sequence"],
                "pose_paths": [pose["path"] for pose in poses],
                "pose_count": len(poses),
                "multi_pose_ready": len(poses) >= 2,
            })
        manifest = {
            "schema_version": 1,
            "status": "expanded_experimental_pose_manifest_complete",
            "classification": config["classification"],
            "component_count": len(manifest_components),
            "components": manifest_components,
            "future_confirmation_eligible_count": 0,
            "claim_boundary": config["claim_boundary"],
        }
        admission = {
            "schema_version": 1,
            "status": "expanded_sampling_admission_ready",
            "classification": config["classification"],
            "components": admission_components,
            "future_confirmation_eligible_count": 0,
            "claim_boundary": config["claim_boundary"],
        }
        readiness = {
            "schema_version": 1,
            "status": "generation_ready" if all(
                row["multi_pose_ready"] for row in readiness_components
            ) else "generation_blocked",
            "classification": config["classification"],
            "component_count": len(readiness_components),
            "h3_ready_components": len(readiness_components),
            "multi_pose_ready_components": sum(
                row["multi_pose_ready"] for row in readiness_components
            ),
            "components": readiness_components,
            "future_confirmation_eligible_count": 0,
            "claim_boundary": config["claim_boundary"],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
        admission_path.write_text(json.dumps(admission, indent=2) + "\n", encoding="ascii")
        readiness_path.write_text(json.dumps(readiness, indent=2) + "\n", encoding="ascii")
        return manifest, admission, readiness
    except Exception:
        for path in sorted(pose_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        pose_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_pose_extraction.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    outputs = config["outputs"]
    manifest, _admission, readiness = extract(
        config_path,
        ROOT / outputs["pose_directory"],
        ROOT / outputs["manifest"],
        ROOT / outputs["sampling_admission"],
        ROOT / outputs["generation_readiness"],
    )
    print(json.dumps({
        "manifest_status": manifest["status"],
        "components": manifest["component_count"],
        "multi_pose_ready": readiness["multi_pose_ready_components"],
        "sampling_required": (
            readiness["component_count"] - readiness["multi_pose_ready_components"]
        ),
        "future_confirmation_eligible": readiness[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
