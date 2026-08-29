#!/usr/bin/env python
"""Score fixed expanded-IDP candidates with an orthogonal contact scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.h3_interface_contacts import (  # noqa: E402
    BACKBONE_ATOMS,
    InterfaceContact,
    ResidueID,
    score_h3_sequence,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residue_rows(chain):
    rows = []
    for residue in chain:
        if "CA" not in residue or not is_aa(residue, standard=True):
            continue
        rows.append({
            "id": ResidueID(chain.id, residue.id[1], residue.id[2].strip()),
            "aa": seq1(residue.resname),
            "atoms": {
                atom.name: np.asarray(atom.coord, dtype=float)
                for atom in residue if atom.element != "H"
            },
        })
    return rows


def minimum_distance(left, right):
    values = [
        (float(np.linalg.norm(left_coordinate - right_coordinate)),
         (left_name, right_name))
        for left_name, left_coordinate in left.items()
        for right_name, right_coordinate in right.items()
    ]
    return min(values, key=lambda row: row[0])


def build_interface(pose, native_h3, contact_cutoff, sidechain_cutoff):
    model = next(iter(PDBParser(QUIET=True).get_structure(
        pose["pose_id"], str(ROOT / pose["path"])
    )))
    heavy = residue_rows(model[pose["antibody_chains"][0]])
    antigen = [
        row for chain_id in pose["antigen_chains"]
        for row in residue_rows(model[chain_id])
    ]
    sequence = "".join(row["aa"] for row in heavy)
    start = sequence.find(native_h3)
    if start < 0:
        raise ValueError(f"Native H3 not found in pose: {pose['pose_id']}")
    h3 = heavy[start:start + len(native_h3)]
    contacts = []
    for h3_index, h3_residue in enumerate(h3):
        for epitope_index, epitope_residue in enumerate(antigen):
            distance, atom_pair = minimum_distance(
                h3_residue["atoms"], epitope_residue["atoms"]
            )
            if distance > contact_cutoff:
                continue
            h3_sidechain = {
                name: coordinate for name, coordinate in h3_residue["atoms"].items()
                if name not in BACKBONE_ATOMS
            }
            epitope_sidechain = {
                name: coordinate
                for name, coordinate in epitope_residue["atoms"].items()
                if name not in BACKBONE_ATOMS
            }
            sidechain_distance = sidechain_pair = None
            if h3_sidechain and epitope_sidechain:
                candidate_distance, candidate_pair = minimum_distance(
                    h3_sidechain, epitope_sidechain
                )
                if candidate_distance <= sidechain_cutoff:
                    sidechain_distance = candidate_distance
                    sidechain_pair = candidate_pair
            contacts.append(InterfaceContact(
                h3_index=h3_index,
                h3_residue=h3_residue["id"],
                h3_aa=h3_residue["aa"],
                epitope_index=epitope_index,
                epitope_residue=epitope_residue["id"],
                epitope_aa=epitope_residue["aa"],
                min_heavy_atom_distance=distance,
                heavy_atom_pair=atom_pair,
                min_sidechain_distance=sidechain_distance,
                sidechain_atom_pair=sidechain_pair,
            ))
    if not contacts:
        raise ValueError(f"Pose has no H3-antigen contacts: {pose['pose_id']}")
    return {
        "h3_sequence": native_h3,
        "contacts": contacts,
    }


def score(config_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent scorer: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    candidates_path = ROOT / config["inputs"]["candidates"]
    poses_path = ROOT / config["inputs"]["poses"]
    if sha256(candidates_path) != config["inputs"]["candidates_sha256"]:
        raise RuntimeError("Candidates changed after scorer freeze")
    if sha256(poses_path) != config["inputs"]["poses_sha256"]:
        raise RuntimeError("Pose manifest changed after scorer freeze")
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    poses = json.loads(poses_path.read_text(encoding="ascii"))
    pose_rows = {row["component_id"]: row["poses"] for row in poses["components"]}
    scorer = config["scorer"]
    records, paired = [], []
    for component_id, component in candidates["components"].items():
        native = component["native_h3"]
        interfaces = [
            build_interface(
                pose, native,
                scorer["heavy_atom_contact_cutoff_angstrom"],
                scorer["sidechain_contact_cutoff_angstrom"],
            )
            for pose in pose_rows[component_id]
        ]
        scores = {}
        for arm_name, arm in component["arms"].items():
            for candidate in arm["candidates"]:
                sequence = candidate["sequence"]
                pose_scores = [
                    score_h3_sequence(sequence, interface)["chemistry_score"]
                    for interface in interfaces
                ]
                key = (arm_name, candidate["substitution_bucket"])
                scores[key] = statistics.mean(pose_scores)
                records.append({
                    "component_id": component_id,
                    "target": component["target"],
                    "origin_arm": arm_name,
                    "substitution_bucket": candidate["substitution_bucket"],
                    "sequence": sequence,
                    "pose_count": len(interfaces),
                    "pose_chemistry_scores": pose_scores,
                    "mean_contact_chemistry_score": scores[key],
                })
        for bucket in component["substitution_buckets"]:
            difference = scores[("ensemble", bucket)] - scores[("single_state", bucket)]
            paired.append({
                "component_id": component_id,
                "target": component["target"],
                "substitution_bucket": bucket,
                "ensemble_minus_single_state_contact_chemistry": difference,
                "ensemble_favored_by_contact_chemistry": difference > 0,
            })
    payload = {
        "schema_version": 1,
        "status": "expanded_independent_scorer_complete",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "candidate_count": len(records),
        "matched_pair_count": len(paired),
        "ensemble_favored_pair_count": sum(
            row["ensemble_favored_by_contact_chemistry"] for row in paired
        ),
        "median_ensemble_minus_single_state_contact_chemistry": statistics.median(
            row["ensemble_minus_single_state_contact_chemistry"] for row in paired
        ),
        "records": records,
        "matched_bucket_contrasts": paired,
        "future_confirmation_eligible_count": 0,
        "decision": "compare_with_proteinmpnn_without_weight_tuning",
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
            "configs/benchmarks/idp_ensemble_expanded_development_v2_independent_scorer.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    result = score(config_path, ROOT / config["output"])
    print(json.dumps({
        "status": result["status"],
        "candidates": result["candidate_count"],
        "matched_pairs": result["matched_pair_count"],
        "ensemble_favored": result["ensemble_favored_pair_count"],
        "median_difference": result[
            "median_ensemble_minus_single_state_contact_chemistry"
        ],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
