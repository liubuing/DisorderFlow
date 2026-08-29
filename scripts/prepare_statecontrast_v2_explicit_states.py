#!/usr/bin/env python
"""Materialize target/peptide-stripped-apo state pairs for fixed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _KeepChains(Select):
    def __init__(self, chains):
        self.chains = set(chains)

    def accept_chain(self, chain):
        return chain.id in self.chains


def write_apo_pose(target_path, antibody_chains, output_path):
    """Remove antigen chains while preserving the deposited antibody frame."""
    structure = PDBParser(QUIET=True).get_structure("apo", str(target_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(output_path), _KeepChains(antibody_chains))
    return sha256(output_path)


def antigen_sequence(pose):
    model = PDBParser(QUIET=True).get_structure(
        pose["pose_id"], str(ROOT / pose["path"]))[0]
    return "".join(
        seq1(residue.resname)
        for chain_id in pose["antigen_chains"] for residue in model[chain_id]
        if is_aa(residue, standard=True) and "CA" in residue
    )


def select_source_balanced_poses(poses, maximum=5):
    """Deterministically round-robin broad pose sources before score access."""
    if len(poses) <= maximum:
        return list(poses)
    grouped = {"experimental": [], "sampled": []}
    for pose in sorted(poses, key=lambda row: (
            int(row.get("source_priority", 1)), row["pose_id"])):
        source = (
            "sampled" if pose["source"] == "restrained_local_interface_sampling"
            else "experimental"
        )
        grouped[source].append(pose)
    selected = []
    depth = 0
    while len(selected) < maximum:
        added = False
        for source in ("experimental", "sampled"):
            if depth < len(grouped[source]):
                selected.append(grouped[source][depth])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def prepare(candidates_path, poses_path, scorer_path, plan_path, output_dir,
            maximum_poses_per_component=5):
    candidates = json.loads(candidates_path.read_text(encoding="ascii"))
    poses = json.loads(poses_path.read_text(encoding="ascii"))
    scorer = json.loads(scorer_path.read_text(encoding="ascii"))
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    lineage = {
        row["component_id"]: row["antibody_lineage"] for row in plan["components"]
    }
    teacher = {
        (row["component_id"], row["origin_arm"], int(row["substitution_bucket"])):
        float(row["mean_contact_chemistry_score"])
        for row in scorer["records"]
    }
    pose_rows = {
        row["component_id"]: select_source_balanced_poses(
            row["poses"], maximum=maximum_poses_per_component)
        for row in poses["components"]
    }
    target_by_component = {
        component_id: candidates["components"][component_id]["target"]
        for component_id in candidates["components"]
    }
    antigen_by_component = {
        component_id: antigen_sequence(rows[0])
        for component_id, rows in pose_rows.items()
    }
    donor_by_component = {}
    component_ids = sorted(candidates["components"])
    for component_id in component_ids:
        alternatives = [
            donor for donor in component_ids
            if target_by_component[donor] != target_by_component[component_id]
            and antigen_by_component[donor]
        ]
        donor_by_component[component_id] = min(
            alternatives,
            key=lambda donor: (
                abs(len(antigen_by_component[donor])
                    - len(antigen_by_component[component_id])), donor),
        )
    apo_dir = output_dir / "apo_poses"
    apo_cache = {}
    components = []
    for component_id, component in candidates["components"].items():
        for arm_name, arm in component["arms"].items():
            for candidate in arm["candidates"]:
                bucket = int(candidate["substitution_bucket"])
                states = []
                teacher_score = teacher[(component_id, arm_name, bucket)]
                donor_component = donor_by_component[component_id]
                donor_sequence = antigen_by_component[donor_component]
                for pose in pose_rows[component_id]:
                    target_path = ROOT / pose["path"]
                    cache_key = (component_id, pose["pose_id"])
                    if cache_key not in apo_cache:
                        apo_path = apo_dir / component_id / f"{pose['pose_id']}_apo.pdb"
                        apo_cache[cache_key] = {
                            "path": apo_path,
                            "sha256": write_apo_pose(
                                target_path, pose["antibody_chains"], apo_path),
                        }
                    target_source = pose["source"]
                    source_panel = (
                        "sampled" if target_source
                        == "restrained_local_interface_sampling" else "experimental"
                    )
                    states.extend([
                        {
                            "state_id": f"target|{pose['pose_id']}",
                            "pose_id": pose["pose_id"],
                            "state_type": "target",
                            "source": target_source,
                            "source_panel": source_panel,
                            "prior_weight": 1.0,
                            "coordinate_path": target_path.relative_to(ROOT).as_posix(),
                            "coordinate_sha256": sha256(target_path),
                            "independent_teacher_score": teacher_score,
                            "independent_teacher_required": True,
                            "antibody_chains": pose["antibody_chains"],
                            "antigen_chains": pose["antigen_chains"],
                        },
                        {
                            "state_id": f"apo|{pose['pose_id']}",
                            "pose_id": f"{pose['pose_id']}_apo",
                            "state_type": "apo",
                            "source": f"paired_apo:{target_source}",
                            "source_panel": source_panel,
                            "prior_weight": 1.0,
                            "coordinate_path": apo_cache[cache_key]["path"].relative_to(
                                ROOT).as_posix(),
                            "coordinate_sha256": apo_cache[cache_key]["sha256"],
                            "independent_teacher_score": None,
                            "independent_teacher_required": False,
                            "antibody_chains": pose["antibody_chains"],
                            "antigen_chains": [],
                        },
                        {
                            "state_id": f"off_target|{pose['pose_id']}|{donor_component}",
                            "pose_id": f"{pose['pose_id']}_off_target",
                            "state_type": "off_target",
                            "source": "synthetic_cross_target_sequence_mismatch",
                            "source_panel": source_panel,
                            "prior_weight": 1.0,
                            "state_training_weight": 0.25,
                            "coordinate_path": target_path.relative_to(ROOT).as_posix(),
                            "coordinate_sha256": sha256(target_path),
                            "independent_teacher_score": None,
                            "independent_teacher_required": False,
                            "antibody_chains": pose["antibody_chains"],
                            "antigen_chains": pose["antigen_chains"],
                            "antigen_sequence_override": donor_sequence,
                            "donor_component": donor_component,
                            "evidence_tier": "synthetic_weak_not_experimental_nonbinder",
                        },
                    ])
                components.append({
                    "component_id": component_id,
                    "target": component["target"],
                    "antibody_lineage_cluster": lineage[component_id],
                    "global_sequence_cluster": f"{component['target']}|{lineage[component_id]}",
                    "candidate_sequence": candidate["sequence"],
                    "native_h3": component["native_h3"],
                    "dataset_source": "expanded_idp_development_v2",
                    "origin_arm": arm_name,
                    "substitution_bucket": bucket,
                    "states": states,
                })
    payload = {
        "schema_version": 1,
        "status": "explicit_states_frozen_before_training",
        "classification": "exposed_expanded_idp_target_apo_development",
        "inputs": {
            "candidates": {"path": str(candidates_path), "sha256": sha256(candidates_path)},
            "poses": {"path": str(poses_path), "sha256": sha256(poses_path)},
            "independent_scorer": {"path": str(scorer_path), "sha256": sha256(scorer_path)},
            "candidate_plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        },
        "component_candidate_count": len(components),
        "unique_component_count": len(candidates["components"]),
        "state_record_count": sum(len(row["states"]) for row in components),
        "maximum_poses_per_component": maximum_poses_per_component,
        "pose_selection": "deterministic_experimental_sampled_round_robin",
        "available_state_types": ["target", "apo", "off_target"],
        "missing_state_types": [],
        "components": components,
        "claim_boundary": (
            "Exposed target-versus-peptide-stripped-apo development states; "
            "apo is not an experimentally observed unbound ensemble"
        ),
    }
    output_path = output_dir / "explicit_states.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path(
        "reviewer_outputs/idp_ensemble_expanded_development_v2/candidates.json"))
    parser.add_argument("--poses", type=Path, default=Path(
        "reviewer_outputs/idp_ensemble_expanded_development_v2/merged_pose_manifest_v2.json"))
    parser.add_argument("--scorer", type=Path, default=Path(
        "reviewer_outputs/idp_ensemble_expanded_development_v2/independent_scorer_v2.json"))
    parser.add_argument("--plan", type=Path, default=Path(
        "reviewer_outputs/idp_ensemble_expanded_development_v2/candidate_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "reviewer_outputs/statecontrast_v2_expanded_states_v1"))
    parser.add_argument("--maximum-poses-per-component", type=int, default=5)
    args = parser.parse_args()
    result, path = prepare(
        ROOT / args.candidates, ROOT / args.poses, ROOT / args.scorer,
        ROOT / args.plan, ROOT / args.output_dir,
        maximum_poses_per_component=args.maximum_poses_per_component)
    print(json.dumps({
        "status": result["status"],
        "component_candidates": result["component_candidate_count"],
        "states": result["state_record_count"],
        "available": result["available_state_types"],
        "missing": result["missing_state_types"],
        "output": str(path.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
