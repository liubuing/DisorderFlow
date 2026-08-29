#!/usr/bin/env python
"""Verify donor heavy/H3 sequences and freeze independent-scorer inputs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from Bio.PDB import MMCIFParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_idp_ensemble_expanded_pose_sources_v2 import (  # noqa: E402
    chain_sequence,
)
from scripts.audit_idp_ensemble_expanded_structural_v2 import (  # noqa: E402
    number_h3,
)
from scripts.reaudit_idp_ensemble_expanded_development_v2 import (  # noqa: E402
    resolve_antigen_chains,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_identity(left, right):
    return difflib.SequenceMatcher(None, left, right).ratio()


def audit(config_path, donor_output, scorer_output):
    for path in (donor_output, scorer_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite scorer input artifact: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    for key, hash_key in (
        ("structural_panel", "structural_panel_sha256"),
        ("pose_source_audit", "pose_source_audit_sha256"),
        ("donor_manifest", "donor_manifest_sha256"),
    ):
        path = ROOT / config["inputs"][key]
        if sha256(path) != config["inputs"][hash_key]:
            raise RuntimeError(f"Frozen scorer input changed: {key}")
    structural_path = ROOT / config["inputs"]["structural_panel"]
    source_path = ROOT / config["inputs"]["pose_source_audit"]
    donor_manifest_path = ROOT / config["inputs"]["donor_manifest"]
    registry_path = ROOT / config["inputs"]["registry"]
    structural = json.loads(structural_path.read_text(encoding="ascii"))
    source = json.loads(source_path.read_text(encoding="ascii"))
    donor_manifest = json.loads(donor_manifest_path.read_text(encoding="ascii"))
    registry = json.loads(registry_path.read_text(encoding="ascii"))
    source_rows = {row["component_id"]: row for row in source["components"]}
    donor_paths = {
        row["pdb_id"]: Path(row["path"]) for row in donor_manifest["records"]
    }
    registry_groups = defaultdict(list)
    for row in registry["records"]:
        registry_groups[(row["target"], row["lineage_proxy"], row["pdb_id"])].append(row)
    parser = MMCIFParser(QUIET=True)
    best_records, donor_sequences = {}, {}
    for component in structural["components"]:
        reference_heavy = component["heavy_sequence"]
        source_row = source_rows[component["component_id"]]
        for pdb_id in source_row["same_lineage_target_pdb_ids"]:
            if pdb_id not in donor_paths:
                continue
            path = donor_paths[pdb_id]
            model = next(iter(parser.get_structure(pdb_id, str(path))))
            available = {chain.id: chain for chain in model}
            candidates = []
            for registry_row in registry_groups[
                (component["target"], component["lineage_proxy"], pdb_id)
            ]:
                heavy_chain = registry_row["heavy_chain"]
                if heavy_chain not in available:
                    continue
                sequence = chain_sequence(available[heavy_chain])
                identity = sequence_identity(reference_heavy, sequence)
                antigen_chains, antigen_method = resolve_antigen_chains(
                    registry_row, available
                )
                antigen_ready = bool(antigen_chains)
                candidates.append((
                    identity,
                    heavy_chain,
                    registry_row,
                    sequence,
                    antigen_chains,
                    antigen_method,
                    antigen_ready,
                ))
            if not candidates:
                best_records[(component["component_id"], pdb_id)] = None
                continue
            best = max(candidates, key=lambda row: (row[0], row[1]))
            best_records[(component["component_id"], pdb_id)] = best
            donor_sequences[f"{component['component_id']}::{pdb_id}"] = best[3]
    numbered = number_h3(donor_sequences, [93, 102]) if donor_sequences else {}
    components = []
    scorer_components = []
    threshold = config["donor_admission"]["minimum_heavy_sequence_identity"]
    minimum_poses = config["donor_admission"]["minimum_experimental_poses"]
    for component in structural["components"]:
        source_row = source_rows[component["component_id"]]
        donors = []
        for pdb_id in source_row["same_lineage_target_pdb_ids"]:
            if pdb_id not in donor_paths:
                continue
            best = best_records.get((component["component_id"], pdb_id))
            if best is None:
                donors.append({
                    "pdb_id": pdb_id,
                    "accepted": False,
                    "reasons": ["no_matching_configured_heavy_chain"],
                })
                continue
            identity, heavy_chain, registry_row, _sequence, antigen_chains, antigen_method, antigen_ready = best
            numbering = numbered[f"{component['component_id']}::{pdb_id}"]
            reasons = []
            if identity < threshold:
                reasons.append("heavy_sequence_identity_below_threshold")
            if numbering.get("h3_sequence") != component["numbering"]["h3_sequence"]:
                reasons.append("h3_sequence_mismatch")
            if not antigen_ready:
                reasons.append("no_observed_protein_or_peptide_antigen")
            donors.append({
                "pdb_id": pdb_id,
                "coordinate_path": str(donor_paths[pdb_id]),
                "coordinate_sha256": sha256(donor_paths[pdb_id]),
                "heavy_chain": heavy_chain,
                "light_chain": registry_row["light_chain"],
                "antigen_chains": antigen_chains,
                "antigen_chain_resolution": antigen_method,
                "heavy_sequence_identity": round(identity, 6),
                "h3_sequence": numbering.get("h3_sequence"),
                "accepted": not reasons,
                "reasons": reasons,
            })
        accepted = [row for row in donors if row["accepted"]]
        components.append({
            "component_id": component["component_id"],
            "target": component["target"],
            "lineage_proxy": component["lineage_proxy"],
            "reference_h3_sequence": component["numbering"]["h3_sequence"],
            "donors": donors,
            "accepted_donor_count": len(accepted),
            "experimental_pose_ready": len(accepted) >= minimum_poses,
            "restrained_sampling_required": len(accepted) < minimum_poses,
        })
        scorer_components.append({
            "component_id": component["component_id"],
            "target": component["target"],
            "lineage_proxy": component["lineage_proxy"],
            "reference": {
                "coordinate_path": component["coordinate_path"],
                "coordinate_sha256": component["coordinate_sha256"],
                "heavy_chain": component["heavy_chain"],
                "light_chain": component["light_chain"],
                "antigen_chains": component["antigen_chains"],
                "h3_sequence": component["numbering"]["h3_sequence"],
            },
            "accepted_experimental_donors": accepted,
            "independent_scorer_reference_ready": True,
            "experimental_ensemble_ready": len(accepted) >= minimum_poses,
            "restrained_sampling_required": len(accepted) < minimum_poses,
        })
    donor_audit = {
        "schema_version": 1,
        "status": "expanded_donor_sequence_audit_complete",
        "classification": config["classification"],
        "component_count": len(components),
        "experimental_pose_ready_count": sum(
            row["experimental_pose_ready"] for row in components
        ),
        "restrained_sampling_required_count": sum(
            row["restrained_sampling_required"] for row in components
        ),
        "components": components,
        "future_confirmation_eligible_count": 0,
        "decision": "use_sequence_verified_donors_only",
        "claim_boundary": config["claim_boundary"],
    }
    scorer = {
        "schema_version": 1,
        "status": "expanded_independent_scorer_inputs_frozen",
        "classification": config["classification"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "component_count": len(scorer_components),
        "reference_ready_count": sum(
            row["independent_scorer_reference_ready"] for row in scorer_components
        ),
        "experimental_ensemble_ready_count": sum(
            row["experimental_ensemble_ready"] for row in scorer_components
        ),
        "components": scorer_components,
        "future_confirmation_eligible_count": 0,
        "decision": "run_independent_scorer_on_fixed_reference_inputs",
        "claim_boundary": config["claim_boundary"],
    }
    donor_output.parent.mkdir(parents=True, exist_ok=True)
    donor_output.write_text(json.dumps(donor_audit, indent=2) + "\n", encoding="ascii")
    scorer_output.write_text(json.dumps(scorer, indent=2) + "\n", encoding="ascii")
    return donor_audit, scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(
            "configs/benchmarks/idp_ensemble_expanded_development_v2_scorer_inputs.yml"
        ),
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="ascii"))
    donor, scorer = audit(
        config_path,
        ROOT / config["outputs"]["donor_audit"],
        ROOT / config["outputs"]["scorer_inputs"],
    )
    print(json.dumps({
        "donor_audit_status": donor["status"],
        "components": donor["component_count"],
        "experimental_pose_ready": donor["experimental_pose_ready_count"],
        "restrained_sampling_required": donor[
            "restrained_sampling_required_count"
        ],
        "independent_scorer_reference_ready": scorer["reference_ready_count"],
        "future_confirmation_eligible": scorer[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
