#!/usr/bin/env python
"""Reaudit expanded coordinates using type-filtered SAbDab antigen chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from Bio.PDB import MMCIFParser

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_idp_ensemble_expanded_development_v2 import (  # noqa: E402
    residue_count,
)

ALLOWED_ANTIGEN_TYPES = {"PEPTIDE", "PROTEIN"}


def typed_antigen_chains(component):
    pairs = zip(
        component["antigen_chains"], component["antigen_type"], strict=False
    )
    return [chain for chain, antigen_type in pairs if antigen_type in ALLOWED_ANTIGEN_TYPES]


def resolve_antigen_chains(component, available):
    chains = component["antigen_chains"]
    types = component["antigen_type"]
    if len(chains) == len(types):
        return typed_antigen_chains(component), "metadata_chain_type_pairing"
    resolved = [
        chain for chain in chains
        if chain in available
        and chain != component["heavy_chain"]
        and chain != component["light_chain"]
        and residue_count(available[chain]) > 0
    ]
    return resolved, "coordinate_observed_amino_acid_fallback"


def audit(panel_path, coordinate_dir, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite coordinate audit v2: {output}")
    panel = json.loads(panel_path.read_text(encoding="ascii"))
    parser = MMCIFParser(QUIET=True)
    rows = []
    for component in panel["components"]:
        path = coordinate_dir / f"{component['pdb_id'].upper()}.cif"
        model = next(iter(parser.get_structure(component["component_id"], str(path))))
        available = {chain.id: chain for chain in model}
        heavy = component["heavy_chain"]
        light = component["light_chain"]
        antigens, antigen_resolution = resolve_antigen_chains(component, available)
        missing = [
            chain for chain in [heavy, light, *antigens]
            if chain and chain not in available
        ]
        heavy_count = residue_count(available[heavy]) if heavy in available else 0
        light_count = residue_count(available[light]) if light in available else None
        antigen_counts = {
            chain: residue_count(available[chain]) if chain in available else 0
            for chain in antigens
        }
        checks = {
            "typed_configured_chains_present": not missing,
            "heavy_chain_minimum_90_residues": heavy_count >= 90,
            "light_chain_minimum_80_or_absent": (
                light is None or (light_count or 0) >= 80
            ),
            "typed_antigen_has_observed_residues": bool(antigens) and any(
                count > 0 for count in antigen_counts.values()
            ),
        }
        rows.append({
            "component_id": component["component_id"],
            "pdb_id": component["pdb_id"],
            "target": component["target"],
            "lineage_proxy": component["lineage_proxy"],
            "typed_antigen_chains": antigens,
            "antigen_chain_resolution": antigen_resolution,
            "ignored_nonprotein_antigen_chains": [
                chain for chain in component["antigen_chains"]
                if chain not in antigens
            ],
            "missing_configured_chains": missing,
            "heavy_residue_count": heavy_count,
            "light_residue_count": light_count,
            "antigen_residue_counts": antigen_counts,
            "checks": checks,
            "coordinate_ready": all(checks.values()),
        })
    payload = {
        "schema_version": 3,
        "status": "expanded_coordinate_audit_v3_complete",
        "classification": "retrospective_expanded_idp_development_audit",
        "supersedes": (
            "coordinate_audit_v2.json: v2 zipped unequal antigen-chain and "
            "antigen-type lists, which misassigned the 7QCQ tau peptide chain"
        ),
        "component_count": len(rows),
        "coordinate_ready_count": sum(row["coordinate_ready"] for row in rows),
        "blocked_count": sum(not row["coordinate_ready"] for row in rows),
        "components": rows,
        "future_confirmation_eligible_count": 0,
        "decision": "use_coordinate_ready_components_for_development_only",
        "claim_boundary": panel["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "balanced_panel.json"
        ),
    )
    parser.add_argument(
        "--coordinates", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/coordinates"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reviewer_outputs/idp_ensemble_expanded_development_v2/"
            "coordinate_audit_v3.json"
        ),
    )
    args = parser.parse_args()
    result = audit(ROOT / args.panel, ROOT / args.coordinates, ROOT / args.output)
    print(json.dumps({
        "status": result["status"],
        "components": result["component_count"],
        "coordinate_ready": result["coordinate_ready_count"],
        "blocked": result["blocked_count"],
        "future_confirmation_eligible": result[
            "future_confirmation_eligible_count"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
