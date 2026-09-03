#!/usr/bin/env python
"""Freeze the calibration-extension membership manifest from the consolidated audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(audit_path: Path, merged_path: Path, contract_path: Path, output: Path) -> dict:
    audit_path = Path(audit_path)
    merged_path = Path(merged_path)
    contract_path = Path(contract_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite extension manifest: {output}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("gate_passed"):
        raise ValueError("Consolidated audit gate did not pass")
    if audit.get("status") != "frozen_model_free_successor_v3_isolation_audit":
        raise ValueError("Audit artifact is not the frozen isolation audit")
    if len(audit["components"]) < 5:
        raise ValueError("Fewer than 5 independent components")

    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    by_instance = {str(row["instance"]): row for row in merged["records"]}

    members = []
    for component in audit["components"]:
        representative = component["representative_id"]
        row = by_instance[representative]
        members.append({
            "component_id": component["component_id"],
            "representative_instance": representative,
            "pdb_id": row["pdb_id"],
            "viral_antigen": bool(row.get("viral_antigen")),
            "resolution": row["resolution"],
            "antigen_sequence": row["antigen_sequence"],
            "cdr_h3_sequence": row["cdr_h3_sequence"],
            "vh_sequence": row["vh_sequence"],
            "vl_sequence": row["vl_sequence"],
            "paired_cdr_sequence": row["paired_cdr_sequence"],
            "n_contacting_h3_positions": row["n_contacting_h3_positions"],
            "n_h3_antigen_residue_contacts": row["n_h3_antigen_residue_contacts"],
            "pdb_path": row["pdb_path"],
            "pdb_sha256": row["pdb_sha256"],
            "member_instances": component["members"],
        })

    manifest = {
        "schema_version": "candidate_interface_calibration_extension_manifest_v1",
        "classification": "preregistered_development_only",
        "status": "frozen_calibration_extension_membership",
        "claim_boundary": (
            "model-free structural eligibility and isolation only; no confidence "
            "target access; calibration extension members are excluded from training, "
            "checkpoint selection, and sealed test"),
        "frozen_split_manifest_sha256": json.loads(
            contract_path.read_text(encoding="utf-8")).get(
            "frozen_split_manifest_sha256"),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "merged_manifest_path": str(merged_path),
        "merged_manifest_sha256": sha256_file(merged_path),
        "membership": {
            "train": [],
            "calibration": [member["component_id"] for member in members],
            "test": [],
        },
        "counts": {
            "independent_components": len(members),
            "minimum_required": 5,
            "representatives": len(members),
        },
        "components": members,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.audit, args.merged_manifest, args.contract,
                            args.out)["counts"], indent=2))


if __name__ == "__main__":
    main()
