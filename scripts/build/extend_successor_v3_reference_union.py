#!/usr/bin/env python
"""Extend the successor-v3 reference union with all newly exposed exploratory IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_sha256(values):
    normalized = sorted({str(value).casefold().strip() for value in values})
    return hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest()


def normalize_pdb(value):
    text = str(value).casefold().strip()
    if text.startswith("pdb_"):
        text = text[4:]
    text = text.split("-", 1)[0].split("_", 1)[0].lstrip("0")
    if len(text) != 4 or not text[0].isdigit() or not text.isalnum():
        raise ValueError(f"Invalid PDB ID: {value!r}")
    return text


def extend(base_path, discovery_path, structural_path, rcsb_path, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite extended reference union: {output}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    rcsb = json.loads(rcsb_path.read_text(encoding="utf-8"))
    if base.get("status") != "frozen_successor_v3_confirmatory_reference_union":
        raise ValueError("Base reference union has the wrong status")
    if discovery.get("status") != "retrospective_exploratory_metadata_cohort":
        raise ValueError("Exploratory discovery has the wrong status")
    if structural.get("classification") != "model_free_structural_eligibility":
        raise ValueError("Structural manifest has the wrong classification")
    if rcsb.get("classification") != "metadata_entry_ids_only":
        raise ValueError("RCSB acquisition has the wrong classification")

    exploratory_pdbs = {normalize_pdb(row["pdb_id"])
                        for row in discovery["candidates"]}
    rcsb_pdbs = {normalize_pdb(value) for value in rcsb["entry_ids"]}
    exact_exposed = (
        {normalize_pdb(value) for value in base["exact_exposed_pdb_ids"]}
        | exploratory_pdbs | rcsb_pdbs)
    references = list(base["records"])
    existing_ids = {str(row["id"]).casefold() for row in references}
    added = []
    for row in sorted(structural["records"], key=lambda item: item["instance"]):
        identifier = str(row["instance"])
        if identifier.casefold() in existing_ids:
            continue
        added.append({
            "id": identifier,
            "pdb_id": normalize_pdb(row["pdb_id"]),
            "vh_sequence": row["vh_sequence"],
            "vl_sequence": row["vl_sequence"],
            "paired_cdr_sequence": row["paired_cdr_sequence"],
            "cdr_h3_sequence": row["cdr_h3_sequence"],
            "antigen_sequence": row["antigen_sequence"],
            "sources": ["successor_v3_retrospective_exploratory_2026_08_07"],
        })
    references.extend(added)
    references.sort(key=lambda row: (str(row["id"]).casefold(), str(row["id"])))
    for index, row in enumerate(references, 1):
        row["reference_id"] = f"SV3R{index:05d}"

    payload = {
        "schema_version": 2,
        "status": "frozen_successor_v3_confirmatory_reference_union",
        "generation": "v3_post_exploratory_exposure_update",
        "classification": "future_confirmatory_exclusion_reference; no performance claims",
        "created_by": "scripts/build/extend_successor_v3_reference_union.py",
        "base_reference_union": {
            "path": str(base_path), "sha256": sha256(base_path)},
        "new_exposure_inputs": {
            "exploratory_discovery": {
                "path": str(discovery_path), "sha256": sha256(discovery_path)},
            "exploratory_structural_manifest": {
                "path": str(structural_path), "sha256": sha256(structural_path)},
            "rcsb_acquisition": {
                "path": str(rcsb_path), "sha256": sha256(rcsb_path)},
        },
        "exposure_policy": {
            "all_exploratory_metadata_candidate_pdbs": "exact_exclude",
            "all_previously_viewed_rcsb_entry_ids": "exact_exclude",
            "all_structurally_eligible_exploratory_sequences": "five_axis_exclude",
        },
        "counts": {
            "base_reference_records": len(base["records"]),
            "added_exploratory_sequence_records": len(added),
            "deduplicated_reference_records": len(references),
            "base_exact_exposed_pdb_ids": len(base["exact_exposed_pdb_ids"]),
            "exploratory_metadata_pdb_ids": len(exploratory_pdbs),
            "viewed_rcsb_entry_ids": len(rcsb_pdbs),
            "exact_exposed_pdb_ids": len(exact_exposed),
        },
        "normalized_sets_sha256": {
            "reference_ids": set_sha256(row["id"] for row in references),
            "reference_pdb_ids": set_sha256(row["pdb_id"] for row in references),
            "exact_exposed_pdb_ids": set_sha256(exact_exposed),
        },
        "exact_exposed_pdb_ids": sorted(exact_exposed),
        "records": references,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=Path("data/successor_v3_confirmatory/reference_union_manifest_v2.json"))
    parser.add_argument(
        "--discovery", type=Path,
        default=Path("data/successor_v3_exploratory/discovery.json"))
    parser.add_argument(
        "--structural-manifest", type=Path,
        default=Path("data/successor_v3_exploratory/structures_final/structural_manifest.json"))
    parser.add_argument(
        "--rcsb-acquisition", type=Path,
        default=Path("data/successor_v3_rcsb_snapshot_2026_08_07/acquisition.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = extend(
        ROOT / args.base, ROOT / args.discovery, ROOT / args.structural_manifest,
        ROOT / args.rcsb_acquisition, ROOT / args.output)
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
