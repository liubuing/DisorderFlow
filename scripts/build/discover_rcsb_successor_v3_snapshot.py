#!/usr/bin/env python
"""Exclude current SAbDab and all frozen reference PDBs from an RCSB snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REFERENCE_UNION_V3_SHA256 = (
    "61d63d6006a1ae1b0d6543572f78ca37502d4a804ef3bf34cbbea7ccea01d900")


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_pdb(value):
    text = str(value).strip().casefold()
    if text.startswith("pdb_"):
        text = text[4:]
    text = text.split("-", 1)[0].split("_", 1)[0]
    return text[-4:]


def sabdab_pdbs(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {normalize_pdb(row["PDB"]) for row in csv.DictReader(handle)}


def write_json_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def discover(acquisition_path, sabdab_summary, reference_path, output,
             expected_reference_sha256=REFERENCE_UNION_V3_SHA256):
    observed_reference_hash = sha256(reference_path)
    if (reference_path.name != "reference_union_manifest_v3.json"
            or observed_reference_hash != expected_reference_sha256):
        raise RuntimeError(
            "Successor-v3 discovery requires the frozen exposure-aware reference union v3")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite RCSB discovery: {output}")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if acquisition.get("classification") != "metadata_entry_ids_only":
        raise ValueError("RCSB acquisition has the wrong classification")
    sabdab = sabdab_pdbs(sabdab_summary)
    exposed = set(reference["exact_exposed_pdb_ids"])
    reference_pdbs = {row["pdb_id"] for row in reference["records"]}
    candidate_ids, exclusions = [], []
    for entry_id in acquisition["entry_ids"]:
        pdb = normalize_pdb(entry_id)
        reasons = []
        if pdb in sabdab:
            reasons.append("latest_sabdab")
        if pdb in exposed:
            reasons.append("exact_prior_exposure")
        if pdb in reference_pdbs:
            reasons.append("reference_union_pdb")
        if reasons:
            exclusions.append({"entry_id": entry_id, "reasons": reasons})
        else:
            candidate_ids.append(entry_id)
    payload = {
        "schema_version": 1,
        "status": ("RCSB_exact_new_entries_found" if candidate_ids
                   else "no_RCSB_exact_new_entries"),
        "classification": "model_free_exact_ID_discovery",
        "claim_boundary": "entry ID exclusion only; no structure or model access",
        "inputs": {
            "acquisition": str(acquisition_path),
            "acquisition_sha256": sha256(acquisition_path),
            "latest_sabdab_summary": str(sabdab_summary),
            "latest_sabdab_summary_sha256": sha256(sabdab_summary),
            "reference_union": str(reference_path),
            "reference_union_sha256": observed_reference_hash,
        },
        "counts": {
            "queried_entries": len(acquisition["entry_ids"]),
            "latest_sabdab_pdbs": len(sabdab),
            "reference_union_pdbs": len(reference_pdbs),
            "exact_exposed_pdbs": len(exposed),
            "excluded_entries": len(exclusions),
            "exact_new_entries": len(candidate_ids),
        },
        "minimum_required_entries": 12,
        "metadata_feasibility_passed": len(candidate_ids) >= 12,
        "candidate_entry_ids": candidate_ids,
        "exclusions": exclusions,
    }
    write_json_once(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sabdab-summary", type=Path,
        default=Path("data/sabdab2_snapshot_2026_08_06_successor_v3/all-summary.csv"))
    parser.add_argument(
        "--reference", type=Path,
        default=Path("data/successor_v3_confirmatory/reference_union_manifest_v3.json"))
    args = parser.parse_args()
    payload = discover(
        ROOT / args.acquisition, ROOT / args.sabdab_summary,
        ROOT / args.reference, ROOT / args.output)
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in {"candidate_entry_ids", "exclusions"}}, indent=2))


if __name__ == "__main__":
    main()
