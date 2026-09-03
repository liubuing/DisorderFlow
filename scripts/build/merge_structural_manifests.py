#!/usr/bin/env python
"""Merge multiple structural manifests into one consolidated candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "instance", "pdb_id", "vh_sequence", "vl_sequence", "paired_cdr_sequence",
    "cdr_h3_sequence", "antigen_sequence", "n_contacting_h3_positions",
    "n_h3_antigen_residue_contacts", "resolution",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(manifests: list[Path], output: Path) -> dict:
    manifests = [Path(path) for path in manifests]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged manifest: {output}")
    if len(manifests) < 1:
        raise ValueError("At least one manifest is required")

    records: list[dict] = []
    sources = []
    seen_instances: set[str] = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(payload["records"]),
        })
        for row in payload["records"]:
            missing = sorted(REQUIRED_FIELDS - row.keys())
            if missing:
                raise ValueError(f"{path} record {row.get('instance')} lacks {missing}")
            instance = str(row["instance"])
            if instance in seen_instances:
                raise ValueError(f"Duplicate instance across manifests: {instance}")
            seen_instances.add(instance)
            records.append(row)

    merged = {
        "schema_version": "candidate_interface_extension_merged_manifest_v1",
        "status": "merged_extension_structural_manifest",
        "classification": "model_free_structural_eligibility",
        "claim_boundary": "sequence and structure metadata only; no model or result imports",
        "sources": sources,
        "counts": {
            "source_manifests": len(manifests),
            "total_records": len(records),
        },
        "records": sorted(records, key=lambda row: str(row["instance"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.manifests, args.output)["counts"], indent=2))


if __name__ == "__main__":
    main()
