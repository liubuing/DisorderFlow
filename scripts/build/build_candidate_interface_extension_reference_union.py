#!/usr/bin/env python
"""Build the calibration-extension reference union from the frozen successor-v3 union plus round-1 exposures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UNION_STATUS = "frozen_successor_v3_confirmatory_reference_union"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AXES_FIELDS = ("vh_sequence", "vl_sequence", "paired_cdr_sequence",
               "cdr_h3_sequence", "antigen_sequence")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_pdb(value) -> str:
    text = str(value or "").casefold().strip()
    if text.startswith("pdb_"):
        text = text[4:]
    if len(text) > 4 and text.startswith("0"):
        text = text.lstrip("0")
    if len(text) > 4:
        text = text.split("_", 1)[0].split("-", 1)[0]
    return text


def clean_sequence(value) -> str:
    return "".join(char for char in str(value).upper() if char in AA)


def snapshot_entry_ids(snapshot_dir: Path) -> list[str]:
    response = json.loads((snapshot_dir / "search_response.json").read_text(encoding="ascii"))
    result_set = response.get("result_set")
    if not isinstance(result_set, list):
        raise ValueError("Round-1 snapshot search response lacks result_set")
    ids = [
        normalize_pdb(item["identifier"] if isinstance(item, dict) else item)
        for item in result_set
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Round-1 snapshot contains duplicate entry IDs")
    return sorted(ids)


def reference_records(structural_manifest: Path) -> list[dict[str, str]]:
    manifest = json.loads(structural_manifest.read_text(encoding="utf-8"))
    records = []
    seen: set[str] = set()
    for row in manifest["records"]:
        reference_id = f"EXTR{len(records) + 1:05d}"
        for field in AXES_FIELDS:
            if not clean_sequence(row[field]):
                raise ValueError(f"{row['instance']} has empty {field}")
        if reference_id in seen:
            raise ValueError(f"Duplicate reference id {reference_id}")
        seen.add(reference_id)
        records.append({
            "reference_id": reference_id,
            "id": reference_id,
            "pdb_id": normalize_pdb(row["pdb_id"]),
            "instance": row["instance"],
            **{field: clean_sequence(row[field]) for field in AXES_FIELDS},
        })
    return records


def build(base_manifest: Path, round1_snapshot: Path, round1_structural: Path,
          output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference union: {output}")
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    if base.get("status") != UNION_STATUS:
        raise ValueError("Base manifest is not a frozen successor-v3 reference union")

    added_records = reference_records(round1_structural)
    base_records = base["records"]
    base_ids = {row["reference_id"] for row in base_records}
    if base_ids & {row["reference_id"] for row in added_records}:
        raise ValueError("Extension reference ids collide with base union ids")

    added_pdb_ids = snapshot_entry_ids(round1_snapshot)
    exact_exposed = sorted(set(
        normalize_pdb(value) for value in base.get("exact_exposed_pdb_ids", [])) | set(added_pdb_ids))

    union = {
        "schema_version": 1,
        "status": UNION_STATUS,
        "classification": "reference_union",
        "created_by": "scripts/build/build_candidate_interface_extension_reference_union.py",
        "base_reference_union": {
            "path": base_manifest.resolve().as_posix(),
            "sha256": sha256_file(base_manifest),
        },
        "extension_round_inputs": {
            "round1_snapshot": {
                "path": round1_snapshot.resolve().as_posix(),
                "search_response_sha256": sha256_file(round1_snapshot / "search_response.json"),
                "entry_ids": len(added_pdb_ids),
            },
            "round1_structural_manifest": {
                "path": round1_structural.resolve().as_posix(),
                "sha256": sha256_file(round1_structural),
                "records": len(added_records),
            },
        },
        "exposure_policy": {
            "all_round1_snapshot_entry_ids": "exact_exclude",
            "all_round1_structurally_eligible_sequences": "five_axis_exclude",
        },
        "counts": {
            "base_reference_records": len(base_records),
            "added_extension_records": len(added_records),
            "deduplicated_reference_records": len(base_records) + len(added_records),
            "base_exact_exposed_pdb_ids": len(base.get("exact_exposed_pdb_ids", [])),
            "round1_snapshot_pdb_ids": len(added_pdb_ids),
            "exact_exposed_pdb_ids": len(exact_exposed),
        },
        "exact_exposed_pdb_ids": exact_exposed,
        "records": base_records + added_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(union, handle)
        handle.write("\n")
    return union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--round1-snapshot", type=Path, required=True)
    parser.add_argument("--round1-structural", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    union = build(
        ROOT / args.base_manifest,
        ROOT / args.round1_snapshot,
        ROOT / args.round1_structural,
        ROOT / args.out,
    )
    print(json.dumps(union["counts"], indent=2))


if __name__ == "__main__":
    main()
