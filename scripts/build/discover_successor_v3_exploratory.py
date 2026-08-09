#!/usr/bin/env python
"""Build a deterministic retrospective exploratory SAbDab peptide cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_PDBS = 240
FIXED_PDBS = {"3stb", "4hix"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_pdb(value):
    text = str(value or "").strip().casefold()
    if text.startswith("pdb_"):
        text = text[4:]
    return text.split("-", 1)[0].split("_", 1)[0][-4:]


def candidate_row(row, pdb):
    resolution = row.get("resolution", "").strip()
    return {
        "instance": row["INSTANCE"],
        "pdb_id": pdb,
        "sabdab_id": row["SABDAB_ID"],
        "heavy_chain": row.get("Hchain"),
        "light_chain": row.get("Lchain"),
        "antigen_chains": sorted(set(
            value for value in row.get("antigen_chain", "").split("|") if value)),
        "antigen_types": sorted(set(
            value for value in row["antigen_type"].split("|") if value)),
        "antigen_name": row.get("antigen_name"),
        "deposition_date": row.get("SABDABdepo_date"),
        "update_date": row.get("SABDABupdate_date"),
        "resolution": None if resolution in {"", "NA"} else float(resolution),
    }


def ranking(row):
    date = "".join(char for char in str(row.get("SABDABdepo_date", "")) if char.isdigit())
    resolution = row.get("resolution", "").strip()
    resolution_rank = float(resolution) if resolution not in {"", "NA"} else float("inf")
    return (-int(date or 0), resolution_rank, row["INSTANCE"].casefold())


def discover(summary_path, reference_path, output, max_pdbs=MAX_PDBS):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite exploratory discovery: {output}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    excluded_pdbs = {
        normalize_pdb(row["pdb_id"]) for row in reference["records"]
    } | {normalize_pdb(value) for value in reference["exact_exposed_pdb_ids"]}
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    eligible_metadata, reason_counts = [], {
        "reference_or_prior_exposure_pdb": 0,
        "fixed_pdb": 0,
        "not_peptide": 0,
        "unpaired_heavy_light": 0,
    }
    for row in rows:
        pdb = normalize_pdb(row["PDB"])
        antigen_types = {value.strip().upper() for value in row["antigen_type"].split("|")}
        reasons = {
            "reference_or_prior_exposure_pdb": pdb in excluded_pdbs,
            "fixed_pdb": pdb in FIXED_PDBS,
            "not_peptide": "PEPTIDE" not in antigen_types,
            "unpaired_heavy_light": row.get("Hchain") in {None, "", "NA"}
            or row.get("Lchain") in {None, "", "NA"},
        }
        for reason, matched in reasons.items():
            reason_counts[reason] += int(matched)
        if not any(reasons.values()):
            eligible_metadata.append(row)

    selected = []
    selected_pdbs = set()
    for row in sorted(eligible_metadata, key=ranking):
        pdb = normalize_pdb(row["PDB"])
        if pdb in selected_pdbs:
            continue
        selected.append(candidate_row(row, pdb))
        selected_pdbs.add(pdb)
        if len(selected_pdbs) == max_pdbs:
            break

    payload = {
        "schema_version": 1,
        "status": "retrospective_exploratory_metadata_cohort",
        "classification": "metadata_only",
        "claim_boundary": (
            "retrospective exploratory cohort selection only; candidate metadata were "
            "available before analysis and cannot support confirmatory or external claims"),
        "selection_rule": {
            "source": "latest frozen SAbDab summary",
            "peptide_antigen_required": True,
            "paired_heavy_light_required": True,
            "reference_union_pdb_excluded": True,
            "one_instance_per_pdb": True,
            "ranking": ["deposition_date_desc", "resolution_asc", "instance_asc"],
            "maximum_pdbs": max_pdbs,
        },
        "inputs": {
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "reference_union": str(reference_path),
            "reference_union_sha256": sha256(reference_path),
        },
        "counts": {
            "summary_rows": len(rows),
            "metadata_eligible_instances": len(eligible_metadata),
            "metadata_eligible_pdbs": len({normalize_pdb(row["PDB"])
                                           for row in eligible_metadata}),
            "selected_instances": len(selected),
            "selected_pdbs": len(selected_pdbs),
        },
        "exclusion_match_counts": reason_counts,
        "candidates": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=Path("data/sabdab2_snapshot_2026_08_06_successor_v3/all-summary.csv"))
    parser.add_argument(
        "--reference", type=Path,
        default=Path("data/successor_v3_confirmatory/reference_union_manifest_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pdbs", type=int, default=MAX_PDBS)
    args = parser.parse_args()
    payload = discover(
        ROOT / args.summary, ROOT / args.reference, ROOT / args.output, args.max_pdbs)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
