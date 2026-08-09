#!/usr/bin/env python
"""Discover peptide-antibody structures absent from the previously exposed archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalize_pdb(value):
    value = str(value).strip().casefold()
    if value.startswith("pdb_"):
        value = value[4:]
    return value[-4:]


def archive_pdbs(path):
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile("splits_final/abag_split.csv")
        if handle is None:
            raise RuntimeError("SAbDab2 archive lacks splits_final/abag_split.csv")
        lines = (line.decode("utf-8") for line in handle)
        return {normalize_pdb(row["PDB_ID"]) for row in csv.DictReader(lines)}


def exposed_pdbs(audit_paths):
    exposed = {"4hix", "3stb"}
    for path in audit_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split_records in payload.get("records", {}).values():
            for record in split_records:
                values = record.get("axis_values", {}).get("pdb_id", [])
                exposed.update(normalize_pdb(value) for value in values)
    return exposed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="data/sabdab2_current/all-summary.csv")
    parser.add_argument("--archive", default="data/sabdab2_current/splits.tar.gz")
    parser.add_argument(
        "--audit", action="append",
        default=["data/peptide_h3_publication_split_v4/audit.json",
                 "data/peptide_h3_temporal_split_v3/audit.json"])
    parser.add_argument(
        "--out", default="data/multiscaffold_confirmatory_v1/discovery.json")
    args = parser.parse_args()

    summary_path = ROOT / args.summary
    archive_path = ROOT / args.archive
    prior_archive = archive_pdbs(archive_path)
    exposed = exposed_pdbs([ROOT / value for value in args.audit])
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    candidates = {}
    for row in rows:
        antigen_types = {value.strip().upper() for value in row["antigen_type"].split("|")}
        pdb_id = normalize_pdb(row["PDB"])
        if "PEPTIDE" not in antigen_types or pdb_id in prior_archive or pdb_id in exposed:
            continue
        key = (pdb_id, row["SABDAB_ID"])
        candidates.setdefault(key, {
            "instance": row["INSTANCE"],
            "pdb_id": pdb_id,
            "sabdab_id": row["SABDAB_ID"],
            "heavy_chain": row["Hchain"],
            "light_chain": None if row["Lchain"] == "NA" else row["Lchain"],
            "antigen_chains": row["antigen_chain"].split("|"),
            "antigen_name": row["antigen_name"],
            "deposition_date": row["SABDABdepo_date"],
            "update_date": row["SABDABupdate_date"],
            "resolution": None if row["resolution"] == "NA" else float(row["resolution"]),
        })

    ordered = sorted(candidates.values(), key=lambda row: (row["pdb_id"], row["instance"]))
    status = "metadata_candidates_found" if ordered else "insufficient_new_data"
    report = {
        "schema_version": 1,
        "status": status,
        "claim_boundary": "metadata discovery only; no model inference and no homology-independence claim",
        "source": {
            "summary": args.summary,
            "summary_sha256": sha256(summary_path),
            "archive": args.archive,
            "archive_sha256": sha256(archive_path),
        },
        "exclusion": {
            "previous_archive_pdbs": len(prior_archive),
            "previously_exposed_pdbs": len(exposed),
            "audits": args.audit,
            "fixed_extra_pdbs": ["3stb", "4hix"],
        },
        "n_candidate_instances": len(ordered),
        "n_candidate_pdbs": len({row["pdb_id"] for row in ordered}),
        "minimum_required_antigen_components": 12,
        "candidates": ordered,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
