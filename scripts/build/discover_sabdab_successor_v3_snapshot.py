#!/usr/bin/env python
"""Discover exact-diff successor-v3 SAbDab metadata candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLAIM_BOUNDARY = (
    "metadata-only exact-diff discovery; no structure access, homology-independence "
    "claim, model inference, or confirmatory claim"
)
FIXED_PDBS = {"3stb", "4hix"}
INSTANCE_PDB = re.compile(r"^pdb_([0-9a-z]+)(?:[-_]|$)", re.IGNORECASE)


def normalize_pdb(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("pdb_"):
        text = text[4:]
    text = text.split("-", 1)[0].split("_", 1)[0]
    return text[-4:]


def normalize_instance(value: object) -> str:
    return str(value or "").strip().casefold()


def pdb_from_instance(value: object) -> str | None:
    match = INSTANCE_PDB.match(str(value or "").strip())
    return normalize_pdb(match.group(1)) if match else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"INSTANCE", "PDB", "SABDAB_ID", "antigen_type"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Summary {path} lacks required columns: {sorted(missing)}")
    instances = [normalize_instance(row["INSTANCE"]) for row in rows]
    if any(not value for value in instances):
        raise ValueError(f"Summary {path} contains an empty INSTANCE")
    return rows


def archive_pdbs(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        source = archive.extractfile("splits_final/abag_split.csv")
        if source is None:
            raise RuntimeError(f"{path} lacks splits_final/abag_split.csv")
        rows = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
        return {normalize_pdb(row["PDB_ID"]) for row in rows}


def pdbs_in_json(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in {"pdb", "pdb_id"}:
                values = child if isinstance(child, list) else [child]
                found.update(normalize_pdb(item) for item in values if item)
            found.update(pdbs_in_json(child))
    elif isinstance(value, list):
        for child in value:
            found.update(pdbs_in_json(child))
    elif isinstance(value, str):
        pdb = pdb_from_instance(value)
        if pdb:
            found.add(pdb)
    return found


def instance_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            found.update(instance_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(instance_ids(child))
    elif isinstance(value, str) and pdb_from_instance(value):
        found.add(normalize_instance(value))
    return found


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Acquisition timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def candidate_row(row: dict[str, str], pdb: str) -> dict[str, object]:
    resolution = row.get("resolution", "").strip()
    return {
        "instance": row["INSTANCE"],
        "pdb_id": pdb,
        "sabdab_id": row["SABDAB_ID"],
        "heavy_chain": row.get("Hchain"),
        "light_chain": None if row.get("Lchain") in {None, "", "NA"} else row["Lchain"],
        "antigen_chains": [value for value in row.get("antigen_chain", "").split("|") if value],
        "antigen_types": [value for value in row["antigen_type"].split("|") if value],
        "antigen_name": row.get("antigen_name"),
        "deposition_date": row.get("SABDABdepo_date"),
        "update_date": row.get("SABDABupdate_date"),
        "resolution": None if resolution in {"", "NA"} else float(resolution),
    }


def merge_candidate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in sorted(rows, key=lambda item: (
            normalize_instance(item["instance"]), str(item["antigen_name"] or ""))):
        key = normalize_instance(row["instance"])
        if key not in merged:
            merged[key] = dict(row)
            continue
        current = merged[key]
        for field in ("antigen_chains", "antigen_types"):
            current[field] = sorted(set(current[field]) | set(row[field]))
        names = {value for value in (current.get("antigen_name"), row.get("antigen_name"))
                 if value}
        current["antigen_name"] = "|".join(sorted(names)) if names else None
        resolutions = [value for value in (current.get("resolution"), row.get("resolution"))
                       if value is not None]
        current["resolution"] = min(resolutions) if resolutions else None
    return sorted(merged.values(), key=lambda row: (
        row["pdb_id"], normalize_instance(row["instance"])))


def discover(
    snapshot_dir: Path,
    output: Path,
    baseline: Path,
    archive: Path,
    audits: list[Path],
    v1_discovery: Path,
    v2_holdout: Path,
    v3_development: Path,
    now=lambda: datetime.now(timezone.utc),
):
    paths = {
        "snapshot_summary": Path(snapshot_dir) / "all-summary.csv",
        "snapshot_acquisition": Path(snapshot_dir) / "acquisition.json",
        "baseline_summary": Path(baseline),
        "historical_splits_archive": Path(archive),
        **{f"peptide_h3_audit_{index + 1}": path for index, path in enumerate(audits)},
        "v1_discovery": Path(v1_discovery),
        "v2_holdout": Path(v2_holdout),
        "v3_development": Path(v3_development),
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite discovery artifact: {output}")
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    acquisition = json.loads(paths["snapshot_acquisition"].read_text(encoding="ascii"))
    snapshot_hash = sha256(paths["snapshot_summary"])
    if acquisition.get("classification") != "metadata_only":
        raise ValueError("Snapshot acquisition is not classified as metadata_only")
    if acquisition.get("sha256") != snapshot_hash:
        raise ValueError("Snapshot hash does not match its acquisition artifact")
    discovered_at = now().astimezone(timezone.utc)
    if parse_utc(acquisition["retrieved_at_utc"]) >= discovered_at:
        raise ValueError("Snapshot acquisition artifact must predate discovery")

    snapshot_rows = read_summary(paths["snapshot_summary"])
    baseline_rows = read_summary(paths["baseline_summary"])
    baseline_instances = {normalize_instance(row["INSTANCE"]) for row in baseline_rows}
    baseline_pdbs = {normalize_pdb(row["PDB"]) for row in baseline_rows}
    historical_pdbs = archive_pdbs(paths["historical_splits_archive"])
    audit_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in audits]
    audit_pdb_sets = [pdbs_in_json(payload) for payload in audit_payloads]
    audit_pdbs = set().union(*audit_pdb_sets)

    v1 = json.loads(paths["v1_discovery"].read_text(encoding="utf-8"))
    v1_records = v1.get("candidates", [])
    if len(v1_records) != 111:
        raise ValueError(f"Expected 111 v1 discovery candidates, found {len(v1_records)}")
    v1_instances = {normalize_instance(row["instance"]) for row in v1_records}
    v1_pdbs = {normalize_pdb(row["pdb_id"]) for row in v1_records}

    v2 = json.loads(paths["v2_holdout"].read_text(encoding="utf-8"))
    v2_members = {
        normalize_instance(member)
        for component in v2.get("components", [])
        for member in component.get("members", [])
    }
    expected_v2 = int(v2.get("n_reference_independent_records", len(v2_members)))
    if len(v2_members) != expected_v2:
        raise ValueError(f"Expected {expected_v2} v2 holdout records, found {len(v2_members)}")
    v2_pdbs = {pdb_from_instance(value) for value in v2_members}
    v2_pdbs.discard(None)

    v3 = json.loads(paths["v3_development"].read_text(encoding="utf-8"))
    v3_records = v3.get("records", [])
    if len(v3_records) != 258:
        raise ValueError(f"Expected 258 successor-v3 development records, found {len(v3_records)}")
    v3_instances = {normalize_instance(row["id"]) for row in v3_records}
    v3_pdbs = {pdb_from_instance(value) for value in v3_instances}
    v3_pdbs.discard(None)

    reason_counts = {
        key: 0 for key in (
            "baseline_instance", "baseline_pdb", "historical_archive_pdb",
            "peptide_h3_audit_pdb", "v1_discovery_instance", "v1_discovery_pdb",
            "v2_holdout_instance", "v2_holdout_pdb", "v3_development_instance",
            "v3_development_pdb", "fixed_pdb", "not_peptide",
        )
    }
    candidate_rows = []
    exact_new_instances: set[str] = set()
    exact_new_pdbs: set[str] = set()
    for row in snapshot_rows:
        instance = normalize_instance(row["INSTANCE"])
        pdb = normalize_pdb(row["PDB"])
        antigen_types = {value.strip().upper() for value in row["antigen_type"].split("|")}
        matches = {
            "baseline_instance": instance in baseline_instances,
            "baseline_pdb": pdb in baseline_pdbs,
            "historical_archive_pdb": pdb in historical_pdbs,
            "peptide_h3_audit_pdb": pdb in audit_pdbs,
            "v1_discovery_instance": instance in v1_instances,
            "v1_discovery_pdb": pdb in v1_pdbs,
            "v2_holdout_instance": instance in v2_members,
            "v2_holdout_pdb": pdb in v2_pdbs,
            "v3_development_instance": instance in v3_instances,
            "v3_development_pdb": pdb in v3_pdbs,
            "fixed_pdb": pdb in FIXED_PDBS,
            "not_peptide": "PEPTIDE" not in antigen_types,
        }
        for reason, matched in matches.items():
            reason_counts[reason] += int(matched)
        if not matches["baseline_instance"]:
            exact_new_instances.add(instance)
        if not matches["baseline_pdb"]:
            exact_new_pdbs.add(pdb)
        if not any(matches.values()):
            candidate_rows.append(candidate_row(row, pdb))

    candidates = merge_candidate_rows(candidate_rows)
    report = {
        "schema_version": 1,
        "classification": "metadata_only",
        "status": "metadata_candidates_found" if candidates else "no_unexposed_metadata_candidates",
        "claim_boundary": CLAIM_BOUNDARY,
        "discovered_at_utc": discovered_at.isoformat().replace("+00:00", "Z"),
        "acquisition_predates_discovery": True,
        "inputs": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in paths.items()
        },
        "source_exposure_counts": {
            "baseline_instances": len(baseline_instances),
            "baseline_pdbs": len(baseline_pdbs),
            "historical_archive_pdbs": len(historical_pdbs),
            "peptide_h3_audit_pdbs": {
                str(path): len(values) for path, values in zip(audits, audit_pdb_sets)
            },
            "v1_discovery_instances": len(v1_instances),
            "v1_discovery_pdbs": len(v1_pdbs),
            "v2_holdout_records": len(v2_members),
            "v2_holdout_pdbs": len(v2_pdbs),
            "v3_development_records": len(v3_instances),
            "v3_development_pdbs": len(v3_pdbs),
            "fixed_pdbs": len(FIXED_PDBS),
        },
        "counts": {
            "snapshot_rows": len(snapshot_rows),
            "snapshot_instances": len({normalize_instance(row["INSTANCE"]) for row in snapshot_rows}),
            "snapshot_duplicate_instance_rows": (
                len(snapshot_rows)
                - len({normalize_instance(row["INSTANCE"]) for row in snapshot_rows})),
            "snapshot_pdbs": len({normalize_pdb(row["PDB"]) for row in snapshot_rows}),
            "instances_absent_from_baseline": len(exact_new_instances),
            "pdbs_absent_from_baseline": len(exact_new_pdbs),
            "candidate_instances": len(candidates),
            "candidate_pdbs": len({row["pdb_id"] for row in candidates}),
        },
        "exclusion_match_counts": reason_counts,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/sabdab2_current/all-summary.csv")
    parser.add_argument("--archive", type=Path, default=ROOT / "data/sabdab2_current/splits.tar.gz")
    parser.add_argument(
        "--audit", type=Path, action="append",
        default=[ROOT / "data/peptide_h3_publication_split_v4/audit.json",
                 ROOT / "data/peptide_h3_temporal_split_v3/audit.json"],
    )
    parser.add_argument("--v1-discovery", type=Path, default=ROOT / "data/multiscaffold_confirmatory_v1/discovery.json")
    parser.add_argument("--v2-holdout", type=Path, default=ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json")
    parser.add_argument("--v3-development", type=Path, default=ROOT / "data/successor_v3_development/manifest.json")
    args = parser.parse_args()
    report = discover(
        args.snapshot_dir, args.output, args.baseline, args.archive, args.audit,
        args.v1_discovery, args.v2_holdout, args.v3_development,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
