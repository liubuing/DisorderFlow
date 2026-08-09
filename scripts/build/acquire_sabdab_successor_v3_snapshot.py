#!/usr/bin/env python
"""Acquire a write-once SAbDab summary snapshot for successor-v3 discovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary"
SUMMARY_NAME = "all-summary.csv"
MANIFEST_NAME = "acquisition.json"
CLAIM_BOUNDARY = (
    "raw SAbDab metadata acquisition only; no candidate selection, structure "
    "download, model inference, or confirmatory claim"
)


def normalize_pdb(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("pdb_"):
        text = text[4:]
    text = text.split("-", 1)[0].split("_", 1)[0]
    return text[-4:]


def parse_summary(content: bytes) -> tuple[list[dict[str, str]], dict[str, object]]:
    text = content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError("Downloaded SAbDab summary contains no data rows")
    required = {"INSTANCE", "PDB", "SABDABupdate_date"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"SAbDab summary lacks required columns: {sorted(missing)}")

    instances = [row["INSTANCE"].strip().casefold() for row in rows]
    if any(not value for value in instances):
        raise ValueError("SAbDab summary contains an empty INSTANCE")
    pdbs = {normalize_pdb(row["PDB"]) for row in rows}
    if "" in pdbs:
        raise ValueError("SAbDab summary contains an empty PDB identifier")
    update_dates = sorted(
        value for row in rows
        if (value := row["SABDABupdate_date"].strip()) not in {"", "NA"}
    )
    if not update_dates:
        raise ValueError("SAbDab summary contains no update dates")
    return rows, {
        "row_count": len(rows),
        "instance_count": len(set(instances)),
        "duplicate_instance_rows": len(rows) - len(set(instances)),
        "pdb_count": len(pdbs),
        "minimum_update_date": update_dates[0],
        "maximum_update_date": update_dates[-1],
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Retrieval timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def acquire_snapshot(output_dir: Path, opener=urllib.request.urlopen, now=utc_now):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot directory: {output_dir}")

    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "DisorderFlow-successor-v3-snapshot/1"},
    )
    with opener(request, timeout=120) as response:
        content = response.read()
    _, counts = parse_summary(content)
    retrieved_at = iso_utc(now())
    manifest = {
        "schema_version": 1,
        "classification": "metadata_only",
        "status": "write_once_raw_snapshot_acquired",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_url": SOURCE_URL,
        "retrieved_at_utc": retrieved_at,
        "summary_file": SUMMARY_NAME,
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        **counts,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        (output_dir / SUMMARY_NAME).write_bytes(content)
        (output_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    except Exception:
        # The directory was created by this invocation and is not a valid snapshot.
        for path in output_dir.iterdir():
            path.unlink()
        output_dir.rmdir()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory to create; existing paths are refused",
    )
    args = parser.parse_args()
    manifest = acquire_snapshot(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
