#!/usr/bin/env python
"""Materialize exact-new RCSB PDB coordinates after metadata readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(readiness_path, discovery_path, output_dir, opener=urllib.request.urlopen):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite coordinate snapshot: {output_dir}")
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    discovery = json.loads(discovery_path.read_text(encoding="ascii"))
    if readiness.get("ready") is not True:
        raise RuntimeError("Metadata readiness has not passed")
    if readiness.get("source_discovery_sha256") != sha256(discovery_path):
        raise RuntimeError("Readiness and discovery hashes do not match")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for entry_id in discovery.get("candidate_entry_ids", []):
            url = f"https://files.rcsb.org/download/{entry_id}.cif"
            request = urllib.request.Request(
                url, headers={"User-Agent": "DisorderFlow-successor-v3-coordinates/1"}
            )
            with opener(request, timeout=120) as response:
                content = response.read()
            path = output_dir / f"{entry_id}.cif"
            path.write_bytes(content)
            records.append({
                "entry_id": entry_id,
                "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "source_url": url,
            })
        payload = {
            "schema_version": 1,
            "status": "exact_new_coordinates_materialized",
            "classification": "post_readiness_coordinate_snapshot",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "readiness": str(readiness_path),
            "readiness_sha256": sha256(readiness_path),
            "discovery": str(discovery_path),
            "discovery_sha256": sha256(discovery_path),
            "record_count": len(records),
            "records": records,
            "claim_boundary": "coordinate curation only; no model or performance result",
        }
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        return payload
    except Exception:
        for path in output_dir.glob("*"):
            path.unlink()
        output_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.readiness, args.discovery, args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
