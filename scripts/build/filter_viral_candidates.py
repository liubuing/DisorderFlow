#!/usr/bin/env python
"""Extract the viral-antigen candidate subset from a discovery artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_viral(discovery_path: Path, output: Path) -> dict:
    discovery_path = Path(discovery_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite filtered discovery: {output}")
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    if discovery.get("classification") != "metadata_only":
        raise ValueError("Discovery artifact is not metadata-only")
    viral_entries = [
        entry for entry in discovery["entries"]
        if entry.get("status") == "candidate" and entry.get("viral_antigen") is True
    ]
    artifact = {
        "schema_version": discovery.get("schema_version"),
        "classification": "metadata_only",
        "status": "write_once_viral_filtered_discovery",
        "claim_boundary": "metadata-only; viral-antigen candidate subset only",
        "source_discovery": str(discovery_path),
        "source_discovery_sha256": sha256_file(discovery_path),
        "snapshot_dir": discovery["snapshot_dir"],
        "snapshot_manifest_sha256": discovery["snapshot_manifest_sha256"],
        "eligibility": discovery.get("eligibility", {}),
        "counts": {
            "viral_antigen_candidates": len(viral_entries),
        },
        "entries": viral_entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(filter_viral(args.discovery, args.output)["counts"], indent=2))


if __name__ == "__main__":
    main()
