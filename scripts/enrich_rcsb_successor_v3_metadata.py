#!/usr/bin/env python
"""Acquire write-once RCSB entry and polymer-entity metadata for exact-new IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://data.rcsb.org/rest/v1/core"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_json(url, opener=urllib.request.urlopen):
    request = urllib.request.Request(
        url, headers={"User-Agent": "DisorderFlow-successor-v3-metadata/1"}
    )
    with opener(request, timeout=120) as response:
        return json.loads(response.read())


def entity_record(entry_id, entity_id, opener):
    payload = fetch_json(f"{ENDPOINT}/polymer_entity/{entry_id}/{entity_id}", opener)
    identifiers = payload.get("rcsb_polymer_entity_container_identifiers", {})
    source = payload.get("rcsb_entity_source_organism", [])
    return {
        "entity_id": str(entity_id),
        "asym_ids": identifiers.get("asym_ids", []),
        "auth_asym_ids": identifiers.get("auth_asym_ids", []),
        "description": payload.get("rcsb_polymer_entity", {}).get("pdbx_description"),
        "polymer_type": payload.get("entity_poly", {}).get("type"),
        "source_organisms": [row.get("ncbi_scientific_name") for row in source],
    }


def enrich(discovery_path, output, opener=urllib.request.urlopen, now=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metadata enrichment: {output}")
    discovery = json.loads(discovery_path.read_text(encoding="ascii"))
    records = []
    for entry_id in discovery.get("candidate_entry_ids", []):
        entry = fetch_json(f"{ENDPOINT}/entry/{entry_id}", opener)
        identifiers = entry.get("rcsb_entry_container_identifiers", {})
        entities = [
            entity_record(entry_id, entity_id, opener)
            for entity_id in identifiers.get("polymer_entity_ids", [])
        ]
        records.append({
            "entry_id": entry_id,
            "title": entry.get("struct", {}).get("title"),
            "initial_release_date": entry.get("rcsb_accession_info", {}).get(
                "initial_release_date"
            ),
            "experimental_methods": [
                row.get("method") for row in entry.get("exptl", [])
            ],
            "polymer_entities": entities,
        })
    retrieved = now or datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "status": "exact_new_metadata_enriched",
        "classification": "entry_and_polymer_entity_metadata_only",
        "retrieved_at_utc": retrieved.astimezone(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "source_discovery": str(discovery_path),
        "source_discovery_sha256": sha256(discovery_path),
        "record_count": len(records),
        "records": records,
        "claim_boundary": "metadata curation only; no coordinate or model access",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = enrich(args.discovery, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
