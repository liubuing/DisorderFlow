#!/usr/bin/env python
"""Acquire a write-once RCSB metadata snapshot for calibration extension discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_URL = "https://data.rcsb.org/graphql"
USER_AGENT = "DisorderFlow-candidate-interface-extension/1"
GRAPHQL_BATCH_SIZE = 500
GRAPHQL_QUERY = """
query CandidateComplexes($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_accession_info { initial_release_date }
    exptl { method }
    rcsb_entry_info { resolution_combined }
    polymer_entities {
      rcsb_id
      rcsb_polymer_entity { pdbx_description }
      entity_poly {
        rcsb_entity_polymer_type
        rcsb_sample_sequence_length
        pdbx_seq_one_letter_code_can
      }
      rcsb_polymer_entity_container_identifiers {
        entity_id
        asym_ids
        auth_asym_ids
      }
      rcsb_cluster_membership { cluster_id identity }
    }
  }
}
"""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def post_json(url: str, payload: object, opener=urllib.request.urlopen) -> object:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("ascii"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with opener(request, timeout=120) as response:
        return json.loads(response.read())


def result_ids(search_response: dict[str, object]) -> list[str]:
    result_set = search_response.get("result_set")
    if not isinstance(result_set, list):
        raise ValueError("RCSB search response lacks result_set")
    ids = [
        str(item["identifier"] if isinstance(item, dict) else item).upper()
        for item in result_set
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("RCSB search response contains duplicate entry IDs")
    if search_response.get("total_count") != len(ids):
        raise ValueError("RCSB search response was truncated")
    return ids


def fetch_entries(ids: list[str], opener=urllib.request.urlopen) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for start in range(0, len(ids), GRAPHQL_BATCH_SIZE):
        payload = {
            "query": GRAPHQL_QUERY,
            "variables": {"ids": ids[start:start + GRAPHQL_BATCH_SIZE]},
        }
        response = post_json(DATA_URL, payload, opener=opener)
        if not isinstance(response, dict) or response.get("errors"):
            raise RuntimeError(f"RCSB Data API error: {response}")
        batch = response.get("data", {}).get("entries")
        if not isinstance(batch, list) or any(entry is None for entry in batch):
            raise ValueError("RCSB Data API returned missing entries")
        entries.extend(batch)
    by_id = {str(entry["rcsb_id"]).upper(): entry for entry in entries}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise ValueError(f"RCSB Data API omitted entries: {missing}")
    return [by_id[entry_id] for entry_id in ids]


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Retrieval timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def acquire(query_path: Path, protocol_path: Path, output_dir: Path,
            opener=urllib.request.urlopen, now=lambda: datetime.now(timezone.utc),
            protocol_section: str = "discovery_contract"):
    query_path = Path(query_path)
    protocol_path = Path(protocol_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot directory: {output_dir}")

    query_bytes = query_path.read_bytes()
    query_hash = sha256_bytes(query_bytes)
    protocol = json.loads(protocol_path.read_text(encoding="ascii"))
    section = protocol.get(protocol_section)
    if not isinstance(section, dict):
        raise ValueError(f"Protocol lacks section: {protocol_section}")
    expected_hash = section.get("query_config_sha256")
    if query_hash != expected_hash:
        raise ValueError(
            f"Frozen RCSB query hash does not match extension protocol "
            f"section {protocol_section}")
    query = json.loads(query_bytes)

    search_response = post_json(SEARCH_URL, query, opener=opener)
    if not isinstance(search_response, dict):
        raise ValueError("RCSB search response is not an object")
    ids = result_ids(search_response)
    entries = fetch_entries(ids, opener=opener)

    search_bytes = canonical_json(search_response)
    entries_bytes = canonical_json({"entries": entries})
    manifest = {
        "schema_version": "candidate_interface_extension_rcsb_snapshot_v1",
        "classification": "metadata_only",
        "status": "write_once_raw_snapshot_acquired",
        "claim_boundary": "metadata only; no coordinate access, homology claim, model inference, or target access",
        "retrieved_at_utc": iso_utc(now()),
        "query_config": str(query_path.as_posix()),
        "query_config_sha256": query_hash,
        "protocol": str(protocol_path.as_posix()),
        "protocol_sha256": sha256_bytes(protocol_path.read_bytes()),
        "protocol_section": protocol_section,
        "search_url": SEARCH_URL,
        "data_url": DATA_URL,
        "entry_count": len(entries),
        "files": {
            "search_response.json": {
                "byte_count": len(search_bytes),
                "sha256": sha256_bytes(search_bytes),
            },
            "entries.json": {
                "byte_count": len(entries_bytes),
                "sha256": sha256_bytes(entries_bytes),
            },
        },
    }
    manifest_bytes = canonical_json(manifest)

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        (output_dir / "search_response.json").write_bytes(search_bytes)
        (output_dir / "entries.json").write_bytes(entries_bytes)
        (output_dir / "acquisition.json").write_bytes(manifest_bytes)
    except Exception:
        for path in output_dir.iterdir():
            path.unlink()
        output_dir.rmdir()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-section", default="discovery_contract",
        help="Protocol section holding the frozen query hash",
    )
    args = parser.parse_args()
    print(json.dumps(acquire(args.query, args.protocol, args.output_dir,
                            protocol_section=args.protocol_section), indent=2))


if __name__ == "__main__":
    main()
