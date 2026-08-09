#!/usr/bin/env python
"""Acquire a deterministic write-once RCSB antibody entry-ID snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"
PAGE_ROWS = 10000
QUERY_SPEC = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "node_id": 0,
                "service": "full_text",
                "parameters": {"value": "antibody"},
            },
            {
                "type": "terminal",
                "node_id": 1,
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "greater_or_equal",
                    "value": "2026-07-14",
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {
        "results_content_type": ["experimental"],
        "results_verbosity": "compact",
        "sort": [{"sort_by": "rcsb_id", "direction": "asc"}],
        "paginate": {"rows": PAGE_ROWS},
    },
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def query_sha256():
    return hashlib.sha256(canonical_json(QUERY_SPEC).encode("ascii")).hexdigest()


def page_request(start):
    payload = json.loads(canonical_json(QUERY_SPEC))
    payload["request_options"]["paginate"]["start"] = int(start)
    return payload


def fetch_page(start, opener=urllib.request.urlopen):
    body = canonical_json(page_request(start)).encode("ascii")
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "User-Agent": "DisorderFlow-successor-v3-RCSB/1"},
        method="POST")
    with opener(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        content = response.read()
    if status == 204:
        return {"total_count": 0, "result_set": []}, content, status
    if status != 200:
        raise RuntimeError(f"RCSB search returned HTTP {status}")
    return json.loads(content), content, status


def result_ids(payload):
    values = payload.get("result_set", [])
    ids = [value if isinstance(value, str) else value.get("identifier") for value in values]
    if any(not value for value in ids):
        raise ValueError("RCSB result_set contains a missing entry identifier")
    return [str(value).upper() for value in ids]


def write_json_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def acquire(output_dir, opener=urllib.request.urlopen, now=None):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite RCSB snapshot: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir()
    expected_total, collected, pages = None, [], []
    try:
        start = 0
        while expected_total is None or len(collected) < expected_total:
            payload, raw, status = fetch_page(start, opener=opener)
            total = int(payload.get("total_count", 0))
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise RuntimeError(
                    f"RCSB total_count changed during pagination: {expected_total} -> {total}")
            ids = result_ids(payload)
            if ids != sorted(ids) or len(ids) != len(set(ids)):
                raise RuntimeError("RCSB page IDs are not unique and ascending")
            if collected and ids and ids[0] <= collected[-1]:
                raise RuntimeError("RCSB IDs are not strictly ascending across pages")
            page_path = pages_dir / f"page_{start:08d}.json"
            page_path.write_bytes(raw)
            pages.append({
                "start": start, "http_status": status, "count": len(ids),
                "total_count": total,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "query_id": payload.get("query_id"),
            })
            collected.extend(ids)
            if not ids:
                break
            start += PAGE_ROWS
        if len(collected) != expected_total or len(collected) != len(set(collected)):
            raise RuntimeError(
                f"Incomplete RCSB snapshot: expected {expected_total}, got {len(collected)}")
        retrieved = (now or datetime.now(timezone.utc))
        manifest = {
            "schema_version": 1,
            "status": "write_once_RCSB_entry_snapshot_acquired",
            "classification": "metadata_entry_ids_only",
            "claim_boundary": "RCSB entry-ID query only; no structure or model access",
            "api_version_observed": "2.6.0",
            "endpoint": ENDPOINT,
            "retrieved_at_utc": retrieved.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "query_spec": QUERY_SPEC,
            "query_sha256": query_sha256(),
            "total_count": expected_total,
            "entry_ids_sha256": hashlib.sha256("\n".join(collected).encode("ascii")).hexdigest(),
            "entry_ids": collected,
            "pages": pages,
        }
        write_json_once(output_dir / "acquisition.json", manifest)
        return manifest
    except Exception:
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        output_dir.rmdir()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = acquire(args.output_dir)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"entry_ids", "query_spec", "pages"}}, indent=2))


if __name__ == "__main__":
    main()
