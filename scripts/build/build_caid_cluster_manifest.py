#!/usr/bin/env python3
"""Build an audited CAID target-to-UniRef50 cluster manifest."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fasta_ids(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [line[1:].split()[0] for line in handle if line.startswith(">")]


def fetch_uniref50(accession, retries=3):
    query = urllib.parse.urlencode({
        "query": f"identity:0.5 AND uniprot_id:{accession}",
        "format": "json",
        "size": 1,
    })
    url = f"https://rest.uniprot.org/uniref/search?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                results = json.load(response).get("results", [])
            if results and results[0].get("entryType") == "UniRef50":
                return str(results[0]["id"])
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == retries:
                return None
            time.sleep(2**attempt)
    return None


def build(entries_path, sequences_path, output, audit_path, threads=8):
    envelope = json.loads(Path(entries_path).read_text(encoding="utf-8"))
    entries = envelope.get("data", envelope)
    by_id = {str(entry["disprot_id"]): entry for entry in entries}
    target_ids = fasta_ids(sequences_path)
    manifest = {}
    sources = {}
    pending = {}

    for target_id in target_ids:
        entry = by_id.get(target_id)
        if entry and entry.get("uniref50"):
            manifest[target_id] = str(entry["uniref50"])
            sources[target_id] = "disprot_snapshot"
        elif entry and entry.get("acc"):
            pending[target_id] = str(entry["acc"])

    with ThreadPoolExecutor(max_workers=max(1, min(threads, 8))) as executor:
        jobs = {
            executor.submit(fetch_uniref50, accession): target_id
            for target_id, accession in pending.items()
        }
        for job in as_completed(jobs):
            target_id = jobs[job]
            cluster = job.result()
            if cluster:
                manifest[target_id] = cluster
                sources[target_id] = "uniprot_rest"

    unresolved = []
    for target_id in target_ids:
        if target_id not in manifest:
            manifest[target_id] = f"unresolved:{target_id}"
            sources[target_id] = "unresolved"
            unresolved.append(target_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    counts = {}
    for source in sources.values():
        counts[source] = counts.get(source, 0) + 1
    audit = {
        "schema_version": 1,
        "target_count": len(target_ids),
        "source_counts": counts,
        "resolved_count": len(target_ids) - len(unresolved),
        "unresolved_count": len(unresolved),
        "unresolved_ids": sorted(unresolved),
        "manifest": str(output),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, default=Path("data/disprot_current/entries.json"))
    parser.add_argument("--sequences", type=Path,
                        default=Path("data/caid2/disorder_nox/sequences.fasta"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/caid2/disorder_nox_clusters.json"))
    parser.add_argument("--audit", type=Path,
                        default=Path("data/caid2/disorder_nox_clusters.audit.json"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    audit = build(args.entries, args.sequences, args.output, args.audit, args.threads)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
