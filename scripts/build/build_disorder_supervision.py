#!/usr/bin/env python3
"""Build split-scoped disorder supervision from provenance-rich JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.disorder_supervision import (  # noqa: E402
    audit_cluster_splits,
    merge_evidence,
)


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                missing = {"id", "sequence", "values", "source", "cluster_id"} - set(record)
                if missing:
                    raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
                records.append(record)
    return records


def build_artifact(inputs, split, output, held_out_clusters=()):
    records = [record for path in inputs for record in read_jsonl(path)]
    held_out = {str(value) for value in held_out_clusters}
    leaking = sorted({str(r["cluster_id"]) for r in records} & held_out)
    if leaking:
        raise ValueError(f"Held-out clusters present in {split}: {leaking[:10]}")
    grouped = defaultdict(list)
    sequences = {}
    for record in records:
        sample_id = str(record["id"])
        sequence = str(record["sequence"]).upper()
        if sample_id in sequences and sequences[sample_id] != sequence:
            raise ValueError(f"Conflicting sequences for {sample_id}")
        sequences[sample_id] = sequence
        grouped[sample_id].append(record)
    profiles = {}
    provenance = {}
    for sample_id, evidence_records in grouped.items():
        merged = merge_evidence(evidence_records, len(sequences[sample_id]))
        profiles[sample_id] = {
            "values": merged.values,
            "mask": merged.mask,
            "confidence": merged.confidence,
        }
        provenance[sample_id] = {
            "sources": list(merged.sources),
            "cluster_id": merged.cluster_id,
            "sequence_sha256": hashlib.sha256(sequences[sample_id].encode("ascii")).hexdigest(),
            "n_supervised": int(merged.mask.sum()),
        }
    artifact = {
        "schema_version": 3,
        "split": split,
        "source_contract": {
            "claim_scope": "confidence-weighted residue-level disorder supervision",
            "experimental_sources": ["disprot_experimental", "mobidb_experimental"],
            "teacher_sources_are_ground_truth": False,
        },
        "profiles": profiles,
        "provenance": provenance,
        "stats": {
            "n_sequences": len(profiles),
            "n_clusters": len({p["cluster_id"] for p in provenance.values()}),
            "n_supervised_residues": sum(p["n_supervised"] for p in provenance.values()),
            "sources": sorted({source for p in provenance.values() for source in p["sources"]}),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    output.with_suffix(".audit.json").write_text(
        json.dumps({k: v for k, v in artifact.items() if k not in {"profiles", "provenance"}}, indent=2) + "\n",
        encoding="ascii",
    )
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "dev", "external_test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--held-out-clusters", type=Path)
    args = parser.parse_args()
    held_out = []
    if args.held_out_clusters:
        held_out = [line.strip() for line in args.held_out_clusters.read_text().splitlines() if line.strip()]
    artifact = build_artifact(args.input, args.split, args.output, held_out)
    print(json.dumps({"output": str(args.output), **artifact["stats"]}, indent=2))


if __name__ == "__main__":
    main()
