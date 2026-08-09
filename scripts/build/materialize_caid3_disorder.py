#!/usr/bin/env python
"""Materialize frozen CAID3 three-line disorder FASTA without changing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError("CAID3 disorder reference is not a three-line FASTA")
    records = []
    for index in range(0, len(lines), 3):
        header, sequence, labels = lines[index : index + 3]
        if not header.startswith(">") or len(sequence) != len(labels):
            raise ValueError(f"Malformed CAID3 record at line {index + 1}")
        if set(labels) - {"0", "1", "-"}:
            raise ValueError(f"Unexpected CAID3 labels for {header}")
        records.append({"id": header[1:].split()[0], "sequence": sequence, "labels": labels})
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate CAID3 target IDs")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(ROOT / "data/caid3/frozen_raw/disorder_nox.fasta"))
    parser.add_argument("--manifest", default=str(ROOT / "data/caid3/frozen_raw/manifest.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/caid3/disorder_nox"))
    args = parser.parse_args()
    raw = Path(args.raw)
    manifest = json.loads(Path(args.manifest).read_text())
    expected = next(item["sha256"] for item in manifest["files"] if item["name"] == raw.name)
    observed = sha256(raw)
    if observed != expected:
        raise ValueError(f"Frozen CAID3 hash mismatch: {observed} != {expected}")
    records = load_records(raw)
    out_dir = Path(args.out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sequences.fasta").open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['id']}\n{record['sequence']}\n")
            tokens = ("nan" if label == "-" else label for label in record["labels"])
            (labels_dir / f"{record['id']}.label").write_text(
                " ".join(tokens) + "\n", encoding="ascii"
            )
    summary = {
        "schema_version": 1,
        "status": "materialized_from_hash_verified_frozen_raw",
        "source": str(raw),
        "source_sha256": observed,
        "targets": len(records),
        "residues": sum(len(record["sequence"]) for record in records),
        "defined_labels": sum(label != "-" for record in records for label in record["labels"]),
        "positive_labels": sum(label == "1" for record in records for label in record["labels"]),
    }
    (out_dir / "materialization.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
