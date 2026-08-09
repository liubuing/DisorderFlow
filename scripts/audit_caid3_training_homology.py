#!/usr/bin/env python
"""Audit CAID3 against the exact balanced-v4 training sequence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[1]


def clean_sequence(sequence):
    return "".join(residue for residue in sequence.upper() if residue in "ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = []
    target_id = None
    sequence = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if target_id is not None:
                records.append({"id": target_id, "sequence": "".join(sequence)})
            target_id = line[1:].split()[0]
            sequence = []
        elif line:
            sequence.append(line)
    if target_id is not None:
        records.append({"id": target_id, "sequence": "".join(sequence)})
    return records


def load_training(lookup_path, lmdb_path):
    envelope = pickle.loads(Path(lookup_path).read_bytes())
    record_ids = sorted(envelope["profiles"])
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, subdir=True)
    records = []
    missing = []
    with env.begin() as transaction:
        for record_id in record_ids:
            raw = transaction.get(str(record_id).encode("ascii"))
            if raw is None:
                missing.append(record_id)
                continue
            record = pickle.loads(raw)
            sequence = clean_sequence(str(record.get("sequence", "")))
            if sequence:
                records.append({"id": str(record_id), "sequence": sequence})
    env.close()
    if missing:
        raise ValueError(f"Training LMDB is missing lookup records: {missing[:10]}")
    return records, envelope


def write_fasta(path, records):
    with Path(path).open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(f">{record['id']}\n{clean_sequence(record['sequence'])}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caid", default=str(ROOT / "data/caid3/disorder_nox/sequences.fasta"))
    parser.add_argument(
        "--lookup", default=str(ROOT / "data/disorder_supervision/train_afdb_balanced_v3.pkl")
    )
    parser.add_argument("--lmdb", default=str(ROOT / "data/confidence_disprot_afdb_v1"))
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--out", default=str(ROOT / "data/caid3/disorder_nox_training_homology.json")
    )
    parser.add_argument(
        "--training-clusters-out",
        default=str(ROOT / "data/caid3/balanced_v4_training_clusters.txt"),
    )
    parser.add_argument(
        "--cluster-manifest",
        default=str(ROOT / "data/caid3/disorder_nox_clusters.json"),
    )
    args = parser.parse_args()
    if shutil.which(args.mmseqs) is None:
        raise FileNotFoundError(f"MMseqs2 not found: {args.mmseqs}")
    caid = read_fasta(args.caid)
    training, envelope = load_training(args.lookup, args.lmdb)
    with tempfile.TemporaryDirectory(prefix="caid3_homology_") as temporary:
        temporary = Path(temporary)
        query = temporary / "caid3.fasta"
        target = temporary / "balanced_v4_train.fasta"
        result = temporary / "hits.tsv"
        write_fasta(query, caid)
        write_fasta(target, training)
        command = [
            args.mmseqs,
            "easy-search",
            str(query),
            str(target),
            str(result),
            str(temporary / "work"),
            "--min-seq-id",
            "0.3",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--format-output",
            "query,target,fident,qcov,tcov,evalue",
            "--threads",
            str(args.threads),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:])
        hits = {}
        if result.exists():
            for line in result.read_text().splitlines():
                query_id, target_id, identity, qcov, tcov, evalue = line.split("\t")
                hits.setdefault(query_id, []).append(
                    {
                        "training_record": target_id,
                        "identity_percent": float(identity),
                        "query_coverage": float(qcov),
                        "target_coverage": float(tcov),
                        "evalue": float(evalue),
                    }
                )
    independent = sorted(record["id"] for record in caid if record["id"] not in hits)
    cluster_manifest = json.loads(Path(args.cluster_manifest).read_text())
    unresolved = sorted(
        target_id
        for target_id in independent
        if str(cluster_manifest.get(target_id, "unresolved:")).startswith("unresolved:")
    )
    eligible = sorted(set(independent) - set(unresolved))
    training_clusters = sorted(
        {
            str(item.get("cluster_id"))
            for item in envelope["provenance"].values()
            if item.get("cluster_id")
        }
    )
    output = {
        "schema_version": 1,
        "status": "sequence_homology_audited_before_external_scoring",
        "threshold": {"minimum_identity": 0.3, "coverage": 0.8, "coverage_mode": 0},
        "caid_targets": len(caid),
        "training_records": len(training),
        "training_uniref50_clusters": len(training_clusters),
        "training_cluster_ids": training_clusters,
        "homology_excluded": len(hits),
        "homology_independent": len(independent),
        "independent_ids": independent,
        "unresolved_cluster_ids": unresolved,
        "eligible_ids": eligible,
        "hits": dict(sorted(hits.items())),
        "inputs": {
            "caid_sha256": hashlib.sha256(Path(args.caid).read_bytes()).hexdigest(),
            "lookup_sha256": hashlib.sha256(Path(args.lookup).read_bytes()).hexdigest(),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    Path(args.training_clusters_out).write_text(
        "\n".join(training_clusters) + "\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"training_cluster_ids", "independent_ids", "eligible_ids", "hits"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
