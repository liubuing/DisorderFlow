#!/usr/bin/env python
"""Exclude holdout-antigen homologs from a disorder-supervision lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import lmdb

ROOT = Path(__file__).resolve().parents[2]


def clean_sequence(value):
    return "".join(aa for aa in str(value).upper() if aa in "ACDEFGHIKLMNPQRSTVWY")


def write_fasta(path, rows):
    with path.open("w", encoding="ascii") as handle:
        for row_id, sequence in rows:
            handle.write(f">{row_id}\n{clean_sequence(sequence)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout", default="data/multiscaffold_confirmatory_v2/holdout_manifest.json")
    parser.add_argument(
        "--lookup", default="data/disorder_supervision/train_afdb_balanced_v3.pkl")
    parser.add_argument("--lmdb", default="data/confidence_disprot_afdb_v1")
    parser.add_argument(
        "--mmseqs",
        default=("C:/biological/Metabolic model prediction/"
                 "Integrated_Yeast_MetaTwin_Deployment/tools/mmseqs2/mmseqs/bin/mmseqs.exe"))
    parser.add_argument(
        "--out", default="data/disorder_supervision/train_afdb_multiscaffold_v2.pkl")
    parser.add_argument(
        "--audit-out", default="data/multiscaffold_confirmatory_v2/disorder_lookup_audit.json")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    holdout_path = ROOT / args.holdout
    lookup_path = ROOT / args.lookup
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    envelope = pickle.loads(lookup_path.read_bytes())
    record_ids = sorted(envelope["profiles"])
    environment = lmdb.open(
        str(ROOT / args.lmdb), readonly=True, lock=False, readahead=False, subdir=True)
    training = []
    with environment.begin() as transaction:
        for record_id in record_ids:
            raw = transaction.get(record_id.encode("ascii"))
            if raw is None:
                raise KeyError(f"Missing confidence record: {record_id}")
            record = pickle.loads(raw)
            training.append((record_id, record["sequence"]))
    environment.close()
    antigens = [
        (component["representative_id"], component["representative"]["antigen_sequence"])
        for component in holdout["components"]]

    with tempfile.TemporaryDirectory(prefix="multiscaffold_lookup_") as tmp:
        temporary = Path(tmp)
        query = temporary / "holdout_antigens.fasta"
        target = temporary / "disorder_training.fasta"
        output = temporary / "hits.tsv"
        write_fasta(query, antigens)
        write_fasta(target, training)
        command = [
            args.mmseqs, "easy-search", str(query), str(target), str(output),
            str(temporary / "work"), "--min-seq-id", "0.3", "-c", "0.8",
            "--cov-mode", "2",
            "--format-output", "query,target,fident,qcov,tcov,evalue",
            "--threads", str(args.threads),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-2000:])
        hits = []
        if output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                query_id, target_id, identity, qcov, tcov, evalue = line.split("\t")
                hits.append({
                    "holdout": query_id,
                    "training_record": target_id,
                    "identity": float(identity),
                    "query_coverage": float(qcov),
                    "target_coverage": float(tcov),
                    "evalue": float(evalue),
                })
        command = [value.replace(str(temporary), "<TEMP>") for value in command]
    excluded = {row["training_record"] for row in hits}
    profiles = {
        key: value for key, value in envelope["profiles"].items() if key not in excluded}
    provenance = {
        key: value for key, value in envelope["provenance"].items() if key not in excluded}
    filtered = {
        **envelope,
        "source_contract": {
            **envelope["source_contract"],
            "multiscaffold_v2_holdout_sha256": hashlib.sha256(
                holdout_path.read_bytes()).hexdigest(),
            "antigen_homology_filter": {
                "minimum_identity": 0.3,
                "minimum_query_coverage": 0.8,
                "coverage_mode": 2,
            },
        },
        "profiles": profiles,
        "provenance": provenance,
        "stats": {
            **envelope["stats"],
            "n_sequences": len(profiles),
            "n_clusters": len({row.get("cluster_id") for row in provenance.values()}),
            "n_supervised_residues": sum(row.get("n_supervised", 0) for row in provenance.values()),
        },
    }
    out_path = ROOT / args.out
    out_path.write_bytes(pickle.dumps(filtered, protocol=pickle.HIGHEST_PROTOCOL))
    audit = {
        "schema_version": 1,
        "status": "holdout_antigen_query_coverage_filter_complete",
        "holdout_sha256": hashlib.sha256(holdout_path.read_bytes()).hexdigest(),
        "input_lookup_sha256": hashlib.sha256(lookup_path.read_bytes()).hexdigest(),
        "output_lookup_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "n_input_records": len(record_ids),
        "n_excluded_records": len(excluded),
        "n_output_records": len(profiles),
        "excluded_ids": sorted(excluded),
        "hits": hits,
        "mmseqs_version": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
        "command": command,
    }
    audit_path = ROOT / args.audit_out
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in audit.items() if key != "hits"}, indent=2))


if __name__ == "__main__":
    main()
