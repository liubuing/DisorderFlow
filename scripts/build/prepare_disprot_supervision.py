#!/usr/bin/env python3
"""Prepare CAID-excluded, UniRef50-split DisProt experimental supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import lmdb


def fasta_ids(path):
    ids = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
    return ids


def load_caid_ids(caid_root):
    return {
        target_id
        for fasta in Path(caid_root).glob("*/sequences.fasta")
        for target_id in fasta_ids(fasta)
    }


def experimental_profile(entry):
    """Use only explicit experimental disorder/order regions; leave all else unknown."""
    length = len(entry["sequence"])
    positive = [False] * length
    negative = [False] * length
    evidence_codes = set()
    for region in entry.get("regions", []):
        if region.get("term_namespace") != "Structural state":
            continue
        state = str(region.get("term_name", "")).casefold()
        if state not in {"disorder", "order"}:
            continue
        if region.get("ec_go") not in {"EXP", "IDA"}:
            continue
        start = max(0, int(region["start"]) - 1)
        end = min(length, int(region["end"]))
        target = positive if state == "disorder" else negative
        target[start:end] = [True] * (end - start)
        if region.get("ec_id"):
            evidence_codes.add(str(region["ec_id"]))
    conflict = [p and n for p, n in zip(positive, negative, strict=True)]
    mask = [(p or n) and not c for p, n, c in zip(positive, negative, conflict, strict=True)]
    values = [1.0 if p and m else 0.0 for p, m in zip(positive, mask, strict=True)]
    return values, mask, sorted(evidence_codes)


def stable_dev_cluster(cluster_id, dev_fraction):
    digest = hashlib.sha256(str(cluster_id).encode("utf-8")).hexdigest()
    value = int(digest[:15], 16) / float(16**15)
    return value < dev_fraction


def load_lmdb_index(lmdb_path):
    """Map exact (accession, sequence) pairs to local structure record IDs."""
    if lmdb_path is None:
        return None
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, subdir=True)
    index = {}
    with env.begin() as transaction:
        for key, value in transaction.cursor():
            if key == b"__len__":
                continue
            record = pickle.loads(value)
            accession = str(record.get("pdb_id", ""))
            sequence = str(record.get("sequence", ""))
            plddt = record.get("af2_plddt")
            if hasattr(plddt, "tolist"):
                plddt = plddt.tolist()
            index.setdefault((accession, sequence), []).append({
                "id": key.decode("ascii"),
                "af2_plddt": plddt,
            })
    env.close()
    return index


def prepare(entries_path, caid_root, output_dir, dev_fraction=0.1, lmdb_path=None,
            afdb_ordered_threshold=None):
    entries_envelope = json.loads(Path(entries_path).read_text(encoding="utf-8"))
    entries = entries_envelope.get("data", entries_envelope)
    caid_ids = load_caid_ids(caid_root)
    by_id = {str(entry["disprot_id"]): entry for entry in entries}
    caid_clusters = {
        str(by_id[target_id].get("uniref50"))
        for target_id in caid_ids
        if target_id in by_id and by_id[target_id].get("uniref50")
    }
    lmdb_index = load_lmdb_index(lmdb_path)
    split_records = {"train": [], "dev": []}
    excluded_ids = 0
    excluded_clusters = 0
    no_labels = 0
    no_structure_match = 0
    for entry in entries:
        target_id = str(entry["disprot_id"])
        cluster = str(entry.get("uniref50") or "")
        if target_id in caid_ids:
            excluded_ids += 1
            continue
        if cluster and cluster in caid_clusters:
            excluded_clusters += 1
            continue
        values, mask, evidence_codes = experimental_profile(entry)
        if not any(mask):
            no_labels += 1
            continue
        sequence = str(entry["sequence"])
        local_matches = [{"id": target_id, "af2_plddt": None}]
        if lmdb_index is not None:
            local_matches = lmdb_index.get((str(entry.get("acc", "")), sequence), [])
            if not local_matches:
                no_structure_match += 1
                continue
        split = "dev" if stable_dev_cluster(cluster or target_id, dev_fraction) else "train"
        for local_match in local_matches:
            local_id = local_match["id"]
            split_records[split].append({
                "id": local_id,
                "disprot_id": target_id,
                "accession": entry.get("acc"),
                "sequence": sequence,
                "values": values,
                "mask": mask,
                "source": "disprot_experimental",
                "cluster_id": cluster or target_id,
                "evidence_codes": evidence_codes,
            })
            plddt = local_match.get("af2_plddt")
            if afdb_ordered_threshold is not None and plddt is not None:
                if len(plddt) != len(sequence):
                    raise ValueError(f"AFDB pLDDT length mismatch for {local_id}")
                proxy_mask = [
                    not supervised and float(score) >= afdb_ordered_threshold
                    for score, supervised in zip(plddt, mask, strict=True)
                ]
                if any(proxy_mask):
                    split_records[split].append({
                        "id": local_id,
                        "disprot_id": target_id,
                        "accession": entry.get("acc"),
                        "sequence": sequence,
                        "values": [0.0] * len(sequence),
                        "mask": proxy_mask,
                        "source": "afdb_plddt_ordered_proxy",
                        "cluster_id": cluster or target_id,
                        "afdb_plddt_threshold": afdb_ordered_threshold,
                    })
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="ascii", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        clusters = sorted({record["cluster_id"] for record in records})
        (output_dir / f"{split}_clusters.txt").write_text("\n".join(clusters) + "\n", encoding="ascii")
    train_clusters = {record["cluster_id"] for record in split_records["train"]}
    dev_clusters = {record["cluster_id"] for record in split_records["dev"]}
    audit = {
        "schema_version": 1,
        "source": "DisProt experimental Structural state regions",
        "caid_ids": len(caid_ids),
        "caid_uniref50_clusters": len(caid_clusters),
        "excluded_direct_caid_ids": excluded_ids,
        "excluded_caid_clusters": excluded_clusters,
        "records_without_explicit_order_or_disorder": no_labels,
        "records_without_exact_local_structure_match": no_structure_match,
        "lmdb_path": str(lmdb_path) if lmdb_path else None,
        "afdb_ordered_threshold": afdb_ordered_threshold,
        "train_records": len(split_records["train"]),
        "dev_records": len(split_records["dev"]),
        "train_clusters": len(train_clusters),
        "dev_clusters": len(dev_clusters),
        "cross_split_clusters": sorted(train_clusters & dev_clusters),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="ascii")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, default=Path("data/disprot_current/entries.json"))
    parser.add_argument("--caid-root", type=Path, default=Path("data/caid2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/disorder_supervision/disprot_v1"))
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--lmdb", type=Path,
                        help="Optional confidence_idp_unified LMDB for exact accession+sequence mapping")
    parser.add_argument("--afdb-ordered-threshold", type=float,
                        help="Add low-weight ordered proxy labels above this pLDDT threshold")
    args = parser.parse_args()
    audit = prepare(args.entries, args.caid_root, args.output_dir, args.dev_fraction, args.lmdb,
                    args.afdb_ordered_threshold)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
