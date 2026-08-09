#!/usr/bin/env python3
"""Build a BFN-ready AFDB structure store for CAID-excluded DisProt records."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.build.build_idp_dataset_v3 import download_af2_pdb, preprocess_af2_pdb


def load_candidates(paths, min_length, max_length):
    candidates = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sequence = str(record["sequence"]).upper()
                accession = str(record.get("accession") or "")
                if accession and min_length <= len(sequence) <= max_length:
                    candidates[(accession, sequence)] = record
    return list(candidates.values())


def pdb_plddt(path):
    values = []
    seen = set()
    with Path(path).open(encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            residue = (line[21], line[22:26], line[26])
            if residue in seen:
                continue
            seen.add(residue)
            values.append(float(line[60:66]) / 100.0)
    return values


def save_lmdb(entries, output):
    import lmdb

    output.mkdir(parents=True, exist_ok=True)
    estimated = max(64 * 1024 * 1024, sum(len(pickle.dumps(entry)) for entry in entries) * 2)
    env = lmdb.open(str(output), map_size=estimated, subdir=True)
    with env.begin(write=True) as transaction:
        for index, entry in enumerate(entries):
            transaction.put(f"{index:08d}".encode("ascii"), pickle.dumps(entry))
        transaction.put(b"__len__", pickle.dumps(len(entries)))
    env.close()


def build(inputs, output, cache, report_path, min_length=20, max_length=350, threads=8,
          afdb_version=6):
    candidates = load_candidates(inputs, min_length, max_length)
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    with ThreadPoolExecutor(max_workers=max(1, min(threads, 8))) as executor:
        jobs = {
            executor.submit(download_af2_pdb, record["accession"], str(cache), afdb_version): record
            for record in candidates
        }
        for job in as_completed(jobs):
            record = jobs[job]
            downloaded[record["accession"]] = job.result()

    entries = []
    failures = {"no_afdb_structure": 0, "preprocess_failed": 0, "sequence_mismatch": 0,
                "plddt_mismatch": 0}
    for index, record in enumerate(candidates, 1):
        accession = str(record["accession"])
        expected_sequence = str(record["sequence"]).upper()
        pdb_path = downloaded.get(accession)
        if pdb_path is None:
            failures["no_afdb_structure"] += 1
            continue
        batch, structure_sequence, length = preprocess_af2_pdb(pdb_path)
        if batch is None:
            failures["preprocess_failed"] += 1
            continue
        if structure_sequence != expected_sequence or length != len(expected_sequence):
            failures["sequence_mismatch"] += 1
            continue
        plddt = pdb_plddt(pdb_path)
        if len(plddt) != length:
            failures["plddt_mismatch"] += 1
            continue
        entries.append({
            "pdb_id": accession,
            "sequence": expected_sequence,
            "batch": batch,
            "length": length,
            "af2_plddt": torch.tensor(plddt, dtype=torch.float32),
            "af2_iptm": torch.tensor(0.5, dtype=torch.float32),
            "af2_pae_matrix": torch.zeros(length, length, dtype=torch.float32),
            "is_idp": True,
            "source": "DisProt_experimental_AFDB_structure",
        })
        if index % 100 == 0:
            print(f"Processed {index}/{len(candidates)} candidates; retained {len(entries)}")

    save_lmdb(entries, output)
    report = {
        "schema_version": 1,
        "candidate_records": len(candidates),
        "retained_records": len(entries),
        "min_length": min_length,
        "max_length": max_length,
        "afdb_version": afdb_version,
        "failures": failures,
        "output": str(output),
        "cache": str(cache),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/confidence_disprot_afdb_v1"))
    parser.add_argument("--cache", type=Path, default=Path("data/disprot_afdb_cache"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/confidence_disprot_afdb_v1.audit.json"))
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=350)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--afdb-version", type=int, default=6)
    args = parser.parse_args()
    report = build(args.input, args.output, args.cache, args.report, args.min_length,
                   args.max_length, args.threads, args.afdb_version)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
