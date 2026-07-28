#!/usr/bin/env python3
"""Audit CAID2 Disorder-NOX against training data and cache exact AFDB models."""

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_dataset_splits import load_lmdb  # noqa: E402
from benchmark_caid import load_caid_reference  # noqa: E402
from prepare_sabdab2_external import mmseqs_hits, write_fasta  # noqa: E402


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)


def map_target(target):
    disprot = fetch_json(f"https://disprot.org/api/{target['id']}")
    accession = disprot.get("acc")
    if not accession or disprot.get("sequence") != target["sequence"]:
        return {"id": target["id"], "status": "disprot_sequence_mismatch"}
    prediction = fetch_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}")
    if not prediction:
        return {"id": target["id"], "accession": accession, "status": "afdb_missing"}
    model = prediction[0]
    if model.get("sequence") != target["sequence"]:
        return {"id": target["id"], "accession": accession, "status": "afdb_sequence_mismatch"}
    return {
        "id": target["id"],
        "accession": accession,
        "length": len(target["sequence"]),
        "pdb_url": model["pdbUrl"],
        "afdb_version": model.get("latestVersion"),
        "global_plddt": model.get("globalMetricValue"),
        "status": "mapped",
    }


def download(item, output_dir):
    destination = output_dir / f"{item['id']}.pdb"
    if destination.exists() and destination.stat().st_size > 0:
        return True
    try:
        with urllib.request.urlopen(item["pdb_url"], timeout=90) as response:
            data = response.read()
        if not data.startswith(b"HEADER") and b"ATOM" not in data[:10000]:
            return False
        destination.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caid-dir", default="data/caid2/disorder_nox")
    parser.add_argument("--conformation-train", default="data/confidence_conformation_v5_clustered/confidence_train.lmdb")
    parser.add_argument("--phase3-train", default="data/phase3_v5_1_pair_clustered/train.lmdb")
    parser.add_argument("--output-dir", default="results/caid_workdir")
    parser.add_argument("--report", default="data/caid2/disorder_nox_afdb_audit.json")
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=380)
    args = parser.parse_args()

    targets = load_caid_reference(args.caid_dir)
    candidate_records = [{"id": target["id"], "sequence": target["sequence"]} for target in targets]
    training = load_lmdb(args.conformation_train)
    phase3 = load_lmdb(args.phase3_train)
    training_records = [
        {"id": f"conformation:{index}", "sequence": record.get("sequence", "")}
        for index, record in enumerate(training)
    ] + [
        {"id": f"phase3:{index}", "sequence": record.get("antigen_sequence", "")}
        for index, record in enumerate(phase3)
    ]

    with tempfile.TemporaryDirectory(prefix="caid2_audit_") as tmp:
        tmp_dir = Path(tmp)
        query = tmp_dir / "caid.fasta"
        database = tmp_dir / "training.fasta"
        hits_path = tmp_dir / "hits.tsv"
        write_fasta(query, candidate_records, "sequence", "id")
        write_fasta(database, training_records, "sequence", "id")
        hits = mmseqs_hits(
            args.mmseqs, query, database, hits_path, 0.3, tmp_dir / "work", args.threads)

    independent = [target for target in targets if target["id"] not in hits]
    mappings = []
    with ThreadPoolExecutor(max_workers=min(args.threads, 8)) as executor:
        jobs = {executor.submit(map_target, target): target["id"] for target in independent}
        for job in as_completed(jobs):
            target_id = jobs[job]
            try:
                mappings.append(job.result())
            except Exception as error:  # noqa: BLE001
                mappings.append({"id": target_id, "status": "mapping_error", "error": str(error)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [item for item in mappings if item["status"] == "mapped" and item["length"] <= args.max_length]
    with ThreadPoolExecutor(max_workers=min(args.threads, 8)) as executor:
        jobs = {executor.submit(download, item, output_dir): item for item in eligible}
        for job in as_completed(jobs):
            item = jobs[job]
            item["status"] = "cached" if job.result() else "download_error"

    report = {
        "benchmark": "CAID2 Disorder-NOX",
        "caid_targets": len(targets),
        "training_sources": {
            "conformation_train": args.conformation_train,
            "phase3_antigen_train": args.phase3_train,
        },
        "homology_threshold": {"minimum_identity": 0.3, "coverage": 0.8, "coverage_mode": 0},
        "homology_excluded": len(hits),
        "homology_independent": len(independent),
        "model_max_length": args.max_length,
        "cached_structures": sum(item["status"] == "cached" for item in mappings),
        "status_counts": {},
        "mappings": sorted(mappings, key=lambda item: item["id"]),
        "homology_hits": hits,
    }
    for item in mappings:
        report["status_counts"][item["status"]] = report["status_counts"].get(item["status"], 0) + 1
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("mappings", "homology_hits")}, indent=2))


if __name__ == "__main__":
    main()
