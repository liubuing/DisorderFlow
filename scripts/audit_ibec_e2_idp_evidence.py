#!/usr/bin/env python
"""Audit evidence tiers in the frozen 1,289-record conformation dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_lmdb(path):
    environment = lmdb.open(
        str(path), readonly=True, lock=False, readahead=False, max_readers=1
    )
    records = []
    with environment.begin() as transaction:
        for key, value in transaction.cursor():
            if key.startswith(b"__"):
                continue
            record = pickle.loads(value)
            records.append({"lmdb_key": key.decode("ascii"), **record})
    environment.close()
    return records


def canonical_fingerprint(records):
    lines = [f"{record.get('pdb_id')}\t{record.get('sequence')}" for record in records]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def disorder_intervals(disprot_entries, sequence_length):
    intervals = []
    disprot_ids = []
    accessions = []
    for entry in disprot_entries:
        disprot_ids.append(entry.get("disprot_id"))
        accessions.append(entry.get("acc"))
        for region in entry.get("regions", []):
            if (str(region.get("term_namespace", "")).lower() != "structural state"
                    or str(region.get("term_name", "")).lower() != "disorder"):
                continue
            start = max(1, int(region["start"]))
            end = min(sequence_length, int(region["end"]))
            if end >= start:
                intervals.append((start, end))
    return intervals, sorted(set(filter(None, disprot_ids))), sorted(set(filter(None, accessions)))


def interval_metrics(intervals, sequence_length):
    covered = set()
    for start, end in intervals:
        covered.update(range(start, end + 1))
    return {
        "disorder_residues": len(covered),
        "disorder_fraction": len(covered) / sequence_length if sequence_length else 0.0,
        "longest_disorder_region": max((end - start + 1 for start, end in intervals), default=0),
        "disorder_region_count": len(intervals),
    }


def classify(config, source_record, exact_entries):
    sequence = source_record["sequence"]
    intervals, disprot_ids, accessions = disorder_intervals(exact_entries, len(sequence))
    metrics = interval_metrics(intervals, len(sequence))
    classification = config["classification"]
    fraction = metrics["disorder_fraction"]
    longest = metrics["longest_disorder_region"]
    if exact_entries and intervals:
        if fraction >= float(classification["fully_disordered_minimum_fraction"]):
            tier = "experimentally_curated_fully_disordered"
        elif fraction >= float(classification["partially_disordered_minimum_fraction"]):
            tier = "experimentally_curated_partially_disordered"
        elif longest >= int(classification["idr_minimum_length"]):
            tier = "experimentally_curated_contains_idr"
        else:
            tier = "experimentally_curated_short_disorder"
    elif exact_entries:
        tier = "disprot_exact_without_disorder_region"
    else:
        tier = classification["source_label_semantics"].get(
            source_record.get("disorder_source"), "no_disorder_evidence"
        )
    return {
        "evidence_tier": tier,
        "exact_disprot_sequence_match": bool(exact_entries),
        "disprot_ids": ";".join(disprot_ids),
        "disprot_accessions": ";".join(accessions),
        **metrics,
    }


def validate_inputs(config):
    manifest_path = ROOT / config["inputs"]["frozen_manifest"]
    disprot_path = ROOT / config["inputs"]["disprot_entries"]
    if sha256(manifest_path) != config["inputs"]["frozen_manifest_sha256"]:
        raise ValueError("Frozen manifest hash mismatch")
    if sha256(disprot_path) != config["inputs"]["disprot_entries_sha256"]:
        raise ValueError("DisProt entry hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["count"] != int(config["inputs"]["conformation_records"]):
        raise ValueError("Frozen manifest count mismatch")


def build_report(summary):
    counts = summary["evidence_tier_counts"]
    lines = [
        "# iBEC E2 IDP Evidence Audit",
        "",
        "## Dataset Identity",
        "",
        f"- Source LMDB: {summary['source_records']} records.",
        f"- Frozen conformation LMDB: {summary['conformation_records']} records.",
        "- Every retained record contains exactly five conformations.",
        "- Twelve final source records (length 495-500) were mechanically omitted because the build stopped at 1,289; this was not biological filtering.",
        "",
        "## Evidence Tiers",
        "",
        "| Tier | Count |",
        "|---|---:|",
    ]
    for tier, count in counts.items():
        lines.append(f"| {tier} | {count} |")
    lines.extend([
        "",
        "## Corrected Competition Statement",
        "",
        f"> We built a 1,289-record, five-conformation protein dataset. {summary['experimental_disorder_evidence_records']} records have exact-sequence DisProt disorder-region evidence; the remaining records are separated into prediction-only, conformational-proxy, and no-evidence tiers.",
        "",
        "The previous phrase '800+ natural IDPs' is not supported and must not be used.",
        "",
        "## Claim Boundary",
        "",
        "MobiDB-lite predictions and AF2 pLDDT fallbacks are not experimental disorder annotations. Exact DisProt membership without a disorder region is also not counted as experimental disorder evidence.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/benchmarks/ibec_e2_idp_evidence_audit_v1.yml"
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_inputs(config)
    source_records = load_lmdb(ROOT / config["inputs"]["source_lmdb"])
    conformation_records = load_lmdb(ROOT / config["inputs"]["conformation_lmdb"])
    if len(source_records) != int(config["inputs"]["source_records"]):
        raise ValueError("Source record count mismatch")
    if len(conformation_records) != int(config["inputs"]["conformation_records"]):
        raise ValueError("Conformation record count mismatch")
    if canonical_fingerprint(source_records) != config["inputs"]["source_canonical_fingerprint"]:
        raise ValueError("Source canonical fingerprint mismatch")
    if canonical_fingerprint(conformation_records) != config["inputs"]["conformation_canonical_fingerprint"]:
        raise ValueError("Conformation canonical fingerprint mismatch")

    source_by_identity = {
        (record["pdb_id"], record["sequence"]): record for record in source_records
    }
    conformation_identities = {
        (record["pdb_id"], record["sequence"]) for record in conformation_records
    }
    missing = [
        record for record in source_records
        if (record["pdb_id"], record["sequence"]) not in conformation_identities
    ]
    if len(missing) != int(config["truncation"]["expected_missing_records"]):
        raise ValueError("Unexpected source-to-v5 missing count")
    if [record["lmdb_key"] for record in missing] != config["truncation"]["expected_source_keys"]:
        raise ValueError("Missing records are not the frozen mechanical tail")

    disprot = json.loads((ROOT / config["inputs"]["disprot_entries"]).read_text(
        encoding="utf-8"))["data"]
    if len(disprot) != int(config["inputs"]["disprot_entry_count"]):
        raise ValueError("DisProt entry count mismatch")
    disprot_by_sequence = defaultdict(list)
    for entry in disprot:
        if entry.get("sequence"):
            disprot_by_sequence[entry["sequence"]].append(entry)

    rows = []
    for conformation in conformation_records:
        source = source_by_identity[(conformation["pdb_id"], conformation["sequence"])]
        evidence = classify(config, source, disprot_by_sequence.get(conformation["sequence"], []))
        rows.append({
            "conformation_lmdb_key": conformation["lmdb_key"],
            "source_lmdb_key": source["lmdb_key"],
            "pdb_id": conformation["pdb_id"],
            "sequence_length": len(conformation["sequence"]),
            "sequence_sha256": sequence_sha256(conformation["sequence"]),
            "n_conformations": conformation["n_conformations"],
            "source_disorder_source": source.get("disorder_source"),
            "source_is_idp": source.get("is_idp"),
            "source_disorder_fraction": source.get("disorder_fraction"),
            **evidence,
        })
    tier_order = config["classification"]["priority"]
    raw_counts = Counter(row["evidence_tier"] for row in rows)
    counts = {tier: raw_counts.get(tier, 0) for tier in tier_order}
    experimental = sum(
        count for tier, count in counts.items()
        if tier.startswith("experimentally_curated_")
    )
    summary = {
        "schema_version": 1,
        "status": "ibec_e2_evidence_audit_complete",
        "config": args.config,
        "config_sha256": sha256(config_path),
        "runner_sha256": sha256(Path(__file__)),
        "source_records": len(source_records),
        "conformation_records": len(conformation_records),
        "records_with_five_conformations": sum(row["n_conformations"] == 5 for row in rows),
        "exact_disprot_sequence_matches": sum(row["exact_disprot_sequence_match"] for row in rows),
        "experimental_disorder_evidence_records": experimental,
        "evidence_tier_counts": counts,
        "source_disorder_source_counts": dict(sorted(Counter(
            str(row["source_disorder_source"]) for row in rows).items())),
        "truncation": {
            "missing_records": len(missing),
            "source_keys": [record["lmdb_key"] for record in missing],
            "minimum_length": min(len(record["sequence"]) for record in missing),
            "maximum_length": max(len(record["sequence"]) for record in missing),
            "interpretation": config["truncation"]["interpretation"],
        },
        "corrected_statement": (
            f"1,289 five-conformation protein records; {experimental} records have "
            "exact-sequence DisProt disorder-region evidence."
        ),
        "deprecated_statement": "800+ natural IDPs",
        "claim_boundary": config["claim_boundary"],
    }
    output_dir = ROOT / (args.out or config["outputs"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / config["outputs"]["records_csv"]
    with records_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing_payload = [{
        "source_lmdb_key": record["lmdb_key"],
        "pdb_id": record["pdb_id"],
        "sequence_length": len(record["sequence"]),
        "sequence_sha256": sequence_sha256(record["sequence"]),
        "disorder_source": record.get("disorder_source"),
        "is_idp": record.get("is_idp"),
    } for record in missing]
    (output_dir / config["outputs"]["missing_records_json"]).write_text(
        json.dumps(missing_payload, indent=2) + "\n", encoding="ascii"
    )
    summary["records_csv_sha256"] = sha256(records_path)
    (output_dir / config["outputs"]["summary_json"]).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    (output_dir / config["outputs"]["report_markdown"]).write_text(
        build_report(summary), encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "records": summary["conformation_records"],
        "exact_disprot_matches": summary["exact_disprot_sequence_matches"],
        "experimental_disorder_evidence": experimental,
        "evidence_tiers": counts,
        "missing_tail": summary["truncation"]["missing_records"],
    }, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
