#!/usr/bin/env python
"""Build a post-ProteinMPNN-training temporal flexible-peptide H3 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sabdab2_external_lmdb import chain_ids  # noqa: E402
from preprocess_sabdab_phase3 import build_lmdb, preprocess_one  # noqa: E402
from scripts.build.build_peptide_h3_publication_split import (  # noqa: E402
    annotate_record,
    clean_sequence,
    cluster_tokens,
    digest_ids,
    exclude_conflicts,
    exclude_homology_conflicts,
    serializable_record,
    write_manifest,
)


def parse_date(value):
    return date.fromisoformat(str(value).strip())


def read_official_rows(archive_path, member):
    with tarfile.open(archive_path, "r:gz") as archive:
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(f"Missing {member} in {archive_path}")
        return list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8")))


def metadata_record(row):
    instance = row["INSTANCE"]
    antigen = clean_sequence(str(row.get("agresolvedseqs", "")).replace("/", ""))
    return {
        "id": instance,
        "row": row,
        "axis_values": {
            "pdb_id": (str(row.get("PDB_ID", "")).casefold(),),
            "official_ab_cluster": cluster_tokens(row.get("ab_cluster")),
            "official_cdrh3_cluster": cluster_tokens(row.get("cdrh3_cluster")),
            "official_antigen_cluster": cluster_tokens(row.get("agclusters")),
            "vh_sequence_exact": (clean_sequence(
                row.get("VH_numerable_seq", "") or row.get("Hseq", "")),),
            "vl_sequence_exact": (clean_sequence(
                row.get("VL_numerable_seq", "") or row.get("Lseq", "")),),
            "cdr_h3_sequence_exact": (clean_sequence(row.get("CDRH3", "")),),
            "antigen_sequence_exact": (antigen,),
        },
    }


def metadata_eligible(row, config, training_cutoff):
    try:
        deposition = parse_date(row.get("PDBdepo", ""))
    except ValueError:
        return False
    antigen_types = str(row.get("agtypes", "")).upper().split("/")
    antigen = clean_sequence(str(row.get("agresolvedseqs", "")).replace("/", ""))
    h3 = clean_sequence(row.get("CDRH3", ""))
    return bool(
        deposition > training_cutoff
        and row.get("holo") == "True"
        and "PEPTIDE" in antigen_types
        and "PROTEIN" not in antigen_types
        and int(config["minimum_antigen_length"]) <= len(antigen)
        <= int(config["maximum_antigen_length"])
        and int(config["minimum_h3_length"]) <= len(h3)
        <= int(config["maximum_h3_length"])
        and clean_sequence(row.get("VH_numerable_seq", "") or row.get("Hseq", ""))
    )


def scored_ids(paths):
    values = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.update(str(record["id"]) for record in payload["results"])
    return values


def extract_cifs(archive_path, records, output_dir):
    wanted = {
        f"splits_final/{record['id']}.cif": record["id"] for record in records
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            instance = wanted.get(member.name)
            if instance is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with (output_dir / f"{instance}.cif").open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.add(instance)
    missing = sorted(set(wanted.values()) - extracted)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected CIF files")


def preprocess_records(records, cif_dir):
    processed = []
    failures = []
    for index, metadata in enumerate(records, 1):
        row = metadata["row"]
        instance = metadata["id"]
        result = preprocess_one(
            instance,
            cif_dir / f"{instance}.cif",
            row.get("Hchain") or None,
            row.get("Lchain") or None,
            chain_ids(row.get("agchains", "")),
        )
        if result is None or result.get("antigen") is None:
            failures.append(instance)
        else:
            result["pdb_id"] = str(row.get("PDB_ID", "")).casefold().removeprefix("pdb_")
            result["official_instance"] = instance
            result["external_split"] = "post_proteinmpnn_temporal"
            processed.append(result)
        if index % 50 == 0:
            print(f"Preprocessed {index}/{len(records)}", flush=True)
    return processed, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "benchmarks" / "peptide_h3_temporal_split_v1.yml"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "peptide_h3_temporal_split_v1"))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite temporal benchmark: {output}")
    output.mkdir(parents=True)
    archive_path = (ROOT / config["source"]["archive"]).resolve()
    rows = read_official_rows(archive_path, config["source"]["official_csv_member"])
    official = {row["INSTANCE"]: row for row in rows}
    training_cutoff = parse_date(config["proteinmpnn_training_cutoff_exclusive"])
    split_mode = config.get("split_mode", "date_boundary")
    final_cutoff = (
        parse_date(config["temporal_final_cutoff_inclusive"])
        if split_mode == "date_boundary" else None)
    previously_scored = scored_ids([
        (ROOT / path).resolve() for path in config["exclude_scored_results"]
    ])
    prior = [metadata_record(official[record_id]) for record_id in sorted(previously_scored)]
    candidates = [
        metadata_record(row) for row in rows
        if row["INSTANCE"] not in previously_scored
        and metadata_eligible(row, config["eligibility"], training_cutoff)
    ]
    fresh, prior_exact_conflicts = exclude_conflicts(candidates, prior)
    fresh, prior_homology_conflicts = exclude_homology_conflicts(
        fresh, prior, config["homology_isolation"])
    if split_mode == "all_fresh_as_sealed_final":
        development = []
        final_candidates = fresh
        temporal_exact_conflicts = []
        temporal_homology_conflicts = []
    else:
        development = [
            record for record in fresh
            if parse_date(record["row"]["PDBdepo"]) < final_cutoff
        ]
        final_candidates = [
            record for record in fresh
            if parse_date(record["row"]["PDBdepo"]) >= final_cutoff
        ]
        final_candidates, temporal_exact_conflicts = exclude_conflicts(
            final_candidates, development)
        final_candidates, temporal_homology_conflicts = exclude_homology_conflicts(
            final_candidates, development, config["homology_isolation"])

    selected_metadata = [*development, *final_candidates]
    cif_dir = output / "cif"
    extract_cifs(archive_path, selected_metadata, cif_dir)
    processed, preprocess_failures = preprocess_records(selected_metadata, cif_dir)
    processed_by_id = {record["id"]: record for record in processed}
    annotated = {
        record_id: annotate_record(
            processed_by_id[record_id], official, config["eligibility"])
        for record_id in processed_by_id
    }
    structural_dev = [
        annotated[record["id"]] for record in development
        if record["id"] in annotated and annotated[record["id"]]["eligible"]
    ]
    structural_final = [
        annotated[record["id"]] for record in final_candidates
        if record["id"] in annotated and annotated[record["id"]]["eligible"]
    ]
    structural_final, poststructure_exact_conflicts = exclude_conflicts(
        structural_final, structural_dev)
    structural_final, poststructure_homology_conflicts = exclude_homology_conflicts(
        structural_final, structural_dev, config["homology_isolation"])
    dev_ids = {record["id"] for record in structural_dev}
    final_ids = {record["id"] for record in structural_final}
    dev_payload = [processed_by_id[record_id] for record_id in sorted(dev_ids)]
    final_payload = [processed_by_id[record_id] for record_id in sorted(final_ids)]
    minimum_development = 0 if split_mode == "all_fresh_as_sealed_final" else 20
    if len(dev_payload) < minimum_development or len(final_payload) < 20:
        raise RuntimeError(
            f"Temporal benchmark too small after structural audit: "
            f"{len(dev_payload)}/{len(final_payload)}")
    if dev_payload:
        build_lmdb(dev_payload, str(output / "development.lmdb"))
    build_lmdb(final_payload, str(output / "sealed_final.lmdb"))
    write_manifest(output / "development_manifest.csv", structural_dev)
    write_manifest(output / "sealed_final_manifest.csv", structural_final)
    audit = {
        "schema_version": 1,
        "status": "temporal development available; temporal final sealed; no model forward",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "proteinmpnn_training_cutoff_exclusive": str(training_cutoff),
        "split_mode": split_mode,
        "temporal_final_cutoff_inclusive": (
            str(final_cutoff) if final_cutoff is not None else None),
        "counts": {
            "official_rows": len(rows),
            "previously_scored_ids": len(previously_scored),
            "post_training_candidates_before_isolation": len(candidates),
            "fresh_after_prior_isolation": len(fresh),
            "development_metadata": len(development),
            "final_metadata": len(final_candidates),
            "preprocess_failures": len(preprocess_failures),
            "development_structural": len(structural_dev),
            "sealed_final_structural": len(structural_final),
        },
        "ids_sha256": {
            "development": digest_ids(structural_dev),
            "sealed_final": digest_ids(structural_final),
        },
        "conflict_counts": {
            "prior_exact": len(prior_exact_conflicts),
            "prior_homology": len(prior_homology_conflicts),
            "temporal_exact": len(temporal_exact_conflicts),
            "temporal_homology": len(temporal_homology_conflicts),
            "poststructure_exact": len(poststructure_exact_conflicts),
            "poststructure_homology": len(poststructure_homology_conflicts),
        },
        "preprocess_failures": preprocess_failures,
        "records": {
            "development": [serializable_record(record) for record in structural_dev],
            "sealed_final": [serializable_record(record) for record in structural_final],
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "counts": audit["counts"],
        "ids_sha256": audit["ids_sha256"],
        "conflict_counts": audit["conflict_counts"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
