#!/usr/bin/env python3
"""Build an instance-keyed authoritative IMGT structural parent cache."""

import argparse
import csv
import gzip
import io
import json
import pickle
import sys
import tarfile
from pathlib import Path

import lmdb
import torch
from Bio.PDB import MMCIFParser


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.datasets.statecontrast_structural import (
    assign_official_imgt_cdrs,
    require_full_antigen_mapping,
)
from disorderflow.data_factory import sha256_file
from disorderflow.utils.protein.parsers import parse_biopython_structure


def chain_ids(value):
    return [item.strip() for item in str(value).split("/") if item.strip() and item.strip() != "+"]


def observed_records(records_dir, split, limit):
    manifest = json.loads((records_dir / "manifest.json").read_text(encoding="utf-8"))
    records = []
    for shard in manifest["shards"]:
        with gzip.open(records_dir / shard["path"], "rt", encoding="ascii") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] != split or record["lineage"]["operation"] != "observed_bound":
                    continue
                records.append(record)
                if limit is not None and len(records) >= limit:
                    return manifest, records
    return manifest, records


def merge_chains(chains):
    if len(chains) == 1:
        return chains[0]
    merged = {}
    for key in chains[0]:
        values = [chain[key] for chain in chains]
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = sum(values, start=[])
        else:
            merged[key] = values[0]
    return merged


def parse_parent(source, row, record):
    text = io.StringIO(source.read().decode("utf-8"))
    model = MMCIFParser(QUIET=True).get_structure(row["INSTANCE"], text)[0]
    heavy_id = row.get("Hchain")
    light_id = row.get("Lchain")
    antigen_ids = chain_ids(row.get("agchains", ""))
    if not heavy_id or not light_id or heavy_id not in model or light_id not in model:
        raise ValueError("Declared heavy/light chains are absent")
    if not antigen_ids or any(chain_id not in model for chain_id in antigen_ids):
        raise ValueError("Declared antigen chains are absent")
    if len({heavy_id, light_id, *antigen_ids}) != 2 + len(antigen_ids):
        raise ValueError("Antibody and antigen chain roles overlap")

    heavy, _ = parse_biopython_structure(model[heavy_id])
    light, _ = parse_biopython_structure(model[light_id])
    antigen_parts = [parse_biopython_structure(model[chain_id])[0] for chain_id in antigen_ids]
    antigen = merge_chains(antigen_parts)

    antibody = record["antibody"]
    heavy_cdrs = {name: antibody["cdrs"][name] for name in ("H1", "H2", "H3")}
    light_cdrs = {name: antibody["cdrs"][name] for name in ("L1", "L2", "L3")}
    heavy_spans = {name: antibody["cdr_spans"][name] for name in heavy_cdrs}
    light_spans = {name: antibody["cdr_spans"][name] for name in light_cdrs}
    cdr_indices = {}
    cdr_indices.update(assign_official_imgt_cdrs(
        heavy, antibody["vh"], heavy_cdrs, heavy_spans))
    cdr_indices.update(assign_official_imgt_cdrs(
        light, antibody["vl"], light_cdrs, light_spans))
    antigen["cdr_flag"] = torch.zeros_like(antigen["aa"])
    antigen_indices = require_full_antigen_mapping(antigen, record["antigen"]["sequence"])

    for chain in (heavy, light, antigen):
        if not torch.isfinite(chain["pos_heavyatom"]).all():
            raise ValueError("Non-finite atom coordinates")
    return {
        "id": row["INSTANCE"],
        "pdb_id": row["PDB_ID"],
        "official_instance": row["INSTANCE"],
        "heavy": heavy,
        "light": light,
        "antigen": antigen,
        "factory_mapping": {
            "scheme": "IMGT",
            "source": "SAbDab2 official abag_split.csv",
            "cdr_indices": cdr_indices,
            "antigen_indices": antigen_indices,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", default="data/statecontrast_factory_v2_1")
    parser.add_argument("--archive", default="data/sabdab2_current/splits.tar.gz")
    parser.add_argument("--output", default="data/statecontrast_structural_smoke/parents.lmdb")
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--max-success", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    records_dir = ROOT / args.records_dir
    archive = ROOT / args.archive
    output = ROOT / args.output
    ids_path = Path(str(output) + "-ids")
    progress_path = output.with_suffix(".progress.json")
    if not args.resume and (output.exists() or ids_path.exists()):
        raise FileExistsError(f"Refusing to overwrite structural cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, parents = observed_records(records_dir, args.split, args.candidate_limit)
    by_instance = {record["provenance"]["source_record_id"]: record for record in parents}

    with tarfile.open(archive, "r:gz") as tar:
        split_file = tar.extractfile("splits_final/abag_split.csv")
        if split_file is None:
            raise FileNotFoundError("SAbDab2 abag_split.csv is absent")
        rows_by_key = {}
        for row in csv.DictReader(io.TextIOWrapper(split_file, encoding="utf-8")):
            key = row["INSTANCE"].casefold()
            if key in by_instance:
                rows_by_key.setdefault(key, row)
        rows = {row["INSTANCE"]: row for row in rows_by_key.values()}

    env = lmdb.open(str(output), subdir=False, map_size=4 * 1024**3)
    with env.begin() as txn:
        successful = sorted(key.decode() for key, _ in txn.cursor())
    failures = {}
    if args.resume and progress_path.exists():
        previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous_progress.get("factory_schema") != manifest["schema_version"]:
            raise ValueError("Resume progress schema does not match current factory")
        failures = dict(previous_progress.get("failures", {}))
    if args.retry_failures:
        failures = {}

    def write_progress(status="running"):
        ids_path.write_bytes(pickle.dumps(successful))
        progress = {
            "status": status,
            "factory_schema": manifest["schema_version"],
            "split": args.split,
            "candidate_limit": args.candidate_limit,
            "requested_successes": args.max_success,
            "successful_parents": len(successful),
            "failed_parents": len(failures),
            "parent_ids": successful,
            "failures": failures,
            "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        }
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")
        return progress

    write_progress()
    completed = set(successful)
    if not args.retry_failures:
        completed.update(key.casefold() for key in failures)
    wanted = {
        f"splits_final/{instance}.cif": (instance, rows.get(instance))
        for instance in rows if instance.casefold() not in completed
    }
    examined = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if len(successful) >= args.max_success:
                break
            selected = wanted.get(member.name)
            if selected is None:
                continue
            examined += 1
            instance, row = selected
            source = tar.extractfile(member)
            if source is None or row is None:
                failures[instance] = "missing archive row or CIF stream"
                continue
            try:
                parent = parse_parent(source, row, by_instance[instance.casefold()])
                with env.begin(write=True) as txn:
                    if txn.get(instance.casefold().encode()) is not None:
                        raise ValueError("duplicate structural parent key")
                    txn.put(instance.casefold().encode(), pickle.dumps(parent))
                successful.append(instance.casefold())
                successful.sort()
            except Exception as error:
                failures[instance] = f"{type(error).__name__}: {error}"
            if examined % 50 == 0:
                write_progress()
                print(json.dumps({
                    "examined_this_run": examined,
                    "successful_parents": len(successful),
                    "failed_parents": len(failures),
                }), flush=True)
    env.close()
    report = write_progress("pass" if successful else "fail")
    report.update({
        "examined_this_run": examined,
        "parent_lmdb_sha256": sha256_file(output),
        "parent_ids_sha256": sha256_file(ids_path),
    })
    output.with_suffix(".build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not successful:
        raise RuntimeError("No authoritative structural parents were built")


if __name__ == "__main__":
    main()
