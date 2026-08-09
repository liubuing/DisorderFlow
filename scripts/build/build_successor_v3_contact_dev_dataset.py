#!/usr/bin/env python
"""Build component-split LMDBs for exposed successor-v3 contact development."""

from __future__ import annotations

import argparse
import json
import lmdb
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from disorderflow.utils.data import PaddingCollate  # noqa: E402
from disorderflow.utils.protein.constants import Fragment  # noqa: E402
from scripts.audit_successor_v3_exploratory import (  # noqa: E402
    DEFAULT_MMSEQS,
    mmseqs_version,
    run_search,
)
from scripts.audit_successor_v3_isolation import (  # noqa: E402
    AXES,
    connected_components,
    set_sha256,
    sha256_file,
    write_fasta,
)
from scripts.evaluate_successor_v3_confirmatory import record_batch  # noqa: E402


AA = "ACDEFGHIKLMNPQRSTVWY"


def assign_folds(components, by_id, folds):
    sizes = [0] * folds
    assignment = {}
    ordered = sorted(components, key=lambda members: (
        -sum(len(by_id[value]["cdr_h3_sequence"]) for value in members), members[0]))
    for members in ordered:
        fold = min(range(folds), key=lambda value: (sizes[value], value))
        for identifier in members:
            assignment[identifier] = fold
        sizes[fold] += sum(len(by_id[value]["cdr_h3_sequence"]) for value in members)
    return assignment, sizes


def select_donors(records, component_by_id):
    donors = {}
    for record in records:
        options = [other for other in records
                   if component_by_id[other["instance"]] != component_by_id[record["instance"]]
                   and other["antigen_sequence"] != record["antigen_sequence"]]
        donors[record["instance"]] = min(options, key=lambda other: (
            abs(len(other["antigen_sequence"]) - len(record["antigen_sequence"])),
            other["instance"]))["instance"]
    return donors


def attach_hard_mismatch(batch, donor_sequence):
    mismatch = batch["aa"].clone()
    antigen = batch["fragment_type"] == int(Fragment.Antigen)
    indices = torch.where(antigen)[0]
    donor = torch.tensor([AA.index(value) for value in donor_sequence], dtype=mismatch.dtype)
    mapped = torch.linspace(0, len(donor) - 1, len(indices)).round().long()
    mismatch[indices] = donor[mapped]
    batch["hard_antigen_mismatch_aa"] = mismatch
    batch["hard_antigen_mismatch_valid"] = torch.tensor(
        not torch.equal(mismatch[indices], batch["aa"][indices]))
    return batch


def write_lmdb(path, rows):
    working = path.with_suffix(path.suffix + ".working")
    environment = lmdb.open(str(working), subdir=False, map_size=8 * 1024 ** 3)
    with environment.begin(write=True) as transaction:
        for index, row in enumerate(rows):
            transaction.put(f"{index:08d}".encode(), pickle.dumps(row))
    environment.sync()
    environment.close()
    source = lmdb.open(str(working), subdir=False, readonly=True, lock=False)
    source.copy(str(path), compact=True)
    source.close()
    working.unlink()
    lock = working.with_name(working.name + "-lock")
    if lock.exists():
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--eval-fold", type=int, default=4)
    parser.add_argument("--mmseqs-wsl", default=DEFAULT_MMSEQS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    manifest_path, output_dir = ROOT / args.manifest, ROOT / args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite contact development dataset: {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    by_id = {row["instance"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Structural manifest instance IDs are not unique")
    version, _ = mmseqs_version(args.mmseqs_wsl)

    hits = {}
    with tempfile.TemporaryDirectory(
            prefix="successor_v3_contact_dev_", ignore_cleanup_errors=True) as name:
        temporary = Path(name)
        for axis, (field, threshold) in AXES.items():
            fasta = temporary / f"{axis}.fasta"
            write_fasta(fasta, records, field, "instance")
            hits[axis], _ = run_search(
                args.mmseqs_wsl, fasta, fasta, temporary / f"{axis}.tsv",
                temporary / f"work_{axis}", threshold, args.threads)
    components = connected_components(list(by_id), hits)
    assignment, fold_h3_residues = assign_folds(components, by_id, args.folds)
    component_by_id = {
        member: f"SV3D{index:03d}"
        for index, members in enumerate(components, 1) for member in members}
    donors = select_donors(records, component_by_id)

    output_dir.mkdir(parents=True)
    try:
        split_rows = {"train": [], "eval": []}
        split_records = {"train": [], "eval": []}
        exclusions = []
        for index, record in enumerate(sorted(records, key=lambda row: row["instance"]), 1):
            try:
                batch = attach_hard_mismatch(
                    record_batch(record),
                    by_id[donors[record["instance"]]]["antigen_sequence"])
            except Exception as error:  # noqa: BLE001
                exclusions.append({
                    "instance": record["instance"],
                    "reason": f"record_batch_failed: {error}",
                })
                print(f"Excluded {record['instance']}: {error}", flush=True)
                continue
            expected_h3 = len(record["cdr_h3_sequence"])
            if int(batch["generate_flag"].sum()) != expected_h3:
                raise ValueError(f"H3 mapping mismatch for {record['instance']}")
            split = "eval" if assignment[record["instance"]] == args.eval_fold else "train"
            split_rows[split].append(batch)
            split_records[split].append({
                "instance": record["instance"],
                "pdb_id": record["pdb_id"],
                "component_id": component_by_id[record["instance"]],
                "fold": assignment[record["instance"]],
                "h3_length": expected_h3,
                "antigen_length": len(record["antigen_sequence"]),
                "hard_negative_instance": donors[record["instance"]],
            })
            print(f"Prepared {index}/{len(records)}", flush=True)
        for split in ("train", "eval"):
            write_lmdb(output_dir / f"{split}.lmdb", split_rows[split])
        with (output_dir / "meta_indices.pkl").open("xb") as handle:
            pickle.dump({3: list(range(len(split_rows["train"])))}, handle)
        PaddingCollate()([split_rows["train"][0], split_rows["eval"][0]])
        payload = {
            "schema_version": 1,
            "status": "exposed_contact_development_dataset_frozen",
            "classification": "retrospective exposed development; not confirmatory or external",
            "source_manifest": args.manifest.as_posix(),
            "source_manifest_sha256": sha256_file(manifest_path),
            "mmseqs_version": version,
            "thresholds": {axis: threshold for axis, (_field, threshold) in AXES.items()},
            "coverage": 0.8,
            "folds": args.folds,
            "eval_fold": args.eval_fold,
            "fold_h3_residues": fold_h3_residues,
            "counts": {
                "records": len(records), "components": len(components),
                "preprocessing_excluded": len(exclusions),
                "train_records": len(split_rows["train"]),
                "eval_records": len(split_rows["eval"]),
            },
            "normalized_sets_sha256": {
                "train_ids": set_sha256(row["instance"] for row in split_records["train"]),
                "eval_ids": set_sha256(row["instance"] for row in split_records["eval"]),
            },
            "components": [{
                "component_id": f"SV3D{index:03d}", "members": members,
                "fold": assignment[members[0]],
            } for index, members in enumerate(components, 1)],
            "records": split_records["train"] + split_records["eval"],
            "exclusions": exclusions,
        }
        with (output_dir / "manifest.json").open("x", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(json.dumps(payload["counts"], indent=2))
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
