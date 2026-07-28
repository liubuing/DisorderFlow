#!/usr/bin/env python3
"""Prepare a training-disjoint SAbDab2 abag external test set."""

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import lmdb


AXES = {
    "antigen": ("antigen_sequence", 0.3),
    "vh": ("vh_sequence", 0.5),
    "vl": ("vl_sequence", 0.5),
    "cdr_h3": ("cdr_h3_sequence", 0.5),
}


def clean_sequence(value):
    return "".join(char for char in str(value).upper() if char in "ACDEFGHIKLMNPQRSTVWY")


def antigen_sequence(row):
    return "".join(clean_sequence(part) for part in row.get("agresolvedseqs", "").split("/"))


def pdb_code(value):
    value = str(value).casefold().strip()
    return value[-4:] if value.startswith("pdb_") else value


def load_training_records(lmdb_path):
    ids_path = Path(f"{lmdb_path}-ids")
    with ids_path.open("rb") as handle:
        ids = pickle.load(handle)
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False)
    records = []
    with env.begin() as txn:
        for sample_id in ids:
            raw = txn.get(str(sample_id).encode())
            if raw is not None:
                records.append(pickle.loads(raw))
    env.close()
    return records


def write_fasta(path, records, field, key_field):
    count = 0
    with path.open("w", encoding="ascii") as handle:
        for record in records:
            sequence = clean_sequence(record.get(field, ""))
            if not sequence:
                continue
            handle.write(f">{record[key_field]}\n{sequence}\n")
            count += 1
    return count


def mmseqs_hits(mmseqs, query, target, output, threshold, work_dir, threads):
    command = [
        mmseqs, "easy-search", str(query), str(target), str(output), str(work_dir),
        "--min-seq-id", str(threshold), "-c", "0.8", "--cov-mode", "0",
        "--format-output", "query,target,fident,qcov,tcov,evalue", "--threads", str(threads),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"MMseqs2 failed for {query.name}: {result.stderr[-2000:]}")
    hits = {}
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                query_id, target_id, identity, qcov, tcov, evalue = line.rstrip().split("\t")
                hits.setdefault(query_id, []).append({
                    "target": target_id,
                    "identity": float(identity),
                    "query_coverage": float(qcov),
                    "target_coverage": float(tcov),
                    "evalue": float(evalue),
                })
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/sabdab2_current/splits.tar.gz")
    parser.add_argument("--train-lmdb", default="data/phase3_v5_1_pair_clustered/train.lmdb")
    parser.add_argument("--output-dir", default="data/sabdab2_abag_external")
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    archive = Path(args.archive)
    train_lmdb = Path(args.train_lmdb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which(args.mmseqs) is None:
        raise RuntimeError(f"MMseqs2 executable not found: {args.mmseqs}")

    with tarfile.open(archive, "r:gz") as tar:
        split_file = tar.extractfile("splits_final/abag_split.csv")
        if split_file is None:
            raise RuntimeError("abag_split.csv is missing from the verified archive")
        rows = list(csv.DictReader(line.decode("utf-8") for line in split_file))

    candidates = []
    for row in rows:
        agtypes = row.get("agtypes", "").upper()
        if row.get("ab_ag_split") != "test" or row.get("holo") != "True":
            continue
        if not any(kind in agtypes for kind in ("PROTEIN", "PEPTIDE")):
            continue
        item = dict(row)
        item["instance"] = row["INSTANCE"]
        item["pdb_code"] = pdb_code(row["PDB_ID"])
        item["antigen_sequence"] = antigen_sequence(row)
        item["vh_sequence"] = clean_sequence(row.get("VH_numerable_seq", "") or row.get("Hseq", ""))
        item["vl_sequence"] = clean_sequence(row.get("VL_numerable_seq", "") or row.get("Lseq", ""))
        item["cdr_h3_sequence"] = clean_sequence(row.get("CDRH3", ""))
        if item["antigen_sequence"] and item["vh_sequence"] and item["cdr_h3_sequence"]:
            candidates.append(item)

    training = load_training_records(train_lmdb)
    training_pdbs = {
        pdb_code(record.get("pdb_id", record.get("id", ""))) for record in training
    }
    exact_overlap = {item["instance"] for item in candidates if item["pdb_code"] in training_pdbs}

    all_hits = {}
    with tempfile.TemporaryDirectory(prefix="sabdab2_audit_") as tmp:
        tmp_dir = Path(tmp)
        for axis, (field, threshold) in AXES.items():
            query = tmp_dir / f"test_{axis}.fasta"
            target = tmp_dir / f"train_{axis}.fasta"
            result = tmp_dir / f"hits_{axis}.tsv"
            work = tmp_dir / f"work_{axis}"
            write_fasta(query, candidates, field, "instance")
            n_train = write_fasta(target, training, field, "id")
            all_hits[axis] = {} if n_train == 0 else mmseqs_hits(
                args.mmseqs, query, target, result, threshold, work, args.threads)

    strict_independent = []
    interface_eligible = []
    audit_rows = []
    for item in candidates:
        instance = item["instance"]
        failed_axes = [axis for axis, hits in all_hits.items() if instance in hits]
        if instance in exact_overlap:
            failed_axes.insert(0, "pdb_id")
        is_independent = not failed_axes
        audit_rows.append({
            "instance": instance,
            "pdb_code": item["pdb_code"],
            "independent": is_independent,
            "failed_axes": failed_axes,
            "best_hits": {
                axis: hits.get(instance, [])[:3] for axis, hits in all_hits.items()
            },
        })
        if is_independent:
            strict_independent.append(item)
        if not any(axis in failed_axes for axis in ("pdb_id", "antigen", "cdr_h3")):
            interface_eligible.append(item)

    cif_dir = output_dir / "cif"
    cif_dir.mkdir(exist_ok=True)
    extracted = 0
    wanted = {
        f"splits_final/{item['instance']}.cif": item for item in interface_eligible
    }
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            item = wanted.get(member.name)
            if item is None:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            destination = cif_dir / f"{item['instance']}.cif"
            with destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted += 1

    selected_fields = [
        "instance", "pdb_code", "Hchain", "Lchain", "agchains", "agtypes",
        "antigen_sequence", "vh_sequence", "vl_sequence", "cdr_h3_sequence",
    ]
    with (output_dir / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(interface_eligible)

    failed_counts = Counter(axis for row in audit_rows for axis in row["failed_axes"])
    report = {
        "source": str(archive),
        "source_split": "splits_final/abag_split.csv:ab_ag_split=test",
        "selection": "holo protein/peptide antigen with VH, CDR-H3 and antigen sequence",
        "training_lmdb": str(train_lmdb),
        "thresholds": {
            axis: {"minimum_identity": threshold, "coverage": 0.8, "coverage_mode": 0}
            for axis, (_, threshold) in AXES.items()
        },
        "official_rows": len(rows),
        "candidate_test_complexes": len(candidates),
        "training_complexes": len(training),
        "strict_four_axis_independent_complexes": len(strict_independent),
        "interface_eligible_complexes": len(interface_eligible),
        "interface_eligibility": {
            "isolated": ["pdb_id", "antigen", "cdr_h3"],
            "not_isolated": ["vh", "vl"],
            "classification": "secondary external interface test, not a fully antibody-disjoint blind test",
        },
        "extracted_cif": extracted,
        "excluded_by_axis": dict(failed_counts),
        "audit": audit_rows,
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "audit"}, indent=2))
    if extracted != len(interface_eligible):
        raise RuntimeError(
            f"Only extracted {extracted}/{len(interface_eligible)} interface-eligible CIF files")


if __name__ == "__main__":
    main()
