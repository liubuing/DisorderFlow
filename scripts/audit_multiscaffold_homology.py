#!/usr/bin/env python
"""Audit fresh multi-scaffold candidates against all structural model exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_sabdab2_external import load_training_records  # noqa: E402

AXES = {
    "vh": ("vh_sequence", 0.5),
    "vl": ("vl_sequence", 0.5),
    "cdr_h3": ("cdr_h3_sequence", 0.5),
    "antigen": ("antigen_sequence", 0.3),
}


def clean_sequence(value):
    return "".join(aa for aa in str(value).upper() if aa in "ACDEFGHIKLMNPQRSTVWY")


def write_fasta(path, rows, field, id_field):
    count = 0
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            sequence = clean_sequence(row.get(field, ""))
            if not sequence:
                continue
            handle.write(f">{row[id_field]}\n{sequence}\n")
            count += 1
    return count


def easy_search(mmseqs, query, target, output, work, identity, threads):
    command = [
        mmseqs, "easy-search", str(query), str(target), str(output), str(work),
        "--min-seq-id", str(identity), "-c", "0.8", "--cov-mode", "0",
        "--format-output", "query,target,fident,qcov,tcov,evalue",
        "--threads", str(threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 search failed: {completed.stderr[-2000:]}")
    hits = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            query_id, target_id, identity_value, qcov, tcov, evalue = line.split("\t")
            hits.setdefault(query_id, []).append({
                "target": target_id,
                "identity_percent": float(identity_value),
                "query_coverage": float(qcov),
                "target_coverage": float(tcov),
                "evalue": float(evalue),
            })
    return hits, command


def cluster_antigens(mmseqs, rows, temporary, threads):
    fasta = temporary / "eligible_antigens.fasta"
    write_fasta(fasta, rows, "antigen_sequence", "instance")
    prefix = temporary / "antigen_components"
    work = temporary / "cluster_work"
    command = [
        mmseqs, "easy-cluster", str(fasta), str(prefix), str(work),
        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
        "--cluster-mode", "2", "--threads", str(threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 clustering failed: {completed.stderr[-2000:]}")
    clusters = {}
    for line in Path(f"{prefix}_cluster.tsv").read_text(encoding="utf-8").splitlines():
        representative, member = line.split("\t")
        clusters[member] = representative
    return clusters, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="data/multiscaffold_confirmatory_v1/structural_manifest.json")
    parser.add_argument(
        "--reference-lmdb", action="append", default=[
            "data/phase3_v5_1_pair_clustered/train.lmdb",
            "data/sabdab2_abag_project_pool_v2/all.lmdb",
            "data/peptide_h3_temporal_split_v3/sealed_final.lmdb",
        ])
    parser.add_argument(
        "--mmseqs",
        default=("C:/biological/Metabolic model prediction/"
                 "Integrated_Yeast_MetaTwin_Deployment/tools/mmseqs2/mmseqs/bin/mmseqs.exe"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--out", default="data/multiscaffold_confirmatory_v1/homology_audit.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    candidates = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    references = []
    for source_index, value in enumerate(args.reference_lmdb):
        for row in load_training_records(ROOT / value):
            references.append({**row, "reference_id": f"r{source_index}:{row['id']}"})
    references = list({row["reference_id"]: row for row in references}.values())

    all_hits = {}
    commands = []
    with tempfile.TemporaryDirectory(prefix="multiscaffold_homology_") as tmp:
        temporary = Path(tmp)
        for axis, (field, identity) in AXES.items():
            query = temporary / f"query_{axis}.fasta"
            target = temporary / f"reference_{axis}.fasta"
            output = temporary / f"hits_{axis}.tsv"
            write_fasta(query, candidates, field, "instance")
            write_fasta(target, references, field, "reference_id")
            all_hits[axis], command = easy_search(
                args.mmseqs, query, target, output, temporary / f"work_{axis}",
                identity, args.threads)
            commands.append(command)

        audit_rows = []
        independent = []
        for row in candidates:
            failed_axes = [axis for axis, hits in all_hits.items() if row["instance"] in hits]
            audit = {
                "instance": row["instance"],
                "pdb_id": row["pdb_id"],
                "independent": not failed_axes,
                "failed_axes": failed_axes,
                "best_hits": {
                    axis: all_hits[axis].get(row["instance"], [])[:3] for axis in AXES},
            }
            audit_rows.append(audit)
            if not failed_axes:
                independent.append(row)
        components, cluster_command = cluster_antigens(
            args.mmseqs, independent, temporary, args.threads) if independent else ({}, None)

    for row in independent:
        row["antigen_component"] = components[row["instance"]]
    representatives = []
    by_component = {}
    for row in independent:
        by_component.setdefault(row["antigen_component"], []).append(row)
    for _component, rows in sorted(by_component.items()):
        selected = min(rows, key=lambda row: (
            row["resolution"] is None,
            row["resolution"] if row["resolution"] is not None else 999.0,
            row["instance"],
        ))
        representatives.append({**selected, "component_size": len(rows)})

    minimum = 12
    passed = len(by_component) >= minimum
    failed_by_axis = Counter(
        axis for row in audit_rows for axis in row["failed_axes"])
    report = {
        "schema_version": 1,
        "status": "homology_audit_complete",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "thresholds": {
            axis: {"minimum_identity": identity, "minimum_coverage": 0.8, "coverage_mode": 0}
            for axis, (_, identity) in AXES.items()},
        "reference_lmdbs": args.reference_lmdb,
        "n_reference_records": len(references),
        "n_structural_candidates": len(candidates),
        "n_strict_four_axis_independent": len(independent),
        "n_antigen_components": len(by_component),
        "excluded_by_axis": dict(failed_by_axis),
        "minimum_required_antigen_components": minimum,
        "confirmatory_panel_gate_passed": passed,
        "decision": (
            "freeze representatives before generation" if passed
            else "insufficient independent antigen components; do not run confirmatory generation"),
        "mmseqs_version": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
        "commands": commands,
        "cluster_command": cluster_command,
        "representatives": representatives,
        "independent_records": independent,
        "audit": audit_rows,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"representatives", "independent_records", "audit"}}, indent=2))


if __name__ == "__main__":
    main()
