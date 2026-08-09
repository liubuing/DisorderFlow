#!/usr/bin/env python
"""Model-free feasibility audit for revised antibody homology isolation rules."""

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

from scripts.build.materialize_multiscaffold_candidates import (  # noqa: E402
    number_variable_domains,
)
from scripts.prepare_sabdab2_external import load_training_records  # noqa: E402

AXIS_FIELDS = {
    "vh": "vh_sequence",
    "vl": "vl_sequence",
    "cdr_h3": "cdr_h3_sequence",
    "paired_cdr": "paired_cdr_sequence",
    "antigen": "antigen_sequence",
}
SEARCH_MINIMUMS = {
    "vh": 0.5,
    "vl": 0.5,
    "cdr_h3": 0.5,
    "paired_cdr": 0.5,
    "antigen": 0.3,
}


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def clean_sequence(value):
    return "".join(aa for aa in str(value).upper() if aa in "ACDEFGHIKLMNPQRSTVWY")


def write_fasta(path, rows, field, id_field):
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            sequence = clean_sequence(row.get(field, ""))
            if sequence:
                handle.write(f">{row[id_field]}\n{sequence}\n")


def annotate_reference_cdrs(references, cache_path):
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload["reference_ids_sha256"] == ids_sha256(references):
            return payload["records"]
    paired = [row for row in references if row.get("vh_sequence") and row.get("vl_sequence")]
    inputs = [{
        "instance": row["reference_id"],
        "heavy_sequence": clean_sequence(row["vh_sequence"]),
        "light_sequence": clean_sequence(row["vl_sequence"]),
    } for row in paired]
    domains = number_variable_domains(inputs)
    annotated = []
    for row in references:
        item = domains.get(row["reference_id"], {})
        heavy = item.get("H", {})
        light = item.get("L", {})
        cdrs = {**heavy.get("cdrs", {}), **light.get("cdrs", {})}
        paired_cdr = "".join(
            cdrs.get(name, "") for name in ("H1", "H2", "H3", "L1", "L2", "L3"))
        annotated.append({**row, "paired_cdr_sequence": paired_cdr})
    payload = {
        "schema_version": 1,
        "tool": {"name": "ANARCII", "version": "2.0.8", "scheme": "chothia"},
        "reference_ids_sha256": ids_sha256(references),
        "n_records": len(annotated),
        "n_paired_cdr_annotated": sum(bool(row["paired_cdr_sequence"]) for row in annotated),
        "records": annotated,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return annotated


def ids_sha256(rows):
    value = "\n".join(sorted(row["reference_id"] for row in rows))
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def easy_search(mmseqs, query, target, output, work, identity, threads):
    command = [
        mmseqs, "easy-search", str(query), str(target), str(output), str(work),
        "--min-seq-id", str(identity), "-c", "0.8", "--cov-mode", "0",
        "--format-output", "query,target,fident,qcov,tcov,evalue",
        "--threads", str(threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    rows = []
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            query_id, target_id, identity_value, qcov, tcov, evalue = line.split("\t")
            rows.append({
                "query": query_id,
                "target": target_id,
                "identity": float(identity_value),
                "query_coverage": float(qcov),
                "target_coverage": float(tcov),
                "evalue": float(evalue),
            })
    return rows, command


def component_counts(rows, self_hits, thresholds):
    ids = [row["instance"] for row in rows]
    all_axes = UnionFind(ids)
    antigen = UnionFind(ids)
    for axis, hits in self_hits.items():
        threshold = thresholds[axis]
        for hit in hits:
            if (hit["query"] not in all_axes.parent or hit["target"] not in all_axes.parent
                    or hit["query"] == hit["target"] or hit["identity"] < threshold):
                continue
            all_axes.union(hit["query"], hit["target"])
            if axis == "antigen":
                antigen.union(hit["query"], hit["target"])
    return (
        len({all_axes.find(value) for value in ids}),
        len({antigen.find(value) for value in ids}),
    )


def freeze_component_representatives(rows, self_hits, thresholds):
    ids = [row["instance"] for row in rows]
    connected = UnionFind(ids)
    for axis, hits in self_hits.items():
        for hit in hits:
            if (hit["query"] not in connected.parent or hit["target"] not in connected.parent
                    or hit["query"] == hit["target"]
                    or hit["identity"] < thresholds[axis]):
                continue
            connected.union(hit["query"], hit["target"])
    members = {}
    by_id = {row["instance"]: row for row in rows}
    for value in ids:
        members.setdefault(connected.find(value), []).append(value)
    components = []
    for index, component_members in enumerate(
            sorted((sorted(values) for values in members.values()), key=lambda values: values[0]), 1):
        representative_id = min(component_members, key=lambda value: (
            by_id[value]["resolution"] is None,
            by_id[value]["resolution"] if by_id[value]["resolution"] is not None else 999.0,
            value,
        ))
        components.append({
            "component_id": f"V2C{index:03d}",
            "members": component_members,
            "representative_id": representative_id,
            "representative": by_id[representative_id],
        })
    return components


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
        "--reference-cdr-cache",
        default="data/multiscaffold_confirmatory_v2/reference_cdr_annotations.json")
    parser.add_argument(
        "--mmseqs",
        default=("C:/biological/Metabolic model prediction/"
                 "Integrated_Yeast_MetaTwin_Deployment/tools/mmseqs2/mmseqs/bin/mmseqs.exe"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--out", default="data/multiscaffold_confirmatory_v2/feasibility_audit.json")
    parser.add_argument(
        "--holdout-out", default="data/multiscaffold_confirmatory_v2/holdout_manifest.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    candidates = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
    references = []
    for source_index, value in enumerate(args.reference_lmdb):
        for row in load_training_records(ROOT / value):
            references.append({
                "reference_id": f"r{source_index}:{row['id']}",
                "id": str(row["id"]),
                "pdb_id": str(row.get("pdb_id", row["id"])),
                "vh_sequence": clean_sequence(row.get("vh_sequence", "")),
                "vl_sequence": clean_sequence(row.get("vl_sequence", "")),
                "cdr_h3_sequence": clean_sequence(row.get("cdr_h3_sequence", "")),
                "antigen_sequence": clean_sequence(row.get("antigen_sequence", "")),
            })
    references = list({row["reference_id"]: row for row in references}.values())
    references = annotate_reference_cdrs(references, ROOT / args.reference_cdr_cache)

    reference_hits = {}
    self_hits = {}
    commands = []
    with tempfile.TemporaryDirectory(prefix="multiscaffold_v2_") as tmp:
        temporary = Path(tmp)
        for axis, field in AXIS_FIELDS.items():
            query = temporary / f"candidate_{axis}.fasta"
            target = temporary / f"reference_{axis}.fasta"
            write_fasta(query, candidates, field, "instance")
            write_fasta(target, references, field, "reference_id")
            reference_hits[axis], command = easy_search(
                args.mmseqs, query, target, temporary / f"reference_hits_{axis}.tsv",
                temporary / f"reference_work_{axis}", SEARCH_MINIMUMS[axis], args.threads)
            commands.append(command)
            self_hits[axis], command = easy_search(
                args.mmseqs, query, query, temporary / f"self_hits_{axis}.tsv",
                temporary / f"self_work_{axis}", SEARCH_MINIMUMS[axis], args.threads)
            commands.append(command)
        temporary_prefix = str(temporary)
        commands = [
            [value.replace(temporary_prefix, "<TEMP>") for value in command]
            for command in commands]

    schemes = {}
    for variable_identity in (0.8, 0.9, 0.95):
        for paired_cdr_identity in (0.6, 0.7, 0.8):
            name = f"vhvl_{int(variable_identity * 100)}_pairedcdr_{int(paired_cdr_identity * 100)}"
            thresholds = {
                "vh": variable_identity,
                "vl": variable_identity,
                "cdr_h3": 0.5,
                "paired_cdr": paired_cdr_identity,
                "antigen": 0.3,
            }
            hit_ids = {
                axis: {hit["query"] for hit in hits if hit["identity"] >= thresholds[axis]}
                for axis, hits in reference_hits.items()}
            independent = [
                row for row in candidates
                if not any(row["instance"] in hit_ids[axis] for axis in AXIS_FIELDS)]
            all_components, antigen_components = component_counts(
                independent, self_hits, thresholds) if independent else (0, 0)
            schemes[name] = {
                "thresholds": thresholds,
                "n_independent_records": len(independent),
                "n_all_axis_components": all_components,
                "n_antigen_components": antigen_components,
                "gate_passed": antigen_components >= 12,
                "excluded_by_axis": {
                    axis: len(hit_ids[axis]) for axis in AXIS_FIELDS},
                "independent_ids": sorted(row["instance"] for row in independent),
            }

    proposed_name = "vhvl_90_pairedcdr_70"
    proposed = schemes[proposed_name]
    proposed_ids = set(proposed["independent_ids"])
    proposed_rows = [row for row in candidates if row["instance"] in proposed_ids]
    frozen_components = freeze_component_representatives(
        proposed_rows, self_hits, proposed["thresholds"])
    representative_ids = [row["representative_id"] for row in frozen_components]
    representative_ids_sha256 = hashlib.sha256(
        "\n".join(sorted(representative_ids)).encode("ascii")).hexdigest()
    report = {
        "schema_version": 1,
        "status": "model_free_connectivity_feasibility_complete",
        "claim_boundary": "threshold feasibility only; no generator or scorer outputs accessed",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reference_ids_sha256": ids_sha256(references),
        "n_candidates": len(candidates),
        "n_references": len(references),
        "coverage": 0.8,
        "proposed_scheme": proposed_name,
        "proposed_gate_passed": proposed["gate_passed"],
        "decision": (
            "eligible to freeze v2 before holdout selection" if proposed["gate_passed"]
            else "proposed v2 remains infeasible; do not select a confirmatory holdout"),
        "schemes": schemes,
        "mmseqs_version": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
        "commands": commands,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    holdout = {
        "schema_version": 1,
        "status": "frozen_before_new_lineage_training_or_candidate_generation",
        "classification": "prospective_internal_holdout; not external confirmation",
        "source_feasibility_audit": args.out,
        "scheme": proposed_name,
        "thresholds": proposed["thresholds"],
        "coverage": 0.8,
        "n_reference_independent_records": len(proposed_rows),
        "n_all_axis_components": len(frozen_components),
        "n_representatives": len(representative_ids),
        "representative_ids": representative_ids,
        "representative_ids_sha256": representative_ids_sha256,
        "components": frozen_components,
        "prohibitions": [
            "no use in training, calibration, threshold selection, or early stopping",
            "no generator or scorer output before training lineage and candidate rules are frozen",
            "do not describe as a new external dataset",
        ],
    }
    holdout_path = ROOT / args.holdout_out
    holdout_path.write_text(json.dumps(holdout, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"schemes", "commands"}}, indent=2))
    print(json.dumps({name: {key: value for key, value in row.items()
                            if key != "independent_ids"}
                      for name, row in schemes.items()}, indent=2))


if __name__ == "__main__":
    main()
