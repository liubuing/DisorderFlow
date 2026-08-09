#!/usr/bin/env python
"""Run the model-free successor-v3 confirmatory isolation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
AXES = {
    "vh": ("vh_sequence", 0.90),
    "vl": ("vl_sequence", 0.90),
    "paired_cdr": ("paired_cdr_sequence", 0.70),
    "h3": ("cdr_h3_sequence", 0.50),
    "antigen": ("antigen_sequence", 0.30),
}
COVERAGE = 0.80
COVERAGE_MODE = 0
MINIMUM_COMPONENTS = 12


def normalize_pdb(value) -> str:
    text = str(value or "").casefold().strip()
    if text.startswith("pdb_"):
        text = text[4:]
    if len(text) > 4 and text.startswith("0"):
        text = text.lstrip("0")
    if len(text) > 4:
        text = text.split("_", 1)[0].split("-", 1)[0]
    if len(text) != 4 or not text[0].isdigit() or not text.isalnum():
        raise ValueError(f"Invalid PDB ID: {value!r}")
    return text


def clean_sequence(value) -> str:
    return "".join(char for char in str(value).upper() if char in AA)


def exact_pdb_exclusion(candidates: list[dict], exposed_pdb_ids) -> tuple[list[dict], list[dict]]:
    exposed = {normalize_pdb(value) for value in exposed_pdb_ids}
    eligible, excluded = [], []
    for row in sorted(candidates, key=lambda item: str(item["instance"])):
        pdb_id = normalize_pdb(row.get("pdb_id", row["instance"]))
        (excluded if pdb_id in exposed else eligible).append(row)
    return eligible, excluded


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def identity_fraction(value) -> float:
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def connected_components(candidate_ids, hits_by_axis: dict[str, list[dict]]) -> list[list[str]]:
    ids = sorted(set(candidate_ids))
    connected = UnionFind(ids)
    for axis in sorted(hits_by_axis):
        threshold = AXES[axis][1]
        for hit in hits_by_axis[axis]:
            left, right = hit["query"], hit["target"]
            if (left in connected.parent and right in connected.parent and left != right
                    and identity_fraction(hit["identity"]) >= threshold):
                connected.union(left, right)
    groups = {}
    for value in ids:
        groups.setdefault(connected.find(value), []).append(value)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])


def resolution_rank(value) -> tuple[bool, float]:
    if value is None:
        return True, math.inf
    return False, float(value)


def select_representative(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: (
        -int(row["n_contacting_h3_positions"]),
        -int(row["n_h3_antigen_residue_contacts"]),
        *resolution_rank(row.get("resolution")),
        str(row["instance"]),
    ))


def select_donor(record: dict, representatives: list[dict], component_by_id: dict[str, str]) -> dict:
    record_component = component_by_id[record["instance"]]
    options = [row for row in representatives
               if component_by_id[row["instance"]] != record_component]
    if not options:
        raise ValueError("A donor requires a different component")
    return min(options, key=lambda row: (
        abs(len(clean_sequence(row["antigen_sequence"]))
            - len(clean_sequence(record["antigen_sequence"]))),
        str(row["instance"]),
    ))


def write_json_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_fasta(path: Path, rows: list[dict], field: str, id_field: str) -> int:
    count = 0
    with path.open("x", encoding="ascii", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: str(item[id_field])):
            sequence = clean_sequence(row.get(field, ""))
            if sequence:
                handle.write(f">{row[id_field]}\n{sequence}\n")
                count += 1
    return count


def run_mmseqs_search(mmseqs: str, query: Path, target: Path, output: Path,
                      work: Path, threshold: float, threads: int) -> tuple[list[dict], list[str]]:
    command = [
        mmseqs, "easy-search", str(query), str(target), str(output), str(work),
        "--min-seq-id", str(threshold), "-c", str(COVERAGE),
        "--cov-mode", str(COVERAGE_MODE),
        "--format-output", "query,target,fident,qcov,tcov,evalue",
        "--threads", str(threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"MMseqs2 failed: {completed.stderr[-2000:]}")
    hits = []
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            query_id, target_id, identity, qcov, tcov, evalue = line.split("\t")
            hits.append({
                "query": query_id,
                "target": target_id,
                "identity": float(identity),
                "query_coverage": float(qcov),
                "target_coverage": float(tcov),
                "evalue": float(evalue),
            })
    return hits, command


def mmseqs_version(mmseqs: str) -> tuple[str, list[str]]:
    command = [mmseqs, "version"]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Cannot obtain MMseqs2 version: {completed.stderr[-2000:]}")
    version = (completed.stdout or completed.stderr).strip()
    if not version:
        raise RuntimeError("MMseqs2 returned an empty version")
    return version, command


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_sha256(values) -> str:
    return hashlib.sha256("\n".join(sorted(set(values))).encode("ascii")).hexdigest()


def validate_candidates(records: list[dict]) -> None:
    required = {
        "instance", "pdb_id", "vh_sequence", "vl_sequence", "paired_cdr_sequence",
        "cdr_h3_sequence", "antigen_sequence", "n_contacting_h3_positions",
        "n_h3_antigen_residue_contacts", "resolution",
    }
    ids = []
    for row in records:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Candidate {row.get('instance')} lacks fields: {missing}")
        ids.append(str(row["instance"]))
        for field, _threshold in AXES.values():
            if not clean_sequence(row[field]):
                raise ValueError(f"Candidate {row['instance']} has empty {field}")
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate instance IDs are not unique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    candidate_path = ROOT / args.candidate_manifest
    reference_path = ROOT / args.reference_manifest
    out_path = ROOT / args.out
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen audit: {out_path}")

    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference_payload.get("status") != "frozen_successor_v3_confirmatory_reference_union":
        raise ValueError("Reference manifest is not a frozen successor-v3 reference union")
    candidates = candidate_payload["records"]
    references = reference_payload["records"]
    validate_candidates(candidates)
    exact_eligible, exact_excluded = exact_pdb_exclusion(
        candidates, reference_payload["exact_exposed_pdb_ids"])
    version, version_command = mmseqs_version(args.mmseqs)

    reference_hits, self_hits, commands = {}, {}, []
    with tempfile.TemporaryDirectory(prefix="successor_v3_isolation_") as temporary_name:
        temporary = Path(temporary_name)
        for axis, (field, threshold) in AXES.items():
            query = temporary / f"candidate_{axis}.fasta"
            target = temporary / f"reference_{axis}.fasta"
            write_fasta(query, exact_eligible, field, "instance")
            n_references = write_fasta(target, references, field, "reference_id")
            if n_references and exact_eligible:
                reference_hits[axis], command = run_mmseqs_search(
                    args.mmseqs, query, target, temporary / f"reference_{axis}.tsv",
                    temporary / f"reference_work_{axis}", threshold, args.threads)
                commands.append(command)
            else:
                reference_hits[axis] = []
            if exact_eligible:
                self_hits[axis], command = run_mmseqs_search(
                    args.mmseqs, query, query, temporary / f"candidate_{axis}.tsv",
                    temporary / f"candidate_work_{axis}", threshold, args.threads)
                commands.append(command)
            else:
                self_hits[axis] = []
        temporary_prefix = str(temporary)
        commands = [[value.replace(temporary_prefix, "<TEMP>") for value in command]
                    for command in commands]

    failed_by_id = {}
    best_hits = {}
    for axis, hits in reference_hits.items():
        for hit in sorted(hits, key=lambda row: (
                row["query"], -identity_fraction(row["identity"]), row["target"])):
            failed_by_id.setdefault(hit["query"], set()).add(axis)
            if len(best_hits.setdefault((hit["query"], axis), [])) < 3:
                best_hits[(hit["query"], axis)].append(hit)
    independent = [row for row in exact_eligible if row["instance"] not in failed_by_id]
    components = connected_components(
        [row["instance"] for row in independent], self_hits) if independent else []
    gate_passed = len(components) >= MINIMUM_COMPONENTS

    by_id = {row["instance"]: row for row in independent}
    component_by_id = {
        member: f"SV3C{index:03d}"
        for index, members in enumerate(components, 1) for member in members
    }
    representatives = [select_representative([by_id[value] for value in members])
                       for members in components]
    frozen_components = []
    for index, (members, representative) in enumerate(zip(components, representatives), 1):
        donor = (select_donor(representative, representatives, component_by_id)
                 if len(components) >= 2 else None)
        frozen_components.append({
            "component_id": f"SV3C{index:03d}",
            "members": members,
            "representative_id": representative["instance"],
            "donor_id": donor["instance"] if donor else None,
            "donor_component_id": (
                component_by_id[donor["instance"]] if donor else None),
        })

    exact_ids = {row["instance"] for row in exact_excluded}
    audit_rows = []
    for row in sorted(candidates, key=lambda item: item["instance"]):
        axes = sorted(failed_by_id.get(row["instance"], set()))
        if row["instance"] in exact_ids:
            axes.insert(0, "pdb_id")
        audit_rows.append({
            "instance": row["instance"],
            "pdb_id": normalize_pdb(row["pdb_id"]),
            "independent": not axes,
            "failed_axes": axes,
            "best_reference_hits": {
                axis: best_hits.get((row["instance"], axis), []) for axis in AXES},
        })
    report = {
        "schema_version": 1,
        "status": ("frozen_model_free_successor_v3_isolation_audit"
                   if gate_passed else "frozen_model_free_feasibility_failure"),
        "claim_boundary": "sequence and structure metadata only; no model or result imports",
        "inputs": {
            "candidate_manifest": args.candidate_manifest.as_posix(),
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "reference_manifest": args.reference_manifest.as_posix(),
            "reference_manifest_sha256": sha256_file(reference_path),
        },
        "normalized_sets_sha256": {
            "candidate_ids": set_sha256(row["instance"] for row in candidates),
            "candidate_pdb_ids": set_sha256(normalize_pdb(row["pdb_id"]) for row in candidates),
            "independent_ids": set_sha256(row["instance"] for row in independent),
            "representative_ids": set_sha256(row["instance"] for row in representatives),
        },
        "thresholds": {
            axis: {"minimum_identity": threshold, "minimum_coverage": COVERAGE,
                   "coverage_mode": COVERAGE_MODE}
            for axis, (_field, threshold) in AXES.items()
        },
        "mmseqs": {"version": version, "version_command": version_command},
        "commands": commands,
        "counts": {
            "candidates": len(candidates),
            "exact_pdb_excluded": len(exact_excluded),
            "reference_homology_excluded": len(failed_by_id),
            "independent": len(independent),
            "components": len(components),
            "representatives": len(representatives),
        },
        "excluded_by_axis": dict(sorted(Counter(
            axis for axes in failed_by_id.values() for axis in axes).items())),
        "minimum_required_components": MINIMUM_COMPONENTS,
        "gate_passed": gate_passed,
        "decision": ("eligible_for_one_shot_confirmatory_evaluation"
                     if gate_passed
                     else "insufficient_independent_components; model_access_forbidden"),
        "components": frozen_components,
        "representatives": representatives,
        "audit": audit_rows,
    }
    write_json_once(out_path, report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
